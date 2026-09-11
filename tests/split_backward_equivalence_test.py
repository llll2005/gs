"""The split backward must not change model gradients.

`CityGSV2MetricsModule.get_train_metrics` splits the objective while densification is running:

    loss       = lambda*(1 - ssim) + dist + normal + d_reg
    extra_loss = (1 - lambda) * rgb_diff

`gaussian_splatting.py` then backwards them separately with `retain_graph=True`, so that the
viewspace gradient after the first backward is the SSIM-only one -- the signal gradient-based ADC
wants (CityGaussianV2's DGD).

MCMC does not read the viewspace gradient at all; `mcmc_2dgs_density_controller.py:58` says so in
as many words. For MCMC configs the split therefore buys nothing and costs an extra backward plus a
retained graph, and the retained graph is the "backward activation" term of our VRAM peak model on
a 6GB card.

Skipping it is only safe if the two paths produce identical parameter gradients. They do -- the
split is a partition of the same sum -- but "obviously equivalent" is exactly what was said about
the depth loss before it silently poisoned a week of runs, so it is asserted here instead.
"""
import unittest

import torch


class SplitBackwardEquivalenceTest(unittest.TestCase):
    """`backward(a) + backward(b)` vs `backward(a + b)` on shared parameters."""

    def _objective(self, p, lam=0.2):
        """Stand-ins for the four terms, all sharing one parameter."""
        rgb = (p ** 2).mean()
        ssim = torch.sigmoid(p).mean()
        dist = (p.abs()).mean()
        normal = (p ** 3).mean()
        d_reg = torch.relu(p).mean()
        split_loss = lam * (1.0 - ssim) + dist + normal + d_reg
        extra_loss = (1.0 - lam) * rgb
        return split_loss, extra_loss

    def test_gradients_match(self):
        torch.manual_seed(0)
        init = torch.randn(64, 3)

        a = init.clone().requires_grad_(True)
        split_loss, extra_loss = self._objective(a)
        split_loss.backward(retain_graph=True)
        extra_loss.backward()

        b = init.clone().requires_grad_(True)
        split_loss_b, extra_loss_b = self._objective(b)
        (split_loss_b + extra_loss_b).backward()

        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_loss_value_matches_unsplit_branch(self):
        """The densify-phase branch and the post-densify branch optimise the same objective.

        Post-densify, `metrics["loss"]` is the parent's
        `(1-lambda)*rgb + lambda*(1-ssim) + dist + normal`, plus `d_reg`. During densify the metric
        rebuilds it as `lambda*(1-ssim) + dist + normal + d_reg` with `(1-lambda)*rgb` moved out.
        If those two ever stop summing to the same thing, the objective silently changes at
        `densify_until_iter` and every curve has a step in it.
        """
        torch.manual_seed(0)
        p = torch.randn(64, 3)
        lam = 0.2
        rgb = (p ** 2).mean()
        ssim = torch.sigmoid(p).mean()
        dist, normal, d_reg = p.abs().mean(), (p ** 3).mean(), torch.relu(p).mean()

        during = (lam * (1.0 - ssim) + dist + normal + d_reg) + ((1.0 - lam) * rgb)
        after = ((1.0 - lam) * rgb + lam * (1.0 - ssim) + dist + normal) + d_reg

        torch.testing.assert_close(during, after)


class ViewspaceGradConsumersTest(unittest.TestCase):
    """Which density controllers actually read the viewspace gradient.

    The skip is conditioned on this, so if a controller starts reading the gradient without
    declaring it, the skip would silently feed it zeros. This pins the current set.
    """

    def test_mcmc_DOES_read_viewspace_grad_since_absgrad(self):
        """⚠ 2026-09-11 翻轉：這個測試原本斷言「MCMC 不讀 viewspace grad」，而且一直綠燈 ——
        因為它用**字串比對**找 `outputs["viewspace_points"].grad`，
        而 `_absgrad_report` 寫的是 `outputs.get("viewspace_points")` + `getattr(vp, "grad", None)`
        ⇒ **偵測方式被繞過**。`absgrad_densify`（2026-08-25）確實在讀它。

        後果：`_strip_forward_backward` 的 `outputs = dict(last_outputs)` 讓 K>1 時 absgrad
        只看得到最後一條帶（見該函式的 docstring）。改成偵測「同一個模組裡同時出現
        viewspace_points 與 .grad 的取用」，字串換寫法也躲不掉。
        """
        import inspect
        from internal.density_controllers import mcmc_2dgs_density_controller as m
        code = "\n".join(l for l in inspect.getsource(m).splitlines()
                          if not l.strip().startswith("#"))
        reads = ('viewspace_points"].grad' in code
                 or ('get("viewspace_points")' in code and 'getattr(vp, "grad"' in code)
                 or "viewspace_point_tensor.grad" in code)
        self.assertTrue(reads, "偵測失效：改了取用寫法就抓不到，請更新偵測條件")

    def test_strip_path_sums_viewspace_grad_across_strips(self):
        """因為 MCMC 會讀（上一個測試），條帶路徑就必須把各條帶的梯度加起來。

        條帶是**不相交的像素集合**，且每條的 loss 已乘上 `h/H`
        ⇒ 加權梯度之和 == 全幀梯度。這裡直接驗算術，並釘住原始碼裡真的有做加總。
        """
        import inspect
        from internal import gaussian_splatting as gsp
        src = inspect.getsource(gsp.GaussianSplatting._strip_forward_backward)
        self.assertIn("agg_vsgrad", src)
        self.assertIn('outputs["viewspace_points"] = vp_agg', src)

        # 算術：三條不相交的貢獻，加總 == 一次算完
        n = 8
        full = torch.zeros(n, 3)
        parts = [torch.zeros(n, 3) for _ in range(3)]
        g = torch.Generator().manual_seed(0)
        for i, pt in enumerate(parts):
            pt[:, 2] = torch.rand(n, generator=g)      # 每條帶各自的 |g|
            full[:, 2] += pt[:, 2]
        acc = None
        for pt in parts:
            acc = pt.clone() if acc is None else acc + pt
        self.assertTrue(torch.allclose(acc, full))
        # 而「只取最後一條」會漏掉多少：這就是修正前的行為
        self.assertFalse(torch.allclose(parts[-1], full))

    def test_vanilla_does_read_viewspace_grad(self):
        """Guards the test itself: if this stops being true the detection above proves nothing."""
        import inspect
        from internal.density_controllers import vanilla_density_controller as v
        self.assertIn("viewspace_point_tensor.grad", inspect.getsource(v))


class ReadsViewspaceGradFlagTest(unittest.TestCase):
    """The flag `gaussian_splatting._split_backward_needed` dispatches on.

    Getting it backwards is silent: a controller that reads the gradient but is marked False would
    densify on a gradient nobody produced, and nothing would raise.
    """

    def test_flags(self):
        from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl
        from internal.density_controllers.gg_mcmc_2dgs_density_controller import GGMCMC2DGSDensityControllerImpl
        from internal.density_controllers.vanilla_density_controller import VanillaDensityControllerImpl

        # mainline MCMC: skip the split
        self.assertFalse(getattr(MCMC2DGSDensityControllerImpl, "READS_VIEWSPACE_GRAD", True))
        # the gradient-guided variant reads it, and inherits from the MCMC impl -> must re-enable
        self.assertTrue(getattr(GGMCMC2DGSDensityControllerImpl, "READS_VIEWSPACE_GRAD", True))
        # anything that never declares defaults to the safe side
        self.assertTrue(getattr(VanillaDensityControllerImpl, "READS_VIEWSPACE_GRAD", True))

    def test_declared_flag_matches_the_source(self):
        """Cross-check the declaration against what the code actually does."""
        import inspect
        from internal.density_controllers import gg_mcmc_2dgs_density_controller as gg
        from internal.density_controllers import mcmc_2dgs_density_controller as mc
        gg_src = inspect.getsource(gg)
        mc_code = "\n".join(l for l in inspect.getsource(mc).splitlines()
                            if not l.strip().startswith("#"))
        self.assertIn('viewspace_points"].grad', gg_src)      # declared True, and it does read
        # ⚠ 2026-09-11：MCMC 宣告 False **仍然正確**，但理由不是「它不讀」。
        # `READS_VIEWSPACE_GRAD` 問的是「要不要**兩次 backward 之間**的那個梯度」
        # （SSIM-only 的那份，給 gradient-based ADC 用）。MCMC 不要那個。
        # 而 `absgrad_densify` 讀的是**全部 backward 之後**的總梯度 —— 單次融合 backward
        # 也拿得到 ⇒ 旗標不必改。兩者是不同的東西，別再把它當成「MCMC 完全不碰 outputs」。
        self.assertNotIn("retain_graph", mc_code)


if __name__ == "__main__":
    unittest.main()
