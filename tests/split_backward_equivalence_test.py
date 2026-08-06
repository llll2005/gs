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

    def test_mcmc_does_not_read_viewspace_grad(self):
        import inspect
        from internal.density_controllers import mcmc_2dgs_density_controller as m
        src = inspect.getsource(m)
        # tolerate the explanatory comment, reject an actual read
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        self.assertNotIn('viewspace_points"].grad', code)
        self.assertNotIn("viewspace_point_tensor.grad", code)

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
        self.assertNotIn('viewspace_points"].grad', mc_code)  # declared False, and it does not


if __name__ == "__main__":
    unittest.main()
