"""★ 2026-11-12 規格 4 §1 兩輪流程（density round2_*、metric opacity_entropy_round2_*）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl
from internal.metrics.geom_gate import quat_normals
from hole_fill_inplane_test import _gm


def _impl(**cfg):
    c = dict(densify_until_iter=60, round2_from=0, round2_cap_frac=0.7, round2_densify_until=0, round2_surface_opacity=0.5,
             round2_free=False, cap_max=1000, inplane_offset=0.0, inplane_hold=0)
    c.update(cfg)
    impl = object.__new__(MCMC2DGSDensityControllerImpl)
    impl.config = SimpleNamespace(**c)
    return impl


def _on(**kw):
    return _impl(round2_from=100, round2_densify_until=200, **kw)


class Round2OffTest(unittest.TestCase):
    def test_off_matches_old_schedule(self):
        impl = _impl()
        for s in (0, 59, 60, 100, 150, 250):
            self.assertEqual(impl._growing(s), s < 60)                  # 關的時候＝原本的 global_step < densify_until_iter
            self.assertEqual(impl._grow_end(s), 60)
        gm, _ = _gm(10)
        p = torch.rand(10)
        self.assertIs(impl._round2_probs(gm, p), p)                     # 沒有表面標記 => 原封不動

    def test_extend_multidim_and_1d(self):
        impl = _impl()
        impl._birth = torch.tensor([3, -1]); impl._r2_p0 = torch.ones(2, 3)
        impl._per_particle_extend(2, 4)
        self.assertEqual(impl._birth.tolist(), [3, -1, -1, -1])         # 1 維照舊
        self.assertEqual(tuple(impl._r2_p0.shape), (4, 3))
        self.assertTrue(bool((impl._r2_p0[2:] == 0).all()))


class Round2ScheduleTest(unittest.TestCase):
    def test_growing_windows(self):
        impl = _on()
        self.assertEqual([impl._growing(s) for s in (59, 60, 99, 100, 199, 200)], [True, False, False, True, True, False])
        self.assertEqual(impl._grow_end(30), 60)
        self.assertEqual(impl._grow_end(150), 200)

    def test_round1_quota_and_surface(self):
        impl = _on()
        gm, _ = _gm(6)
        with torch.no_grad():
            gm.gaussians["opacities"].copy_(torch.tensor([3.0, 3.0, 3.0, -3.0, -3.0, 3.0])[:, None])   # o ≈ .95／.05
        gm._trim_state = {"Vlast": torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 2.0])}                    # 第 3 顆不透明但沒出力
        impl._round2_tick(gm, 50)
        self.assertEqual(impl._round_cut, 300)                           # 第一輪：只用 70% 名額
        self.assertIsNone(getattr(impl, "_r2_surf", None))
        impl._round2_tick(gm, 100)
        self.assertEqual(impl._round_cut, 0)                             # 第二輪：長到 cap
        self.assertEqual(impl._r2_surf.tolist(), [True, True, False, False, False, True])
        p = impl._round2_probs(gm, torch.ones(6))
        self.assertEqual(p.tolist(), [1, 1, 0, 0, 0, 1])                  # 非表面顆粒抽不到
        self.assertEqual(impl._round2_probs(gm, torch.ones(2), torch.tensor([2, 5])).tolist(), [0, 1])
        impl._r2_birth(gm, torch.tensor([3]))                             # 第二輪出生的可以當母點
        self.assertEqual(impl._round2_probs(gm, torch.ones(6)).tolist(), [1, 1, 0, 1, 0, 1])

    def test_free_does_not_restrict(self):
        impl = _on(round2_free=True)
        gm, _ = _gm(4)
        gm._trim_state = {"Vlast": torch.zeros(4)}
        impl._round2_tick(gm, 100)
        p = torch.rand(4)
        self.assertIs(impl._round2_probs(gm, p), p)

    def test_all_zero_falls_back(self):
        impl = _on()
        gm, _ = _gm(3)
        gm._trim_state = {"Vlast": torch.zeros(3)}                       # 沒有一顆有出力
        impl._round2_tick(gm, 100)
        p = torch.ones(3)
        self.assertIs(impl._round2_probs(gm, p), p)


class Round2PlaneHoldTest(unittest.TestCase):
    def test_born_moves_only_in_plane(self):
        impl = _on()
        gm, _ = _gm(5)
        with torch.no_grad():
            q = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
            gm.gaussians["rotations"].copy_(q)
        gm._trim_state = {"Vlast": torch.ones(5)}
        impl._round2_tick(gm, 100)
        idx = torch.tensor([1, 3])
        impl._r2_birth(gm, idx)
        p_birth = gm.gaussians["means"].detach().clone()
        with torch.no_grad():
            gm.gaussians["means"].add_(torch.randn(5, 3))                 # optimizer＋雜訊亂推
        moved = gm.gaussians["means"].detach().clone()
        impl._r2_plane_hold(None, None, gm, 101, None)
        m = gm.gaussians["means"].detach()
        n0 = quat_normals(q[idx])
        self.assertTrue(torch.allclose(((m[idx] - p_birth[idx]) * n0).sum(-1), torch.zeros(2), atol=1e-5))   # 法向位移＝0
        tang = (moved[idx] - p_birth[idx]) - ((moved[idx] - p_birth[idx]) * n0).sum(-1, keepdim=True) * n0
        self.assertTrue(torch.allclose(m[idx] - p_birth[idx], tang, atol=1e-5))                                 # 切向位移保留
        keep = torch.tensor([0, 2, 4])
        self.assertTrue(torch.equal(m[keep], moved[keep]))                                                       # 其他顆粒不動


class Round2CapTest(unittest.TestCase):
    def test_add_respects_round_cut(self):
        from internal.density_controllers.mcmc_density_controller import MCMCDensityControllerImpl
        impl = object.__new__(MCMCDensityControllerImpl)
        impl.config = SimpleNamespace(cap_max=1000, add_ratio=1.05, cost_budget=0.0, cost_budget_ref_csv="")
        impl._round_cut = 300
        gm = SimpleNamespace(n_gaussians=700)
        self.assertEqual(impl.add_new_gs(gm, []), 0)                      # 第一輪滿 700 = 0.7 cap => 不再加


class Round2EntropyTest(unittest.TestCase):
    def _m(self, **kw):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        base = dict(opacity_entropy_reg=0.002, opacity_entropy_side="down", opacity_entropy_both_from_iter=10, opacity_entropy_both_reg=0.008)
        base.update(kw)
        return MCMCCityGSV2Metrics(**base).instantiate()

    def test_off_is_old_schedule(self):
        m = self._m()
        self.assertEqual(m._entropy_mode(5), (0.002, "down", ""))
        self.assertEqual(m._entropy_mode(10), (0.008, "both", "both"))
        self.assertEqual(self._m(opacity_entropy_reg=0.0)._entropy_mode(50)[0], 0.0)      # 原本外層就跳過

    def test_round2_down_only_and_off(self):
        m = self._m(opacity_entropy_round2_from=20, opacity_entropy_round2_reg=0.002)
        self.assertEqual(m._entropy_mode(15), (0.008, "both", "both"))                    # 第一輪照舊
        self.assertEqual(m._entropy_mode(20), (0.002, "down", "r2"))                      # 第二輪只往 0、權重回原值
        self.assertEqual(self._m(opacity_entropy_round2_from=20, opacity_entropy_round2_reg=0.0)._entropy_mode(25)[0], 0.0)   # r2ent0
        self.assertEqual(self._m(opacity_entropy_round2_from=20)._entropy_mode(25), (0.002, "down", "r2"))                    # -1＝沿用 reg


if __name__ == "__main__":
    unittest.main()
