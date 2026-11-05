"""SfM 錨點與 SfM 增生限制（sfm_*；2026-11-05 使用者「鎖住 SfM 位置＋增生只在 SfM 附近＋合併優先併偏離的」）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl
from merge_prune_test import _plane


def _impl(**cfg):
    c = dict(merge_k=12, merge_max_frac=0.2, merge_color_tol=1.0, merge_from_iter=3000, merge_until_iter=-1, merge_prune=True,
             sfm_lock=True, sfm_track=False, sfm_birth_radius=0.05, sfm_birth_eps=0.01, sfm_birth_from_iter=0, sfm_birth_until_iter=-1,
             sfm_merge_eps=1.0, sfm_dist_voxel=0.01, sfm_report_interval=0, geom_gate_sampling=False)
    c.update(cfg)
    impl = object.__new__(MCMC2DGSDensityControllerImpl)
    impl.config = SimpleNamespace(**c)
    impl._max_radii2D = torch.zeros(1)
    impl._max_tiles2D = torch.zeros(1)
    return impl


def _built(**cfg):
    """20x20 平面（格距 0.01）；SfM＝偶數號顆粒的位置（=> 那些是錨點）＋一顆在平面上方 0.09 的孤點（距離場只蓋顆粒範圍＋0.1）。"""
    gm, opts = _plane()
    X0 = gm.gaussians["means"].detach().clone()
    sfm = np.concatenate([X0[::2].numpy().astype(np.float64), [[0.1, 0.1, 0.09]]])
    impl = _impl(**cfg)
    impl._sfm_build(sfm, gm.gaussians["means"])
    return impl, gm, opts, X0


class SfMAnchorTest(unittest.TestCase):
    def test_anchors_are_exact_sfm_matches(self):
        impl, gm, _, _ = _built()
        a = impl._anchor
        self.assertEqual(int((a >= 0).sum()), 200)
        self.assertTrue(bool((a[::2] >= 0).all()))
        self.assertTrue(bool((a[1::2] < 0).all()))

    def test_distance_field(self):
        impl, gm, _, X0 = _built()
        q = torch.tensor([[0.05, 0.05, 0.0],     # 在平面上（旁邊就有 SfM）
                          [0.05, 0.05, 0.08],    # 平面上方 0.08
                          [0.1, 0.1, 0.05],      # 孤點下方 0.04（平面上方 0.05）
                          [9.0, 9.0, 9.0]])      # 距離場外
        d = impl._sfm_dist(q)
        self.assertLess(float(d[0]), 0.011)
        self.assertAlmostEqual(float(d[1]), 0.08, delta=0.011)
        self.assertAlmostEqual(float(d[2]), 0.04, delta=0.011)
        self.assertTrue(math.isinf(float(d[3])))

    def test_restore_after_move(self):
        impl, gm, _, X0 = _built()
        with torch.no_grad():
            gm.gaussians["means"].add_(0.5)
        impl._sfm_restore(None, None, gm, 1, None)
        m = gm.gaussians["means"].detach()
        self.assertTrue(torch.equal(m[::2], X0[::2]))                     # 錨點回原位
        self.assertTrue(torch.allclose(m[1::2], X0[1::2] + 0.5))          # 非錨點照樣移動

    def test_prune_keeps_anchors_and_remaps(self):
        impl, gm, opts, X0 = _built()
        mask = torch.zeros(gm.n_gaussians, dtype=torch.bool)
        mask[:100] = True                                                 # 前 100 顆：50 個錨點、50 個非錨點
        impl._prune_points(mask, gm, opts)
        self.assertEqual(gm.n_gaussians, 400 - 50)
        self.assertEqual(impl._anchor.shape[0], 350)
        self.assertEqual(int((impl._anchor >= 0).sum()), 200)
        with torch.no_grad():
            gm.gaussians["means"].add_(1.0)
        impl._sfm_restore(None, None, gm, 1, None)
        am = impl._anchor >= 0
        m = gm.gaussians["means"].detach()
        self.assertTrue(torch.equal(m[am], impl._anchor_xyz[impl._anchor[am]]))

    def test_exclude_from_relocation_and_release_nonfinite(self):
        impl, gm, _, _ = _built()
        dead = torch.zeros(gm.n_gaussians, dtype=torch.bool)
        dead[:10] = True                                                  # 0,2,4,6,8 是錨點
        with torch.no_grad():
            gm.gaussians["scales"][4, 0] = float("nan")
        out = impl._anchor_exclude(dead, gm)
        self.assertEqual(out.nonzero(as_tuple=True)[0].tolist(), [1, 3, 4, 5, 7, 9])
        self.assertEqual(int(impl._anchor[4]), -1)                        # 非有限 => 解除鎖定

    def test_birth_probs_factor_and_window(self):
        impl, gm, _, _ = _built(sfm_birth_from_iter=1000, sfm_birth_until_iter=5000)
        with torch.no_grad():
            gm.gaussians["means"][1, 2] = 0.2                             # 離平面 0.2 > r 0.05
        p = torch.ones(gm.n_gaussians)
        impl._ip_step = 500
        self.assertTrue(torch.equal(impl._geom_gate_probs(gm, p), p))   # 窗口外不改
        impl._ip_step = 2000
        q = impl._geom_gate_probs(gm, p)
        self.assertAlmostEqual(float(q[1]), 0.01, places=6)
        self.assertEqual(int((q < 1).sum()), 1)
        idx = torch.tensor([0, 1, 2])
        q2 = impl._geom_gate_probs(gm, torch.ones(3), idx)                # 搬家路徑（alive_indices）
        self.assertTrue(torch.allclose(q2, torch.tensor([1.0, 0.01, 1.0])))

    def test_track_mode_records_fates_without_protecting(self):
        impl, gm, opts, X0 = _built(sfm_lock=False, sfm_track=True)
        self.assertEqual(int((impl._anchor >= 0).sum()), 200)            # 錨點照樣找出來
        self.assertIsNone(impl._anchor_mask(gm.n_gaussians))              # 但不保護
        dead = torch.zeros(gm.n_gaussians, dtype=torch.bool); dead[:10] = True
        self.assertTrue(torch.equal(impl._anchor_exclude(dead, gm), dead))
        impl._anchor_fate(dead, "搬家")                                   # 0,2,4,6,8 被搬走 => 解除身分、記帳
        self.assertEqual(impl._anchor_fates["搬家"], 5)
        self.assertEqual(int((impl._anchor >= 0).sum()), 195)
        mask = torch.zeros(gm.n_gaussians, dtype=torch.bool); mask[10:20] = True   # 10,12,14,16,18 被剪
        impl._prune_points(mask, gm, opts)
        self.assertEqual(gm.n_gaussians, 390)
        self.assertEqual(impl._anchor_fates["trim 剪掉"], 5)
        self.assertEqual(int((impl._anchor >= 0).sum()), 190)

    def test_off_by_default_keeps_old_behaviour(self):
        impl = _impl(sfm_lock=False, sfm_track=False, sfm_birth_radius=0.0)
        self.assertFalse(impl._sfm_on())
        gm, _ = _plane()
        p = torch.rand(gm.n_gaussians)
        self.assertTrue(torch.equal(impl._geom_gate_probs(gm, p), p))
        self.assertIsNone(impl._anchor_mask(gm.n_gaussians))


class MergeExcludePrioTest(unittest.TestCase):
    def test_exclude(self):
        from internal.density_controllers.merge_ops import merge_step
        gm, _ = _plane()
        ex = torch.zeros(gm.n_gaussians, dtype=torch.bool)
        ex[:200] = True
        I, J, info = merge_step(gm, k=12, max_frac=0.2, color_tol=1.0, exclude=ex)
        self.assertGreater(I.numel(), 0)
        self.assertFalse(bool(ex[I].any() or ex[J].any()))

    def test_prio_picks_flagged_pairs_first(self):
        from internal.density_controllers.merge_ops import merge_step
        gm, _ = _plane()
        n = gm.n_gaussians
        far = gm.gaussians["means"].detach()[:, 0] > 0.095                # 右半邊
        prio = torch.where(far, torch.full((n,), 0.01), torch.ones(n))
        I, J, _ = merge_step(gm, k=12, max_frac=0.05, color_tol=1.0, prio=prio)
        self.assertGreater(I.numel(), 0)
        self.assertTrue(bool((far[I] | far[J]).all()))                    # 額度有限 => 全給偏離的配對
        gm2, _ = _plane()
        I2, J2, _ = merge_step(gm2, k=12, max_frac=0.05, color_tol=1.0)   # 沒有 prio：兩邊都有
        self.assertFalse(bool((far[I2] | far[J2]).all()))


if __name__ == "__main__":
    unittest.main()
