"""合併事件刪地下顆粒 merge_vis_under（2026-11-10：v4 判準搬進合併事件）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl


class _FakeGM:
    """只提供 controller／Utils 會用到的介面（同 merge_prune_test）。"""

    def __init__(self, props):
        self._p = dict(props)

    @property
    def properties(self):
        return self._p

    @properties.setter
    def properties(self, v):
        self._p = dict(v)

    @property
    def gaussians(self):
        return self._p

    @property
    def property_names(self):
        return list(self._p.keys())

    def get_property(self, k):
        return self._p[k]

    @property
    def n_gaussians(self):
        return self._p["means"].shape[0]


def _scene(n_side=20, spacing=0.01, n_under=5):
    """z=0 平面上的格點面元（同 merge_prune_test._plane）＋ n_under 顆在平面下方 3～7（互相離很遠、不會被合併）。"""
    g = torch.arange(n_side, dtype=torch.float) * spacing
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    P = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(xx.numel())], -1)
    U = torch.tensor([[0.04 * i, 0.05, -3.0 - i] for i in range(n_under)])
    X = torch.cat([P, U])
    n = X.shape[0]
    props = {
        "means": X,
        "scales": torch.full((n, 2), math.log(spacing / 2)),
        "rotations": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.full((n, 1), math.log(0.9 / 0.1)),
        "shs_dc": torch.zeros(n, 1, 3),
        "shs_rest": torch.zeros(n, 15, 3),
    }
    props = {k: torch.nn.Parameter(v) for k, v in props.items()}
    opt = torch.optim.Adam([{"params": [v], "name": k, "lr": 1e-3} for k, v in props.items()])
    sum((v ** 2).sum() for v in props.values()).backward()
    opt.step()
    opt.zero_grad()
    return _FakeGM(props), [opt], P.shape[0]


def _vis(k=4):
    """k 台相機在高度 10 往下看（x_cam = R x + T），10 px 一格、整片 SfM 平面 z=0 => 每格深度 10。"""
    C = torch.tensor([[0.1 + 0.3 * i, 0.1, 10.0] for i in range(k)])
    R = torch.diag(torch.tensor([1.0, -1.0, -1.0]))[None].repeat(k, 1, 1)
    T = -(R @ C[:, :, None])[:, :, 0]
    o = lambda v: torch.full((k,), float(v))
    d = torch.full((k, 10, 10), 10.0, dtype=torch.float16)
    return {"R": R, "T": T, "fx": o(100), "fy": o(100), "cx": o(50), "cy": o(50), "W": o(100), "H": o(100),
            "dmin": d, "dmax": d.clone(), "cell": 10, "step": -1}


def _impl(**cfg):
    c = dict(merge_k=12, merge_max_frac=0.2, merge_color_tol=1.0, merge_from_iter=3000, merge_until_iter=-1, merge_prune=True,
             merge_vis_under=True, sfm_vis=False, sfm_vis_px=4.0, sfm_vis_free_min=2, sfm_vis_free_frac=0.5, sfm_vis_rep=0.1,
             sfm_vis_behind_min=3, sfm_vis_behind_frac=0.9, sfm_vis_behind_max=0.0, sfm_vis_eps=0.01, sfm_vis_cams=0,
             densify_until_iter=15000)
    c.update(cfg)
    impl = object.__new__(MCMC2DGSDensityControllerImpl)
    impl.config = SimpleNamespace(**c)
    impl._max_radii2D = torch.zeros(1)
    impl._max_tiles2D = torch.zeros(1)
    impl._vis = _vis()
    return impl


class MergeVisUnderTest(unittest.TestCase):
    def test_growth_phase_relocates_without_cap_cut(self):
        # 增生期（step < densify_until_iter）：合併的 j 照舊刪＋扣 cap；地下顆粒回傳遮罩交給 relocation、不扣 cap
        gm, opts, n_plane = _scene()
        impl = _impl()
        n0 = gm.n_gaussians
        out = impl._merge_event(gm, opts, 3000, prune=True)
        cut = impl._cap_cut
        self.assertGreater(cut, 0)                                               # 平面有合併
        self.assertEqual(gm.n_gaussians, n0 - cut)                              # 只有合併的 j 被刪
        self.assertEqual(out.shape[0], gm.n_gaussians)                          # 遮罩對齊合併後的 N（呼叫端直接 OR 進 dead_mask）
        self.assertEqual(int(out.sum()), 5)
        self.assertTrue(bool(out[-5:].all()))
        self.assertEqual(impl._mvu_moved, 5)

    def test_harvest_phase_deletes_underground(self):
        gm, opts, n_plane = _scene()
        impl = _impl()
        n0 = gm.n_gaussians
        out = impl._merge_event(gm, opts, 16500, prune=True)
        self.assertIsNone(out)
        z = gm.gaussians["means"].detach()[:, 2]
        self.assertFalse(bool((z < -1).any()))                                  # 地下 5 顆全刪
        self.assertEqual(impl._mvu_cut, 5)
        self.assertEqual(gm.n_gaussians, n0 - impl._cap_cut - 5)                # cap 只扣合併的 j
        for group in opts[0].param_groups:                                      # 參數與 Adam 狀態一起裁
            p = group["params"][0]
            self.assertEqual(opts[0].state[p]["exp_avg"].shape[0], gm.n_gaussians, group["name"])
        self.assertIsNone(impl._max_radii2D)

    def test_relocate_mode_returns_mask(self):
        gm, opts, n_plane = _scene()
        impl = _impl(merge_prune=False)
        n0 = gm.n_gaussians
        out = impl._merge_event(gm, opts, 3000, prune=False)
        self.assertEqual(gm.n_gaussians, n0)
        self.assertTrue(bool(out[n_plane:].all()))                              # 地下 5 顆一定在遮罩裡（合併的 j 也在）
        self.assertEqual(int(getattr(impl, "_cap_cut", 0) or 0), 0)

    def test_off_keeps_underground(self):
        gm, opts, n_plane = _scene()
        impl = _impl(merge_vis_under=False)
        impl._merge_event(gm, opts, 3000, prune=True)
        z = gm.gaussians["means"].detach()[:, 2]
        self.assertEqual(int((z < -1).sum()), 5)

    def test_too_few_cameras_is_not_evidence(self):
        gm, opts, n_plane = _scene()
        impl = _impl()
        impl._vis = _vis(k=2)                                                   # 被擋的相機 < sfm_vis_behind_min（3）=> 不判地下
        out = impl._merge_event(gm, opts, 3000, prune=True)
        self.assertIsNone(out)
        self.assertEqual(int(getattr(impl, "_mvu_moved", 0) or 0), 0)

    def test_sfm_vis_probs_uses_same_criterion(self):
        # 抽出 _sfm_vis_classify 之後，v4 的增生壓制照舊：地下顆粒的母點機率 x eps、平面上的不變
        gm, opts, n_plane = _scene()
        impl = _impl(sfm_vis=True, sfm_birth_from_iter=0, sfm_birth_until_iter=-1)
        impl._ip_step = 3000
        p = impl._sfm_vis_probs(gm, torch.ones(gm.n_gaussians))
        self.assertTrue(torch.allclose(p[n_plane:], torch.full((5,), 0.01)))
        self.assertTrue(torch.equal(p[:n_plane], torch.ones(n_plane)))


if __name__ == "__main__":
    unittest.main()
