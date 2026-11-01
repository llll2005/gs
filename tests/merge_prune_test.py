"""訓練中合併的 merge_prune（2026-11-02：合併完就刪 j、N 減少、cap 扣掉）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl


class _FakeGM:
    """只提供 controller／Utils 會用到的介面（properties、gaussians、get_property、n_gaussians）。"""

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


def _plane(n_side=20, spacing=0.01):
    """z=0 平面上的格點面元：同色、不透明、法向朝 z、長軸＝半個格距 => 相鄰兩顆可以合併。"""
    g = torch.arange(n_side, dtype=torch.float) * spacing
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    n = xx.numel()
    props = {
        "means": torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(n)], -1),
        "scales": torch.full((n, 2), math.log(spacing / 2)),
        "rotations": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.full((n, 1), math.log(0.9 / 0.1)),
        "shs_dc": torch.zeros(n, 1, 3),
        "shs_rest": torch.zeros(n, 15, 3),
    }
    props = {k: torch.nn.Parameter(v) for k, v in props.items()}
    opt = torch.optim.Adam([{"params": [v], "name": k, "lr": 1e-3} for k, v in props.items()])
    sum((v ** 2).sum() for v in props.values()).backward()
    opt.step()                                          # 讓每個參數都有 Adam 狀態（exp_avg／exp_avg_sq）
    opt.zero_grad()
    return _FakeGM(props), [opt]


def _impl(**cfg):
    c = dict(merge_k=12, merge_max_frac=0.2, merge_color_tol=1.0, merge_from_iter=3000, merge_until_iter=-1, merge_prune=True)
    c.update(cfg)
    impl = object.__new__(MCMC2DGSDensityControllerImpl)
    impl.config = SimpleNamespace(**c)
    impl._max_radii2D = torch.zeros(1)
    impl._max_tiles2D = torch.zeros(1)
    return impl


class MergePruneTest(unittest.TestCase):
    def test_prune_removes_j_and_cuts_cap(self):
        gm, opts = _plane()
        impl = _impl()
        n0 = gm.n_gaussians
        impl._absgrad_accum = torch.arange(n0, dtype=torch.float)
        out = impl._merge_event(gm, opts, 3000, prune=True)
        self.assertIsNone(out)
        cut = impl._cap_cut
        self.assertGreater(cut, 0)
        self.assertEqual(gm.n_gaussians, n0 - cut)
        for group in opts[0].param_groups:              # 參數與 Adam 狀態一起裁（只裁參數會靜默壞掉）
            p = group["params"][0]
            self.assertEqual(p.shape[0], n0 - cut, group["name"])
            st = opts[0].state[p]
            self.assertEqual(st["exp_avg"].shape[0], n0 - cut, group["name"])
            self.assertIs(gm.properties[group["name"]], p)
        self.assertIsNone(impl._max_radii2D)
        self.assertIsNone(impl._max_tiles2D)
        self.assertEqual(impl._absgrad_accum.shape[0], n0 - cut)    # |g| 跟著剪
        # 第二次累加
        impl._merge_event(gm, opts, 4500, prune=True)
        self.assertEqual(gm.n_gaussians, n0 - impl._cap_cut)
        self.assertGreater(impl._cap_cut, cut)

    def test_relocate_mode_keeps_n(self):
        gm, opts = _plane()
        impl = _impl(merge_prune=False)
        n0 = gm.n_gaussians
        out = impl._merge_event(gm, opts, 3000, prune=False)
        self.assertEqual(gm.n_gaussians, n0)
        self.assertEqual(out.shape[0], n0)
        self.assertGreater(int(out.sum()), 0)
        self.assertEqual(int(getattr(impl, "_cap_cut", 0) or 0), 0)

    def test_window(self):
        impl = _impl(merge_from_iter=16500, merge_until_iter=27000)
        self.assertFalse(impl._merge_window(15000))
        self.assertTrue(impl._merge_window(16500))
        self.assertTrue(impl._merge_window(27000))
        self.assertFalse(impl._merge_window(28500))
        self.assertTrue(_impl(merge_from_iter=3000, merge_until_iter=-1)._merge_window(29000))


if __name__ == "__main__":
    unittest.main()
