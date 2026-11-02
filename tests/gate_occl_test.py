"""深度支撐閘門的投票規則／無同步累加，與遮擋回收（occl_*）（2026-11-02）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl
from internal.metrics.geom_gate import DepthSupport


class _RefDepthSupport:
    """11-01 版（nonzero／布林索引）的 update，只留平均規則 => 拿來對照新版的數值。"""

    def __init__(self, df=0.05, db=0.05, window=5, cmin=3, near=0.2):
        self.df, self.db, self.window, self.cmin, self.near = df, db, window, cmin, near
        self.sum = self.cnt = self.mn = None
        self.t0 = None

    def update(self, step, means, w2c, fx, fy, cx, cy, width, height, inv_depth):
        n = means.shape[0]
        if self.sum is None:
            self.sum = torch.zeros(n); self.cnt = torch.zeros(n); self.mn = torch.full((n,), float("inf"))
        if self.t0 is None:
            self.t0 = step
        p = torch.cat([means, torch.ones_like(means[:, :1])], 1) @ w2c
        z = p[:, 2]
        ok = z > self.near
        u = fx * p[:, 0] / z.clamp_min(1e-6) + cx
        v = fy * p[:, 1] / z.clamp_min(1e-6) + cy
        hd, wd = inv_depth.shape[-2:]
        ud = (u * (wd / float(width))).floor().long()
        vd = (v * (hd / float(height))).floor().long()
        ok &= (ud >= 0) & (ud < wd) & (vd >= 0) & (vd < hd)
        idx = ok.nonzero(as_tuple=True)[0]
        dinv = inv_depth.reshape(hd, wd)[vd[idx], ud[idx]]
        good = dinv > 1e-6
        idx, dinv = idx[good], dinv[good]
        zm = 1.0 / dinv
        r = (z[idx] - zm) / zm
        self.sum.index_add_(0, idx, r)
        self.cnt.index_add_(0, idx, torch.ones_like(r))
        self.mn[idx] = torch.minimum(self.mn[idx], r.abs())
        if step - self.t0 >= self.window:
            seen = self.cnt >= self.cmin
            front = seen & (self.sum / self.cnt.clamp_min(1) < -self.df)
            never = seen & (self.mn > self.db)
            return ~(front | never)
        return None


def _w2c(t):
    M = torch.eye(4)
    M[:3, 3] = torch.tensor(t)
    return M.T


class DepthSupportTest(unittest.TestCase):
    def test_masked_update_matches_reference(self):
        torch.manual_seed(0)
        n, W, H = 3000, 32, 24
        means = torch.randn(n, 3) * torch.tensor([1.0, 1.0, 0.3])
        new, ref = DepthSupport(window=5), _RefDepthSupport(window=5)
        for step in range(0, 6):
            w2c = _w2c([0.1 * step, 0.0, 2.0 + 0.05 * step])
            inv = 1.0 / (2.0 + 0.3 * torch.rand(H, W))
            inv[torch.rand(H, W) < 0.1] = 0.0                               # 無效的單目深度
            d1 = new.update(step, means, w2c, 30.0, 30.0, W / 2, H / 2, W, H, inv)
            s2 = ref.update(step, means, w2c, 30.0, 30.0, W / 2, H / 2, W, H, inv)
            if s2 is None:
                self.assertFalse(d1)
                self.assertTrue(torch.allclose(new.sum, ref.sum, atol=1e-5))
                self.assertTrue(torch.equal(new.cnt, ref.cnt))
            else:
                self.assertTrue(d1)
                self.assertTrue(torch.equal(new.support, s2))
                self.assertGreater(int((~s2).sum()), 0)                     # 不是平凡的全支撐

    def test_vote_ignores_one_bad_view(self):
        # 一顆在相機 z=2 的位置；單目深度 5 個視角裡 4 個說 2.0（貼面）、1 個說 4.0（殘差 −50% 的離群視角）
        means = torch.tensor([[0.0, 0.0, 0.0]])
        W = H = 8
        res = {}
        for rule in ("mean", "vote"):
            ds = DepthSupport(window=4, rule=rule)
            for step, zm in enumerate([2.0, 2.0, 4.0, 2.0, 2.0]):
                ds.update(step, means, _w2c([0, 0, 2.0]), 10.0, 10.0, W / 2, H / 2, W, H, torch.full((H, W), 1.0 / zm))
            res[rule] = bool(ds.support[0])
        self.assertFalse(res["mean"])          # 平均 −10% => 被判在前方（誤壓）
        self.assertTrue(res["vote"])           # 5 票只有 1 票說前方 => 有支撐

    def test_rule_validated(self):
        with self.assertRaises(ValueError):
            DepthSupport(rule="median")


class _FakeGM:
    def __init__(self, means):
        self._p = {"means": torch.nn.Parameter(means)}

    @property
    def gaussians(self):
        return self._p

    @property
    def n_gaussians(self):
        return self._p["means"].shape[0]


class OcclusionTest(unittest.TestCase):
    def _setup(self):
        W, H = 20, 20
        cam = SimpleNamespace(fx=torch.tensor(20.0), fy=torch.tensor(20.0), cx=torch.tensor(10.0), cy=torch.tensor(10.0),
                              width=torch.tensor(W), height=torch.tensor(H), world_to_camera=_w2c([0, 0, 2.0]))
        # 表面在相機 z=2（世界 z=0）；左半不透明、右半半透明
        z = torch.full((1, H, W), 2.0)
        a = torch.ones(1, H, W); a[:, :, 10:] = 0.5
        means = torch.tensor([[-0.3, 0.0, 0.5],    # 0 左半、在表面後 25% => 被擋
                              [0.3, 0.0, 0.5],     # 1 右半、在半透明表面後 => 可能透出（可見）
                              [-0.3, 0.0, 0.0],    # 2 在表面上 => 可見
                              [-0.3, 0.0, -0.5],   # 3 在表面前 => 可見
                              [-0.3, 0.0, 0.02]])  # 4 在表面後 1%（< margin 3%）=> 可見
        impl = object.__new__(MCMC2DGSDensityControllerImpl)
        impl.config = SimpleNamespace(occl_recycle_min=3, occl_margin=0.03)
        return impl, _FakeGM(means), cam, {"rend_alpha": a, "surf_depth": z}

    def test_recycles_only_always_hidden(self):
        impl, gm, cam, out = self._setup()
        dead = torch.zeros(5, dtype=torch.bool)
        for step in range(3):
            res = impl._occl_step(gm, out, (cam,), dead, 3000 + 100 * step)
            if step < 2:
                self.assertFalse(bool(res.any()))
        self.assertEqual(res.tolist(), [True, False, False, False, False])
        self.assertEqual(impl._occ_n[0].item(), 0.0)                        # 回收後計數重來
        self.assertEqual(impl._see_n.tolist(), [0.0, 3.0, 3.0, 3.0, 3.0])

    def test_seen_once_never_recycled(self):
        impl, gm, cam, out = self._setup()
        impl._occ_n = torch.zeros(5); impl._see_n = torch.tensor([1.0, 0, 0, 0, 0])
        dead = torch.zeros(5, dtype=torch.bool)
        for step in range(5):
            res = impl._occl_step(gm, out, (cam,), dead, 3000 + 100 * step)
        self.assertFalse(bool(res[0]))

    def test_counts_follow_prune_and_growth(self):
        impl, _, _, _ = self._setup()
        impl._occ_n = torch.arange(5.0); impl._see_n = torch.zeros(5); impl._birth = None
        impl._per_particle_extend(5, 7)
        self.assertEqual(impl._occ_n.tolist(), [0, 1, 2, 3, 4, 0, 0])
        self.assertEqual(impl._see_n.shape[0], 7)


if __name__ == "__main__":
    unittest.main()
