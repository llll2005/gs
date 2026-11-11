"""MCMC 補洞放點（hole_fill_*）與增生方向限制（inplane_*）（2026-11-02）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import (
    MCMC2DGSDensityControllerImpl, backproject, frame_from_normal, hole_candidates, mono_normals, ramp)
from internal.density_controllers.merge_ops import _qmat
from internal.metrics.geom_gate import quat_normals


def _w2c(R, t):
    """標準 [R|t]（相機 = R·世界 + t）=> gspl 的列向量版 world_to_camera（[p,1] @ w2c）。"""
    M = torch.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M.T


def _cam(W=40, H=30, f=50.0):
    R = torch.eye(3)
    t = torch.tensor([0.0, 0.0, 2.0])                  # 相機在世界 (0,0,-2)，看 +z
    w2c = _w2c(R, t)
    return SimpleNamespace(fx=torch.tensor(f), fy=torch.tensor(f), cx=torch.tensor(W / 2), cy=torch.tensor(H / 2),
                           width=torch.tensor(W), height=torch.tensor(H), world_to_camera=w2c,
                           camera_center=torch.tensor([0.0, 0.0, -2.0]))


class _FakeGM:
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

    def get_opacities(self):
        return torch.sigmoid(self._p["opacities"])


def _gm(n=50):
    props = {
        "means": torch.randn(n, 3),
        "scales": torch.full((n, 2), math.log(0.01)),
        "rotations": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.full((n, 1), 2.0),
        "shs_dc": torch.zeros(n, 1, 3),
        "shs_rest": torch.zeros(n, 15, 3),
    }
    props = {k: torch.nn.Parameter(v) for k, v in props.items()}
    opt = torch.optim.Adam([{"params": [v], "name": k, "lr": 1e-3} for k, v in props.items()])
    sum((v ** 2).sum() for v in props.values()).backward()
    opt.step()
    opt.zero_grad()
    return _FakeGM(props), [opt]


def _impl(**cfg):
    c = dict(hole_fill_frac=0.5, hole_fill_margin=0.1, hole_fill_scale_px=2.0, hole_fill_opacity=0.5,
             inplane_offset=0.0, inplane_hold=0, inplane_ramp_from=1000, inplane_ramp_until=10000)
    c.update(cfg)
    impl = object.__new__(MCMC2DGSDensityControllerImpl)
    impl.config = SimpleNamespace(**c)
    return impl


class GeometryHelpersTest(unittest.TestCase):
    def test_backproject_inverts_depth_support_projection(self):
        cam = _cam()
        P = torch.tensor([[0.3, -0.2, 1.0], [-0.5, 0.4, 3.0]])
        pc = torch.cat([P, torch.ones(2, 1)], 1) @ cam.world_to_camera      # geom_gate.DepthSupport.update 同一套
        z = pc[:, 2]
        u = cam.fx * pc[:, 0] / z + cam.cx
        v = cam.fy * pc[:, 1] / z + cam.cy
        Q = backproject(u, v, z, float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy), cam.world_to_camera)
        self.assertTrue(torch.allclose(Q, P, atol=1e-5))

    def test_hole_candidates(self):
        a = torch.tensor([[0.2, 1.0, 1.0, 0.2]])
        z = torch.tensor([[1.0, 2.0, 1.0, 1.0]])
        inv = torch.tensor([[1.0, 1.0, 1.0, 0.0]])                       # 最後一格單目無效
        A = torch.tensor([[True, True, True, True]])
        holes, deeper = hole_candidates(a, z, inv, A, 0.1)
        self.assertEqual(holes.tolist(), [[True, False, False, False]])
        self.assertEqual(deeper.tolist(), [[False, True, False, False]])

    def test_mono_normals_face_camera_and_follow_tilt(self):
        cam = _cam()
        H, W = 30, 40
        zm = torch.full((H, W), 3.0)                                       # 正對相機的平面（相機 z 固定）
        rows, cols = torch.tensor([10, 20]), torch.tensor([10, 30])
        n = mono_normals(zm, rows, cols, float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy), cam.world_to_camera, cam.camera_center)
        self.assertTrue(torch.allclose(n, torch.tensor([[0.0, 0, -1]]).expand(2, 3), atol=1e-4))
        R = frame_from_normal(n)
        self.assertTrue(torch.allclose(R[:, :, 2], n, atol=1e-6))
        self.assertTrue(torch.allclose(R.transpose(1, 2) @ R, torch.eye(3).expand(2, 3, 3), atol=1e-5))

    def test_ramp(self):
        self.assertEqual(ramp(500, 1000, 10000), 0.0)
        self.assertAlmostEqual(ramp(5500, 1000, 10000), 0.5)
        self.assertEqual(ramp(20000, 1000, 10000), 1.0)


class Mat2QuatTest(unittest.TestCase):
    def test_roundtrip_including_180_degree(self):
        from internal.density_controllers.merge_ops import _mat2q
        torch.manual_seed(0)
        q = torch.nn.functional.normalize(torch.randn(2000, 4), dim=-1)
        q = torch.cat([q, torch.tensor([[0.0, 1, 0, 0], [0.0, 0, 1, 0], [0.0, 0, 0, 1], [0.0, 0.6, 0.8, 0], [1e-7, 0, 0.6, 0.8]])])   # 轉 180°
        R = _qmat(q)
        R2 = _qmat(_mat2q(R))
        self.assertLess(float((R2 - R).abs().max()), 1e-5)
        n = torch.tensor([[0.0, 0, -1], [0.0, 0, 1], [1.0, 0, 0], [0.0, -1, 0]])
        self.assertTrue(torch.allclose(_qmat(_mat2q(frame_from_normal(n)))[:, :, 2], n, atol=1e-6))


class HoleFillTest(unittest.TestCase):
    def test_places_dead_particles_at_mono_depth(self):
        cam = _cam()
        H, W = 30, 40
        gm, opts = _gm(50)
        impl = _impl(hole_fill_frac=0.2)                                   # 最多 10 顆
        alpha = torch.ones(1, H, W); alpha[0, 5:8, 5:8] = 0.1             # 9 個洞
        z = torch.full((1, H, W), 2.0)
        inv = torch.full((H, W), 1 / 2.0)
        mask = torch.ones(H, W, dtype=torch.uint8)
        gt = torch.rand(3, H, W)
        pl = SimpleNamespace(metric=SimpleNamespace(_bm_cur=mask))
        dead = torch.zeros(50, dtype=torch.bool); dead[:20] = True
        out = impl._hole_fill(gm, opts, {"rend_alpha": alpha, "surf_depth": z}, (cam, ("x", gt), inv), pl, dead, 3000)
        placed = torch.nonzero(dead & ~out, as_tuple=True)[0]
        self.assertEqual(placed.numel(), 9)                                 # 9 個洞、死顆粒 20 顆夠用
        P = gm.gaussians["means"][placed].detach()
        pc = torch.cat([P, torch.ones(9, 1)], 1) @ cam.world_to_camera
        self.assertTrue(torch.allclose(pc[:, 2], torch.full((9,), 2.0), atol=1e-4))      # 在單目深度上
        u = cam.fx * pc[:, 0] / pc[:, 2] + cam.cx; v = cam.fy * pc[:, 1] / pc[:, 2] + cam.cy
        self.assertTrue(bool(((u >= 5) & (u <= 8) & (v >= 5) & (v <= 8)).all()))           # 落回洞的像素
        n = quat_normals(gm.gaussians["rotations"][placed].detach())
        self.assertTrue(torch.allclose(n, torch.tensor([[0.0, 0, -1]]).expand(9, 3), atol=1e-4))
        o = torch.sigmoid(gm.gaussians["opacities"][placed].detach()).reshape(-1)
        self.assertTrue(torch.allclose(o, torch.full((9,), 0.5), atol=1e-4))
        st = opts[0].state[gm.properties["means"]]
        self.assertEqual(float(st["exp_avg"][placed].abs().sum()), 0.0)                   # Adam 歸零

    def test_missing_inputs_is_noop(self):
        gm, opts = _gm(10)
        impl = _impl()
        dead = torch.ones(10, dtype=torch.bool)
        out = impl._hole_fill(gm, opts, {}, (_cam(), ("x", torch.rand(3, 30, 40)), None), SimpleNamespace(metric=None), dead, 3000)
        self.assertTrue(bool(out.all()))


class HoleFillDemandTest(unittest.TestCase):
    """★ 11-11 規格 2 §2 需求地圖補貨（hole_fill_demand）。"""
    H, W = 30, 40

    def _fill(self, Wm=None, alpha=None, seed=0, n=50, frac=0.2):
        torch.manual_seed(seed)
        cam = _cam()
        gm, opts = _gm(n)
        impl = _impl(hole_fill_frac=frac, hole_fill_demand=True)
        alpha = torch.ones(1, self.H, self.W) if alpha is None else alpha
        out = {"rend_alpha": alpha, "surf_depth": torch.full((1, self.H, self.W), 2.0)}
        if Wm is not None:
            out["rend_capw"] = Wm[None]
        pl = SimpleNamespace(metric=SimpleNamespace(_bm_cur=torch.ones(self.H, self.W, dtype=torch.uint8), _cap_map_cur=torch.full((self.H, self.W), 0.5)))
        dead = torch.zeros(n, dtype=torch.bool); dead[:20] = True
        res = impl._hole_fill(gm, opts, out, (cam, ("x", torch.rand(3, self.H, self.W)), torch.full((self.H, self.W), 0.5)), pl, dead, 3000)
        placed = torch.nonzero(dead & ~res, as_tuple=True)[0]
        P = gm.gaussians["means"][placed].detach()
        pc = torch.cat([P, torch.ones(P.shape[0], 1)], 1) @ cam.world_to_camera
        u = cam.fx * pc[:, 0] / pc[:, 2] + cam.cx; v = cam.fy * pc[:, 1] / pc[:, 2] + cam.cy
        return placed, u, v, pc[:, 2]

    def test_zero_demand_and_no_holes_places_nothing(self):
        placed, _, _, _ = self._fill(Wm=torch.zeros(self.H, self.W))
        self.assertEqual(placed.numel(), 0)

    def test_places_only_where_demand(self):
        Wm = torch.zeros(self.H, self.W); Wm[20:25, 30:35] = 0.8
        placed, u, v, z = self._fill(Wm=Wm)
        self.assertEqual(placed.numel(), 10)                                # frac 0.2 x 50
        self.assertTrue(bool(((u >= 30) & (u <= 35) & (v >= 20) & (v <= 25)).all()))
        self.assertTrue(torch.allclose(z, torch.full_like(z, 2.0), atol=1e-4))   # 放在先驗深度

    def test_weighting_prefers_high_demand(self):
        Wm = torch.full((self.H, self.W), 0.002); Wm[20:25, 30:35] = 0.9   # 高需求 25 像素 x 0.9 vs 其餘 1175 x 0.002
        placed, u, v, _ = self._fill(Wm=Wm, n=200, frac=0.1)               # 20 顆
        hit = int(((u >= 30) & (u <= 35) & (v >= 20) & (v <= 25)).sum())
        self.assertGreaterEqual(hit, 12)                                    # 期望約 0.9x25／(0.9x25＋0.002x1175)＝91% 的前幾抽

    def test_holes_kept_and_missing_W_falls_back_to_holes(self):
        alpha = torch.ones(1, self.H, self.W); alpha[0, 5:8, 5:8] = 0.1   # 9 個洞
        placed, u, v, _ = self._fill(Wm=None, alpha=alpha)
        self.assertEqual(placed.numel(), 9)
        self.assertTrue(bool(((u >= 5) & (u <= 8) & (v >= 5) & (v <= 8)).all()))
        placed2, _, _, _ = self._fill(Wm=torch.zeros(self.H, self.W), alpha=alpha)   # W 全 0 但有洞 => 照補洞
        self.assertEqual(placed2.numel(), 9)

    def test_demand_map_step(self):
        impl = _impl(hole_fill_demand=True, hole_fill_frac=0.002, densification_interval=100, hole_fill_from_iter=1000, densify_until_iter=15000)
        self.assertEqual([impl.demand_map_step(s) for s in (500, 3000, 3050, 14900, 15000)], [False, True, False, True, False])
        self.assertFalse(_impl(hole_fill_demand=False, hole_fill_frac=0.002, densification_interval=100, hole_fill_from_iter=1000,
                               densify_until_iter=15000).demand_map_step(3000))

    def test_metric_map_without_loss(self):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        H, W = 6, 8
        batch = (None, ("x.png", torch.rand(3, H, W)), torch.full((H, W), 0.5))
        m = MCMCCityGSV2Metrics(depth_cap_reg=0.0).instantiate()
        self.assertIsNone(m.depth_cap_map(batch, 3000))                     # 沒開、不是補貨步
        cap = m.depth_cap_map(batch, 3000, demand=True)                     # 補貨步：只出地圖
        self.assertTrue(torch.allclose(cap, torch.full((H, W), 0.5 / 1.01)))
        self.assertIsNone(m._cap_cur)                                       # 不加 loss
        self.assertIs(m._cap_map_cur, cap)
        m2 = MCMCCityGSV2Metrics(depth_cap_reg=0.5, depth_cap_interval=8, depth_cap_from_iter=0).instantiate()
        self.assertIsNotNone(m2.depth_cap_map(batch, 16)) ; self.assertIsNotNone(m2._cap_cur)   # loss 步照舊
        self.assertIsNone(m2.depth_cap_map(batch, 17))
        m2.depth_cap_map(batch, 17, demand=True); self.assertIsNone(m2._cap_cur)               # 非 loss 步的補貨圖不加 loss


class InplaneTest(unittest.TestCase):
    def test_birth_offset_stays_in_plane(self):
        gm, _ = _gm(20)
        q = torch.nn.functional.normalize(torch.randn(20, 4), dim=-1)
        gm.gaussians["rotations"].data.copy_(q)
        before = gm.gaussians["means"].detach().clone()
        impl = _impl(inplane_offset=1.0)
        impl._ip_step = 20000                                               # 漸增已滿
        idx = torch.arange(5, 15)
        with torch.no_grad():
            impl._inplane_birth(gm, idx)
        d = gm.gaussians["means"].detach() - before
        n = _qmat(q)[:, :, 2]
        self.assertTrue(torch.allclose((d * n).sum(-1), torch.zeros(20), atol=1e-6))       # 不離開平面
        self.assertTrue(torch.allclose(d[idx].norm(dim=-1), torch.full((10,), 0.01), atol=1e-5))  # 1 × 長軸
        self.assertEqual(float(d[:5].abs().sum()), 0.0)
        self.assertEqual(impl._birth[idx].tolist(), [20000] * 10)
        self.assertEqual(int((impl._birth >= 0).sum()), 10)

    def test_no_offset_before_ramp(self):
        gm, _ = _gm(5)
        before = gm.gaussians["means"].detach().clone()
        impl = _impl(inplane_offset=1.0)
        impl._ip_step = 500
        with torch.no_grad():
            impl._inplane_birth(gm, torch.arange(5))
        self.assertTrue(torch.equal(gm.gaussians["means"].detach(), before))

    def test_project_removes_normal_gradient_of_newborns_only(self):
        gm, _ = _gm(6)
        impl = _impl(inplane_hold=100)
        impl._birth = torch.tensor([-1, 9000, 9950, 9950, -1, 9950])
        gm.gaussians["means"].grad = torch.tensor([[1.0, 2, 3]]).repeat(6, 1)
        impl._inplane_project(gm, 10000)                                   # 漸增滿；9000 出生的已超過 100 步
        g = gm.gaussians["means"].grad
        self.assertEqual(g[[0, 1, 4]].tolist(), [[1.0, 2, 3]] * 3)
        self.assertTrue(torch.allclose(g[[2, 3, 5]], torch.tensor([[1.0, 2, 0]]).repeat(3, 1)))   # 法向 z 分量去掉

    def test_prune_keeps_birth_in_sync(self):
        gm, opts = _gm(8)
        impl = _impl(inplane_hold=100)
        impl._birth = torch.arange(8)
        mask = torch.tensor([True, False, True, False, False, False, False, True])
        impl._prune_points(mask, gm, opts)
        self.assertEqual(impl._birth.tolist(), [1, 3, 4, 5, 6])
        self.assertEqual(gm.n_gaussians, 5)


if __name__ == "__main__":
    unittest.main()
