"""internal/metrics/geom_gate.py 與 MCMCCityGSV2MetricsModule._geom_gate_step（2026-10-29）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.metrics.geom_gate import morton_neighbors, quat_normals, plane_q, plane_loss, DepthSupport, SfMVoxels, sym3_eig, surf_planes, surf_loss, median_nn_distance


def _axis_quat(axis, ang):
    a = torch.tensor(axis, dtype=torch.float) / math.sqrt(sum(x * x for x in axis))
    return torch.cat([torch.tensor([math.cos(ang / 2)]), a * math.sin(ang / 2)])


class GeomGateTest(unittest.TestCase):
    def test_quat_normals(self):
        n = quat_normals(torch.tensor([[1.0, 0, 0, 0], [*_axis_quat([1, 0, 0], math.pi / 2)]]))
        self.assertTrue(torch.allclose(n[0], torch.tensor([0.0, 0, 1]), atol=1e-6))
        self.assertTrue(torch.allclose(n[1].abs(), torch.tensor([0.0, 1, 0]), atol=1e-6))   # 繞 x 轉 90 度 => 法向躺平

    def test_morton_neighbors_are_local(self):
        g = torch.stack(torch.meshgrid(torch.arange(40.), torch.arange(40.), torch.arange(10.), indexing="ij"), -1).reshape(-1, 3)
        nbr = morton_neighbors(g, k=6)
        self.assertEqual(tuple(nbr.shape), (g.shape[0], 12))
        d = (g[nbr] - g[:, None]).norm(dim=-1)
        self.assertLess(float(d.median()), 2.5)          # 近似 kNN：多數鄰居在 1～2 格內
        self.assertGreater(float((d <= 1.01).float().mean()), 0.3)

    def _plane(self, tilt_idx=None, ang=math.pi / 2):
        g = torch.stack(torch.meshgrid(torch.arange(30.), torch.arange(30.), indexing="ij"), -1).reshape(-1, 2) * 0.1
        xyz = torch.cat([g, torch.zeros(g.shape[0], 1)], 1)
        q = torch.tensor([1.0, 0, 0, 0]).repeat(xyz.shape[0], 1)
        if tilt_idx is not None:
            q[tilt_idx] = _axis_quat([1, 0, 0], ang)
        s1 = torch.full((xyz.shape[0],), 0.1)
        op = torch.ones(xyz.shape[0])
        return xyz, q, s1, op

    def test_plane_q_flags_disordered(self):
        xyz, q, s1, op = self._plane(tilt_idx=465)
        nbr = morton_neighbors(xyz, k=6)
        qq, sw = plane_q(xyz, quat_normals(q), s1, op, nbr)
        self.assertGreater(float(qq[sw > 0.05].median()), 0.95)    # 平面上整齊的面元
        self.assertLess(float(qq[465]), 0.2)                         # 法向躺平的那一顆＝亂向

    def test_plane_loss_pushes_perpendicular(self):
        xyz, q, s1, op = self._plane(tilt_idx=465)                   # 正好 90 度：|cos| 版在這裡梯度最大
        q = q.clone().requires_grad_(True)
        plane_loss(xyz, quat_normals(q), s1, op, morton_neighbors(xyz, k=6), torch.arange(xyz.shape[0])).backward()
        self.assertGreater(float(q.grad[465].norm()), 1e-6)

    def test_plane_loss_gradient_aligns(self):
        xyz, q, s1, op = self._plane(tilt_idx=465, ang=math.pi / 3)
        q = q.clone().requires_grad_(True)
        nbr = morton_neighbors(xyz, k=6)
        idx = torch.arange(xyz.shape[0])
        l0 = plane_loss(xyz, quat_normals(q), s1, op, nbr, idx)
        l0.backward()
        with torch.no_grad():
            q2 = q - 50.0 * q.grad
        l1 = plane_loss(xyz, quat_normals(q2), s1, op, nbr, idx)
        self.assertLess(float(l1), float(l0))

    def test_depth_support(self):
        # 相機在原點看 +z；單目反深度＝1/2（平面 z=2）。面上、前方（z=1.5）、後方（z=3）各一顆
        means = torch.tensor([[0.0, 0, 2.0], [0.0, 0, 1.5], [0.0, 0, 3.0]])
        inv = torch.full((40, 40), 0.5)
        ds = DepthSupport(window=5)
        for s in range(7):
            ds.update(s, means, torch.eye(4), 20.0, 20.0, 20.0, 20.0, 40, 40, inv)
        self.assertEqual(ds.support.tolist(), [True, False, False])
        self.assertAlmostEqual(ds.last["front"], 1 / 3, places=5)
        self.assertAlmostEqual(ds.last["never"], 2 / 3, places=5)   # 前方那顆也從沒貼過面
        ds.update(7, torch.cat([means, means[:1]]), torch.eye(4), 20.0, 20.0, 20.0, 20.0, 40, 40, inv)   # 增生：補在尾端
        self.assertEqual(ds.support.shape[0], 4)


class _FakeModel:
    def __init__(self, xyz, q, s, o):
        self._xyz, self._q, self._s = xyz, q, s
        self.opacities = torch.logit(o).reshape(-1, 1).clone().requires_grad_(True)

    @property
    def get_xyz(self):
        return self._xyz

    def get_rotations(self):
        return self._q

    def get_scales(self):
        return self._s


class MetricGateTest(unittest.TestCase):
    def test_gate_step_plane_and_depth(self):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        m = MCMCCityGSV2Metrics(plane_reg=0.01, plane_from_iter=0, plane_every=1, geom_gate_tau=0.5,
                                depth_support=True, depth_support_from_iter=0, depth_support_window=3).instantiate()
        g = torch.stack(torch.meshgrid(torch.arange(30.), torch.arange(30.), indexing="ij"), -1).reshape(-1, 2) * 0.1 - 1.45
        xyz = torch.cat([g, torch.full((g.shape[0], 1), 2.0)], 1)          # 平面 z=2，正好在單目深度上
        xyz[466, 2] = 1.0                                                   # 一顆浮在前方（畫面中央附近）
        q = torch.tensor([1.0, 0, 0, 0]).repeat(xyz.shape[0], 1)
        q[465] = _axis_quat([1, 0, 0], math.pi / 2)                         # 一顆亂向
        q.requires_grad_(True)
        model = _FakeModel(xyz, q, torch.full((xyz.shape[0], 2), 0.1), torch.full((xyz.shape[0],), 0.99))
        cam = SimpleNamespace(world_to_camera=torch.eye(4), fx=torch.tensor(20.0), fy=torch.tensor(20.0),
                              cx=torch.tensor(20.0), cy=torch.tensor(20.0), width=torch.tensor(40), height=torch.tensor(40))
        batch = (cam, ("x", None), torch.full((40, 40), 0.5))
        gate = None
        for s in range(5):
            metrics, pbar = {"loss": torch.zeros(())}, {}
            gate = m._geom_gate_step(model, s, batch, metrics, pbar, torch.sigmoid(model.opacities.reshape(-1)))
        self.assertIn("plane", metrics)
        self.assertFalse(bool(gate[465]))      # 亂向 => 閘門外
        self.assertFalse(bool(gate[466]))      # 一直在單目表面前方 => 閘門外
        self.assertGreater(float(gate.float().mean()), 0.95)
        self.assertIs(model._geom_gate, gate)
        metrics["loss"].backward()
        self.assertIsNotNone(q.grad)

    def test_sfm_voxels_near(self):
        """2026-11-01 深度支撐的 SfM 豁免：r 內一定算到、超過 2√3·r 一定不算。"""
        g = torch.Generator().manual_seed(0)
        P = torch.rand(3000, 3, generator=g) * 3
        X = torch.rand(50000, 3, generator=g) * 3.4 - 0.2
        near = SfMVoxels(P.numpy(), 0.05, "cpu").near(X)
        d = torch.cdist(X, P).min(dim=1).values
        self.assertTrue(bool(near[d <= 0.05].all()))
        self.assertFalse(bool(near[d > 2 * math.sqrt(3) * 0.05 + 1e-6].any()))

    def test_median_nn_distance_is_scale_free(self):
        g = torch.stack(torch.meshgrid(torch.arange(20.), torch.arange(20.), torch.arange(3.), indexing="ij"), -1).reshape(-1, 3).numpy()
        self.assertAlmostEqual(median_nn_distance(g), 1.0, places=6)
        self.assertAlmostEqual(median_nn_distance(g * 0.01), 0.01, places=8)       # 場景縮放 => 半徑跟著縮
        self.assertAlmostEqual(median_nn_distance(g, max_query=100), 1.0, places=6)  # 抽樣查詢

    def test_sym3_eig_matches_eigh(self):
        g = torch.Generator().manual_seed(1)
        A = torch.randn(2000, 3, 3, generator=g); C = A @ A.transpose(1, 2)
        e1, e2, e3, v = sym3_eig(C)
        ev, evec = torch.linalg.eigh(C)
        self.assertTrue(torch.allclose(torch.stack([e3, e2, e1], 1), ev, rtol=1e-3, atol=1e-3))
        self.assertGreater(float((v * evec[:, :, 0]).sum(-1).abs().min()), 0.99)

    def test_sym3_eig_tiny_scale_stays_unit(self):
        """11-01 bug：cov ~1e-11（鄰居幾乎重疊）時舊版叉積平方下溢 => |v| 到 3.5e7（lab 朝向項 −221）。修正後任何尺度都是單位向量且方向對。"""
        g = torch.Generator().manual_seed(3)
        Q, _ = torch.linalg.qr(torch.randn(256, 3, 3, generator=g, dtype=torch.float64))
        A = Q @ torch.diag(torch.tensor([1.0, 0.5, 0.01], dtype=torch.float64)) @ Q.transpose(1, 2)
        for s in (1.0, 1e-8, 1e-11, 1e-12, 1e-14, 1e-20):
            e1, e2, e3, v = sym3_eig((A * s).float())
            self.assertLess(float((v.norm(dim=-1) - 1).abs().max()), 1e-4, f"scale {s:g}")
            self.assertGreater(float((v.double() * Q[:, :, 2]).sum(-1).abs().min()), 0.99, f"scale {s:g}")

    def test_surf_planes_excludes_collapsed_cluster(self):
        """鄰居幾乎疊在同一點（MCMC 搬移後的複本）=> 平面是雜訊 => 平坦度 inf、被排除；其他點不受影響、法向長度 <= 1。"""
        X, O = self._surf_scene()
        X = X.clone(); X[2000:2010] = X[2000] + 1e-7 * torch.randn(10, 3, generator=torch.Generator().manual_seed(4))
        nbr = morton_neighbors(X, 6)
        C, NH, R, F, W = surf_planes(X, O, nbr, 2.0)
        self.assertLessEqual(float(NH.norm(dim=-1).max()), 1 + 1e-5)
        collapsed = (R < 1e-5)
        self.assertGreater(int(collapsed.sum()), 0)
        self.assertTrue(bool(torch.isinf(F[collapsed]).all()))
        self.assertLess(float(F[2]), 0.1)

    def _surf_scene(self):
        # z=0 平面上的隨機點（不透明；格點會讓 Morton 近鄰排成一直線）＋ 第 0 顆抬高 0.003、第 1 顆抬高 0.5（真結構／浮塊，要被截斷）
        g = torch.Generator().manual_seed(2)
        X = torch.cat([torch.rand(3000, 2, generator=g) * 0.3, torch.zeros(3000, 1)], 1)
        X[0] = torch.tensor([0.15, 0.15, 0.003]); X[1] = torch.tensor([0.151, 0.15, 0.5]); X[2] = torch.tensor([0.05, 0.05, 0.0])
        return X, torch.full((X.shape[0],), 0.9)

    def test_surf_planes_and_truncated_loss(self):
        """2026-11-01 鄰域平面約束：平面上的點法向是 z、有效；稍微偏離的被拉回（梯度沿 -z）；偏離 > tau 的梯度 0。"""
        X, O = self._surf_scene()
        C, NH, R, F, W = surf_planes(X, O, morton_neighbors(X, 6), 2.0)
        self.assertGreater(float(NH[2, 2].abs()), 0.99)
        self.assertLess(float(F[2]), 0.1)
        self.assertGreater(float(NH[0, 2].abs()), 0.99)                     # 抬高那顆的平面由鄰居決定（不含自己）
        self.assertGreater(float(F[0]), -1)                                   # 有算到（不是 nan）
        Xg = X.clone().requires_grad_(True)
        nrm = torch.tensor([[0.0, 0, 1]]).repeat(2, 1)
        lp, lo, trunc = surf_loss(Xg[:2], nrm, C[:2], NH[:2], R[:2], tau=2.0)
        lp.backward()
        self.assertGreater(float(Xg.grad[0, 2]), 0)                          # 在平面上方 => 梯度 +z => 梯度下降往下拉回平面
        self.assertEqual(float(Xg.grad[1].abs().sum()), 0.0)                 # 抬高 50 格 => 截斷、不拉
        self.assertAlmostEqual(trunc, 0.5)
        self.assertLess(float(lo), 1e-6)                                      # 法向和平面一致 => 朝向項 0

    def test_surf_step_ramp_and_interval(self):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        X, O = self._surf_scene()
        m = MCMCCityGSV2Metrics(surf_reg=0.01, surf_from_iter=1000, surf_ramp_until=3000, surf_interval=4, surf_orient=1.0).instantiate()
        q = torch.tensor([[1.0, 0, 0, 0]]).repeat(X.shape[0], 1).requires_grad_(True)
        Xg = X.clone().requires_grad_(True)
        model = SimpleNamespace(get_xyz=Xg, get_rotations=lambda: q)
        for step, expect in ((999, False), (1001, False), (2000, True)):
            metrics = {"loss": torch.zeros(())}
            m._surf_step(model, step, metrics, {}, O)
            self.assertEqual("surf" in metrics, expect, step)
        metrics["loss"].backward()
        self.assertIsNotNone(Xg.grad)

    def test_gate_off_by_default(self):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        m = MCMCCityGSV2Metrics().instantiate()
        self.assertIsNone(m._geom_gate_step(None, 5000, None, {}, {}, None))


if __name__ == "__main__":
    unittest.main()
