"""internal/metrics/geom_gate.py 與 MCMCCityGSV2MetricsModule._geom_gate_step（2026-10-29）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.metrics.geom_gate import morton_neighbors, quat_normals, plane_q, plane_loss, DepthSupport


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

    def test_gate_off_by_default(self):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        m = MCMCCityGSV2Metrics().instantiate()
        self.assertIsNone(m._geom_gate_step(None, 5000, None, {}, {}, None))


if __name__ == "__main__":
    unittest.main()
