"""11-09 從 57 臂萃取的三個機制（使用者「都排」）：法向放大、BH 的無因次門檻、熵閘門 ∧ v4 地下票。"""
import os
import sys
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from internal.gaussian_splatting import normal_gain_step  # noqa: E402
from internal.density_controllers.mcmc_2dgs_density_controller import behind_above_params  # noqa: E402
from internal.metrics.mcmc_citygsv2_metrics import vis_gate_off  # noqa: E402


def _normals(q):
    q = q / q.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([2 * (x * z + r * y), 2 * (y * z - r * x), 1 - 2 * (x * x + y * y)], -1)


class NormalGainTest(unittest.TestCase):
    def test_only_normal_component_scaled(self):
        g = torch.Generator().manual_seed(0)
        q = torch.randn(500, 4, generator=g)
        m0 = torch.randn(500, 3, generator=g)
        d = torch.randn(500, 3, generator=g) * 0.01
        m = m0 + d
        normal_gain_step(m, m0, q, 2.0)
        n = _normals(q)
        dn0 = (d * n).sum(-1, keepdim=True)
        new = m - m0
        dn1 = (new * n).sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(dn1, 2.0 * dn0, atol=1e-6))                       # 法向 x2
        self.assertTrue(torch.allclose(new - dn1 * n, d - dn0 * n, atol=1e-6))          # 切向不變

    def test_gain_one_and_mismatch_are_noops(self):
        m0 = torch.zeros(4, 3); m = torch.ones(4, 3); q = torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1)
        normal_gain_step(m, m0, q, 1.0)
        self.assertTrue(torch.equal(m, torch.ones(4, 3)))
        normal_gain_step(m, torch.zeros(3, 3), q, 2.0)                                  # N 對不上（剛增生）=> 不動
        self.assertTrue(torch.equal(m, torch.ones(4, 3)))

    def test_identity_rotation_normal_is_z(self):
        m0 = torch.zeros(1, 3); m = torch.tensor([[0.1, 0.2, 0.3]]); q = torch.tensor([[1.0, 0, 0, 0]])
        normal_gain_step(m, m0, q, 3.0)
        self.assertTrue(torch.allclose(m, torch.tensor([[0.1, 0.2, 0.9]])))


class BehindAboveTest(unittest.TestCase):
    def test_nn_multiples(self):
        xs, ys = np.meshgrid(np.arange(50) * 0.5, np.arange(50) * 0.5)
        P = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], -1)               # 最近鄰間距 0.5
        c = types.SimpleNamespace(sfm_vis_behind_max_above_nn=16.0, sfm_vis_ground_cell_nn=6.0, sfm_vis_behind_max_above=-1.0)
        gc, above, _ = behind_above_params(c, P)
        self.assertAlmostEqual(gc, 3.0, places=6)
        self.assertAlmostEqual(above, 8.0, places=6)

    def test_scene_unit_fallback(self):
        c = types.SimpleNamespace(sfm_vis_behind_max_above_nn=-1.0, sfm_vis_behind_max_above=0.05)
        gc, above, _ = behind_above_params(c, np.zeros((3, 3)))
        self.assertEqual((gc, above), (0.02, 0.05))


class VisGateUnderTest(unittest.TestCase):
    def setUp(self):
        self.VS = torch.tensor([0.001, 0.001, 0.5, 0.001])
        self.VW = torch.tensor([1.0, 1.0, 1.0, 0.0])

    def test_plain_gate_unchanged(self):
        off = vis_gate_off(self.VS, self.VW, 0.002)
        self.assertEqual(off.tolist(), [True, True, False, False])

    def test_needs_under(self):
        under = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)
        off = vis_gate_off(self.VS, self.VW, 0.002, under, need_under=True)
        self.assertEqual(off.tolist(), [True, False, False, False])

    def test_missing_or_misaligned_under_gates_nothing(self):
        self.assertFalse(bool(vis_gate_off(self.VS, self.VW, 0.002, None, need_under=True).any()))
        bad = torch.ones(3, dtype=torch.uint8)
        self.assertFalse(bool(vis_gate_off(self.VS, self.VW, 0.002, bad, need_under=True).any()))


if __name__ == "__main__":
    unittest.main()
