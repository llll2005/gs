"""熵閘門「看得到才往 1 推」（2026-11-07）：renderer 的 VS 快照與 metric 的閘門——純 CPU。"""
import types
import unittest

import torch

from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2MetricsModule as M
from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer as R


class _PC:
    def __init__(self, n):
        self.xyz = torch.zeros(n, 3); self.o = torch.full((n, 1), 0.5)

    @property
    def get_xyz(self):
        return self.xyz

    def get_opacities(self):
        return self.o


def _cfg(tau):
    return types.SimpleNamespace(plane_reg=0.0, geom_gate_tau=-1.0, plane_from_iter=1000, depth_support=False,
                                 depth_support_from_iter=1000, vis_gate_tau=tau, vis_gate_from_iter=1000)


class VisGateTest(unittest.TestCase):
    def setUp(self):
        self.pc = _PC(4)
        self.r = types.SimpleNamespace(_trim_state=R._trim_state)
        # 顆粒：0 全看得到（T·α＝o）；1 大多被擋（T·α＝0.002·o）；2 沒被算到（C＝0）；3 一半
        self.pc._trim_state = {"V": torch.tensor([10 * 0.5, 10 * 0.001, 0., 4 * 0.25]), "C": torch.tensor([10., 10., 0., 4.]), "views": 3}
        R._vis_snap(self.r, self.pc, 1000)

    def test_snapshot(self):
        st = self.pc._trim_state
        self.assertTrue(torch.allclose(st["VS"], torch.tensor([1.0, 0.002, 0.0, 0.5])))
        self.assertEqual(st["VW"].tolist(), [1.0, 1.0, 0.0, 1.0])

    def test_gate_excludes_only_low_vs_with_info(self):
        me = types.SimpleNamespace(config=_cfg(0.01))
        g = M._geom_gate_step(me, self.pc, 2000, (None,), {}, {}, self.pc.o)
        self.assertEqual(g.tolist(), [True, False, True, True])          # 沒資訊（C＝0）的不進閘門外

    def test_gate_pads_new_points_and_off_before_from_iter(self):
        self.pc.xyz = torch.zeros(6, 3); self.pc.o = torch.full((6, 1), 0.5)       # MCMC 加了 2 顆（快照還是 4 顆長）
        me = types.SimpleNamespace(config=_cfg(0.01))
        g = M._geom_gate_step(me, self.pc, 2000, (None,), {}, {}, self.pc.o)
        self.assertEqual(g.tolist(), [True, False, True, True, True, True])
        self.assertIsNone(M._geom_gate_step(types.SimpleNamespace(config=_cfg(0.01)), self.pc, 500, (None,), {}, {}, self.pc.o))
        self.assertIsNone(M._geom_gate_step(types.SimpleNamespace(config=_cfg(-1.0)), self.pc, 2000, (None,), {}, {}, self.pc.o))


if __name__ == "__main__":
    unittest.main()
