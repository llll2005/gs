"""累加式 trim 的判準 trim_running_score（vc／v／inv_vc；2026-11-03）——純 CPU。"""
import unittest
from types import SimpleNamespace

import torch

from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer


class _PC:
    def __init__(self, V, C):
        self.get_xyz = torch.zeros(V.shape[0], 3)
        self._trim_state = {"V": V.clone(), "C": C.clone(), "views": 3}


def _run(score, V, C, ratio=0.25):
    r = object.__new__(SepDepthTrim2DGSRenderer)
    r.prune_ratio = ratio
    r.trim_running_score = score
    r.trim_protect_vc = -1.0
    r.trim_soft_rho = 0.0
    r.trim_merge_first = False
    got = {}
    pc = _PC(V, C)
    dc = SimpleNamespace(_prune_points=lambda mask, pc_, opts: got.setdefault("mask", mask.clone()))
    module = SimpleNamespace(gaussian_model=pc, density_controller=dc, gaussian_optimizers=[])
    r._running_trim(5000, module, 15000)
    return got["mask"]


class TrimScoreTest(unittest.TestCase):
    # 8 顆：V＝總貢獻、C＝覆蓋像素 => V/C＝[1, 2, 3, 4, 0.5, 0.25, 8, 6]；最後一顆沒被看到（C=0）
    V = torch.tensor([10., 20., 30., 40., 50., 25., 16., 0.])
    C = torch.tensor([10., 10., 10., 10., 100., 100., 2., 0.])

    def test_vc_prunes_lowest_value_per_pixel(self):
        m = _run("vc", self.V, self.C)                    # k = 0.25 x 8 = 2 => V/C 最低的兩顆（0.25、0.5）＝大而淡的
        self.assertEqual(m.nonzero().flatten().tolist(), [4, 5])

    def test_v_prunes_lowest_total_value(self):
        m = _run("v", self.V, self.C)                     # V 最低的兩顆（10、16）
        self.assertEqual(m.nonzero().flatten().tolist(), [0, 6])

    def test_inv_vc_prunes_highest_value_per_pixel(self):
        m = _run("inv_vc", self.V, self.C)                # V/C 最高的兩顆（8、4）＝小而實的
        self.assertEqual(m.nonzero().flatten().tolist(), [3, 6])

    def test_unseen_never_pruned(self):
        for s in ("vc", "v", "inv_vc"):
            self.assertFalse(bool(_run(s, self.V, self.C)[7]), s)

    def test_bad_value_rejected(self):
        with self.assertRaises(ValueError):
            SepDepthTrim2DGSRenderer(trim_running_score="cv")


if __name__ == "__main__":
    unittest.main()
