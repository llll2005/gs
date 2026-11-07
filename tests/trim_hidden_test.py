"""trim_hidden（2026-11-07，使用者「沒貢獻的都刪掉」）：在畫面內卻一次都沒被光柵器算到（覆蓋 0）的顆粒＝被擋——純 CPU。"""
import types
import unittest

import torch

from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer as R


class _PC:
    def __init__(self, n):
        self.xyz = torch.zeros(n, 3); self.o = torch.full((n, 1), 0.9)

    @property
    def get_xyz(self):
        return self.xyz

    def get_opacities(self):
        return self.o


class _DC:
    def __init__(self, pc):
        self.pc = pc; self.pruned = None

    def _prune_points(self, mask, pc, opt):
        self.pruned = mask.clone()
        keep = ~mask
        pc.xyz = pc.xyz[keep]; pc.o = pc.o[keep]
        for k, v in list(pc._trim_state.items()):
            if torch.is_tensor(v) and v.shape[0] == mask.shape[0]:
                pc._trim_state[k] = v[keep]


class TrimHiddenTest(unittest.TestCase):
    def setUp(self):
        self.pc = _PC(5)
        self.r = types.SimpleNamespace(trim_hidden=True, trim_hidden_min_views=3, _trim_state=R._trim_state)
        self.mod = types.SimpleNamespace(gaussian_model=self.pc, gaussian_optimizers=None, density_controller=_DC(self.pc))
        # 顆粒：0 看得到；1 被擋（畫面內 4 次、覆蓋 0）；2 畫面內只 2 次且覆蓋 0（不夠 3 次）；3 從沒在畫面內（未觀測）；4 被擋 3 次
        R._acc_add(self.r, self.pc, torch.tensor([1., 0, 0, 0, 0]), torch.tensor([5., 0, 0, 0, 0]), torch.tensor([3, 2, 1, 0, 1]))
        R._acc_add(self.r, self.pc, torch.zeros(5), torch.zeros(5), torch.tensor([0, 2, 1, 0, 1]))
        R._acc_add(self.r, self.pc, torch.zeros(5), torch.zeros(5), torch.tensor([0, 2, 0, 0, 1]))
        R._acc_add(self.r, self.pc, torch.zeros(5), torch.zeros(5), torch.tensor([0, 2, 0, 0, 0]))

    def test_in_view_count(self):
        self.assertEqual(self.pc._trim_state["F"].tolist(), [1, 4, 2, 0, 3])

    def test_post_growth_prunes_only_hidden_enough_views(self):
        R._hidden_prune(self.r, 20000, self.mod, 1720)
        self.assertEqual(self.mod.density_controller.pruned.tolist(), [False, True, False, False, True])
        self.assertEqual(self.pc.get_xyz.shape[0], 3)
        self.assertEqual(float(self.pc._trim_state["F"].sum()), 0.0)          # 窗口重設


if __name__ == "__main__":
    unittest.main()
