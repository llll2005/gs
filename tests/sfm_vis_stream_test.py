"""sfm_vis_stream（v5，2026-11-15）：每步用當前訓練相機投浮塊票、逐顆累積，事件時用累積票判浮塊——純 CPU。"""
import types
import unittest

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl as C


def make_vis():
    """一台相機：看向 +z（x_cam = x_world），fx=fy=100、cx=cy=50、100x100 像素；cell 50 => 2x2 格，每格的 SfM 觀測深度都是 10。"""
    d = torch.full((1, 2, 2), 10.0, dtype=torch.float16)
    return {"R": torch.eye(3)[None], "T": torch.zeros(1, 3), "fx": torch.tensor([100.]), "fy": torch.tensor([100.]),
            "cx": torch.tensor([50.]), "cy": torch.tensor([50.]), "W": torch.tensor([100.]), "H": torch.tensor([100.]),
            "dmin": d, "dmax": d.clone(), "cell": 50, "step": -1, "name2k": {"cam0.png": 0}}


class SfmVisStreamTest(unittest.TestCase):
    def setUp(self):
        cfg = types.SimpleNamespace(sfm_vis_noup=False, sfm_vis_stream=1, sfm_vis_stream_decay=0.5, sfm_vis_px=4.0,
                                    sfm_vis_free_min=2, sfm_vis_free_frac=0.5, sfm_birth_from_iter=0, sfm_birth_until_iter=-1)
        self.s = types.SimpleNamespace(config=cfg, _vis=make_vis(), _ip_step=0)
        self.s._v5_window = types.MethodType(lambda self, step: True, self.s)
        # 第 0 顆：深度 5（在 SfM 點 10 的前面 => 空區票）；第 1 顆：深度 10（在表面上）；第 2 顆：深度 20（在後面）；第 3 顆：畫面外
        X = torch.tensor([[0., 0., 5.], [0., 0., 10.], [0., 0., 20.], [50., 0., 1.]])
        self.gm = types.SimpleNamespace(gaussians={"means": X})

    def vote(self, n=1, name="cam0.png"):
        for i in range(n):
            C._v5_vote(self.s, self.gm, (None, (name, None, None), None), i)

    def test_accumulates_free_votes_only_in_front(self):
        self.vote(3)
        self.assertEqual(self.s._v5_free.tolist(), [3.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.s._v5_cov.tolist(), [3.0, 3.0, 3.0, 0.0])

    def test_classify_floater_and_decay(self):
        self.vote(3)
        fl, un, cv, iv = C._v5_classify(self.s, self.gm.gaussians["means"])
        self.assertEqual(fl.tolist(), [True, False, False, False])
        self.assertFalse(bool(un.any()))                               # 地下票一律不判
        self.assertEqual(self.s._v5_free.tolist(), [1.5, 0.0, 0.0, 0.0])  # 判完 x 0.5
        fl2, _, _, _ = C._v5_classify(self.s, self.gm.gaussians["means"])
        self.assertEqual(fl2.tolist(), [False, False, False, False])  # 1.5 < free_min 2 => 淡出

    def test_one_vote_is_not_enough(self):
        self.vote(1)
        fl, _, _, _ = C._v5_classify(self.s, self.gm.gaussians["means"])
        self.assertFalse(bool(fl.any()))                               # free_min 2

    def test_unknown_camera_does_not_vote(self):
        self.vote(2, name="not_in_list.png")
        self.assertIsNone(getattr(self.s, "_v5_free", None))
        self.assertEqual(self.s._v5_miss, 2)

    def test_inherit_from_parent(self):
        self.vote(3)
        C._v5_inherit(self.s, torch.tensor([1, 3]), torch.tensor([0, 0]))   # 第 1、3 顆是第 0 顆的副本
        self.assertEqual(self.s._v5_free.tolist(), [3.0, 3.0, 0.0, 3.0])
        C._v5_inherit(self.s, torch.tensor([2]), None)                      # 沒有母點對照 => 歸零
        self.assertEqual(self.s._v5_cov.tolist()[2], 0.0)

    def test_length_mismatch_resets(self):
        self.vote(2)
        self.gm.gaussians["means"] = torch.cat([self.gm.gaussians["means"], torch.tensor([[0., 0., 5.]])])
        self.vote(1)
        self.assertEqual(self.s._v5_free.tolist(), [1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(self.s._v5_reset, 1)

    def test_refuses_with_noup(self):
        self.s.config.sfm_vis_noup = True
        with self.assertRaises(RuntimeError):
            self.vote(1)


if __name__ == "__main__":
    unittest.main()
