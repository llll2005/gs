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



class SoftAndGateFallbackTest(unittest.TestCase):
    def test_soft_factor(self):
        cfg = types.SimpleNamespace(sfm_vis_noup=False, sfm_vis_stream=1, sfm_vis_stream_decay=1.0, sfm_vis_px=4.0, sfm_vis_eps=0.01,
                                    sfm_vis_free_min=2, sfm_vis_free_frac=0.5, sfm_vis_stream_soft=2.0)
        s = types.SimpleNamespace(config=cfg, _ip_step=0, _v5_free=torch.tensor([0., 2., 4., 1.]), _v5_cov=torch.tensor([4., 4., 4., 1.]))
        C._v5_classify(s, torch.zeros(4, 3))
        # phi＝0, 0.5, 1, 0（證據 < 2 => 0）=> (1-phi)^2＝1, 0.25, 0 -> 地板 0.01, 1
        self.assertEqual([round(x, 4) for x in s._v5_fac.tolist()], [1.0, 0.25, 0.01, 1.0])


class DepthSupportFallbackTest(unittest.TestCase):
    def setUp(self):
        from internal.metrics.geom_gate import DepthSupport
        self.DS = DepthSupport
        # 相機 x_cam = x_world；fx=fy=10、cx=cy=5、10x10；先驗深度圖 inv：左半邊有效（深度 10），右半邊無效（0）
        self.inv = torch.zeros(10, 10); self.inv[:, :5] = 0.1
        self.w2c = torch.eye(4)
        # 第 0 顆：左半、深度 5（比先驗淺 50% => 在前方）；第 1 顆：右半、深度 5（沒有先驗）；第 2 顆：右半、深度 10
        self.X = torch.tensor([[-1., 0., 5.], [1., 0., 5.], [2., 0., 10.]])

    def run_ds(self, sfm, n=4):
        ds = self.DS(window=3, min_count=3, rule="vote")
        for st in range(n):
            ds.update(st, self.X, self.w2c, 10., 10., 5., 5., 10, 10, self.inv, sfm=sfm)
        return ds

    def test_without_sfm_is_unchanged(self):
        ds = self.run_ds(None)
        self.assertEqual(ds.support.tolist(), [False, True, True])     # 只有左半那顆被先驗判在前方；右半沒觀測 => 不判
        self.assertEqual(ds.cnt.sum().item(), 0.0)                     # 視窗結束已清零

    def test_sfm_fallback_only_where_prior_missing(self):
        sf = (torch.tensor([False, True, False]), torch.tensor([True, True, True]))
        ds = self.run_ds(sf)
        # 第 0 顆有先驗 => 不看 SfM（仍判前方）；第 1 顆沒先驗、SfM 說空區 => 判前方；第 2 顆沒先驗、SfM 說不是空區 => 有支撐
        self.assertEqual(ds.support.tolist(), [False, False, True])
        self.assertGreater(ds.last["alt"], 0.0)

    def test_never_ignores_sfm_only_observations(self):
        sf = (torch.tensor([False, False, False]), torch.tensor([True, True, True]))
        ds = self.run_ds(sf)
        self.assertEqual(ds.support.tolist()[1:], [True, True])       # 只有 SfM 觀測、沒有先驗 => 不會被判「從未貼面」

    def test_mean_rule_refuses_fallback(self):
        ds = self.DS(window=3, min_count=3, rule="mean")
        with self.assertRaises(ValueError):
            ds.update(0, self.X, self.w2c, 10., 10., 5., 5., 10, 10, self.inv, sfm=(torch.zeros(3, dtype=torch.bool),) * 2)


if __name__ == "__main__":
    unittest.main()
