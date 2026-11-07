"""★ 2026-11-08 view_balance（依覆蓋度加權抽訓練視角）的權重與抽樣（CPU）。"""
import unittest

import numpy as np
import torch

from internal.cameras.cameras import Cameras
from internal.utils.view_balance import coverage_view_weights, epoch_order


def _cams(xs):
    """全部朝 +z 看、位在 (x, 0, 0) 的針孔相機（world_to_camera：p_cam = p_world − C）。"""
    n = len(xs)
    R = torch.eye(3).repeat(n, 1, 1)
    T = torch.tensor([[-x, 0.0, 0.0] for x in xs])
    f = torch.full((n,), 100.0)
    return Cameras(R=R, T=T, fx=f, fy=f.clone(), cx=torch.full((n,), 50.0), cy=torch.full((n,), 50.0),
                   width=torch.full((n,), 100, dtype=torch.int16), height=torch.full((n,), 100, dtype=torch.int16),
                   appearance_id=torch.zeros(n, dtype=torch.int), normalized_appearance_id=torch.zeros(n),
                   distortion_params=None, camera_type=torch.zeros(n, dtype=torch.int8))


class ViewBalanceTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # 群 1 在 x≈1（兩台都看得到）、群 2 在 x≈4（只有 x=4 那台看得到）；z＝5、半視野 0.5 => 看得到 |x − C| < 2.5
        self.c1 = np.c_[rng.uniform(0.9, 1.1, 200), rng.uniform(-0.1, 0.1, 200), np.full(200, 5.0)]
        self.c2 = np.c_[rng.uniform(3.9, 4.1, 100), rng.uniform(-0.1, 0.1, 100), np.full(100, 5.0)]
        self.cams = _cams([0.0, 2.0, 4.0])          # 相機 0 看群 1；相機 1 看兩群；相機 2 只看群 2

    def test_rare_points_weigh_more(self):
        w, st = coverage_view_weights(self.cams, np.r_[self.c1, self.c2], gamma=1.0, clip=(0.01, 100.0))
        self.assertTrue(st["ok"])
        # γ=1：群 1 每點 c=2 => 各給相機 0、1 一半；群 2 每點 c=2（相機 1、2）=> 三台都是 100 份＋… 用比例檢查
        raw = np.array([200 / 2, 200 / 2 + 100 / 2, 100 / 2])
        np.testing.assert_allclose(w.numpy(), raw / raw.mean(), rtol=1e-5)
        w0, _ = coverage_view_weights(self.cams, np.r_[self.c1, self.c2], gamma=0.0, clip=(0.01, 100.0))
        raw0 = np.array([200, 300, 100])                  # γ=0：權重 ∝ 畫面內點數
        np.testing.assert_allclose(w0.numpy(), raw0 / raw0.mean(), rtol=1e-5)

    def test_clip_and_empty(self):
        w, st = coverage_view_weights(self.cams, np.r_[self.c1, self.c2], gamma=1.0, clip=(0.9, 1.1))
        self.assertLessEqual(float(w.max()), 1.1 + 1e-6)
        self.assertGreaterEqual(float(w.min()), 0.9 - 1e-6)
        w, st = coverage_view_weights(self.cams, np.zeros((0, 3)), gamma=1.0)
        self.assertFalse(st["ok"])
        self.assertTrue(torch.all(w == 1))

    def test_epoch_order(self):
        g = torch.Generator(); g.manual_seed(0)
        ref = torch.Generator(); ref.manual_seed(0)
        self.assertEqual(epoch_order(7, g, None), torch.randperm(7, generator=ref).tolist())   # 沒權重＝原本的 randperm（同一條亂數流）
        o = epoch_order(50, g, torch.tensor([1.0, 0.0, 3.0]))
        self.assertEqual(len(o), 50)
        self.assertNotIn(1, o)
        self.assertGreater(o.count(2), o.count(0))


if __name__ == "__main__":
    unittest.main()
