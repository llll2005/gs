"""採樣控制顆數（cap_ctrl；2026-11-05）：選驗證視角、加減點的決定——純 CPU。"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from internal.utils import cap_ctrl


def _fake_colmap(n_cam=40):
    """相機 0..n_cam-1；1000 個點每點被 6 台看到（安全）；另 50 個點只被 38、39 兩台＋第三台看到（38、39 有風險）。"""
    imgs = {i + 1: SimpleNamespace(name=f"{i:04d}.png") for i in range(n_cam)}
    pts, k = {}, 0
    for j in range(1000):
        pts[k] = SimpleNamespace(image_ids=[1 + (j + t) % (n_cam - 2) for t in range(6)], error=0.5); k += 1
    for j in range(50):
        pts[k] = SimpleNamespace(image_ids=[39, 40, 1 + j % 10], error=0.5); k += 1
    return imgs, pts


class CapCtrlTest(unittest.TestCase):
    def test_select_avoids_risky_cameras(self):
        imgs, pts = _fake_colmap()
        names = [f"{i:04d}.png" for i in range(40)]
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "sparse", "0"))
            open(os.path.join(d, "sparse", "0", "points3D.bin"), "wb").close()
            with mock.patch("internal.utils.colmap.read_images_binary", return_value=imgs), \
                    mock.patch("internal.utils.colmap.read_points3D_binary", return_value=pts):
                hold, info = cap_ctrl.select_safe_holdout(d, names, frac=0.1, k_min=2, pool_mult=2, min_pts=10)
        self.assertEqual(info["K"], 4)
        self.assertEqual(len(hold), 4)
        self.assertNotIn("0038.png", hold)
        self.assertNotIn("0039.png", hold)

    def test_decide(self):
        kw = dict(cap_hard=2000, cap_min=100, step_frac=0.1, hole_hi=0.02, hole_lo=0.005, gain_hi=0.02, gain_lo=0.005)
        self.assertEqual(cap_ctrl.decide(None, 0.1, 0.0, 1000, 1000, **kw)[:2], ("持平", 1000))
        # 還在明顯進步、cap 綁住 => 加
        self.assertEqual(cap_ctrl.decide({"L1": 0.1, "act": "持平"}, 0.095, 0.0, 990, 1000, **kw)[:2], ("加", 1100))
        # 進步但 cap 沒綁住 => 不加
        self.assertEqual(cap_ctrl.decide({"L1": 0.1, "act": "持平"}, 0.095, 0.0, 800, 1000, **kw)[0], "持平")
        # 破洞多 => 加（即使進步小）
        self.assertEqual(cap_ctrl.decide({"L1": 0.1, "act": "持平"}, 0.0999, 0.05, 1000, 1000, **kw)[0], "加")
        # 上次加了、這次進步 < gain_lo => 加的點沒用，不再加（破洞也不算）
        self.assertEqual(cap_ctrl.decide({"L1": 0.1, "act": "加"}, 0.0999, 0.05, 1000, 1000, **kw)[0], "持平")
        # 平了且沒破洞 => 減
        self.assertEqual(cap_ctrl.decide({"L1": 0.1, "act": "持平"}, 0.0999, 0.001, 1000, 1000, **kw)[:2], ("減", 900))
        # 上下限
        self.assertEqual(cap_ctrl.decide({"L1": 0.1, "act": "持平"}, 0.09, 0.0, 1990, 1990, **kw)[1], 2000)
        self.assertEqual(cap_ctrl.decide({"L1": 0.1, "act": "持平"}, 0.0999, 0.0, 105, 105, **kw)[1], 100)


if __name__ == "__main__":
    unittest.main()
