"""★ 2026-11-11 SfM 三角形校正單目深度（tools/make_tri_depths.py）與 dataparser 的可靠度過濾。"""
import os, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from make_tri_depths import tri_correct   # noqa: E402


def plane_inv(u, v, fx, fy, cx, cy, n, d):
    """平面 n·X = d 的逆深度：1/z = n·K^-1[u,v,1] / d（影像座標的仿射函數）。"""
    return (n[0] * (u - cx) / fx + n[1] * (v - cy) / fy + n[2]) / d


class TriCorrectTest(unittest.TestCase):
    W, H, fx, fy, cx, cy = 640, 360, 500.0, 500.0, 320.0, 180.0

    def _pts(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        return np.stack([rng.uniform(0, self.W, n), rng.uniform(0, self.H, n)], 1)

    def _grid(self, down=2):
        h, w = self.H // down, self.W // down
        return np.meshgrid((np.arange(w) + 0.5) * down, (np.arange(h) + 0.5) * down)

    def test_plane_exact_without_mono(self):
        n, d = np.array([0.2, -0.3, 1.0]), 5.0
        uv = self._pts()
        zc = 1.0 / plane_inv(uv[:, 0], uv[:, 1], self.fx, self.fy, self.cx, self.cy, n, d)
        inv, cls = tri_correct(uv, zc, None, None, self.W, self.H, self.fx, 2)
        uu, vv = self._grid()
        gt = plane_inv(uu, vv, self.fx, self.fy, self.cx, self.cy, n, d)
        ins = (cls & 7) < 4
        self.assertGreater(ins.mean(), 0.8)
        self.assertLess(np.abs(inv[ins] / gt[ins] - 1).max(), 1e-5)

    def test_affine_mono_error_is_removed(self):
        n, d = np.array([0.1, 0.4, 1.0]), 3.0
        uv = self._pts(seed=1)
        true_pts = plane_inv(uv[:, 0], uv[:, 1], self.fx, self.fy, self.cx, self.cy, n, d)
        uu, vv = self._grid()
        gt = plane_inv(uu, vv, self.fx, self.fy, self.cx, self.cy, n, d)
        mono_pts, mono_grid = 0.7 * true_pts + 0.05, 0.7 * gt + 0.05      # 單目有 30% 縮放誤差＋偏移
        inv, cls = tri_correct(uv, 1.0 / true_pts, mono_pts, mono_grid, self.W, self.H, self.fx, 2)
        ins = (cls & 7) < 4
        self.assertLess(np.abs(inv[ins] / gt[ins] - 1).max(), 1e-5)
        self.assertGreater(np.abs(mono_grid[ins] / gt[ins] - 1).min(), 0.05)   # 校正前差很多

    def test_slope_class(self):
        uv = self._pts(seed=2)
        flat = np.full(len(uv), 4.0)                                        # 正對相機的平面：斜率 0
        _, cls = tri_correct(uv, flat, None, None, self.W, self.H, self.fx, 2)
        self.assertTrue(((cls & 7)[(cls & 7) < 4] == 0).all())
        # 高低落差：左半 z=2、右半 z=6 => 跨界的三角形斜率大（>= 8）
        z = np.where(uv[:, 0] < self.W / 2, 2.0, 6.0)
        _, cls = tri_correct(uv, z, None, None, self.W, self.H, self.fx, 2)
        sl = cls & 7
        self.assertGreater((sl == 3).mean(), 0.02)
        self.assertGreater((sl == 0).mean(), 0.5)

    def test_outside_hull_is_class4(self):
        uv = self._pts(seed=3) * 0.5 + np.array([self.W * 0.25, self.H * 0.25])   # 點只在中央
        _, cls = tri_correct(uv, np.full(len(uv), 3.0), None, None, self.W, self.H, self.fx, 2)
        self.assertEqual(int(cls[0, 0] & 7), 4)
        self.assertEqual(int(cls[cls.shape[0] // 2, cls.shape[1] // 2] & 7), 0)


class DepthRelLoadTest(unittest.TestCase):
    def test_unreliable_pixels_zeroed(self):
        from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthBlockColmapDataParser as P
        with tempfile.TemporaryDirectory() as td:
            d = np.full((4, 6), 0.5, np.float32)
            rel = np.zeros((4, 6), np.uint8)
            rel[0, :] = 2            # 斜率級 2
            rel[1, :] = 1 + 8 * 3    # 斜率級 1、大小級 3
            rel[2, :] = 4 + 8 * 3    # 凸包外
            np.save(os.path.join(td, "d.npy"), d); np.save(os.path.join(td, "r.npy"), rel)
            sc = {"scale": 1.0, "offset": 0.0}
            plain = P.load_depth((os.path.join(td, "d.npy"), sc))
            self.assertTrue(bool((plain == 0.5).all()))
            a = P.load_depth((os.path.join(td, "d.npy"), sc, os.path.join(td, "r.npy"), 1, 3)).numpy()
            self.assertTrue((a[0] == 0).all() and (a[1] == 0.5).all() and (a[2] == 0).all() and (a[3] == 0.5).all())
            b = P.load_depth((os.path.join(td, "d.npy"), sc, os.path.join(td, "r.npy"), 1, 2)).numpy()
            self.assertTrue((b[1] == 0).all() and (b[3] == 0.5).all())


if __name__ == "__main__":
    unittest.main()
