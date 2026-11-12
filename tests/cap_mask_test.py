"""★ 2026-11-12 規格 3 §3.1 上限遮罩：metric `depth_cap_mask_dir` 與 `tools/make_cap_masks.py --kind flat`。"""
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
from tools.make_cap_masks import flat_mask, world_normals


def _batch(inv, name="0001.png", hw=(6, 8)):
    return (None, (name, torch.rand(3, *hw), None), inv)


def _scene(scale=1.0, W=320, H=240, down=2):
    """斜 45° 往下看的相機，場景＝地面 z=0 ＋ 一面朝相機的直牆 y=5（高 20）。回傳 (inv [H/down, W/down], 是否打到牆, 相機參數)。"""
    f = np.array([0.0, 1.0, -1.0]) / np.sqrt(2)
    x = np.array([1.0, 0.0, 0.0]); y = np.cross(f, x)
    R = np.stack([x, y, f])                                   # 世界 -> 相機（列＝相機軸）
    C = np.array([0.0, -10.0, 10.0]) * scale
    fx = fy = 200.0; cx, cy = W / 2, H / 2
    h, w = H // down, W // down
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * down, (np.arange(h) + 0.5) * down)
    d = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1) @ R   # 世界方向（相機 z 每單位）
    with np.errstate(divide="ignore", invalid="ignore"):
        sg = np.where(d[..., 2] < 0, -C[2] / d[..., 2], np.inf)
        sw = np.where(d[..., 1] > 0, (5.0 * scale - C[1]) / d[..., 1], np.inf)
    zw = C[2] + sw * d[..., 2]
    sw = np.where((zw >= 0) & (zw <= 20.0 * scale), sw, np.inf)
    s = np.minimum(sg, sw)
    inv = np.where(np.isfinite(s), 1.0 / s, 0.0).astype(np.float32)
    return inv, sw < sg, (fx, fy, cx, cy, 1.0 / down, 1.0 / down, R)


class CapMaskMetricTest(unittest.TestCase):
    def _m(self, **kw):
        return MCMCCityGSV2Metrics(depth_cap_reg=1.0, depth_cap_interval=1, depth_cap_from_iter=0, **kw).instantiate()

    def test_off_is_identical(self):
        inv = torch.rand(6, 8) + 0.1
        a = self._m().depth_cap_map(_batch(inv), 5)
        b = self._m(depth_cap_mask_dir="").depth_cap_map(_batch(inv), 5)
        self.assertTrue(torch.equal(a, b))

    def test_mask_zero_pixels_have_no_cap(self):
        with tempfile.TemporaryDirectory() as d:
            mk = np.full((3, 4), 255, np.uint8); mk[:, :2] = 0                 # 遮罩 3x4（左半 0）、影像 6x8 => 最近鄰放大
            Image.fromarray(mk).save(os.path.join(d, "0001.png"))
            inv = torch.full((6, 8), 0.5)
            m = self._m(depth_cap_mask_dir=d)
            cap = m.depth_cap_map(_batch(inv), 5)
            self.assertTrue(bool((cap[:, :4] == 0).all()))
            self.assertTrue(torch.allclose(cap[:, 4:], torch.full((6, 4), 0.5 / 1.01)))
            self.assertIs(m._cap_cur, cap)                                       # loss 用的就是遮罩後的圖
            self.assertAlmostEqual(m._cm_acc[1] / m._cm_acc[0], 0.5)             # 可觀測性：保留一半

    def test_missing_file_means_no_cap(self):
        with tempfile.TemporaryDirectory() as d:
            m = self._m(depth_cap_mask_dir=d)
            cap = m.depth_cap_map(_batch(torch.full((6, 8), 0.5), name="9999.png"), 5)
            self.assertTrue(bool((cap == 0).all()))
            self.assertEqual(m._cm_miss, {"9999.png"})
            m.depth_cap_map(_batch(torch.full((6, 8), 0.5), name="9999.png"), 6)  # 第二次不重讀、不重複計數
            self.assertEqual(len(m._cm_miss), 1)


class FlatMaskToolTest(unittest.TestCase):
    def test_ground_kept_wall_dropped(self):
        inv, wall, K = _scene()
        n, ok = world_normals(inv, *K)
        m = flat_mask(inv, *K)
        from scipy.ndimage import binary_erosion
        g_in = binary_erosion(~wall & (inv > 0), iterations=5)                   # 離交界 5 px 以上的純地面
        w_in = binary_erosion(wall, iterations=5)
        self.assertGreater(int(g_in.sum()), 100); self.assertGreater(int(w_in.sum()), 100)
        self.assertTrue(np.allclose(np.abs(n[g_in, 2]), 1.0, atol=1e-6))        # 地面法向 = ±z
        self.assertTrue(np.allclose(np.abs(n[w_in, 1]), 1.0, atol=1e-6))        # 牆法向 = ±y
        self.assertTrue(bool((m[g_in] == 255).all()))
        self.assertTrue(bool((m[w_in] == 0).all()))
        self.assertTrue(bool((m[:2] == 0).all() and (m[:, :2] == 0).all()))     # 鄰點不齊的邊框＝0

    def test_no_depth_is_zero(self):
        inv, _, K = _scene()
        inv = inv.copy(); inv[30:40, 50:60] = 0
        m = flat_mask(inv, *K)
        self.assertTrue(bool((m[28:42, 48:62][2:-2, 2:-2] == 0).all()))         # 無深度與其 ±2 px 鄰點都不設

    def test_scene_scale_invariant(self):
        a = flat_mask(*(lambda s: (s[0], *s[2]))(_scene(1.0)))
        b = flat_mask(*(lambda s: (s[0], *s[2]))(_scene(10.0)))
        self.assertTrue(np.array_equal(a, b))                                    # 場景放大 10 倍逐像素相同（沒有場景單位）


if __name__ == "__main__":
    unittest.main()
