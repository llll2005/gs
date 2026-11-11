"""★ 2026-11-11 逐貢獻深度上限（metric depth_cap_*）：上限圖怎麼建、哪些步算、loss 怎麼接。光柵器那一半用 tools/check_depth_cap.py（要 GPU）。"""
import unittest

import torch

from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics


def _batch(inv, hw=(6, 8)):
    return (None, ("0001.png", torch.rand(3, *hw), None), inv)


class DepthCapMetricTest(unittest.TestCase):
    def test_off_by_default_and_interval(self):
        m = MCMCCityGSV2Metrics().instantiate()
        self.assertIsNone(m.depth_cap_map(_batch(torch.full((6, 8), 0.5)), 2000))
        m = MCMCCityGSV2Metrics(depth_cap_reg=1.0, depth_cap_interval=8, depth_cap_from_iter=1000).instantiate()
        self.assertIsNone(m.depth_cap_map(_batch(torch.full((6, 8), 0.5)), 992))     # 還沒到起始步
        self.assertIsNone(m.depth_cap_map(_batch(torch.full((6, 8), 0.5)), 2001))    # 不是 8 的倍數
        self.assertIsNotNone(m.depth_cap_map(_batch(torch.full((6, 8), 0.5)), 2000))

    def test_map_values_invalid_and_resize(self):
        m = MCMCCityGSV2Metrics(depth_cap_reg=1.0, depth_cap_interval=1, depth_cap_from_iter=0, depth_cap_margin=0.02).instantiate()
        inv = torch.full((3, 4), 0.5); inv[0, :] = 0.0                               # 第一列無效（不可靠的像素）
        cap = m.depth_cap_map(_batch(inv), 5)                                        # 深度圖 3x4、影像 6x8 => 最近鄰放大
        self.assertEqual(tuple(cap.shape), (6, 8))
        self.assertTrue(bool((cap[:2] == 0).all()))
        self.assertTrue(torch.allclose(cap[2:], torch.full((4, 8), 0.5 / 1.02)))
        self.assertTrue(cap.is_contiguous() and cap.dtype == torch.float32)
        self.assertIsNone(m.depth_cap_map(_batch(None), 5))                          # 沒有深度圖 => 不算

    def test_loss_added_only_with_map_and_channel(self):
        m = MCMCCityGSV2Metrics(depth_cap_reg=2.0, depth_cap_interval=4, depth_cap_from_iter=0).instantiate()
        inv = torch.full((6, 8), 0.5); inv[:, :4] = 0.0                              # 左半不設上限
        b = _batch(inv)
        B = torch.full((1, 6, 8), 0.01, requires_grad=True)
        metrics = {"loss": torch.zeros(())}; pbar = {}
        m._depth_cap_step(8, b, {"rend_cap": B}, metrics, pbar)                      # 沒先取上限圖 => 不動
        self.assertEqual(float(metrics["loss"]), 0.0)
        m.depth_cap_map(b, 8)
        m._depth_cap_step(8, b, {}, metrics, pbar)                                   # 渲染器沒交出通道 => 不動
        self.assertEqual(float(metrics["loss"]), 0.0)
        m._depth_cap_step(8, b, {"rend_cap": B}, metrics, pbar)
        self.assertAlmostEqual(float(metrics["loss"]), 2.0 * 4 * 0.01, places=6)     # 權重 x 間隔 x 設了上限的像素上的平均
        metrics["loss"].backward()
        self.assertTrue(bool((B.grad[0, :, :4] == 0).all()) and bool((B.grad[0, :, 4:] > 0).all()))
        m.depth_cap_map(b, 9)                                                        # 下一步不算 => 上限圖清掉
        m2 = {"loss": torch.zeros(())}
        m._depth_cap_step(9, b, {"rend_cap": B}, m2, {})
        self.assertEqual(float(m2["loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
