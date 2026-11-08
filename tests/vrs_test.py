import unittest
from types import SimpleNamespace

import torch

from internal.metrics import vrs as V


class VrsTest(unittest.TestCase):
    def test_keep_probs_mean_and_clamp(self):
        g = torch.Generator().manual_seed(0)
        s = torch.rand(57, 100, generator=g) ** 3
        pi = V.keep_probs(s, 0.6, 0.25)
        self.assertAlmostEqual(float(pi.mean()), 0.6, places=3)
        self.assertGreaterEqual(float(pi.min()), 0.25 - 1e-6); self.assertLessEqual(float(pi.max()), 1.0)
        self.assertTrue(torch.allclose(V.keep_probs(s * 1000, 0.6, 0.25), pi, atol=1e-4))     # 只看相對大小
        self.assertTrue(torch.allclose(V.keep_probs(torch.zeros(3, 4), 0.6, 0.25), torch.full((3, 4), 0.6)))

    def test_scale_grad(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = V.scale_grad(x, torch.tensor([3.0, 0.5]))
        self.assertTrue(torch.equal(y.detach(), x.detach()))
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.tensor([3.0, 0.5])))

    def test_tile_helpers(self):
        x = torch.arange(20 * 33, dtype=torch.float32).reshape(20, 33)
        tm = V.tile_mean(x, 16)
        self.assertEqual(tuple(tm.shape), (2, 3))
        self.assertAlmostEqual(float(tm[1, 2]), float(x[16:, 32:].mean()), places=4)
        p = V.tile_to_pix(torch.tensor([[1, 0, 1], [0, 1, 0]]), 20, 33, 16)
        self.assertEqual(tuple(p.shape), (20, 33)); self.assertEqual(int(p[17, 20]), 1); self.assertEqual(int(p[0, 16]), 0)
        flat = torch.zeros(3, 32, 32); flat[:, :, 16:] = torch.rand(3, 32, 16)
        ts = V.texture_score(flat, 16)
        self.assertLess(float(ts[0, 0]), float(ts[0, 1]))

    def test_metric_skip_and_apply(self):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        m = MCMCCityGSV2Metrics(vrs_mode="error", vrs_keep=0.5, vrs_min_keep=0.25, vrs_from_iter=0).instantiate()
        gt = torch.rand(3, 40, 48)
        self.assertIsNone(m.vrs_skip("a", gt, 10))                                # error 模式第一次：沒有紀錄 => 全畫
        r = torch.rand(3, 40, 48, requires_grad=True)
        b, o = m._vrs_apply(10, (None, ("a", gt)), {"render": r})
        self.assertIs(o["render"], r)                                            # 全畫的步不動 loss
        sk = m.vrs_skip("a", gt, 11)
        self.assertIsNotNone(sk); self.assertEqual(sk.numel(), 40 * 48)
        b, o = m._vrs_apply(11, (None, ("a", gt)), {"render": r})
        kp = (sk.view(40, 48) == 0)
        loss = (o["render"] - b[1][1]).abs().sum()
        loss.backward()
        self.assertTrue(bool((r.grad[:, ~kp] == 0).all()))                       # 跳過的像素零梯度
        self.assertTrue(bool((r.grad[:, kp].abs() >= 1.0 - 1e-6).all()))        # 保留的梯度 ×1/π >= 1


if __name__ == "__main__":
    unittest.main()
