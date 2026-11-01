"""internal/metrics/mcmc_citygsv2_metrics.see_through_loss（10-31；11-02 拆項權重）——純 CPU。"""
import unittest

import torch

from internal.metrics.mcmc_citygsv2_metrics import see_through_loss


def _case():
    # 4 個 A 像素：0 沒蓋滿（α 0.2）、1 蓋滿且比單目深 2 倍（看穿）、2 蓋滿且比單目淺（不罰）、3 不在 A
    alpha = torch.tensor([[0.2, 1.0, 1.0, 0.0]], requires_grad=True)
    z = torch.tensor([[1.0, 2.0, 0.5, 1.0]], requires_grad=True)
    inv_gt = torch.ones(1, 4)                  # 單目深度 1
    A = torch.tensor([[True, True, True, False]])
    return alpha, z, inv_gt, A


class SeeThroughTest(unittest.TestCase):
    def test_both_terms_default(self):
        a, z, g, A = _case()
        loss, st = see_through_loss(a, z, g, A, 0.1)
        hole = 0.8 + 0.0 + 0.0
        deep = 1 / 1.1 - 1 / 2.0
        self.assertAlmostEqual(float(loss), (hole + deep) / 3, places=5)
        self.assertAlmostEqual(st["holes"], 1 / 3, places=5)
        self.assertAlmostEqual(st["deeper"], 1 / 3, places=5)

    def test_hole_only_has_no_depth_gradient(self):
        a, z, g, A = _case()
        loss, st = see_through_loss(a, z, g, A, 0.1, w_hole=1.0, w_deep=0.0)
        self.assertAlmostEqual(float(loss), 0.8 / 3, places=5)
        loss.backward()
        self.assertTrue(z.grad is None or float(z.grad.abs().sum()) == 0.0)
        self.assertAlmostEqual(st["deeper"], 1 / 3, places=5)       # 統計照常算

    def test_deeper_only_has_no_alpha_gradient(self):
        a, z, g, A = _case()
        loss, _ = see_through_loss(a, z, g, A, 0.1, w_hole=0.0, w_deep=1.0)
        self.assertAlmostEqual(float(loss), (1 / 1.1 - 1 / 2.0) / 3, places=5)
        loss.backward()
        self.assertTrue(a.grad is None or float(a.grad.abs().sum()) == 0.0)
        self.assertGreater(float(z.grad[0, 1]), 0.0)                # 看穿的像素被往前拉（深度梯度為正 => 減深度）


if __name__ == "__main__":
    unittest.main()
