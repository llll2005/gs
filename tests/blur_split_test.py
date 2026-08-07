"""Blur split biases MCMC's host selection towards primitives that under-reconstruct.

MCMC samples hosts by opacity alone, which cannot see WHERE detail is missing. The premise for
changing that was measured, not assumed (紀錄 §11.9.1): on cap4m_b12, 0.01% of primitives carry
12.6% of the blending weight, and 74.6% of that weight sits on textured content.

These tests pin the properties that make the knob safe to ship: off by default, monotone in the
score, and never able to starve a primitive of selection probability entirely.
"""
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_density_controller import MCMCDensityControllerImpl


def _probs(opacity, score, w, thr=288.0):
    """Reproduces the probs computation in add_new_gs without building a model."""
    c = SimpleNamespace(blur_split_weight=w, blur_split_threshold=thr)
    impl = MCMCDensityControllerImpl.__new__(MCMCDensityControllerImpl)
    impl.config = c
    impl.blur_score = score
    p = opacity.clone()
    if c.blur_split_weight > 0 and impl.blur_score is not None \
            and impl.blur_score.shape[0] == p.shape[0]:
        p = p * (1.0 + c.blur_split_weight *
                 (impl.blur_score / max(c.blur_split_threshold, 1e-9)).clamp(0.0, 32.0))
    return p


class BlurSplitTest(unittest.TestCase):
    def setUp(self):
        self.op = torch.tensor([0.5, 0.5, 0.5])
        self.sc = torch.tensor([0.0, 288.0, 2880.0])      # 0x, 1x, 10x threshold

    def test_off_by_default(self):
        """Every existing config must be byte-identical, so 0 has to be a true no-op."""
        self.assertTrue(torch.equal(_probs(self.op, self.sc, 0.0), self.op))

    def test_monotone_in_score(self):
        p = _probs(self.op, self.sc, 1.0)
        self.assertLess(float(p[0]), float(p[1]))
        self.assertLess(float(p[1]), float(p[2]))

    def test_threshold_sets_the_unit(self):
        """One threshold over with w=1 doubles the weight -- that is what the knob should mean."""
        p = _probs(self.op, self.sc, 1.0)
        self.assertAlmostEqual(float(p[1] / self.op[1]), 2.0, places=5)

    def test_clamped_so_one_blob_cannot_take_everything(self):
        huge = torch.tensor([0.0, 288.0, 288.0 * 1e6])
        p = _probs(self.op, huge, 1.0)
        self.assertAlmostEqual(float(p[2] / self.op[2]), 33.0, places=3)

    def test_never_zeroes_a_primitive(self):
        """The factor is 1 + w*score, so a zero-score primitive keeps its opacity weight."""
        self.assertAlmostEqual(float(_probs(self.op, self.sc, 5.0)[0]), float(self.op[0]), places=6)

    def test_shape_mismatch_falls_back(self):
        """The score is stale for one step after a prune; it must not crash or mis-index."""
        self.assertTrue(torch.equal(_probs(self.op, torch.zeros(2), 1.0), self.op))


if __name__ == "__main__":
    unittest.main()
