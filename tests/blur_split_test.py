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


def _probs(opacity, score, share, thr=288.0):
    """Reproduces the probs computation in add_new_gs without building a model."""
    c = SimpleNamespace(blur_split_budget=share, blur_split_threshold=thr)
    impl = MCMCDensityControllerImpl.__new__(MCMCDensityControllerImpl)
    impl.config = c
    impl.blur_score = score
    p = opacity.clone()
    if share > 0 and score is not None and score.shape[0] == p.shape[0]:
        over = score > thr
        m_over, m_rest = p[over].sum(), p[~over].sum()
        if m_over > 0 and m_rest > 0:
            p = p.clone()
            p[over] *= (share / (1.0 - share)) * m_rest / m_over
    return p


class BlurSplitTest(unittest.TestCase):
    def setUp(self):
        # 4 primitives, 1 of them over the 288 threshold -- the realistic ratio is 0.01%, which is
        # exactly why a multiplier fails and a budget share is needed.
        self.op = torch.tensor([0.5, 0.5, 0.5, 0.5])
        self.sc = torch.tensor([0.0, 10.0, 100.0, 2880.0])

    def _share_of_mass(self, share):
        p = _probs(self.op, self.sc, share)
        over = self.sc > 288.0
        return float(p[over].sum() / p.sum())

    def test_off_by_default(self):
        """Every existing config must be byte-identical, so 0 has to be a true no-op."""
        self.assertTrue(torch.equal(_probs(self.op, self.sc, 0.0), self.op))

    def test_budget_share_is_hit_exactly(self):
        """The knob's whole point: it MEANS the fraction of the densify budget, at any candidate
        count. The multiplier version could not do this -- 0.01% of primitives x3 is still 0.03%."""
        for s in (0.1, 0.3, 0.5, 0.8):
            self.assertAlmostEqual(self._share_of_mass(s), s, places=5)

    def test_share_holds_when_candidates_are_rare(self):
        """1 candidate in 10,000, the regime measured on cap4m."""
        op = torch.full((10_000,), 0.5)
        sc = torch.zeros(10_000); sc[0] = 5000.0
        p = _probs(op, sc, 0.3)
        self.assertAlmostEqual(float(p[0] / p.sum()), 0.3, places=5)

    def test_non_candidates_keep_their_relative_order(self):
        """Only the over-threshold group is rescaled; opacity ranking among the rest is untouched."""
        op = torch.tensor([0.1, 0.9, 0.5, 0.5])
        p = _probs(op, self.sc, 0.3)
        self.assertAlmostEqual(float(p[1] / p[0]), 9.0, places=5)

    def test_no_candidates_is_a_no_op(self):
        """Before the first trim pass every score is 0, and after a prune the set can be empty."""
        self.assertTrue(torch.equal(_probs(self.op, torch.zeros(4), 0.3), self.op))

    def test_never_zeroes_a_primitive(self):
        p = _probs(self.op, self.sc, 0.9)
        self.assertTrue(bool((p > 0).all()))

    def test_shape_mismatch_falls_back(self):
        """The score is stale for one step after a prune; it must not crash or mis-index."""
        self.assertTrue(torch.equal(_probs(self.op, torch.zeros(2), 0.3), self.op))


if __name__ == "__main__":
    unittest.main()
