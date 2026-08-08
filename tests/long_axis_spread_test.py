"""New Gaussians must not all land on top of their host.

MCMC copies the host's position verbatim, so N children start stacked and only SGLD noise
separates them -- and 3dgs-mcmc's own paper calls that walk irrecoverable once a Gaussian leaves
its support region. Eq. 9 corrects opacity and scale so the stack RENDERS like the original, but
says nothing about position. This matters most for blur-split hosts: those are exactly the
primitives that alone cover a large patch, and children stacked at the centre cannot cover it.
"""
import unittest
from types import SimpleNamespace

import torch

from internal.utils.general_utils import build_rotation


def _offset(scales, quats, idxs, spread, seed=0):
    """The offset computed in _get_new_params, isolated from the model plumbing."""
    if spread <= 0:
        return torch.zeros(len(idxs), 3)
    torch.manual_seed(seed)
    s = scales[idxs]
    long_len, long_ax = s.max(dim=-1)
    R = build_rotation(quats[idxs])
    d = torch.gather(R, 2, long_ax.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
    t = torch.rand_like(long_len) * 2.0 - 1.0
    return d * (spread * long_len * t).unsqueeze(-1)


class LongAxisSpreadTest(unittest.TestCase):
    def setUp(self):
        # one primitive, long axis = x (scale 4 vs 1 vs 1), identity rotation
        self.scales = torch.tensor([[4.0, 1.0, 1.0]])
        self.quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.idxs = torch.zeros(64, dtype=torch.long)          # 64 children of the same host

    def test_off_by_default(self):
        self.assertTrue(torch.equal(_offset(self.scales, self.quats, self.idxs, 0.0),
                                    torch.zeros(64, 3)))

    def test_children_move_along_the_long_axis_only(self):
        o = _offset(self.scales, self.quats, self.idxs, 0.5)
        self.assertGreater(float(o[:, 0].abs().max()), 0.0)
        self.assertAlmostEqual(float(o[:, 1].abs().max()), 0.0, places=6)
        self.assertAlmostEqual(float(o[:, 2].abs().max()), 0.0, places=6)

    def test_children_are_spread_not_displaced_as_a_block(self):
        """A fixed +-offset would just move the whole stack. They must land at DIFFERENT points."""
        o = _offset(self.scales, self.quats, self.idxs, 0.5)
        self.assertGreater(float(o[:, 0].std()), 0.1)
        self.assertLess(abs(float(o[:, 0].mean())), 0.5, "should straddle the host, not shift it")

    def test_magnitude_is_bounded_by_the_long_extent(self):
        o = _offset(self.scales, self.quats, self.idxs, 0.5)
        self.assertLessEqual(float(o[:, 0].abs().max()), 0.5 * 4.0 + 1e-6)

    def test_rotation_is_respected(self):
        """Long axis is x in local space; a 90deg rotation about z must send it to y."""
        r2 = 2 ** -0.5
        q = torch.tensor([[r2, 0.0, 0.0, r2]])                  # w, x, y, z
        o = _offset(self.scales, q, self.idxs, 0.5)
        self.assertAlmostEqual(float(o[:, 0].abs().max()), 0.0, places=5)
        self.assertGreater(float(o[:, 1].abs().max()), 0.0)

    def test_picks_the_longest_axis_not_the_first(self):
        scales = torch.tensor([[1.0, 5.0, 1.0]])                # long axis is y
        o = _offset(scales, self.quats, self.idxs, 0.5)
        self.assertAlmostEqual(float(o[:, 0].abs().max()), 0.0, places=6)
        self.assertGreater(float(o[:, 1].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
