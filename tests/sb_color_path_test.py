"""The SB colour fast path must equal the reference path.

`eval_spherical_beta_raw` is documented as a "math-identical fast path" to `eval_spherical_beta`:
it takes RAW `sb_params` (softplus inlined on the rgb slice instead of materialising an activated
copy) and PRE-NORMALISED directions (skipping a second normalise). The training renderer calls the
fast one; every tool that loads a checkpoint calls the slow one through `get_sb_params`.

"Math-identical" is a claim, and three such claims turned out false on 2026-08-06 (the top-K
contribution, the test command, the dust cull's mechanism). Pinned here rather than believed.
"""
import math
import unittest

import torch

from internal.utils.sb_utils import (eval_spherical_beta, eval_spherical_beta_raw,
                                     sb_params_activation)


class SBFastPathEquivalenceTest(unittest.TestCase):
    def _inputs(self, n=256, lobes=3, seed=0):
        torch.manual_seed(seed)
        raw = torch.randn(n, lobes, 6)
        raw[..., 3] = math.pi * torch.rand(n, lobes)        # theta
        raw[..., 4] = 2 * math.pi * torch.rand(n, lobes)    # phi
        dirs = torch.randn(n, 3) * 5.0                      # deliberately unnormalised
        base = torch.rand(n, 3)
        return raw, dirs, base

    def test_fast_path_matches_reference(self):
        raw, dirs, base = self._inputs()
        ref = eval_spherical_beta(sb_params_activation(raw), dirs, base)
        fast = eval_spherical_beta_raw(raw, torch.nn.functional.normalize(dirs, dim=-1), base)
        torch.testing.assert_close(ref, fast, rtol=1e-6, atol=1e-6)

    def test_matches_across_lobe_counts(self):
        for L in (1, 2, 3, 5):
            raw, dirs, base = self._inputs(lobes=L, seed=L)
            ref = eval_spherical_beta(sb_params_activation(raw), dirs, base)
            fast = eval_spherical_beta_raw(raw, torch.nn.functional.normalize(dirs, dim=-1), base)
            torch.testing.assert_close(ref, fast, rtol=1e-6, atol=1e-6, msg=f"L={L}")

    def test_output_is_non_negative(self):
        """Colours feed the rasteriser as `colors_precomp`; a negative would render as garbage."""
        raw, dirs, base = self._inputs()
        out = eval_spherical_beta_raw(raw, torch.nn.functional.normalize(dirs, dim=-1), base)
        self.assertTrue(torch.all(out >= 0))

    def test_zero_init_reduces_to_the_base_plus_a_small_positive_lobe(self):
        """At init raw rgb is 0 and beta is 0, so softplus(0) > 0 gives a small positive lobe on
        top of the diffuse base -- not a hard zero, and not a large perturbation."""
        n = 64
        raw = torch.zeros(n, 2, 6)
        raw[..., 3] = math.pi * torch.rand(n, 2)
        raw[..., 4] = 2 * math.pi * torch.rand(n, 2)
        base = torch.full((n, 3), 0.5)
        out = eval_spherical_beta_raw(raw, torch.nn.functional.normalize(torch.randn(n, 3), dim=-1), base)
        self.assertTrue(torch.all(out >= base - 1e-6), "lobes subtracted from the base")
        self.assertTrue(torch.all(out < base + 1.0), "zero-init lobes dominate the base")

    def test_gradients_reach_both_the_lobes_and_the_direction(self):
        """`dirs = means - campos`, so the colour path is one of the routes by which the view
        direction feeds back into the means (matching the CUDA SH path)."""
        raw, dirs, base = self._inputs(n=32)
        raw.requires_grad_(True)
        dirs.requires_grad_(True)
        eval_spherical_beta_raw(
            raw, torch.nn.functional.normalize(dirs, dim=-1), base).sum().backward()
        self.assertTrue(torch.any(raw.grad != 0), "sb_params got no gradient")
        self.assertTrue(torch.any(dirs.grad != 0), "view direction got no gradient")


if __name__ == "__main__":
    unittest.main()
