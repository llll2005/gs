"""The top-K transmittance that contribution pruning ranks on.

`contribution = mean of the K largest transmittances across views` is the quantity the trim
renderer prunes by, that `tools/prune_curve.py` ranks by, and that every "low contribution" claim
in the record rests on. The implementation in
`sep_depth_trim_2dgs_renderer.py:after_training_step` does not compute it:

    m = trans > top_list[0]          # only compares against slot 0
    for i in range(K - 1):
        top_list[K-1-i][m] = top_list[K-2-i][m]
        top_list[0][m] = trans[m]

Only values that beat the running maximum are inserted, so a value that belongs in the top-K but
is not a new record is dropped. And the seed `top_list = [trans]*K` copies the FIRST camera's value
into all K slots, so it is double-counted until enough records displace it.

These tests state the property first (against a reference implementation on small sequences) so the
defect is demonstrated rather than argued.
"""
import unittest

import torch


def reference_topk_mean(values, K):
    """What the docstring says: mean of the K largest, per primitive."""
    stacked = torch.stack(values)                        # [V, N]
    k = min(K, stacked.shape[0])
    return stacked.topk(k, dim=0).values.mean(0)


def shipped_topk_mean(values, K):
    """Transcription of the renderer's loop, so the test pins the real behaviour."""
    top_list = [None] * K
    for trans in values:
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any():
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                    top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]
    return torch.stack(top_list, dim=-1).mean(-1)


def fixed_topk_mean(values, K):
    """Streaming top-K: insert into the sorted-descending buffer, drop the smallest.

    Kept streaming rather than stacking every view because the renderer walks 284 cameras and an
    [N, 284] buffer at N=2M is 2.3 GB on a 6 GB card.
    """
    top = None
    for trans in values:
        if top is None:
            neg = torch.full((K,) + trans.shape, float("-inf"), dtype=trans.dtype)
            top = neg
        # insert `trans` into the descending buffer
        for k in range(K):
            bigger = trans > top[k]
            if not bigger.any():
                continue
            carry = top[k][bigger]
            top[k][bigger] = trans[bigger]
            trans = trans.clone()
            trans[bigger] = carry
    return torch.where(torch.isinf(top), torch.zeros_like(top), top).mean(0)


class TopKContributionTest(unittest.TestCase):
    K = 3

    def _seq(self, per_view):
        return [torch.tensor([v], dtype=torch.float32) for v in per_view]

    def test_shipped_version_drops_a_non_record_value(self):
        """5, 3, 8, 7 -> true top-3 is (8, 7, 5). The shipped loop gives (8, 8, 5).

        `top_list[0] = trans` sits INSIDE the shift loop, so the later shifts read the slot that
        was just overwritten instead of the value it displaced:

            [a, b, c] + record r
              i=0: top[2]=top[1]->b ; top[0]=r   -> [r, b, b]
              i=1: top[1]=top[0]->r ; top[0]=r   -> [r, r, b]

        The previous maximum `a` is discarded outright and the new record occupies K-1 slots.
        """
        vals = self._seq([5.0, 3.0, 8.0, 7.0])
        ref = float(reference_topk_mean(vals, self.K))
        shipped = float(shipped_topk_mean(vals, self.K))
        self.assertAlmostEqual(ref, (8 + 7 + 5) / 3, places=5)
        self.assertAlmostEqual(shipped, (8 + 8 + 5) / 3, places=5)
        self.assertNotAlmostEqual(ref, shipped, places=3)

    def test_shipped_version_is_effectively_a_running_max(self):
        """With enough records the buffer saturates on the maximum.

        This is the consequential part: the trim is documented and reasoned about as a MULTI-VIEW
        statistic, but ranks by roughly the single best view. A floater placed to explain one
        camera peaks in that camera and survives -- which is exactly the failure the record keeps
        attributing to "the criterion is blind to geometry".
        """
        rising = self._seq([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # steady state is [max, max, previous max] -> (2*max + prev)/K, not the top-3 mean
        self.assertAlmostEqual(float(shipped_topk_mean(rising, self.K)), (6 + 6 + 5) / 3, places=5)
        self.assertAlmostEqual(float(reference_topk_mean(rising, self.K)), (6 + 5 + 4) / 3, places=5)

    def test_shipped_version_double_counts_the_first_view(self):
        """A primitive seen once at 9 and never again reads as 9 in all K slots."""
        vals = self._seq([9.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(float(shipped_topk_mean(vals, self.K)), 9.0, places=5)
        self.assertAlmostEqual(float(reference_topk_mean(vals, self.K)), (9 + 1 + 1) / 3, places=5)

    def test_shipped_version_is_order_dependent(self):
        """The same multiset of views must give the same contribution. It does not."""
        a = float(shipped_topk_mean(self._seq([5.0, 3.0, 8.0, 7.0]), self.K))
        b = float(shipped_topk_mean(self._seq([8.0, 7.0, 5.0, 3.0]), self.K))
        self.assertNotAlmostEqual(a, b, places=3)

    def test_fixed_version_matches_the_reference(self):
        for seq in ([5.0, 3.0, 8.0, 7.0],
                    [9.0, 1.0, 1.0, 1.0],
                    [8.0, 7.0, 5.0, 3.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            vals = self._seq(seq)
            self.assertAlmostEqual(float(fixed_topk_mean(vals, self.K)),
                                   float(reference_topk_mean(vals, self.K)), places=5,
                                   msg=f"sequence {seq}")

    def test_fixed_version_is_order_independent(self):
        import random
        base = [0.2, 0.9, 0.5, 0.1, 0.7, 0.33]
        vals = self._seq(base)
        want = float(fixed_topk_mean(vals, self.K))
        for _ in range(5):
            shuffled = base[:]
            random.shuffle(shuffled)
            self.assertAlmostEqual(float(fixed_topk_mean(self._seq(shuffled), self.K)),
                                   want, places=5)

    def test_fixed_version_handles_many_primitives(self):
        torch.manual_seed(0)
        vals = [torch.rand(1000) for _ in range(12)]
        torch.testing.assert_close(fixed_topk_mean(vals, 5), reference_topk_mean(vals, 5))

    def test_fewer_views_than_K(self):
        """284 cameras in practice, but a block with few images must not read -inf."""
        vals = self._seq([4.0, 6.0])
        got = float(fixed_topk_mean(vals, self.K))
        self.assertTrue(0 <= got <= 6)


class SharedImplementationIsUsedTest(unittest.TestCase):
    """The fix must be in every call site, not just the one that was noticed.

    Four copies of the loop existed: the 2DGS trim renderer (both the start trim and the periodic
    one), the 3DGS trim renderer (two more), and LightGaussian importance pruning. The last one
    shifted correctly but still only inserted record-breakers, so it was wrong in a different way
    from the renderer's -- which is exactly how a second copy survives a fix to the first.
    """

    def test_default_is_max_not_topk_mean(self):
        """Measured, not assumed. Post-hoc pruning of oreg_0p002 (building texture ratio):

            kept        max      legacy(broken)   topk_mean
            160,000    0.303        0.296           0.279
             80,000    0.266        0.262           0.236

        `max` wins at every level; the broken loop was approximating it. So the criterion is `max`,
        named and computed as such, rather than a correct implementation of a worse statistic.
        """
        import inspect
        from internal.utils.topk_contribution import contribution_accumulator
        self.assertEqual(
            inspect.signature(contribution_accumulator).parameters["reduce"].default, "max")

    def test_max_reduce_returns_the_single_best_view(self):
        import torch as t
        from internal.utils.topk_contribution import contribution_accumulator
        push, gather = contribution_accumulator(5, "max")
        for v in (5.0, 3.0, 8.0, 7.0):
            push(t.tensor([v]))
        self.assertAlmostEqual(float(gather()), 8.0, places=5)

    def test_shipped_helper_matches_the_reference(self):
        import torch as t
        from internal.utils.topk_contribution import topk_mean_accumulator
        for seq in ([5.0, 3.0, 8.0, 7.0], [9.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            vals = [t.tensor([v]) for v in seq]
            push, gather = topk_mean_accumulator(3)
            for v in vals:
                push(v)
            self.assertAlmostEqual(float(gather()),
                                   float(reference_topk_mean(vals, 3)), places=5, msg=str(seq))

    def test_no_call_site_still_hand_rolls_it(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        offenders = []
        for f in (root / "internal").rglob("*.py"):
            if f.name == "topk_contribution.py":
                continue
            code = "\n".join(l for l in f.read_text().splitlines()
                             if not l.strip().startswith("#"))
            if "top_list" in code:
                offenders.append(str(f.relative_to(root)))
        self.assertEqual(offenders, [], f"hand-rolled top-K left in: {offenders}")

    def test_fewer_views_than_K_uses_only_the_views_seen(self):
        import torch as t
        from internal.utils.topk_contribution import topk_mean_accumulator
        push, gather = topk_mean_accumulator(5)
        for v in (4.0, 6.0):
            push(t.tensor([v]))
        self.assertAlmostEqual(float(gather()), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()


class MergePerPrimitiveBufferTest(unittest.TestCase):
    """Density-controller buffers that are per-primitive must take the block mask.

    `VanillaDensityController` (and `CityGSV2DensityController` through it) registers
    `max_radii2D`, `xyz_gradient_accum`, `denom` with persistent=True. `merge_citygs_ckpts.py`
    used to append them whole while cropping the gaussians, so the merged checkpoint carried two
    different primitive counts and nothing raised. MCMC never hits it (`binoms` is
    persistent=False); the self-run CityGSV2 baseline would.
    """

    def test_vanilla_controller_has_persistent_per_primitive_buffers(self):
        import inspect
        from internal.density_controllers import vanilla_density_controller as v
        src = inspect.getsource(v)
        for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
            self.assertIn(f'register_buffer("{name}"', src)
        self.assertIn("persistent=True", src)

    def test_mcmc_buffer_is_not_persistent(self):
        import inspect
        from internal.density_controllers import mcmc_2dgs_density_controller as m
        self.assertIn("persistent=False", inspect.getsource(m))

    def test_merge_script_masks_matching_length_buffers(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent.parent
               / "utils" / "merge_citygs_ckpts.py").read_text()
        self.assertIn("value.shape[:1] == mask_preserved.shape[:1]", src)
        self.assertIn("value[mask_preserved]", src)
