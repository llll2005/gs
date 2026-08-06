"""The SB colour port: lobe seeding and where `sb_params` gets attached.

`Gaussian2DSB` is ours (DBS's spherical-beta colour lifted onto 2DGS surfels, without DBS's beta
falloff). It has only ever been smoke-tested, and it currently sits 0.46 dB under SH3 on this
content, so its initialisation is worth pinning down rather than assuming.
"""
import math
import unittest

import torch

from internal.models.gaussian_2d_sb import Gaussian2DSB, Gaussian2DSBModel


def _model(**kw):
    cfg = Gaussian2DSB(**kw)
    m = Gaussian2DSBModel(cfg)
    m.config = cfg
    return m


class LobeInitTest(unittest.TestCase):
    N = 200_000

    def _polar_cap_fraction(self, mode, limit=math.pi / 6):
        torch.manual_seed(0)
        theta = _model(sb_number=2, sb_lobe_init=mode)._init_sb_params(self.N)[..., 3]
        return float((theta < limit).float().mean())

    def test_uniform_angle_is_the_shipped_default(self):
        """Every number in the results table was trained with this. Do not change it silently."""
        self.assertEqual(Gaussian2DSB().sb_lobe_init, "uniform_angle")

    def test_uniform_angle_oversamples_the_poles(self):
        """theta ~ U(0, pi) puts 1/6 of the lobes in a cap the sphere gives 6.7% of."""
        got = self._polar_cap_fraction("uniform_angle")
        self.assertAlmostEqual(got, 1 / 6, delta=0.005)

    def test_uniform_sphere_matches_the_sphere_measure(self):
        """theta = arccos(1 - 2u) -> P(theta < pi/6) = (1 - cos(pi/6))/2."""
        expected = (1 - math.cos(math.pi / 6)) / 2
        self.assertAlmostEqual(self._polar_cap_fraction("uniform_sphere"), expected, delta=0.005)

    def test_beta_is_zero_initialised(self):
        """DBS Sec.4: "we set b to zero at initialization" -- gives a Gaussian-like lobe."""
        p = _model(sb_number=3)._init_sb_params(64)
        self.assertTrue(torch.all(p[..., 5] == 0))

    def test_rgb_amplitudes_start_at_raw_zero(self):
        """Raw 0 -> a small positive amplitude once softplus is applied, not a hard zero."""
        from internal.utils.sb_utils import sb_params_activation
        p = _model(sb_number=2)._init_sb_params(64)
        self.assertTrue(torch.all(p[..., :3] == 0))
        self.assertTrue(torch.all(sb_params_activation(p)[..., :3] > 0))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _model(sb_number=2, sb_lobe_init="spiral")._init_sb_params(8)

    def test_shape_and_lobe_count(self):
        for L in (1, 2, 3):
            self.assertEqual(tuple(_model(sb_number=L)._init_sb_params(7).shape), (7, L, 6))


class ExtraPropertyTest(unittest.TestCase):
    def test_sb_params_is_declared_as_an_extra_property(self):
        """Density control moves per-primitive tensors by this list; missing it means `sb_params`
        would not be pruned/cloned alongside the rest and would desynchronise from `means`."""
        self.assertIn("sb_params", _model(sb_number=2).get_extra_property_names())


if __name__ == "__main__":
    unittest.main()
