"""Texture ratio in the periodic val.

The metric that decides whether a run is blurry was only available offline, so a training run
reported PSNR climbing while its building texture collapsed -- 77k points scored 22.022 against
1.8M's 23.388 (1.37 dB) while recovering 0.145 of GT gradient energy against 0.393 (2.7x).
b12 is half flat water and the average hid it.

These tests pin the two things that make the logged numbers readable:
  - direction: blur < 1 < noise, so "higher is better" is only true below 1
  - weighting: tex_render/tex_gt is energy-weighted, so building views dominate WITHOUT a
    threshold or labels, while a per-image mean ratio is diluted by water
"""
import unittest
from unittest import mock

import torch

from internal.metrics.citygsv2_metrics import CityGSV2MetricsModule, _grad_energy


class GradEnergyTest(unittest.TestCase):
    def test_flat_image_has_no_energy(self):
        self.assertEqual(float(_grad_energy(torch.full((3, 16, 16), 0.4))), 0.0)

    def test_blur_lowers_energy(self):
        torch.manual_seed(0)
        sharp = torch.rand(3, 32, 32)
        blurred = torch.nn.functional.avg_pool2d(sharp.unsqueeze(0), 3, 1, 1).squeeze(0)
        self.assertLess(float(_grad_energy(blurred)), float(_grad_energy(sharp)))

    def test_counts_both_axes(self):
        """A purely vertical edge must register; summing only one axis would miss it."""
        img = torch.zeros(1, 8, 8)
        img[:, :, 4:] = 1.0
        self.assertGreater(float(_grad_energy(img)), 0.0)


class ValidateMetricsTest(unittest.TestCase):
    """Exercises the real added lines with the parent and depth term stubbed out."""

    def _run(self, gt, render):
        m = CityGSV2MetricsModule.__new__(CityGSV2MetricsModule)
        batch = (None, ("img.png", gt, None), None)
        outputs = {"render": render}
        with mock.patch(
                "internal.metrics.gs2d_metrics.GS2DMetricsImpl.get_validate_metrics",
                return_value=({"loss": torch.tensor(0.0)}, {})), \
             mock.patch.object(CityGSV2MetricsModule, "get_inverse_depth_metric",
                               return_value=torch.tensor(0.0)):
            metrics, _ = m.get_validate_metrics(None, None, batch, outputs)
        return metrics

    def test_perfect_render_scores_one(self):
        torch.manual_seed(0)
        gt = torch.rand(3, 24, 24)
        self.assertAlmostEqual(float(self._run(gt, gt.clone())["texratio"]), 1.0, places=5)

    def test_blurry_render_scores_below_one(self):
        torch.manual_seed(0)
        gt = torch.rand(3, 32, 32)
        blurred = torch.nn.functional.avg_pool2d(gt.unsqueeze(0), 5, 1, 2).squeeze(0)
        self.assertLess(float(self._run(gt, blurred)["texratio"]), 0.6)

    def test_noisy_render_scores_above_one(self):
        """Above 1 is a failure mode, not a better score -- the direction must be visible."""
        torch.manual_seed(0)
        gt = torch.rand(3, 32, 32) * 0.1 + 0.5
        noisy = gt + torch.randn_like(gt) * 0.2
        self.assertGreater(float(self._run(gt, noisy)["texratio"]), 1.0)

    def test_flat_gt_does_not_divide_by_zero(self):
        r = self._run(torch.full((3, 16, 16), 0.5), torch.full((3, 16, 16), 0.5))
        self.assertTrue(torch.isfinite(r["texratio"]))

    def test_energy_weighting_beats_the_per_image_mean(self):
        """The whole point: two water views and one building view, all rendered equally blurry.

        The per-image mean says the model is fine because water is easy; the ratio of the logged
        means says it is not, because the building view carries most of the gradient energy.
        """
        torch.manual_seed(0)
        water = torch.rand(3, 32, 32) * 0.02 + 0.5           # nearly flat
        building = torch.rand(3, 32, 32)                      # rich texture
        blur = lambda x: torch.nn.functional.avg_pool2d(x.unsqueeze(0), 5, 1, 2).squeeze(0)

        per_image, gts, renders = [], [], []
        for gt, render in [(water, water.clone()),            # water rendered perfectly
                           (water, water.clone()),
                           (building, blur(building))]:       # building rendered blurry
            r = self._run(gt, render)
            per_image.append(float(r["texratio"]))
            gts.append(float(r["tex_gt"]))
            renders.append(float(r["tex_render"]))

        naive = sum(per_image) / len(per_image)               # what Lightning means over texratio
        weighted = sum(renders) / sum(gts)                    # ratio of the two logged means
        building_only = per_image[-1]                         # what we actually want to know

        # Energy weighting lands far closer to the building reading than the per-image mean does,
        # WITHOUT a threshold or a label -- but it does not equal it. Correctly rendered water
        # still contributes to the numerator, and the blurrier the buildings the larger that
        # share, so the weighted value is an upper bound. Pinned here so nobody reads it as exact.
        self.assertGreaterEqual(weighted, building_only, "upper bound, never below")
        self.assertLess(weighted - building_only, naive - weighted,
                        f"weighted {weighted:.3f} must sit nearer the building reading "
                        f"{building_only:.3f} than the diluted mean {naive:.3f}")
        self.assertGreater(naive, 2 * weighted,
                           f"per-image mean {naive:.3f} should be badly diluted vs {weighted:.3f}")


class LogMetricsRobustnessTest(unittest.TestCase):
    """`log_metrics` used to index prog_bar directly, so adding a metric without a prog_bar entry
    raised KeyError -- and for a validate-only metric that lands at the FIRST val, hours in. The
    unit tests here all passed while the real path crashed, which is why this one exists."""

    def _log_metrics(self, metrics, prog_bar):
        from internal.gaussian_splatting import GaussianSplatting
        m = GaussianSplatting.__new__(GaussianSplatting)
        logged = {}
        m.log = lambda name, value, **kw: logged.__setitem__(name, kw.get("prog_bar"))
        m.batch_size = 1
        GaussianSplatting.log_metrics(m, metrics, prog_bar, "val", on_step=False, on_epoch=True)
        return logged

    def test_missing_prog_bar_entry_does_not_raise(self):
        logged = self._log_metrics({"psnr": 1.0, "texratio": 0.5}, {"psnr": True})
        self.assertEqual(logged["val/texratio"], False)
        self.assertEqual(logged["val/psnr"], True)


if __name__ == "__main__":
    unittest.main()
