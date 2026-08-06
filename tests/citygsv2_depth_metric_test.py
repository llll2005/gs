"""The depth term of `CityGSV2Metrics`, exercised on the inputs that broke it.

Two real defects lived here, and neither raised an exception on the configs we ship:

  1. `1 / (surf_depth + 1e-8)` returns 1e8 wherever nothing was rendered. On 2026-08-06 that
     poisoned 7 runs (`d_reg` spiking to 1.1e6 against a healthy 0.002-0.17) and collapsed
     `cap80k_60k_b12` from val 20.34 to 17.06. It only fires when the cloud stops covering the
     frame, which is why it went unnoticed for a week.
  2. The `(depth, mask)` tuple was unpacked AFTER the `depth_normalized` block called `.clamp()`
     on it. Latent, because every config sets `depth_normalized: False`.

Both are shape/coverage conditions rather than numerical drift, so they are cheap to pin here
rather than discovered on a 6.7-hour run.
"""
import unittest

import torch

from internal.metrics.citygsv2_metrics import CityGSV2Metrics, CityGSV2MetricsModule


class _Cam:
    device = "cpu"


def _module(**kw):
    cfg = CityGSV2Metrics(**kw)
    m = CityGSV2MetricsModule(cfg)
    m.config = cfg
    # `setup` needs a pl_module; wire only the dispatch it would have installed
    m._get_inverse_depth_loss = m._depth_l1_loss
    return m


def _outputs(depth, alpha=None):
    o = {"surf_depth": depth[None]}
    if alpha is not None:
        o["rend_alpha"] = alpha[None]
    return o


class DepthCoverageGateTest(unittest.TestCase):
    def test_uncovered_pixels_do_not_blow_up_the_loss(self):
        """Half the frame renders nothing: without the gate this is ~1e8, with it ~0."""
        H = W = 16
        gt = torch.full((H, W), 0.5)
        depth = torch.full((H, W), 2.0)
        alpha = torch.ones(H, W)
        depth[:8] = 0.0          # nothing rendered on the top half
        alpha[:8] = 0.0

        gated = _module(depth_coverage_eps=1e-4)
        loss_gated = gated.get_inverse_depth_metric((_Cam(), None, gt), _outputs(depth, alpha))

        ungated = _module(depth_coverage_eps=0.0)
        loss_ungated = ungated.get_inverse_depth_metric((_Cam(), None, gt), _outputs(depth, alpha))

        self.assertLess(float(loss_gated), 1.0)
        self.assertGreater(float(loss_ungated), 1e6)

    def test_gate_is_a_noop_when_everything_is_covered(self):
        """The runs already in the results table must not move."""
        H = W = 16
        gt = torch.full((H, W), 0.5)
        depth = torch.full((H, W), 2.0)
        alpha = torch.ones(H, W)

        gated = _module(depth_coverage_eps=1e-4).get_inverse_depth_metric(
            (_Cam(), None, gt), _outputs(depth, alpha))
        ungated = _module(depth_coverage_eps=0.0).get_inverse_depth_metric(
            (_Cam(), None, gt), _outputs(depth, alpha))
        torch.testing.assert_close(gated, ungated)

    def test_gate_is_skipped_when_the_renderer_gives_no_alpha(self):
        """Renderers without `rend_alpha` must still work, just ungated."""
        H = W = 8
        gt = torch.full((H, W), 0.5)
        depth = torch.full((H, W), 2.0)
        loss = _module(depth_coverage_eps=1e-4).get_inverse_depth_metric(
            (_Cam(), None, gt), _outputs(depth))
        self.assertTrue(torch.isfinite(loss))

    def test_gradient_is_zero_on_uncovered_pixels(self):
        """The substituted GT is detached, so empty pixels must not steer anything."""
        H = W = 8
        gt = torch.full((H, W), 0.5)
        # 2.5 -> 1/2.5 = 0.4 against a target of 0.5, so the covered half carries a real error.
        # An earlier version of this test used 2.0, which inverts to exactly 0.5: `abs` has
        # subgradient 0 at zero, so the assertion failed for a reason that had nothing to do
        # with the gate.
        depth = torch.full((H, W), 2.5, requires_grad=True)
        alpha = torch.ones(H, W)
        alpha[:4] = 0.0
        m = _module(depth_coverage_eps=1e-4)
        m.get_inverse_depth_metric((_Cam(), None, gt), _outputs(depth, alpha)).backward()
        self.assertTrue(torch.all(depth.grad[:4] == 0), "uncovered pixels leaked gradient")
        self.assertTrue(torch.any(depth.grad[4:] != 0), "covered pixels lost gradient")


class DepthMaskTupleTest(unittest.TestCase):
    def test_masked_gt_with_normalization_does_not_crash(self):
        """The ordering bug: `.clamp()` used to be called on the tuple."""
        H = W = 8
        gt = torch.full((H, W), 0.5)
        mask = torch.ones(H, W)
        mask[:4] = 0.0
        depth = torch.full((H, W), 2.0)
        loss = _module(depth_normalized=True).get_inverse_depth_metric(
            (_Cam(), None, (gt, mask)), _outputs(depth, torch.ones(H, W)))
        self.assertTrue(torch.isfinite(loss))

    def test_masked_gt_without_normalization_still_applies_the_mask(self):
        H = W = 8
        gt = torch.full((H, W), 0.5)
        depth = torch.full((H, W), 2.0)          # 1/2.0 = 0.5 -> matches gt exactly
        depth[:4] = 0.01                          # 1/0.01 = 100 -> large error, but masked out
        mask = torch.ones(H, W)
        mask[:4] = 0.0
        loss = _module().get_inverse_depth_metric(
            (_Cam(), None, (gt, mask)), _outputs(depth, torch.ones(H, W)))
        self.assertLess(float(loss), 1e-3, "masked region still contributed")


class DepthResizeTest(unittest.TestCase):
    def test_prediction_is_resized_to_the_target(self):
        """The pseudo-depth .npy is baked at 1.2x; the render follows the camera."""
        gt = torch.full((16, 16), 0.5)
        depth = torch.full((8, 8), 2.0)
        loss = _module().get_inverse_depth_metric(
            (_Cam(), None, gt), _outputs(depth, torch.ones(8, 8)))
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 1e-3)


if __name__ == "__main__":
    unittest.main()
