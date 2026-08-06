from typing import Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .gs2d_metrics import GS2DMetrics, GS2DMetricsImpl


@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: int = 30_000


@dataclass
class CityGSV2Metrics(GS2DMetrics):
    lambda_normal: float = 0.05

    normal_regularization_from_iter: int = 7000

    # GS2DMetrics declares these but is not a @dataclass, so jsonargparse can't set them.
    # Re-declare here (same defaults) so lambda_dist/dist_reg are settable via config.
    lambda_dist: float = 0.

    dist_regularization_from_iter: int = 3000

    depth_loss_type: Literal["l1", "l1+ssim", "l2", "kl"] = "l1"

    depth_loss_ssim_weight: float = 0.2

    depth_loss_weight: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    depth_normalized: bool = False

    depth_output_key: str = "inverse_depth"

    depth_coverage_eps: float = 1e-4
    """Pixels whose rendered alpha is below this carry no depth information and are excluded.

    Without it, `1 / (surf_depth + 1e-8)` returns 1e8 wherever nothing was rendered, and the depth
    term stops being a loss and becomes noise. Measured 2026-08-06 on `cap80k_60k_b12` (count
    pinned at 80,000): `d_reg` spiked to 1.1e6 on individual batches against a healthy 0.002-0.17,
    and val collapsed 20.34 -> 17.06 between steps 17,040 and 22,720.

    It does not fire on a cloud that covers the frame -- `aggr24k_b12` and `graded_k4_24k_b12` both
    held `d_reg` in 0.0009-0.0055 for all 24,000 steps, and a single 1e8 pixel would have pushed the
    mean over 1.44M pixels to ~69. That is why the threshold is small: it must stay a no-op on the
    runs already in the results table, and only catch pixels that are genuinely empty.

    Set to 0 to restore the old behaviour."""

    def instantiate(self, *args, **kwargs) -> "CityGSV2MetricsModule":
        return CityGSV2MetricsModule(self)


class CityGSV2MetricsModule(GS2DMetricsImpl):
    config: CityGSV2Metrics

    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)

        if self.config.depth_loss_type == "l1":
            self._get_inverse_depth_loss = self._depth_l1_loss
        elif self.config.depth_loss_type == "l1+ssim":
            # self.depth_ssim = StructuralSimilarityIndexMeasure()
            self.depth_ssim = self._depth_ssim
            self._get_inverse_depth_loss = self._depth_l1_and_ssim_loss
        elif self.config.depth_loss_type == "l2":
            self._get_inverse_depth_loss = self._depth_l2_loss
        # elif self.config.depth_loss_type == "kl":
        #     self._get_inverse_depth_loss = self._depth_kl_loss
        else:
            raise NotImplementedError()

    def _depth_l1_loss(self, a, b):
        return torch.abs(a - b).mean()

    def _depth_l1_and_ssim_loss(self, a, b):
        l1_loss = self._depth_l1_loss(a, b)
        # ssim_metric = self.depth_ssim(a[None, None, ...], b[None, None, ...])
        ssim_metric = self.depth_ssim(a, b)

        return (1 - self.config.depth_loss_ssim_weight) * l1_loss + self.config.depth_loss_ssim_weight * (1 - ssim_metric)

    def _depth_l2_loss(self, a, b):
        return ((a - b) ** 2).mean()

    def _depth_kl_loss(self, a, b):
        pass

    def _depth_ssim(self, a, b):
        from internal.utils.ssim import ssim
        return ssim(a[None], b[None])

    def get_inverse_depth_metric(self, batch, outputs):
        # TODO: apply mask

        camera, _, gt_inverse_depth = batch

        if gt_inverse_depth is None:
            return torch.tensor(0., device=camera.device)
        
        predicted_inverse_depth = 1. / (outputs["surf_depth"].clamp_min(0.).squeeze() + 1e-8)

        # The pseudo-depth .npy files are baked at whatever resolution estimate_dataset_depths.py ran
        # at (1600x900 here, i.e. down_sample_factor 1.2) while the render follows the camera, so
        # changing down_sample_factor makes the two disagree -- "size of tensor a (1600) must match
        # the size of tensor b (960)", which killed the 2026-08-04 official-coarse attempt at 2x.
        # Regenerating 5621 monocular depth maps just to try a resolution is the most expensive step
        # in the pipeline. MCMCInverseDepthMetrics already solves this the same way (resize the
        # PREDICTION to the target's size, not the reverse); matching that keeps one convention in
        # the codebase. No-op when the shapes agree, so no existing result changes.
        _gt_for_shape = gt_inverse_depth[0] if isinstance(gt_inverse_depth, tuple) else gt_inverse_depth
        if predicted_inverse_depth.shape[-2:] != _gt_for_shape.shape[-2:]:
            predicted_inverse_depth = torch.nn.functional.interpolate(
                predicted_inverse_depth[None, None, ...],
                size=_gt_for_shape.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        if self.config.depth_normalized:
            # with torch.no_grad():
            clamp_val = (predicted_inverse_depth.mean() + 2 * predicted_inverse_depth.std()).item()
            predicted_inverse_depth = predicted_inverse_depth.clamp(max=clamp_val) / clamp_val
            gt_inverse_depth = gt_inverse_depth.clamp(max=clamp_val) / clamp_val

        if isinstance(gt_inverse_depth, tuple):
            gt_inverse_depth, gt_inverse_depth_mask = gt_inverse_depth

            gt_inverse_depth = gt_inverse_depth * gt_inverse_depth_mask
            predicted_inverse_depth = predicted_inverse_depth * gt_inverse_depth_mask

        # Coverage gate. `surf_depth` is 0 where nothing was rendered, and 1/(0 + 1e-8) is 1e8 --
        # not a depth, just a division by an epsilon. `rend_alpha` says which pixels actually
        # received something, and it has been sitting in the same output dict all along
        # (sep_depth_trim_2dgs_renderer.py:175).
        #
        # Uncovered pixels take the GT value rather than being dropped: `depth_loss_type` is
        # `l1+ssim` and SSIM is a windowed operator, so punching holes in the tensor leaves it
        # undefined. Substituting a detached GT makes the error and the gradient exactly zero
        # there, which works for both loss forms.
        alpha = outputs.get("rend_alpha") if self.config.depth_coverage_eps > 0 else None
        if alpha is not None:
            alpha = alpha.squeeze()
            if alpha.shape[-2:] != predicted_inverse_depth.shape[-2:]:
                alpha = torch.nn.functional.interpolate(
                    alpha[None, None, ...],
                    size=predicted_inverse_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            covered = alpha > self.config.depth_coverage_eps
            predicted_inverse_depth = torch.where(
                covered, predicted_inverse_depth, gt_inverse_depth.detach())

        return self._get_inverse_depth_loss(gt_inverse_depth, predicted_inverse_depth)

    def get_weight(self, step: int):
        return self.config.depth_loss_weight.init * (self.config.depth_loss_weight.final_factor ** min(step / self.config.depth_loss_weight.max_steps, 1))

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        d_reg_weight = self.get_weight(step)
        d_reg = self.get_inverse_depth_metric(batch, outputs) * d_reg_weight

        metrics["d_reg"] = d_reg
        metrics["d_w"] = d_reg_weight
        pbar["d_reg"] = True
        pbar["d_w"] = True

        if step < pl_module.hparams["density"].densify_until_iter:
            pbar["extra_loss"] = False
            metrics["loss"] = pl_module.hparams["metric"].lambda_dssim * (1. - metrics["ssim"]) + metrics["dist_loss"] + metrics["normal_loss"] + d_reg
            metrics["extra_loss"] = (1.0 - pl_module.hparams["metric"].lambda_dssim) * metrics["rgb_diff"]
        else:
            metrics["loss"] = metrics["loss"] + d_reg

        return metrics, pbar

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        metrics, pbar = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)

        d_reg = self.get_inverse_depth_metric(batch, outputs)

        metrics["loss"] = metrics["loss"] + d_reg
        metrics["d_reg"] = d_reg
        pbar["d_reg"] = True

        return metrics, pbar
