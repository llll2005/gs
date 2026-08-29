from typing import Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .gs2d_metrics import GS2DMetrics, GS2DMetricsImpl


def _grad_energy(img: torch.Tensor) -> torch.Tensor:
    """Total variation of a [C, H, W] image -- how much high-frequency energy it carries."""
    return (img[:, 1:, :] - img[:, :-1, :]).abs().mean() + (img[:, :, 1:] - img[:, :, :-1]).abs().mean()


COARSE_DOC = '''Weight on an L1 computed at 1/2**coarse_l1_levels resolution, added to the loss.

Measured, not guessed. tools/loss_footprint_pricing.py injects a FIXED total amount of error into a
real render and varies only how it is spread; at equal error mass the SSIM penalty relative to L1
falls monotonically with blob radius -- 15.1x at r=2 down to 1.14x at r=64, and 7.47x -> 1.34x on a
second sweep with different amplitude and area. So a large-footprint error is in the blind spot of
BOTH terms: L1 sees a faint tint spread thin, SSIM sees nothing structural. That is how large
floaters survive.

Meanwhile the band decomposition of the fixed model puts 37% of the residual ENERGY at >=32 px
(2.57e-3 of 6.87e-3), even though its relative error there is only 2.5%. Large signal, small
relative error, no prioritisation from either loss.

Downsampling by 8 and taking L1 there prices exactly that band. It costs nothing: avg_pool on an
already-computed image, no extra VRAM.

⚠ Unvalidated. 0 = off. It is a NEW loss term, and this project's record with added regularisers is
poor (AtomGS edge-normal, EdgeAware, FreGS all lost on PSNR *and* LPIPS). What distinguishes this
one is that the gap it targets was measured first.'''


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

    coarse_l1_weight: float = 0.0
    coarse_l1_levels: int = 3
    __doc_coarse__ = COARSE_DOC

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

        # Unpack the optional GT mask up front. It used to be unpacked at the very end, AFTER the
        # `depth_normalized` block called `.clamp()` on it -- which is an AttributeError the moment a
        # dataparser supplies `(depth, mask)` with `depth_normalized=True`. Every config we ship sets
        # it False so the crash never fired, but the ordering was the bug, not the flag.
        gt_inverse_depth_mask = None
        if isinstance(gt_inverse_depth, tuple):
            gt_inverse_depth, gt_inverse_depth_mask = gt_inverse_depth

        # The pseudo-depth .npy files are baked at whatever resolution estimate_dataset_depths.py ran
        # at (1600x900 here, i.e. down_sample_factor 1.2) while the render follows the camera, so
        # changing down_sample_factor makes the two disagree -- "size of tensor a (1600) must match
        # the size of tensor b (960)", which killed the 2026-08-04 official-coarse attempt at 2x.
        # Regenerating 5621 monocular depth maps just to try a resolution is the most expensive step
        # in the pipeline. MCMCInverseDepthMetrics already solves this the same way (resize the
        # PREDICTION to the target's size, not the reverse); matching that keeps one convention in
        # the codebase. No-op when the shapes agree, so no existing result changes.
        if predicted_inverse_depth.shape[-2:] != gt_inverse_depth.shape[-2:]:
            predicted_inverse_depth = torch.nn.functional.interpolate(
                predicted_inverse_depth[None, None, ...],
                size=gt_inverse_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        if self.config.depth_normalized:
            # with torch.no_grad():
            clamp_val = (predicted_inverse_depth.mean() + 2 * predicted_inverse_depth.std()).item()
            predicted_inverse_depth = predicted_inverse_depth.clamp(max=clamp_val) / clamp_val
            gt_inverse_depth = gt_inverse_depth.clamp(max=clamp_val) / clamp_val

        if gt_inverse_depth_mask is not None:
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
        # 2026-08-26：權重為 0 時**跳過計算**。`get_inverse_depth_metric` 是全幅逐像素
        # （1/depth、resize、正規化、L1、**外加一個 SSIM 窗積分**）且帶 grad 進 loss
        # ⇒ backward 也走一遍。現行配方 `depth_loss_weight.init = 0.0` ⇒ 全是純浪費。
        # ⚠ 位元級等價：只在深度指標含 inf/nan 時不同（0*inf = nan）⇒ 加閘門更穩健。
        if d_reg_weight > 0:
            d_reg = self.get_inverse_depth_metric(batch, outputs) * d_reg_weight
        else:
            d_reg = metrics["loss"].new_zeros(())

        metrics["d_reg"] = d_reg
        metrics["d_w"] = d_reg_weight
        pbar["d_reg"] = True
        pbar["d_w"] = True

        # Coarse-scale L1: prices the band both photometric terms ignore. See COARSE_DOC.
        if self.config.coarse_l1_weight > 0:
            _, image_info, _ = batch
            _, gt_image, _ = image_info
            pred = outputs["render"]
            k = 2 ** self.config.coarse_l1_levels
            lo = lambda x: torch.nn.functional.avg_pool2d(x.unsqueeze(0), k).squeeze(0)
            c_l1 = (lo(pred) - lo(gt_image)).abs().mean()
            metrics["loss"] = metrics["loss"] + self.config.coarse_l1_weight * c_l1
            metrics["c_l1"] = c_l1
            pbar["c_l1"] = False

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

        # Texture ratio = render TV / GT TV: the one metric verified to have resolution on this
        # scene (7x separation on the known-bad controls, where building LPIPS fully overlapped).
        # It reads as "how much of the GT's high-frequency energy did we recover", so BELOW 1 means
        # blurry and higher is better; ABOVE 1 means we are producing MORE detail than exists
        # (noise/speckle/floaters) and is bad. It compares energy, not agreement -- correctly-scaled
        # noise also scores 1.0 -- so it is only trustworthy alongside LPIPS.
        #
        # `tex_render`/`tex_gt` are logged separately on purpose. Lightning averages each over the
        # val set, so their RATIO is the energy-weighted texture ratio, which leans towards the
        # building views because their GT gradient is 5-10x the water views' (0.051-0.061 vs
        # 0.005-0.013). `texratio` alone is the per-image mean, which flat water dilutes -- the
        # same averaging trap that hid a 2.7x texture gap behind 1.37 dB of PSNR (紀錄 2026-08-06).
        #
        # ⚠ The weighted ratio is an UPPER BOUND on the building-only reading, not equal to it:
        # the blurrier the buildings, the smaller their share of the NUMERATOR, so correctly
        # rendered water inflates it. Estimated on b12 that is ~0.27 against a true ~0.145. It is
        # biased consistently, so it is sound for tracking a run and for comparing runs on the same
        # val set; for a number to report, use tools/rescore_by_content.py, which splits by content.
        _, image_info, _ = batch
        _, gt_image, _ = image_info
        g_gt = _grad_energy(gt_image)
        g_render = _grad_energy(outputs["render"])
        metrics["tex_gt"] = g_gt
        metrics["tex_render"] = g_render
        metrics["texratio"] = g_render / g_gt.clamp_min(1e-8)
        pbar["tex_gt"] = pbar["tex_render"] = False
        pbar["texratio"] = True

        return metrics, pbar
