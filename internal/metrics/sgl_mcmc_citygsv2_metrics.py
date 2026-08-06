"""
Selective Gradient Loss (SGL) on top of MCMCCityGSV2Metrics.

From SOGS (Second-Order Anchor for Advanced 3DGS, eq.12-16). SGL computes Sobel gradient
maps of the rendered image vs GT, and weights the gradient-domain loss by the per-pixel
gradient *discrepancy* -> the optimization dynamically focuses on hard-to-render textured /
structured regions (edges, fine detail), which plain L1+SSIM under-emphasizes.

Representation-agnostic (pure image-space loss) -> drops onto our current 2DGS pipeline with
no anchor machinery. This is the cheap, self-contained piece of SOGS; the anchor+MLP
representation (Scaffold-GS base) is the separate big port (see 紀錄/SOGS移植藍圖.md).

Implementation note: SOGS eq.15-16 use the discrepancy map w as a per-pixel weight ("dynamic
region selection"). We realize that intent as a self-weighted gradient loss
  L_sgl = mean( w_x * |dx_pred - dx_gt| ) + mean( w_y * |dy_pred - dy_gt| ),  w = discrepancy.detach()
i.e. pixels where the rendered gradient disagrees most with GT get the most loss. The detach
keeps w a selection weight (not double-counted in the gradient).
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .mcmc_citygsv2_metrics import MCMCCityGSV2Metrics, MCMCCityGSV2MetricsModule


@dataclass
class SGLMCMCCityGSV2Metrics(MCMCCityGSV2Metrics):
    lambda_sgl: float = 1.0
    """Weight of the selective gradient loss. 0 disables. Watch the logged `sgl` value vs
    `rgb_diff` to calibrate — SGL is a squared gradient-discrepancy so its raw scale is small."""

    def instantiate(self, *args, **kwargs) -> "SGLMCMCCityGSV2MetricsModule":
        return SGLMCMCCityGSV2MetricsModule(self)


class SGLMCMCCityGSV2MetricsModule(MCMCCityGSV2MetricsModule):
    config: SGLMCMCCityGSV2Metrics

    def _sobel(self, device, dtype):
        s = getattr(self, "_sobel_cache", None)
        if s is None or s[0].device != device or s[0].dtype != dtype:
            sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=dtype)
            sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=dtype)
            # depthwise (groups=3) conv weights: [3,1,3,3]
            s = (sx.view(1, 1, 3, 3).repeat(3, 1, 1, 1), sy.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
            self._sobel_cache = s
        return s

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        if self.config.lambda_sgl > 0:
            pred = outputs["render"]                       # [3,H,W], (0,1)
            _, image_info, _ = batch
            _, gt_image, _ = image_info
            gt = gt_image.to(pred.dtype)

            sx, sy = self._sobel(pred.device, pred.dtype)
            p = pred.unsqueeze(0)
            g = gt.unsqueeze(0)
            dx = (F.conv2d(p, sx, padding=1, groups=3) - F.conv2d(g, sx, padding=1, groups=3)).abs()
            dy = (F.conv2d(p, sy, padding=1, groups=3) - F.conv2d(g, sy, padding=1, groups=3)).abs()
            # selective weight = per-pixel discrepancy (detached) -> emphasize hard-to-render regions
            sgl = (dx.detach() * dx).mean() + (dy.detach() * dy).mean()

            metrics["loss"] = metrics["loss"] + self.config.lambda_sgl * sgl
            metrics["sgl"] = sgl.detach()
            pbar["sgl"] = True

        return metrics, pbar
