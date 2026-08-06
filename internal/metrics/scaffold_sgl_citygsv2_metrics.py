"""ScaffoldCityGSV2Metrics + SOGS Selective Gradient Loss (for Scaffold-2DGS config 4)."""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .scaffold_citygsv2_metrics import ScaffoldCityGSV2Metrics, ScaffoldCityGSV2MetricsModule


@dataclass
class ScaffoldSGLCityGSV2Metrics(ScaffoldCityGSV2Metrics):
    lambda_sgl: float = 0.2

    def instantiate(self, *args, **kwargs) -> "ScaffoldSGLCityGSV2MetricsModule":
        return ScaffoldSGLCityGSV2MetricsModule(self)


class ScaffoldSGLCityGSV2MetricsModule(ScaffoldCityGSV2MetricsModule):
    def _sobel(self, device, dtype):
        s = getattr(self, "_sobel_cache", None)
        if s is None or s[0].device != device:
            sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
            sy = torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype)
            s = (sx.view(1, 1, 3, 3).repeat(3, 1, 1, 1), sy.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
            self._sobel_cache = s
        return s

    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        if self.config.lambda_sgl > 0:
            pred = outputs["render"]
            _, image_info, _ = batch
            _, gt, _ = image_info
            g = gt.to(pred.dtype)
            sx, sy = self._sobel(pred.device, pred.dtype)
            dx = (F.conv2d(pred[None], sx, padding=1, groups=3) - F.conv2d(g[None], sx, padding=1, groups=3)).abs()
            dy = (F.conv2d(pred[None], sy, padding=1, groups=3) - F.conv2d(g[None], sy, padding=1, groups=3)).abs()
            sgl = (dx.detach() * dx).mean() + (dy.detach() * dy).mean()
            metrics["loss"] = metrics["loss"] + self.config.lambda_sgl * sgl
            metrics["sgl"] = sgl.detach()
            pbar["sgl"] = True
        return metrics, pbar
