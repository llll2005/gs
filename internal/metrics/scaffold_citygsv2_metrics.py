"""
CityGSV2 metrics for Scaffold-2DGS: same photometric + depth/normal regularization, but
WITHOUT the grad-densify (DGD) loss/extra_loss split, which references
density.densify_until_iter (absent on StaticDensityController, and meaningless for an
anchor+MLP model that doesn't grad-densify). We always use the simple combined loss.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .citygsv2_metrics import CityGSV2Metrics, CityGSV2MetricsModule
from .gs2d_metrics import GS2DMetricsImpl


@dataclass
class ScaffoldCityGSV2Metrics(CityGSV2Metrics):
    def instantiate(self, *args, **kwargs) -> "ScaffoldCityGSV2MetricsModule":
        return ScaffoldCityGSV2MetricsModule(self)


class ScaffoldCityGSV2MetricsModule(CityGSV2MetricsModule):
    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        # grandparent (GS2D): loss = photometric + dist + normal  (no densify split)
        metrics, pbar = GS2DMetricsImpl.get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs)
        d_reg = self.get_inverse_depth_metric(batch, outputs) * self.get_weight(step)
        metrics["loss"] = metrics["loss"] + d_reg
        metrics["d_reg"] = d_reg.detach()
        metrics["d_w"] = self.get_weight(step)
        pbar["d_reg"] = True
        pbar["d_w"] = False
        return metrics, pbar
