"""AtomGS Edge-Aware Normal Loss added on top of MCMCCityGSV2Metrics (the 22.18 baseline metric).

Pairs with ANY density controller (MCMC or EdgeMCMC) since metric and density are independent
config slots -- so the same metric serves both the loss-only sweep and the module-combination arm.
Adds  lambda_ea_normal * edge_aware_normal_loss(surf_normal, gt)  to the loss. Keeps the original
2DGS normal-consistency term (lambda_normal, kept low per 2DGS-R's "consistency hurts PSNR"); this
edge-aware curvature term is a complementary geometry-smoothness prior, not a replacement.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .mcmc_citygsv2_metrics import MCMCCityGSV2Metrics, MCMCCityGSV2MetricsModule
from internal.utils.atom_normal_loss import edge_aware_normal_loss


@dataclass
class AtomNormalMCMCCityGSV2Metrics(MCMCCityGSV2Metrics):
    lambda_ea_normal: float = 0.05
    """weight of the AtomGS edge-aware normal (curvature) loss. 0 disables (== baseline)."""
    ea_normal_q: int = 2
    """ω(x)=(x-1)^q sharpness; larger q = sharper flat/edge separation (must be even)."""
    ea_normal_from_iter: int = 0
    """start applying the edge-aware normal loss from this step."""

    def instantiate(self, *args, **kwargs) -> "AtomNormalMCMCCityGSV2MetricsModule":
        return AtomNormalMCMCCityGSV2MetricsModule(self)


class AtomNormalMCMCCityGSV2MetricsModule(MCMCCityGSV2MetricsModule):
    config: AtomNormalMCMCCityGSV2Metrics

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        if self.config.lambda_ea_normal > 0 and step >= self.config.ea_normal_from_iter:
            ea = edge_aware_normal_loss(
                outputs["surf_normal"], batch[1][1], q=self.config.ea_normal_q
            )
            metrics["loss"] = metrics["loss"] + self.config.lambda_ea_normal * ea
            metrics["ea_n"] = ea.detach()
            pbar["ea_n"] = False
        return metrics, pbar
