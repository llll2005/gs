"""
CityGSV2 metrics + MCMC's two L1 regularizers (opacity_reg, scale_reg).

3DGS-MCMC adds `opacity_reg * |opacity| + scale_reg * |scale|` to the loss (train.py).
In this framework losses live in the metric, so we add them here. Per the code-grounded
mapping in 紀錄/主線_Gaussian效率.md §3, these two L1 terms replace the two pruning
mechanisms MCMC removes: opacity_reg ↔ opacity-prune (<0.005, feeds MCMC's dead_mask),
scale_reg ↔ size-prune (keeps surfels small).

Immune moat (protects the alpha=0.99 depth-init prior): opacity_reg is applied ONLY to
Gaussians whose CURRENT opacity is below `immune_opacity_threshold`. High-opacity surfels
(load-bearing, incl. the depth-init alpha=0.99 ones) are exempt, so the sparsity pressure
only pushes UNCERTAIN surfels toward death (where MCMC recycles them) without dragging down
decided geometry. This protects by current state, not origin — no per-Gaussian buffer needed,
and a depth-init surfel that legitimately decayed is not immune forever. scale_reg is uniform.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from .citygsv2_metrics import CityGSV2Metrics, CityGSV2MetricsModule


@dataclass
class MCMCCityGSV2Metrics(CityGSV2Metrics):
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    immune_opacity_threshold: float = 0.9
    """Opacity above this -> exempt from opacity_reg (covers alpha=0.99 depth-init)."""

    def instantiate(self, *args, **kwargs) -> "MCMCCityGSV2MetricsModule":
        return MCMCCityGSV2MetricsModule(self)


class MCMCCityGSV2MetricsModule(CityGSV2MetricsModule):
    config: MCMCCityGSV2Metrics

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        opacities = gaussian_model.get_opacities().squeeze(-1)  # [N], activated in (0, 1)
        non_immune = opacities.detach() <= self.config.immune_opacity_threshold
        if bool(non_immune.any()):
            opacity_reg = opacities[non_immune].mean()        # |opacity| == opacity in (0,1)
        else:
            opacity_reg = opacities.sum() * 0.0               # grad-safe zero
        scale_reg = gaussian_model.get_scales().abs().mean()

        metrics["loss"] = metrics["loss"] \
            + self.config.opacity_reg * opacity_reg \
            + self.config.scale_reg * scale_reg

        metrics["op_reg"] = opacity_reg.detach()
        metrics["sc_reg"] = scale_reg.detach()
        pbar["op_reg"] = False
        pbar["sc_reg"] = False
        return metrics, pbar
