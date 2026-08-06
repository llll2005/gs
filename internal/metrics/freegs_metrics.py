"""FreGS frequency loss added on top of CityGSV2Metrics (grad-densify) and MCMCCityGSV2Metrics (MCMC)."""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .citygsv2_metrics import CityGSV2Metrics, CityGSV2MetricsModule
from .mcmc_citygsv2_metrics import MCMCCityGSV2Metrics, MCMCCityGSV2MetricsModule
from internal.utils.freegs_loss import frequency_loss


def _add_freegs(self, metrics, step, batch, outputs):
    if self.config.lambda_freq <= 0:
        return
    pred = outputs["render"]
    _, image_info, _ = batch
    _, gt, _ = image_info
    total = getattr(self, "_total_steps", 30000)
    fl = frequency_loss(pred, gt.to(pred.dtype), step, total)
    metrics["loss"] = metrics["loss"] + self.config.lambda_freq * fl
    metrics["freg"] = fl.detach()


@dataclass
class FreGSCityGSV2Metrics(CityGSV2Metrics):
    lambda_freq: float = 0.05

    def instantiate(self, *args, **kwargs):
        return FreGSCityGSV2MetricsModule(self)


class FreGSCityGSV2MetricsModule(CityGSV2MetricsModule):
    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)
        self._total_steps = getattr(pl_module.trainer, "max_steps", 30000) or 30000

    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        _add_freegs(self, metrics, step, batch, outputs)
        pbar["freg"] = False
        return metrics, pbar


@dataclass
class FreGSMCMCCityGSV2Metrics(MCMCCityGSV2Metrics):
    lambda_freq: float = 0.05

    def instantiate(self, *args, **kwargs):
        return FreGSMCMCCityGSV2MetricsModule(self)


class FreGSMCMCCityGSV2MetricsModule(MCMCCityGSV2MetricsModule):
    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)
        self._total_steps = getattr(pl_module.trainer, "max_steps", 30000) or 30000

    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        _add_freegs(self, metrics, step, batch, outputs)
        pbar["freg"] = False
        return metrics, pbar
