from typing import Tuple, Dict, Any

from .metric import MetricImpl
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


class GS2DMetrics(VanillaMetrics):
    lambda_normal: float = 0.05
    lambda_dist: float = 0.
    normal_regularization_from_iter: int = 7000
    dist_regularization_from_iter: int = 3000

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return GS2DMetricsImpl(self)


class GS2DMetricsImpl(VanillaMetricsImpl):
    def train_metrics(self, pl_module, step: int, batch, outputs, basic_metrics: Tuple[Dict, Dict]):
        metrics, prog_bar = basic_metrics

        # regularization
        lambda_normal = self.config.lambda_normal if step > self.config.normal_regularization_from_iter else 0.0
        lambda_dist = self.config.lambda_dist if step > self.config.dist_regularization_from_iter else 0.0

        # 2026-08-26：權重為 0 時**跳過計算**，不要算完再乘 0。
        # 原本兩項都是無條件全幅逐像素計算、且帶 grad 進 loss ⇒ backward 也走一遍。
        # 現行配方 `lambda_normal = 0` 且 `lambda_dist = 0`（預設）⇒ 兩項全是純浪費。
        # ⚠ 位元級等價：`loss + 0.0 * X` 與 `loss + 0` 只在 X 含 inf/nan 時不同
        #   （那時 0*inf = nan 會毒化整個 loss）⇒ 加閘門反而更穩健。
        zero = metrics["loss"].new_zeros(())
        if lambda_normal > 0:
            rend_normal = outputs['rend_normal']
            surf_normal = outputs['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
        else:
            normal_loss = zero
        if lambda_dist > 0:
            dist_loss = lambda_dist * (outputs["rend_dist"]).mean()
        else:
            dist_loss = zero

        # update metrics
        metrics["loss"] = metrics["loss"] + dist_loss + normal_loss
        metrics["normal_loss"] = normal_loss
        prog_bar["normal_loss"] = False
        metrics["dist_loss"] = dist_loss
        prog_bar["dist_loss"] = False

        return metrics, prog_bar

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        basic_metrics = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)
        return self.train_metrics(
            pl_module=pl_module,
            step=1 << 30,
            batch=batch,
            outputs=outputs,
            basic_metrics=basic_metrics,
        )

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        basic_metrics = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        return self.train_metrics(
            pl_module=pl_module,
            step=step,
            batch=batch,
            outputs=outputs,
            basic_metrics=basic_metrics,
        )
