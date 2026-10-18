from typing import Tuple, Dict, Any

from .metric import MetricImpl
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


class GS2DMetrics(VanillaMetrics):
    lambda_normal: float = 0.05
    lambda_dist: float = 0.
    normal_regularization_from_iter: int = 7000
    dist_regularization_from_iter: int = 3000
    lambda_dist_ramp_until: int = 0
    """★ 2026-10-18 使用者「堆疊懲罰從 0 逐步加強」：>0 時 dist 權重從 dist_regularization_from_iter 線性升到此步才達 lambda_dist
    （0＝原行為：到 from_iter 一步跳到 lambda_dist）。動機：dist 1000 從頭固定 => −2~3 dB；增生初期幾何未成形時不宜強壓。"""

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return GS2DMetricsImpl(self)


class GS2DMetricsImpl(VanillaMetricsImpl):
    def train_metrics(self, pl_module, step: int, batch, outputs, basic_metrics: Tuple[Dict, Dict]):
        metrics, prog_bar = basic_metrics

        # regularization
        lambda_normal = self.config.lambda_normal if step > self.config.normal_regularization_from_iter else 0.0
        lambda_dist = self.config.lambda_dist if step > self.config.dist_regularization_from_iter else 0.0
        _ru = int(getattr(self.config, "lambda_dist_ramp_until", 0) or 0)
        if _ru > 0 and lambda_dist > 0:
            _f0 = self.config.dist_regularization_from_iter
            lambda_dist = lambda_dist * min(1.0, max(0.0, (step - _f0) / max(_ru - _f0, 1)))
            if not getattr(self, "_dist_ramp_announced", False) and lambda_dist > 0 and step <= _ru:   # sanity check 用假步數，排除
                self._dist_ramp_announced = True
                print(f"[dist-ramp] ✅ 首次觸發 step={step}：權重 {lambda_dist:.4g}（線性升到 step {_ru} 達 {self.config.lambda_dist:g}）", flush=True)

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
        metrics["dist_w"] = zero + lambda_dist
        prog_bar["dist_w"] = False
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
