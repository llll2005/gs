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
    lambda_dist_interval: int = 1
    """★ 2026-10-24 >1 時 dist 只在 step % k == 0 的步算、權重乘 k（期望值不變）；其餘步光柵器走不含 distortion 的 lean
    （renderer.training_forward 用同一個規則決定 _lean_dist）。動機：本機量到 dist100 讓每步 +21%（render forward +4.6 ms、
    backward +10.9 ms；紀錄/加速論文對照.md），而它的作用是幾何（F1、馬路地底），不一定要每步都壓。1＝原行為。"""
    lambda_normal_interval: int = 1
    """★ 2026-10-29 >1 時 normal loss 只在 step % k == 0 的步算、權重乘 k（期望值不變）；其餘步光柵器照走 lean
    （renderer.training_forward 用同一個規則決定該步要不要幾何通道）。動機：法向一致性把 b12 面元朝向一致度 0.80 -> 0.95
    （修立面毛邊），但幾何通道讓每塊 +50~60% 時間。1＝原行為（每步都算；lean 配方下該步自動退回非 lean）。"""

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

        _ni = int(getattr(self.config, "lambda_normal_interval", 1) or 1)
        if _ni > 1 and lambda_normal > 0:
            lambda_normal = lambda_normal * _ni if step % _ni == 0 else 0.0
            if lambda_normal > 0 and step < 10 ** 8 and not getattr(self, "_normal_interval_announced", False):
                self._normal_interval_announced = True
                print(f"[normal-interval] ✅ 首次觸發 step={step}：normal 每 {_ni} 步算一次、權重 {lambda_normal:.4g}（＝{_ni} 倍）", flush=True)

        _di = int(getattr(self.config, "lambda_dist_interval", 1) or 1)
        if _di > 1 and lambda_dist > 0:
            lambda_dist = lambda_dist * _di if step % _di == 0 else 0.0
            if lambda_dist > 0 and step < 10 ** 8 and not getattr(self, "_dist_interval_announced", False):   # sanity check 用假步數（1073741824），排除
                self._dist_interval_announced = True
                print(f"[dist-interval] ✅ 首次觸發 step={step}：dist 每 {_di} 步算一次、權重 {lambda_dist:.4g}（＝{_di} 倍）", flush=True)

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
