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

    opacity_reg_until_iter: int = -1
    """>=0 時，`step >= 此值` 之後把 `opacity_reg` 設為 0（`scale_reg` 不動）。

    動機（§11.61 + §11.62）：MCMC 的回收路徑在 `densify_until_iter` 之後 early-return
    （`mcmc_2dgs_density_controller.py:487`），**opacity L1 卻繼續壓 30,000 步**
    ⇒ 判死比例 3.77% -> 15.03%，而死掉的沒有任何機制回收。

    ⚠ **刻意不是 `freeze_opacity_after_densify`**（把 opacity 的梯度整個歸零）。
    收割期實測在做**兩極化**而非普遍衰減：`o` 中位 0.1083 -> 0.1539（**升**）、
    `o>0.5` 佔比 9.12% -> 23.22%（**翻倍**）、中間帶 22.26% -> 8.99%（排乾）。
    那個往「確信」的遷移很可能正是收割期 +1 dB 的來源 ⇒ 凍結全部梯度會把它一起關掉。
    本旗標只移除**固定的下壓力（L1）**，光度梯度對 opacity 的學習照常 ⇒ 兩極化保留。
    """

    opacity_entropy_reg: float = 0.0
    """>0 時加上 opacity 的**二元熵**正則 H(o) = -o ln o - (1-o) ln(1-o)（對全部顆粒取平均），
    把 o 往 0 或 1 推（兩極化）。2026-10-01 使用者提的構想：兩極化後靠近 0 的會被 trim／MCMC 回收 ⇒
    顆數可降、每個 tile 的前幾名貢獻更集中。⚠ 不是「整體變高」：那個方向已有反證（收割期關 L1 => 四項全輸）。
    梯度 dH/do = ln((1-o)/o)：o=0.5 為 0、o=0.1 約 2.2 ⇒ 權重與 opacity_reg 同量級（0.002）時，極端處約 L1 的 2 倍。"""
    opacity_entropy_from_iter: int = 0
    """從這一步開始加熵正則（要在 densify_until 之前才會被回收機制吃到）。"""

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

        _oreg_w = self.config.opacity_reg
        if 0 <= self.config.opacity_reg_until_iter <= step:
            _oreg_w = 0.0      # 收割期只移除 L1 下壓力，光度梯度照常（見 config docstring）

        metrics["loss"] = metrics["loss"] \
            + _oreg_w * opacity_reg \
            + self.config.scale_reg * scale_reg

        _ew = float(getattr(self.config, "opacity_entropy_reg", 0.0))
        if _ew > 0 and step >= int(getattr(self.config, "opacity_entropy_from_iter", 0)):
            _o = opacities.clamp(1e-6, 1 - 1e-6)
            _H = -(_o * torch.log(_o) + (1 - _o) * torch.log(1 - _o)).mean()
            metrics["loss"] = metrics["loss"] + _ew * _H
            metrics["op_ent"] = _H.detach()
            pbar["op_ent"] = False
            if not getattr(self, "_ent_announced", False):
                self._ent_announced = True
                _od = _o.detach()
                print(f"[opacity-entropy] ✅ 首次觸發 step={step} 權重={_ew:g}  H={float(_H):.4f}  "
                      f"o∈(0.1,0.9) 佔 {100 * float(((_od > 0.1) & (_od < 0.9)).float().mean()):.1f}%", flush=True)

        metrics["op_reg"] = opacity_reg.detach()
        metrics["sc_reg"] = scale_reg.detach()
        pbar["op_reg"] = False
        pbar["sc_reg"] = False
        return metrics, pbar
