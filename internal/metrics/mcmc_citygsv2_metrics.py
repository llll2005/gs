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

    exposure_comp: bool = False
    """★ 2026-10-15 使用者同意（研究總覽 §3.4、§11）：逐影像曝光補償。MatrixCity GT 有逐幀曝光不一致（場景越暗越被提亮，
    log-log 斜率約 −0.5），靜態 3D 模型表達不了 => 每張訓練影像學一個 log 增益 la_k 與偏移 b_k，**只在算訓練光度 loss 時**
    把渲染換成 `exp(la_k − mean(la))·render + (b_k − mean(b))`（減平均＝固定尺度：每塊 log 增益平均 0、偏移平均 0，
    否則增益與模型整體亮度可以互相抵換）。渲染本身、trim、absgrad、驗證的原始 psnr 都不受影響。
    出處：🟨 逐影像外觀／曝光補償是已知做法（NeRF-W 類 appearance embedding、各種 exposure compensation）；這裡只用 2 個純量。
    評測：新視角不知道曝光 => 驗證時另記 `psnr_ex`＝用全域曲線 `la ≈ a·log(渲染亮度) + c`（從訓練影像學到的 la 與其渲染亮度擬合）
    預測增益後的 PSNR；原始 `psnr` 不變。"""
    exposure_lr: float = 1e-3
    exposure_from_iter: int = 1000
    """從這一步開始補償（先讓幾何與顏色大致定下來，避免增益吃掉早期的整體亮度誤差）。"""
    exposure_bias: bool = False
    """另學逐影像偏移 b_k（加法）。預設關：量到的現象是乘法性的（三通道同倍增益），偏移是多餘自由度，
    本機功能測試 300 步就長到 std 0.076（顏色單位），有吸收模型本身顏色誤差的風險。"""

    def instantiate(self, *args, **kwargs) -> "MCMCCityGSV2MetricsModule":
        return MCMCCityGSV2MetricsModule(self)


class MCMCCityGSV2MetricsModule(CityGSV2MetricsModule):
    config: MCMCCityGSV2Metrics

    # ── 逐影像曝光補償（config.exposure_comp） ──
    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)
        if not getattr(self.config, "exposure_comp", False):
            return
        n = len(pl_module.trainer.datamodule.dataparser_outputs.train_set)
        # 參數在 setup 依訓練集大小建立（fit／test 都建 => 從 ckpt 載入時 state_dict 對得上）
        self.ex_logit = torch.nn.Parameter(torch.zeros(n))
        self.ex_bias = torch.nn.Parameter(torch.zeros(n))
        self.register_buffer("ex_lum", torch.full((n,), float("nan")))     # 各訓練影像渲染亮度的 EMA（擬合全域曲線用）
        self.register_buffer("ex_curve", torch.zeros(2))                   # (a, c)：la ≈ a·log L + c
        print(f"[exposure] 逐影像曝光補償：{n} 張訓練影像 x（log 增益" + ("＋偏移" if self.config.exposure_bias else "") +
              f"），lr {self.config.exposure_lr:g}，從 step {self.config.exposure_from_iter} 起", flush=True)

    def training_setup(self, pl_module):
        opt, sch = super().training_setup(pl_module)
        if getattr(self.config, "exposure_comp", False):
            params = [self.ex_logit] + ([self.ex_bias] if self.config.exposure_bias else [])
            opt = list(opt) + [torch.optim.Adam(params, lr=self.config.exposure_lr, eps=1e-15)]
        return opt, sch

    def _exposure_apply(self, step, batch, outputs):
        camera = batch[0]
        k = int(camera.idx)
        r = outputs["render"]
        with torch.no_grad():
            lum = r.detach().mean()
            old = self.ex_lum[k]
            self.ex_lum[k] = lum if torch.isnan(old) else 0.9 * old + 0.1 * lum
        if step < self.config.exposure_from_iter:
            return outputs
        g = torch.exp(self.ex_logit[k] - self.ex_logit.mean())
        out = dict(outputs)
        out["render"] = g * r + ((self.ex_bias[k] - self.ex_bias.mean()) if self.config.exposure_bias else 0.0)
        if not getattr(self, "_ex_announced", False):
            self._ex_announced = True
            print(f"[exposure] ✅ 首次觸發 step={step}：影像 {k} 增益 {float(g):.4f}", flush=True)
        return out

    def _exposure_curve(self):
        """從訓練影像學到的 la（置中）與渲染亮度擬合 la ≈ a·log L + c（新視角用它預測增益）。"""
        with torch.no_grad():
            m = torch.isfinite(self.ex_lum) & (self.ex_lum > 1e-4)
            if int(m.sum()) < 10:
                return
            x = torch.log(self.ex_lum[m]); y = (self.ex_logit - self.ex_logit.mean())[m]
            xm, ym = x.mean(), y.mean()
            a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp_min(1e-12)
            self.ex_curve[0] = a; self.ex_curve[1] = ym - a * xm

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs):
        metrics, pbar = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)
        if getattr(self.config, "exposure_comp", False):
            self._exposure_curve()
            gt = batch[1][1]
            r = outputs["render"]
            g = torch.exp(self.ex_curve[0] * torch.log(r.detach().mean().clamp_min(1e-4)) + self.ex_curve[1])
            mse = ((g * r).clamp(0, 1) - gt).pow(2).mean()
            metrics["psnr_ex"] = -10 * torch.log10(mse.clamp_min(1e-12))
            pbar["psnr_ex"] = True
        return metrics, pbar

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        if getattr(self.config, "exposure_comp", False):
            outputs = self._exposure_apply(step, batch, outputs)
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        if getattr(self.config, "exposure_comp", False) and step >= self.config.exposure_from_iter:
            with torch.no_grad():
                la = self.ex_logit - self.ex_logit.mean()
                metrics["ex_std"] = la.std()
                pbar["ex_std"] = False

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
