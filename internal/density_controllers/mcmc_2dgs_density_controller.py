"""
3DGS-MCMC adapted for 2D Gaussian Surfels (2DGS).

Base: `MCMCDensityController` (internal/density_controllers/mcmc_density_controller.py),
which is written for 3DGS (3D scale). This subclass makes three minimal, code-level
adaptations so MCMC's relocation + noise work on Gaussian2D (2D surfel scale). See
`紀錄/MCMC整合分析.md` and `紀錄/主線_Gaussian效率.md` §3 for the rationale.

Why these three (verified against source, 2026-06-07):
  1. relocation scale dim — gsplat `compute_relocation` requires scales [N, 3] (CUDA loops
     i<3). But the relocation coefficient comes ONLY from opacity (CUDA `cuda_rasterizer/
     utils.cu`: opacity_new = 1-(1-o)^(1/N); coeff = o/denom(o); scale_new = coeff*scale_old),
     so it is scale-dimension-agnostic. We pad the 2D scale with a zero 3rd component, call
     gsplat, then truncate back to 2D — exact for the two real components.
  2. noise geometry — the parent shapes xyz noise by the 3D covariance. For a 2D surfel the
     normal-direction scale is implicitly 0; building a real 3D covariance would push new
     points OFF the surfel plane along the normal, destroying the geometry 2DGS fits. We pad
     the 2D scale with a zero normal component before `compute_cov_3d`, so the covariance has
     zero normal variance and the noise stays in the tangent plane.
  3. setup — the parent only registers the noise hook when `initialize_from is None`, and it
     clobbers opacities/scales (would destroy our alpha=0.99 depth-init prior). We register
     the hook unconditionally and skip the clobber whenever we initialize from a prior.
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, Tuple

import torch
from lightning import LightningModule
from gsplat.relocation import compute_relocation

from internal.utils.gaussian_projection import compute_cov_3d
from .density_controller import DensityControllerImpl, Utils
from .mcmc_density_controller import MCMCDensityController, MCMCDensityControllerImpl


@dataclass
class MCMC2DGSDensityController(MCMCDensityController):
    """MCMC density controller for 2D Gaussian surfels. Same config as the 3DGS MCMC
    controller (cap_max, noise_lr, densify_*, min_opacity, N_max); only the impl differs."""

    screen_size_prune_px: int = -1
    """>0 enables screen-size recycling: at each densify event, Gaussians whose max screen
    radius over the last interval exceeds this many pixels are added to the dead mask and
    recycled by relocation. MCMC has no counterpart of vanilla ADC's max_screen_size prune,
    so mid-opacity near-camera monsters (radius ∝ scale/depth, single splat can cover the
    whole frame) survive both relocation (opacity > min_opacity) and contribution trimming
    (transmittance > 0) while their rasterizer buffers (∝ intersections) blow up 6GB VRAM.
    Measured on MC-aerial block_12: cameras fly z 1.5-5.0 through content airspace, nearest
    point-camera distance 0.017, 106k point-camera pairs within 1.0 → deterministic OOM at
    step ~3-4k. -1 keeps the stock MCMC behavior."""

    densify_grad_scaler: float = 0.0
    """gaussian_splatting.py:433 reads this when the metric emits an `extra_loss` (CityGSV2
    hard-depth reg) to rebalance the viewspace-grad used for *gradient-based* densification.
    MCMC densifies via relocate/add_new (not viewspace grad), so this is irrelevant here.
    0.0 makes `max(scaler * ratio, 1.0) == 1.0` → the viewspace grad is left unchanged (no-op);
    both backward passes still accumulate into the real parameter grads."""

    min_opacity_final: float = -1.
    """Zombie-recycling fix (2026-07-19): if > 0, exponentially anneal the relocation
    death line from `min_opacity` up to this value between densify_from_iter and
    min_opacity_anneal_end_iter. Root cause being fixed: the opacity-L1 equilibrium
    parks crushed points at o~0.01-0.05, ABOVE the 0.005 death line — they never die,
    never relocate, never render (sub-pixel dust), yet occupy cap and model VRAM
    (measured b12: 96.6% of points carry 0.1% of render mass). Raising the line past
    the parking band converts zombies back into relocation supply. The exponential
    curriculum (Gemini round-5 endorsed) avoids a one-shot relocation storm."""

    min_opacity_anneal_end_iter: int = 15_000
    """step at which the annealed death line reaches min_opacity_final"""

    min_opacity_anneal_start_iter: int = -1
    """step at which the death-line annealing starts; -1 = densify_from_iter.
    Gemini round-6 'condensation forcing' recipe sets this to the densify TAIL
    (e.g. 30k -> 42k): losers are pushed over the death line while relocation is
    still on, so they are recycled into probes instead of piling up as zombies
    when relocation shuts off."""

    harvest_dust_trim_interval: int = -1
    """>0 enables periodic dust trimming DURING the harvest phase (after
    densify_until_iter): every this many steps, prune points that are both
    near-transparent (opacity < harvest_dust_opacity) and sub-pixel
    (max tangent scale < harvest_dust_px). Rationale (2026-07-19 audits): dust
    forms progressively during harvest (0.3% at 42k -> 86.5% at 60k) as the
    L1s crush condensation losers, and with relocation off nothing removes the
    corpses — they render nothing (verified -0.000 dB / bit-identical on b12)
    while paying O(N) projection + optimizer + model-VRAM tax for 18k steps.
    -1 keeps stock behavior."""

    harvest_dust_opacity: float = 0.05

    harvest_dust_px: float = 0.00155
    """scene-units size of one pixel at typical view distance (b12-calibrated);
    scale this with your scene if reusing elsewhere"""

    add_ratio: float = 1.05
    """每個 densify 事件的族群成長倍率（`add_new_gs`: N -> min(cap, add_ratio*N)）。
    3dgs-mcmc 原版寫死 1.05；改成可調是為了壓縮排程（總步數砍半 => 事件數也砍半 =>
    每次變動量要提高，族群才到得了 cap）。

    ⚠ 與 `contribution_prune_interval` / `prune_ratio` 有破平衡關係：
        break-even interval = prune_interval x ln(add_ratio) / (-ln(1-prune_ratio))
      densification_interval 大於它 => 族群衰減。預設組（1.05 / 500 / 0.1）的破平衡點是 231.5。
    ⚠ 調高會讓族群更早撞 cap => 撞到之後全是 churn，而 §12.21/§12.25 證實那段是淨負的
      => 調高 add_ratio 時要同步把 `densify_until_iter` 往前拉。
    """

    # ── RTG-SLAM 式的透明修正層（§11.9）────────────────────────────────────────
    transparent_corrector: float = 0.0
    """>0 時：relocation 的目的地若其父代是「高誤差 + 高不透明」，就把新粒子的 opacity
    直接設成本值（例 0.1），而**不是**用 MCMC Eq.9 的分裂公式。0.0 = 關閉（現行行為）。

    來源（研究總覽 §11.9，已逐句核對 `參考論文/3641519.3657455.pdf` §3.1-3.2）：
    RTG-SLAM 對「已承諾的不透明高斯做錯了」的處理**不是解鎖它**（RTG 的 opacity 是二元的
    0.99/0.1 且 `lr_α = 0`，根本沒有梯度），而是
      "we add a transparent (α = 0.1) Gaussian to correct color errors together with the
       stable Gaussian"
    理由（原文）："such Gaussians with low opacity do not cause a significant attenuation of
    light energy, and the color impact to other views is little... during depth rendering,
    they are automatically filtered out by δ_α"。

    為什麼對我方有意義：§11.7 量到**好視角是靠大量半透明粒子做出來的**
    （opacity<0.1 佔 47~49%），壞視角反而高不透明多（>0.9 佔 18~23%）。
    RTG 的設計正好解釋了「半透明多」為什麼是好事：修顏色而不擋光、不干擾幾何。

    ⚠ 這**破壞 MCMC Eq.9 的 alpha 守恆**（那正是重點：RTG 不守恆，它是「加修正」不是「分裂」）。
    ⚠ 與 `err_unlock_frac` 是**互斥的兩個假說**：那個壓低已存在的粒子（我自己發明的），
      這個新增低不透明的伴隨粒子（RTG 的做法）。**不要同時開**，否則分不出是誰的功勞。
    ⚠ RTG 用 RGB-D 感測器的公制深度，我方是單目估計（8.6% 誤差）——機制可搬，精度不會跟著搬。
    """

    transparent_corrector_min_opacity: float = 0.5
    """只有父代 opacity 高於此值才觸發（低不透明的父代本來就沒擋住任何東西）。"""

    transparent_corrector_err_frac: float = 0.3
    """在符合上一條的父代裡，只取 `_err_score` 最高的這個比例。"""

    # ── 錯誤觸發的 opacity 解鎖（RTG-SLAM B4 的想法移植到 MCMC）─────────────────
    err_unlock_frac: float = 0.0
    """>0 時，每個 densify 事件把「持續高誤差 + 高不透明」的粒子 opacity 壓回
    `err_unlock_to`，讓被它們擋住的幾何重新收到梯度。0.0 = 關閉（現行行為）。

    為什麼需要（研究總覽 §11.7）：b12 最差的視角（1729/2999/3007）與最好的（2652/2643/1292）
    內容難度相同、粒子更多、沒有錯位（平移掃描峰值在 (0,0)），唯一分得開的是 **opacity 分布**：
    壞視角 opacity>0.9 的粒子佔 18~23%，好視角只有 8~9%。
    ⇒ 假說：早期把某層推到高 opacity ⇒ 後面的正確幾何永遠收不到梯度 ⇒ 自我強化的局部極小。
    這解釋了為什麼它從 step 15,000 起 45,000 步只改善 4.5%、跨六種配方 Spearman ρ=0.990。

    為什麼是這個做法而不是 vanilla 的 `opacity_reset_interval`：
      * vanilla 3DGS 每 3,000 步把**全部**opacity 壓低 —— 會連 90% 正常的區域一起打斷；
      * RTG-SLAM 的 B4（見 `rtg_stable_density_controller.py` 的 B4 段）是**錯誤觸發**的：
        只把「持續落在高誤差像素上」的粒子退回可塑狀態。本項是它在 MCMC 上的對應物。
      * 零件本來就有：`_err_score` 每一步都在累積逐顆誤差（`_accumulate_error_score`），
        目前只餵給從沒開過的 `err_guided_densify`。

    ⚠ 壓到 `err_unlock_to` 而不是壓到 0：低於 `min_opacity` 會被 MCMC 判定為 dead 並被
      relocate 搬走，那會連位置一起換掉（多一個變數）。留在原地變透明，讓最佳化自己決定。
    """

    err_unlock_to: float = 0.05
    """`err_unlock_frac` 觸發時把 opacity 壓到多少（要 > min_opacity，否則會被 relocate 搬走）。"""

    err_unlock_min_opacity: float = 0.5
    """只解鎖 opacity 高於此值的粒子 —— 低 opacity 的本來就沒擋住任何東西。"""

    err_guided_densify: float = 0.0
    """>0 steers add_new_gs toward unexplained error; 0.0 = DIAGNOSTIC ONLY (measure, change
    nothing). MCMC picks split sites with `probs = get_opacities()` -- there is no error term at
    all, so it densifies wherever mass already is, not where the render is wrong. Every other
    densifier we read uses an error-ish signal (FastGS counts high-error pixels in the footprint,
    Taming weights 8 signals, Mini-Splatting uses max-contribution area). Ours is the least
    informed of them, and the working-set audit says 89% of primitives carry 5% of render mass.

    Signal (FastGS's, approximated at tile resolution so no CUDA change is needed): per step,
    |render - gt| pooled to the 16px tile grid, gathered at each primitive's projected centre and
    accumulated until the next densify event. The buffer self-heals on any N change, which is what
    we want -- an event consumes exactly the error accumulated since the previous one.

    When >0: `probs = opacity * (1 + err_guided_densify * normalised_err)`. Multiplicative, not a
    replacement: MCMC's split formula o_new = 1-(1-o)^(1/N) divides a parent's opacity among its
    children, so sampling a low-opacity parent just makes fainter children. Keeping opacity as
    the base preserves that, and only steers where the mass goes. Taming's score is a weighted
    product for the same reason."""

    dar_lambda: float = 0.0
    """>0 enables DAR-style decoupled, COST-AWARE opacity regularization — the unified
    framework in `紀錄/現行方案與公式.md` §1. Set the metric's opacity_reg
    to 0 when using this (the point is to take the regularizer OUT of the loss).

    Derivation: the VRAM budget is a constraint, not a loss term. Relax the discrete
    "does primitive i exist" by its opacity and the Lagrangian gives

        min ℓ(θ)  s.t. Σ_i c_i·o_i ≤ B   ⇒   ∂L/∂o_i = ∂ℓ/∂o_i + λ·c_i

    so the regularization gradient IS the shadow price times the primitive's cost.
    MCMC's plain opacity L1 is the c_i ≡ 1 special case — the formal statement of this
    project's thesis that COUNT IS THE WRONG UNIT OF ACCOUNT. AdamW-GS (ICLR 2026,
    arXiv 2601.16736) fixes HOW the penalty is applied (decouple it from Adam's moments,
    precondition by 1/√v̂) but keeps c_i ≡ 1; we supply the c_i.

    Applied post-optimizer-step, so Adam's moments track only the photometric gradient:

        o_logit_i −= lr_o · min( λ · ĉ_i · u_i , dar_clip ),   u_i = √ε/√(v̂_i+ε) ∈ (0,1]
        ĉ_i = dar_cost_storage + log1p(min(r_i², H²+W²) / median(r²))

    ĉ_i is capped at the screen diagonal² (a splat cannot cost more than the frame), then
    log-compressed and median-normalised: raw radii² was measured to span 5.8e11× on b12
    because the projection Jacobian is singular as z→0. The constant term is the per-point
    model-state tax (F·4·M bytes, independent of screen footprint) — it keeps dust decaying
    slowly instead of accumulating without bound.

    Predictions (CPU-verified on b12@30k using that run's own Adam state, λ=0.05/C_t=0.25:
    dust dies in 3621 steps vs 64-106 under stock L1+Adam = 34-57x slower, fog in 1614,
    large-footprint in 519, normal geometry in 7051; cost ratio 6.98x, 0% clipped):
    sub-pixel dust gets LESS pressure than plain L1 (its c_i is tiny) so the 63% relocation
    treadmill shrinks; near-camera monsters get far MORE (their c_i explodes) so they die fast — which DAR alone cannot do, since a monster
    covers many pixels and therefore has photometric gradient that protects it.
    Also continuous (every step) rather than event-based, closing the 150-step gap in
    which monsters grow. 0.0 = off."""

    dar_cost_storage: float = 0.1
    """constant term in ĉ_i = the per-point model-state tax (人頭稅); keeps invisible dust
    under slow decay pressure so it cannot accumulate unbounded (the reg=0 failure mode)"""

    dar_eps: float = 1e-8
    """ε inside √(v̂+ε) (AdamW-GS Eq.8). Also the normaliser: u = √ε/√(v̂+ε) ∈ (0,1].
    Measured b12@30k: opacity gradients are so small (v̂ median 1.2e-17 ≪ ε) that u ≈ 1 for
    99% of points — i.e. **the DAR preconditioner is inert in this regime** and ĉ does all
    the work. Keep the term anyway: it costs nothing and engages wherever v̂ ≳ ε (early
    training, other scenes). Note (per review): v̂ is a squared-gradient EMA, so it measures
    the photometric loss's ABSOLUTE SENSITIVITY to the primitive, not a signed utility —
    "wants to be opaque" and "wants to vanish" both give large √v̂."""

    dar_clip: float = 0.25
    """C_t in AdamW-GS Eq.8. Not just numerical safety: it is a rate limit, so it sets a
    GUARANTEED MINIMUM SURVIVAL TIME for any primitive —

        t_death ≥ Δlogit / (lr_o · C_t),   Δlogit = 5.29 for o: 0.5 → 0.005 (min_opacity)

    C_t must sit ABOVE the most expensive group's stride or the clip binds and flattens ĉ's
    spread — the failure this mechanism exists to avoid. Measured b12@30k with λ=0.05:
    strides run 0.015 (normal) / 0.029 (dust) / 0.204 (large footprint), so C_t=0.25 leaves
    0% saturated and preserves the full 6.98x cost ratio, with a floor of 423 steps of life.
    C_t=0.1 clips 30% of points and collapses the ratio to 3.42x."""

    max_relocate_frac: float = 0.0
    """>0 caps how much of the population may be relocated in one densify event — a gate
    MCMC does not have (verified 2026-07-28: `relocate_gs` samples hosts by opacity with no
    volume limit and no gradient gate; Gemini's claim that MCMC-GS gates on positional
    gradient is a misattribution of vanilla ADC's split/clone criterion).

    Why it is needed (問題與應對總表 B0, the death treadmill): on b12 at 30k, 63.4% of points
    sit below the death line and ALL of them are relocated every 150 steps (616k points onto
    355k hosts). The mechanism is forced, not incidental: relocation hands a point back at
    o≈0.11 (measured), the opacity L1 walks its logit down ~lr=0.05 per step, so it crosses
    0.005 again in ~64 steps — well inside the 150-step event gap. The population never
    settles; this is why b12 is slow, why it sits near OOM, and why any mechanism that ADDS
    condemnation (condensation annealing, v/c) OOMs on top of it.

    Which dead points get the budget: the most EXPENSIVE ones first (largest screen footprint
    from the max-radii window), because recycling those is what actually reduces the binning
    load — a cost-aware relocation gate. Falls back to lowest-opacity-first when no radii
    window is available. The rest simply stay put this event (they are dust: they keep paying
    model-state VRAM but stop paying the relocation/churn cost).
    0.0 = off (stock MCMC: relocate everything)."""

    vpc_prune_frac: float = 0.0
    """>0 enables the value-per-cost recycling channel (2026-07-25). Each densify event,
    recycle the bottom `vpc_prune_frac` of Gaussians ranked by

        v_i / c_i  =  [multi-view Σ(T·α) / covered_pixels]  /  [screen tile footprint]

    i.e. render-grounded VALUE over render-grounded COST — the natural completion of the
    cost-aware thesis (we补了成本 but left value at opacity, an intrinsic property).
    Why it matters: the near-camera monster has opacity 0.028 (escapes the opacity death
    line) but a frame-filling footprint, so only a value/cost ratio catches it; the same
    criterion also kills fog (measured 82% of b12's binning intersections). Value comes
    free from the trim rasterizer's `record_transmittance` path (T·α accumulation, no CUDA
    change); cost reuses the max-radii window. 0.0 = off."""

    vpc_interval: int = 1500
    """how often (steps) to refresh the multi-view contribution estimate — it costs one
    transmittance-only render pass over the training cameras, so keep it infrequent"""

    screen_prune_emergency_px: int = -1
    """>0 enables per-STEP emergency monster recycling (decoupled from the
    densify event). After each step, if the largest screen radius seen exceeds
    this many pixels, immediately recycle the offending Gaussians instead of
    waiting up to densification_interval steps. Rationale (2026-07-20 CPU
    prediction on the b12 2M near-OOM ckpt): the binning load is SPREAD over the
    top ~5-8% largest points (not a few killable giants — top 0.1% carry only
    3.4%), which the existing screen_size_prune already targets; the failure mode
    is purely TEMPORAL — monsters grow past the OOM threshold within the 150-step
    event gap. Closing that gap to 1 step is the predicted fix, pure-Python, and
    only pays cost when a monster is actually present. Set this ABOVE the normal
    screen_size_prune_px (e.g. prune=300, emergency=600) so routine recycling
    stays on the event schedule and only true runaways trigger the per-step path.
    Requires screen_size_prune_px > 0 (shares its max-radii buffer). -1 = off."""

    vpc_report: int = 0
    """>0 時，每這麼多步量一次 v/c 並報告（**只讀，不剪枝**）。需要 `screen_size_prune_px > 0`
    （成本來自那個 max-radii 視窗）。

    要驗的是**退化定理的前提**（2026-08-25）：`archive/S11_成本模型_bug下量測_作廢.md` 自述
    「解析論證 `v_i = kappa*c_i => 選擇無自由度` 本身不依賴資料，可能仍成立，但支持它的
    **三個實測已作廢**（bug 期量的）」⇒ **前提從未在當前資料上驗證過**，
    而我一度拿這條定理擋掉整條成本感知剪枝線。

    今天的發現反而指向相反方向（§11.23）：trim 按 v 剪掉最低的 10%，卻**省不到 VRAM**
    （notrim2 多 11% 顆粒、VRAM 反而低 0.06G）⇒ 低 v 的正好是低 c 的 ⇒ 按 `v/c` 剪才會
    剪到「拿得多、給得少」的那批，而省下的 VRAM 可以換更高的 cap（顆數值 +0.541 dB/加倍）。

    報告的決定性數字是 **`bottom-k by v/c` 相對 `bottom-k by v` 多省下幾倍成本**：
      比值 ~1  => 退化成立，選子集沒有自由度 => 這條線收掉，出口只剩「改變集合」
      比值 >>1 => 有自由度 => `vpc_prune_frac` 值得跑一臂"""

    elongation_prune: float = 0.0
    """★ Elongation Filter（路徑 a：**硬剪**）。>0 時在每個 densify 事件把
    `s_max/s_min > 此值` 的粒子**直接刪除**（`_prune_points`），這是 CityGaussian V2 的做法。

    為什麼值得做（`tools/elongation_ceiling.py` + `binning_and_trim_probe.py` 實測）：
    `forward.cu:281` 的 tile 數是 `max(extent.x, extent.y)` 撐出的**正方形外接盒**
    ⇒ 長寬比 k 的粒子有 `1 - 1/k` 的 binning 是空白。speed3_b12 @60k：
    ```
    長寬比 p50=3.17  p90=13.28  p99=40.93          <- 拉長是**常態**不是罕見病理
    Σs_max² vs Σs_max·s_min => 浪費 76.5%
    剔除 >10（15.3% 顆）=> binning -31.6%，真覆蓋只 -7.3%   （4.3:1）
    剔除 >20（ 4.8% 顆）=> binning -14.0%，真覆蓋只 -1.8%   （7.9:1）
    ```
    而逐段計時量到 backward 佔 37~47%、逐顆項佔 77% ⇒ 打的是最大宗。
    ⚠ 但被剔除的那批帶著 15.8% 的不透明度質量 ⇒ 刪掉會改變畫面，不是免費的。
    ⚠ 我方是 MCMC，**沒有** CityGSV2 的 elongation filter（那在 `CityGSV2DensityController`）。
      現有的替代品是 `screen_size_prune_px`（看螢幕半徑）與 `scale_reg`（壓**平均**尺度），
      兩者都不直接看長寬比。訊號本身早就實作在
      `internal/metrics/scale_regularization_metrics.py:58`，但那個 mixin 掛在
      `VanillaMetrics` 上、我們用 `MCMCCityGSV2Metrics` ⇒ 接不到。
    """

    elongation_relocate: float = 0.0
    """★ Elongation Filter（路徑 b：**併入死亡判準、由 MCMC 搬移**）。>0 時把
    `s_max/s_min > 此值` 的粒子 OR 進 `dead_mask` ⇒ 走 relocation（搬到抽中的宿主）而非刪除。

    ⚠⚠ **這條路有一個既有的反證**：`vpc_prune_frac` 用的就是同一個機制（OR 進 dead_mask），
    實測 **-2.34 dB**（§11.34）。當時的結論是「把低 v/c 的粒子搬走會**破壞它們原本在做的事**；
    『改變集合』有效、『破壞集合』無效」。
    ⇒ 先驗上 (a) 硬剪比 (b) 搬移更有機會（(a) 有 CityGSV2 的外部先例，(b) 有我方的內部反證）。
    兩者都測是使用者 2026-09-12 的決定，但報告時必須帶上這個先驗。
    ⚠ 不要與 `elongation_prune` 同時開 —— 那會變成兩個變數。
    """

    absgrad_densify: float = 0.0
    """>0 時把 AbsGS 的 `|g|` 當**增生取樣權重**：`probs *= (1 + w * |g|/mean|g|)`。
    需要光柵器以 `ABSGRAD 1` 編譯（`|g|` 累加在 `viewspace_points.grad[:,2]`）。

    為什麼是這個而不是 `vpc_prune_frac`（§11.34）：後者把 mask 併進 `dead_mask` ⇒ 走
    relocation，把粒子丟到隨機宿主、毀掉它原本在做的事，實測 -2.34 dB。**診斷支持的是
    「決定往哪裡增生（改變集合）」，不是「把既有粒子搬走（破壞集合）」。**

    為什麼相信有空間（§11.30）：我方誤差分數的 `ceiling(top5%)/mean` 只有 **1.22**
    （完美取樣器也只能看到 1.22 倍平均誤差 ⇒ 沒東西可賺，`egd` 也確實無效），
    但 `|g|` 的同一個量是 **16.68**，是它的 13 倍；且 `|g|` 的 top10% 與 opacity 只重疊
    4.4~20.6%（隨機是 10%）⇒ 選的是幾乎不相干的一批 ⇒ **換訊號會換掉增生位置**。
    到 step 600 時 opacity 取樣只吃到 2.76/16.68 = 17%，還有 6 倍空間。"""

    cost_add_densify: float = 0.0
    """**加法式**的成本感知增生：`probs *= (1 + |w| * ŝ)`，ŝ = 訊號/其均值。

    w > 0 => 訊號 = **1/c**（偏好便宜，我方命題）／ w < 0 => 訊號 = **c**（偏好貴，Taming）。
    `c` 取自 trim pass 的 `num_covered_pixels`（**精確**渲染成本，每視角平均），
    而不是 `cost_aware_densify` 用的 `_max_radii2D²` 代理。

    為什麼要有這個旗標（研究總覽 §11.108/§11.109）：
    ```
    冪次式 cost_aware_densify   動態範圍 **300x**   兩個符號都輸（-4.8sd / -11.6sd）
    加法式 absgrad_densify      動態範圍  ~26x      **贏**（現行最佳的一部分）
    且冪次式吃的代理把離散度**誇大 66%**（ceiling 15.26x vs 精確 9.19x）
    ```
    ⇒ 疑點在**形式與強度**，不在訊號本身。本旗標把兩者都換掉。
    ★ 建議值 **2.278**：令 `1 + w*ceiling(1/c)` = 25.8，與 absgrad 同動態範圍
      （`tools/cost_proxy_quality.py` 實測 ceiling(1/ĉ) = 10.89x）。
    ⚠ 需要 trim 開著（精確 c 從那裡來）且 `densify_from_iter` 晚於第一次 trim。
      拿不到訊號時會**大聲印警告**，不會靜默 no-op（§11.101 的教訓）。"""

    cost_aware_densify: float = 0.0

    densify_blind_report: bool = False
    """`[densify-blind]` 診斷（每個 densify 事件印一次「opacity 取樣看到的誤差 / 全體平均」）。

    ⚠ 開啟它會連帶讓 `_accumulate_error_score` **每一步**執行（全幅 avg_pool2d + N x 3 投影）。
    2026-09-05 前這個累積是**無守衛**的 ⇒ 現行配方（三個真正的消費者全為 0）下純屬浪費。
    預設 False：訓練行為完全不變，只是少一行 log。"""

    noise_gate_eps: float = 0.0
    """>0 時，MCMC 位置噪音**只對 `op_sigmoid(1-o) > eps` 的粒子計算**（稀疏化）。

    為什麼可以（2026-09-05 實測，`agd2_b12` 的 14999/29999/60000 三個 ckpt）：
    噪音 = `cov @ (randn * gate * noise_lr * lr)`，而 `gate = sigmoid(100*(0.005-o))`
    對高 opacity 粒子本來就 ~0 —— 我們只是把「算出一個約等於 0 的位移」換成「不算」。
    ```
    eps      可跳過比例（14999/29999/60000）
    1e-4        40.2% / 53.8% / 62.6%
    1e-3        47.6% / 62.8% / **69.0%**
    ```
    安全性用**相對自身尺寸**判定（不可用「相對 Adam 步長」—— 被跳過的多半梯度趨近 0，
    比值必然爆掉，那是判準選錯不是機制有問題）：
    eps=1e-3 時被跳過者每步位移 <= `scale^2 * noise_lr * lr * eps` = 4.15e-8，
    60,000 步隨機遊走累積 sqrt(60000)*4.15e-8 = 1.0e-5 = **自身尺寸的 0.15%**。

    成本：`_add_xyz_noise` 實測 **59.4 ms/step**（總 602 ms 的 9.9%，`tools/step_breakdown.py`），
    成本來自 `compute_cov_3d(N)` + `randn_like(N,3)` + `bmm(N,3,3)` —— 全對整個族群做。
    ⚠ 非位元等價（RNG 串流不同），需要端到端驗分數。
    ⚠ 與 `fast_noise` 互補：那個是代數改寫（不 materialize N x 3 x 3），這個是減少 N。"""

    fast_noise: bool = False
    """MCMC noise 的等價改寫：`cov @ v = R S S^T R^T v = R (s^2 * (R^T v))`
    （`compute_cov_3d` 明文是 `(RS)(RS)^T`，S 對角 ⇒ 不必 materialize N x 3 x 3）。

    **實測（`agd2_b12` 60k ckpt，N=2.34M，CUDA event）**：57.47 -> 31.98 ms，
    省 44.4%（1.80x）⇒ 對每步 602 ms 只佔 **4.2%**。
    ⚠ **非位元級相同**：最大絕對差 9.5e-07，但**最大相對差 25%**（小值上的浮點抵消）。
      作用對象是**隨機**噪音項，統計上應無影響，但無法證明 ⇒ 預設關閉。
    ⚠ **4.2% 低於跨跑次 wall-time 的 5% 噪音底**（§11.64）⇒ 單獨投 9.6h 驗證不划算；
      規劃是等要跑 25 塊時再開（4.2% x 25 = 省約 10 小時，那時才值得驗）。
    """

    ac_densify: float = 0.0
    """★ **顯性高頻觸發**的增生取樣（2026-09-04，梯度盲區確認後的第一個機制）。

    `probs = opacity * (1 + w * AC/mean(AC))`，其中 `AC` = 該顆投影位置所在 tile 的
    **殘差高頻能量**（tile 內殘差的標準差），而**不是** `err_guided_densify` 用的
    `|render-gt|` 池化平均（那是 DC 誤差，天花板只有 1.22x）。

    **為什麼是這個訊號**（§11.80，實測）：失敗區的高頻殘差多 **86%**，
    但位置梯度只有 **41%** ⇒ **每單位殘差的訊號少 4.6 倍**；而且這個盲區
    **在 step 1,499 就存在**（那時失敗區殘差還比成功區低）⇒ **盲區是原因不是結果**。
    數學上：DC 擬合完後殘差是零均值高頻振盪，而 `d(alpha)/d(mu)` 在 footprint 上是
    平滑奇函數 ⇒ 兩者求和正負抵消 ⇒ `|g_mu| > tau` 永不觸發。

    ⇒ **繞過被抵消的梯度通道，直接用高頻殘差當觸發訊號。**
    ⚠ 這與已否證的十次密度控制介入**不同類**：那些調的是「往哪裡分配」，
      而**訊號本身是死的**；本旗標換的是**觸發訊號**。
    ⚠ 動手前先量天花板（`tools/grad_blindspot.py` 會印）——這是 `absgrad` 成功的關鍵步驟。
    """

    ac_shrink: float = 0.0
    """★ 對高 AC 殘差的粒子**直接縮小尺度**（0=關；0.5 = 尺度乘 0.5）。

    **為什麼需要這個、而 `ac_densify` 不夠**（§11.82 實測）：
    `ac_densify` 走 MCMC 的 `probs` 取樣，實測**幾乎沒有搬動族群**
    （失敗區的高斯/像素 0.62 -> 0.64，+3%；半徑與 opacity 完全不變）
    ⇒ 那個實驗檢驗的是「太弱的介入」，不是機制本身。
    原因可算：AC 天花板只有 2.83x，w=1 時最大加權 3.83x，而每次事件搬動的粒子本就少。

    **本旗標繞過取樣瓶頸，直接作用在成因上**：梯度抵消的程度取決於
    **footprint 相對於殘差空間頻率的大小** —— 把 footprint 縮小就能脫離盲區。
    這是 Gemini 原提案的另一半（我先前只實作了「往那裡增生」）。

    ⚠ 與 `err_unlock_frac` 同構（那個壓 opacity，這個壓 scale），都是**直接改參數**。
    ⚠ 風險：MCMC 的 Eq.9 假設 relocation 的子代共位；本旗標不動位置只動尺度，
      不破壞該假設，但會讓被縮小的粒子暫時解釋不了原本的區域（由最佳化補回）。
    """

    ac_shrink_frac: float = 0.05
    """每次 densify 事件，對 AC 分數最高的這個比例做縮小。"""

    ac_shrink_target_px: float = 10.0
    """⛔ **本機制已退役（2026-09-04，研究總覽 §11.91），預設 `ac_shrink=0` 不會觸發。**
    O0 oracle 把失敗 tile 從 corr 0.273 拉到 0.850，過程中**粒子半徑比值 1.019**
    （沒變小，反而微幅變大）、opacity 比值 1.001 ⇒ **解不在 scale 上**。
    程式碼保留是因為冪等目標的寫法本身是對的（§11.85），但**不要再拿它做實驗**。

    以下為原始說明。縮到的**目標投影半徑**（像素）。`scale = target_px * z / (3*fx)`，且只縮不放。

    ⚠ 這取代了「反覆乘 0.5」——後者在兩個版本上都塌陷（見實作處的註解）。
    本式**冪等**：套用兩次與一次相同 ⇒ 不累積、不需要下限。

    ★ 10.0 是**實測**出來的，不是猜的（`tools/radius_cliff.py`，研究總覽 §11.88）：
    ```
       半徑 px        corr 中位
       0.00~ 9.30      0.980
       9.30~12.25      0.980
      12.25~15.87      0.732   <- 懸崖（最大跌幅 = 全距的 49.5%，均勻只會是 14%）
    ```
    ⛔ **我最初設的是 2.0，那是憑空的**：失敗/成功區的半徑實測只差 1.18x
    （20.73 vs 17.51 px），縮到 2px 等於對一個差 18% 的變數做 10x 介入
    ＝破壞集合（§11.34 實測 −2.34 dB）。10.0 落在 corr 仍為 0.980 的箱內且留有餘裕。"""

    harvest_relocate: bool = False
    """★ 收割期回收（使用者 2026-09-01 提案）：`densify_until_iter` 之後**繼續 relocation
    但不成長**（只搬死粒子到 `probs` 抽中的宿主，不呼叫 `add_new_gs`）⇒ N 不變、不動 cap/VRAM。

    動機（§11.61/§11.62）：收割期的 opacity 兩極化持續製造死粒子（判死 3.77% -> 15.03%），
    而回收路徑在 `:487` early-return ⇒ 最終 2.34M 顆裡約 351,500 顆躺著不做事。

    ⚠ **與已知淨負的 churn 是同一個操作**（`notrim2` 的 27,700 步純 churn，§11.26）。
    差別在**送去哪裡**：那時 `probs = opacity`（盲目），現在是 `o*(1+w|g|/mean|g|)`
    ⇒ 送到「梯度說缺細節的地方」，而那是唯一跨塊驗證過有效的機制（w=2 為峰值）。
    ⚠ 成本在**目的地**：`relocate_gs` 會 reset 宿主的 Adam 動量（MCMC 論文 §3.4），
    而收割期的宿主正是在收斂最終外觀的那些 —— 這就是 churn 淨負的具體機制。
    ⚠ 強度小：死亡累積 0.375pp/1000 步 x `densification_interval` 150
    ⇒ 每次事件僅擾動約 **0.056%** 的族群（notrim2 的 churn 遠比這密集）。
    """

    cost_budget: float = 0.0
    """★ 成本預算（§11.60）。>0 時把族群的生長條件從**顆數** `N <= cap_max`
    換成**渲染成本** `max_view Σ_i (2r_i/16)^2 <= cost_budget`。

    依據（`agd2_b12` 軌跡實測，五個 ckpt）：`B/N` 在 7.21~15.98 之間變動（2.2 倍）；
    **15k -> 30k 顆數 +98% 而成本只 +9.4%** ⇒ 模型在 step 15k 就付掉最終渲染成本的 91%，
    卻只用了最終顆數的 56% ⇒ `cap_max` 對「貴的前半」與「幾乎免費的後半」收一樣的價。

    ⚠ **`cap_max` 仍然生效，且必須保留**：VRAM = 逐顆儲存（M*F*4 bytes/顆，只看 N）
    ＋ binning（gamma*tau，只看成本）。本旗標只約束後者，前者仍需顆數上限防 OOM。
    ⇒ 實驗時把 `cap_max` 設高當安全閥，讓 `cost_budget` 成為實際綁住的約束。

    ⚠ 單位與 `tools/cost_budget_probe.py` 的 B **同定義但不同實作路徑**（光柵器 radii
    vs 解析投影）⇒ **不可直接沿用探針的 23.1M**，先用 `cost_budget_report` 短跑標定。
    """

    exact_tile_cost: bool = False
    """★ 2026-09-21：用**光柵器回傳的逐顆 tile 數**當成本，取代 `radii^2` / `Σ(2r/16)^2` 代理。

    代理有三重誤差：假設外接盒是**正方形**、忽略 **tile 量化**（半徑 3px 的顆粒算出 0.14 個
    tile，實際碰 1~4 個）、忽略**螢幕裁切**（`getRect` 會夾到格線範圍，公式不會）。
    本機實測（b6 @21,920，N=1.85M，30 台相機）：
        精確 Σtiles 158,437,005 vs 代理 212,373,630 = **1.340 倍**
        逐視角比值 中位 1.319、**最小 0.852、最大 3.356**、標準差 0.4605
    ⇒ 重點不是「高估 34%」（常數偏差可以靠重新標定預算吸收），而是它**隨視角跳 0.85~3.36 倍**。
      而 `cost_budget` 的閘門正是取**最壞視角**觸發的 => 誤差最大的視角主導了閘門。
      這與「訓練中打印與離線工具差 1.5 倍、跨路徑不可比」是同一件事。

    ⚠⚠ **單位會變**：`cost_budget` 先前標定出來的數值是**代理單位**的，開了這個旗標之後
      同一個數字代表不同的量 => **必須重新標定**（開啟時會印出兩種單位的當期值供換算）。
    ⚠ 需要光柵器回傳 `tiles`（2026-09-21 起）。拿不到會**印警告並退回代理**，不會靜默。
    ⚠ 成本訊號與外接盒耦合：`tiles` 的值取決於 `EXACT_CONIC_AABB` 開不開。"""

    cost_budget_report: int = 0
    """>0 時每 N 步印出線上量到的 `Load`（不改變任何行為），用來標定 `cost_budget`。"""

    churn_report: bool = False
    """（2026-09-14，純打印、不改任何行為）每個 densify 事件印一行 `[churn]`：
    事件前顆數、被 relocate 的 dead 數、新增數、事件後顆數。
    用途：驗「前期寬鬆的預算會不會造成中期 churn」之前，先要量得到 churn。"""

    cost_budget_stat: str = "max"
    """（2026-09-14）閘門比較的統計量：`max`＝區間最壞視角（原行為）／`mean`＝區間平均。
    b6 基準實測 最大/平均 中位 1.72、最高 11.2 倍 => 最大值被少數視角主宰，而**平均與離線工具對得上**。"""

    cost_budget_ref_csv: str = ""
    """（2026-09-14）相對預算：`B(t) = rho(t) x Load_ref(t)`，Load_ref 取自參考跑次（`refrep` 臂）的
    `step,load_mean` CSV（例：`logs/refrep_b6_traj.csv`）。設了它就一律用**平均**判斷，`cost_budget` 常數被忽略。"""

    cost_budget_ratio_start: float = 1.0
    """相對預算在增生期起點（densify_from_iter）的 rho。"""

    cost_budget_ratio_end: float = 1.0
    """相對預算在增生期終點（densify_until_iter）的 rho；中間線性內插。
    start > end＝前鬆後緊（使用者假設一）；start < end＝前緊後鬆（假設二）；相等＝固定比例。"""
    """>0 時按渲染成本折扣增生取樣權重：`probs /= (c/median(c))^w`，`c` = 區間內最大螢幕半徑^2。

    論文命題的直接實作：**既有方法用顆數計價，我方用渲染成本計價**。MCMC 分裂會讓子代
    繼承父代的尺度 ⇒ 抽到大足跡的父代就是在製造大足跡的子代。折扣它 = 讓新增的質量
    往便宜的地方去。前提量測（§11.28）：`rho(v,c)=+0.127` ⇒ v 與 c 幾乎不相關
    ⇒ **選擇有自由度**（退化定理的前提在當前資料上不成立）。

    ⚠ 與 `absgrad_densify` 可獨立開關；同時開就是 `probs ~ o * (1+w*g) / c^w'`
      ＝「每單位渲染成本的預期誤差下降」的貪婪規則。**先各自單獨測，別一次開兩個。**"""

    absgrad_report: int = 0
    """>0 時，每這麼多步報告一次 AbsGS 絕對值位置梯度的統計（**只讀，不改變任何行為**）。

    需要光柵器以 `ABSGRAD 1` 編譯（`cuda_rasterizer/auxiliary.h`）：它把
    `|dL_ds.x| + |dL_ds.y|` 累加進 `dL_dmean2D.z`，該分量從未進入梯度鏈
    ⇒ 渲染與梯度逐位元不變。Python 端在 `viewspace_points.grad[:, 2]`。

    為什麼要先報告而不是直接拿來用（2026-08-25）：AbsGS（arXiv 2404.10484 §3.2）證明
    有符號的位置梯度會因梯度碰撞而低估過度重建的大粒子，但**那是在 3DGS 上證的**。
    在動用它之前要先回答一個決策相關的問題：**這個訊號排出來的名次，跟 MCMC 現在用的
    `probs = opacity` 有沒有實質差別？** 若前 10% 幾乎重疊，整條線就沒有意義，
    不管碰撞是否存在。所以報告的是 Spearman rho 與 top-k 重疊率，不是碰撞比。

    ⚠ 2DGS 的 viewspace 梯度 (`.x`/`.y`) **不是** 3DGS 的對應物：`dL_dmean2D` 只在
    `rho3d > rho2d`（次像素 fallback）分支被寫，解析良好的粒子恆為 0。
    天真移植 3DGS 的梯度式 densify 到 2DGS 只會看到次像素 splat。"""

    freeze_opacity_after_densify: bool = False
    """RTG-style harvest-phase opacity freeze: once global_step >= densify_until_iter,
    zero out opacity grads every step (Adam skips None-grad params entirely). Rationale:
    after densification stops, relocation is off so MCMC no longer needs the opacity
    death-channel, yet the metric's opacity L1 keeps shrinking opacities (arm A lost
    100k points to the end trim, 1.0M -> 0.90M). This tests whether that pressure is
    pure attrition or a needed regularizer during the harvest/settle phase."""

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        assert self.cap_max > 0, "cap_max must > 0"
        return MCMC2DGSDensityControllerImpl(self)


class MCMC2DGSDensityControllerImpl(MCMCDensityControllerImpl):
    config: MCMC2DGSDensityController

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        # Call the grandparent directly to bypass MCMCDensityControllerImpl.setup, which
        # (a) only registers the noise hook for from-scratch init and (b) clobbers
        # opacities/scales (destroying the alpha=0.99 depth-init prior).
        DensityControllerImpl.setup(self, stage, pl_module)
        # Keep a NON-tracked handle for the value-per-cost multi-view pass. A plain
        # attribute would make nn.Module register pl_module as a child of this
        # controller (which is itself pl_module's child) -> infinite recursion in
        # module.apply (RecursionError at .to(device), hit 2026-07-25). object.__setattr__
        # bypasses nn.Module.__setattr__, and a weakref avoids the reference cycle.
        import weakref
        object.__setattr__(self, "_pl_ref", weakref.ref(pl_module))

        N_max = self.config.N_max
        binoms = torch.zeros((N_max, N_max), dtype=torch.float, device=pl_module.device)
        for n in range(N_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        self.register_buffer("binoms", binoms, persistent=False)

        if stage == "fit":
            # Only do MCMC's opacity/scale reset when there is NO prior to preserve.
            if pl_module.hparams["initialize_from"] is None:
                self._opacities_and_scales_initialization(pl_module.gaussian_model)
            # Register the xyz-noise hook regardless of initialize_from.
            pl_module.on_train_batch_end_hooks.append(self._add_xyz_noise)
            if self.config.dar_lambda > 0:
                # runs after optimizer.step() -> Adam moments stay photometric-only
                pl_module.on_train_batch_end_hooks.append(self._dar_cost_decay)

    def after_backward(self, outputs: dict, batch, gaussian_model, optimizers, global_step: int, pl_module: LightningModule) -> None:
        # Same event gating as the parent, plus optional screen-size recycling (see the
        # config docstring). The max-radii buffer is a plain tensor that self-heals on any
        # topology change (trim/relocate/add all shift N), like the gg controller's grad buffer.
        # ⚠ LATENT NO-OP FIXED 2026-08-29: `cost_aware_densify` reads `_max_radii2D` but was
        # NOT in this guard list, so it was a **silent no-op** unless one of the other four
        # flags happened to be on. `cad_b12` only measured anything because its script also
        # passed `screen_size_prune_px 300` — by luck, not by design. Same shape as
        # `err_guided_densify`, which was dead code and burned 9.7 h on `egd_b12`.
        # ⛔ 2026-09-05 更正：上面那句「cad_b12 only measured anything because its script also
        #    passed screen_size_prune_px 300」**是錯的，方向相反**。傳 `screen_size_prune_px`
        #    會讓該機制在 `add_new_gs` 之前把視窗清成 None ⇒ cost_aware 反而**永遠不會觸發**。
        #    ⇒ `cad_b12` 與 `fpd_b12` 都沒量到東西。順序 bug 已修（見 ~685 與 ~716）。
        if self.config.screen_size_prune_px > 0 or self.config.dar_lambda > 0 \
                or self.config.max_relocate_frac > 0 or self.config.vpc_prune_frac > 0 \
                or self.config.cost_aware_densify != 0:
            self._update_max_radii(outputs, gaussian_model)   # cost signal for DAR / gate / v-p-c
        # 成本預算（§11.60）：B 與 N 動態解耦（B/N 在軌跡上變動 2.2 倍；15k->30k 顆數 +98%
        # 而成本只 +9.4%）⇒ `cap_max` 按顆數計價是錯的貨幣。這裡量的是正確的貨幣。
        # ⚠ 守衛必須含 `cost_budget_ref_csv` 自己（本檔踩過「守衛不含自己 => 靜默 no-op」）
        if self.config.cost_budget > 0 or self.config.cost_budget_report > 0 or self.config.cost_budget_ref_csv:
            # 閘門的平均視窗＝最近一個 densify 間隔（與報告同寬）；不這樣做，第一個事件會平均到 step 0 以來的全部歷史
            if global_step % self.config.densification_interval == 1:
                self._gate_sum, self._gate_cnt = 0.0, 0
            self._update_load(outputs)
            _cbr = self.config.cost_budget_report
            if _cbr > 0 and global_step % _cbr == 0:
                _n = gaussian_model.n_gaussians
                _l = getattr(self, "_load_max", 0.0)
                print(f"[cost-budget] step {global_step}: N={_n:,} "
                      f"Load(區間最壞視角)={_l:,.0f} Load/N={_l / max(_n, 1):.2f} "
                      f"預算={self.config.cost_budget:,.0f}"
                      f"{'（未設，純標定）' if self.config.cost_budget <= 0 else ''}"
                      f" ｜報告區間 平均={getattr(self, '_rep_sum', 0.0) / max(getattr(self, '_rep_cnt', 0), 1):,.0f}"
                      f" 最壞={getattr(self, '_rep_max', 0.0):,.0f}（{getattr(self, '_rep_cnt', 0)} 視角）")
                # 報告自己的視窗（2026-09-14）：與 add_new_gs 閘門用的 `_load_max` 分開，互不干擾。
                #   ⚠ 原本的「Load(區間最壞視角)」在沒設預算時從不歸零 => 其實是從頭累積的最大值。
                self._rep_sum, self._rep_cnt, self._rep_max = 0.0, 0, 0.0
        # ★ 2026-09-05 閘門：`_accumulate_error_score` 是**唯一沒有守衛、每步都跑**的助手
        #   （全幅 avg_pool2d + N x 3 的投影矩陣乘）。它的消費者：
        #     err_unlock_frac / err_guided_densify / transparent_corrector  —— 現行配方全是 0
        #     `_report_densify_blindness`（只印字，不影響訓練）
        #   ⇒ 現行配方下它是純診斷成本。守衛含**所有**消費者（稽核清單 §7 的教訓）。
        if (self.config.err_unlock_frac > 0 or self.config.err_guided_densify > 0
                or self.config.transparent_corrector > 0 or self.config.densify_blind_report):
            self._accumulate_error_score(outputs, batch, gaussian_model)
        # ⚠ 守衛必須含 `ac_densify` 自己（`cost_aware_densify` 曾因守衛不含自己而靜默 no-op）
        # ⚠ 守衛必須含**所有**消費者（ac_shrink 也讀 _ac_score）——
        #   cost_aware_densify 就是因為守衛不含自己而靜默 no-op（稽核清單 §7）
        if self.config.ac_densify > 0 or self.config.ac_shrink > 0:
            self._accumulate_ac_score(outputs, batch, gaussian_model)
        if self.config.absgrad_report > 0 or self.config.absgrad_densify > 0:
            # densify 用途下也要累積 `|g|`（原本只有 report 會觸發累積）
            self._absgrad_report(outputs, gaussian_model, global_step)
        if self.config.vpc_report > 0:
            self._vpc_report(gaussian_model, global_step)

        # Option-4 emergency path: per-step monster recycle, decoupled from the
        # densify-event schedule (closes the temporal gap that OOM'd b12@2M K=2).
        # LATENT BUG FIXED 2026-07-28 (never ran, so no past result is affected): this path is
        # PER-STEP, but `_max_radii2D` is a window that gets cleared at every densify event, so
        # for ~150 steps after each event most points read 0 and the emergency prune would
        # silently under-detect exactly when a monster is most likely to be forming. Every
        # other consumer of the window runs AT densify events (where it holds a full interval),
        # which is why only this path and DAR were affected. Use the analytic projection.
        if self.config.screen_prune_emergency_px > 0:
            from internal.utils.strip_cameras import projected_radius
            self._max_radii2D = projected_radius(gaussian_model.get_xyz.detach(),
                                                 gaussian_model.get_scales().detach(), batch[0])
            hot = self._max_radii2D > self.config.screen_prune_emergency_px
            n_hot = int(hot.sum())
            if n_hot > 0:
                print(f"[screen-prune EMERGENCY] step {global_step}: recycling {n_hot} runaway "
                      f"Gaussians (radius > {self.config.screen_prune_emergency_px}px, "
                      f"max {int(self._max_radii2D.max())}px)")
                with torch.no_grad():
                    if global_step < self.config.densify_until_iter:
                        self.relocate_gs(gaussian_model, optimizers, hot)  # recycle into supply
                    else:
                        self._prune_points(hot, gaussian_model, optimizers)  # harvest: just remove
                self._max_radii2D = None
                self._max_tiles2D = None

        if self.config.freeze_opacity_after_densify and global_step >= self.config.densify_until_iter:
            gaussian_model.gaussians["opacities"].grad = None

        if global_step >= self.config.densify_until_iter:
            # ★ 收割期回收（使用者 2026-09-01 提案）：densify 停了，但 opacity 兩極化持續
            # 製造死粒子（判死 3.77% -> 15.03%，§11.62），而回收路徑在這裡 early-return
            # ⇒ 那些粒子躺到結束，什麼也不做。本旗標讓 relocation 繼續，**但不成長**
            # （只呼叫 relocate，不呼叫 add_new_gs ⇒ N 不變 ⇒ 不動 cap/VRAM）。
            #
            # 為什麼值得試（與已知淨負的 churn 的差別）：
            #   ⚠ notrim2 的 27,700 步純 churn 實測淨負（§11.26），**是同一個操作**。
            #   但那時 `probs = opacity`（盲目）；現在 `probs = o*(1+w|g|/mean|g|)`
            #   ⇒ 送去的地方變成「梯度說缺細節的地方」，而那是唯一驗證過有效的機制。
            #   強度也小：死亡累積 0.375pp/1000 步 x 間隔 150 = 每次約 0.056% 的族群。
            # ⚠ 成本在**目的地**：relocate 會 reset 宿主的 Adam 動量（MCMC 論文 §3.4），
            #   而收割期的宿主正是在收斂外觀的那些 ⇒ 這就是 churn 淨負的機制。
            # ⚠ 門檻用 `min_opacity` 即可：實測 `o<0.005` 與 `o<=1/255` 選到同一批
            #   （15.03% vs 14.96%，比 0.995，§11.61）⇒ 再收緊門檻沒有意義。
            if self.config.harvest_relocate and global_step % self.config.densification_interval == 0:
                with torch.no_grad():
                    _dead = (gaussian_model.get_opacities()
                             <= self._current_min_opacity(global_step)).squeeze(-1)
                    _n_dead = int(_dead.sum())
                    if _n_dead > 0:
                        self.relocate_gs(gaussian_model, optimizers, _dead)
                        if global_step % 3000 == 0:
                            print(f"[harvest-relocate] step {global_step}: 回收 {_n_dead} 顆"
                                  f"（{_n_dead / max(gaussian_model.n_gaussians, 1):.3%} 的族群）")
            if self.config.harvest_dust_trim_interval > 0 \
                    and global_step % self.config.harvest_dust_trim_interval == 0:
                with torch.no_grad():
                    o = gaussian_model.get_opacities().squeeze(-1)
                    sc = gaussian_model.get_scales()
                    dust = (o < self.config.harvest_dust_opacity) \
                        & (sc.max(dim=-1).values < self.config.harvest_dust_px)
                    n_dust = int(dust.sum())
                    if n_dust > 0:
                        self._prune_points(dust, gaussian_model, optimizers)
                        print(f"[harvest-dust] step {global_step}: pruned {n_dust} dust points -> N={gaussian_model.n_gaussians}")
            return
        if global_step <= self.config.densify_from_iter:
            return
        if global_step % self.config.densification_interval != 0:
            return

        with torch.no_grad():
            # ★ 路徑 a：Elongation Filter 硬剪（CityGaussian V2 的做法）。
            #   必須在建 dead_mask **之前**做 —— `_prune_points` 會改變 N，
            #   而後面所有 mask 都是按當下的 N 建的（這是本檔踩過的形狀：
            #   `_max_radii2D` 與 dead_mask 長度不一致就靜默跳過）。
            if self.config.elongation_prune > 0:
                _sc = gaussian_model.get_scales()
                _el = _sc.max(dim=-1).values / _sc.min(dim=-1).values.clamp_min(1e-12)
                _em = _el > self.config.elongation_prune
                _n = int(_em.sum())
                if not getattr(self, "_elong_reported", False):
                    self._elong_reported = True
                    print(f"[elongation-prune] ✅ 首次觸發 step {global_step}："
                          f"門檻 {self.config.elongation_prune}，命中 {_n:,} 顆 "
                          f"({100*_n/max(_el.numel(),1):.2f}%)，長寬比中位 "
                          f"{float(_el.median()):.2f}／p99 {float(_el.quantile(0.99)):.1f}", flush=True)
                if _n > 0:
                    self._prune_points(_em, gaussian_model, optimizers)
                    if global_step % 3000 == 0:
                        print(f"[elongation-prune] step {global_step}: -{_n:,} => "
                              f"N={gaussian_model.n_gaussians:,}", flush=True)
                if self._max_radii2D is not None:
                    self._max_radii2D = None
                    self._max_tiles2D = None      # N 變了，舊視窗長度對不上
            death_line = self._current_min_opacity(global_step)
            dead_mask = (gaussian_model.get_opacities() <= death_line).squeeze(-1)
            if self.config.min_opacity_final > 0 and global_step % 1500 == 0:
                print(f"[death-line] step {global_step}: min_opacity {death_line:.4f}, dead {int(dead_mask.sum())}")
            # value-per-cost channel: render-grounded value (multi-view alpha-blending
            # contribution) over render-grounded cost (screen tile footprint). Catches
            # what the opacity death-line cannot: the near-camera monster has o=0.028
            # (not low enough to die) but a frame-filling footprint -> v/c collapses.
            # ORDER MATTERS: this must run BEFORE screen_size_prune clears _max_radii2D,
            # since both read the same radii window (see the 2026-07-27 review — the old
            # code snapshotted the window instead, and the snapshot went stale on every
            # add_new_gs, so v/c silently never fired).
            vpc_mask = self._value_per_cost_mask(gaussian_model, global_step)
            if vpc_mask is not None:
                n_vpc = int((vpc_mask & ~dead_mask).sum())
                dead_mask = dead_mask | vpc_mask
                if n_vpc > 0:
                    print(f"[value-per-cost] step {global_step}: +{n_vpc} recycled (v/c bottom {self.config.vpc_prune_frac:.1%})")
            # ★ 路徑 b：長寬比併入死亡判準 => 由 MCMC relocate（搬移）而非刪除。
            #   ⚠ 與 `vpc_prune_frac`（-2.34 dB）是同一個機制，先驗不利，見 config docstring。
            if self.config.elongation_relocate > 0:
                _sc = gaussian_model.get_scales()
                _el = _sc.max(dim=-1).values / _sc.min(dim=-1).values.clamp_min(1e-12)
                _em = _el > self.config.elongation_relocate
                _n_new = int((_em & ~dead_mask).sum())
                if _n_new > 0:
                    dead_mask = dead_mask | _em
                if not getattr(self, "_elong_reported", False):
                    self._elong_reported = True
                    print(f"[elongation-relocate] ✅ 首次觸發 step {global_step}："
                          f"門檻 {self.config.elongation_relocate}，命中 {int(_em.sum()):,} 顆"
                          f"（新增 {_n_new:,}，長寬比中位 {float(_el.median()):.2f}"
                          f"／p99 {float(_el.quantile(0.99)):.1f}）", flush=True)
                elif global_step % 3000 == 0:
                    print(f"[elongation-relocate] step {global_step}: +{_n_new:,} 搬移", flush=True)
            if self.config.screen_size_prune_px > 0 and self._max_radii2D is not None \
                    and self._max_radii2D.shape[0] == dead_mask.shape[0]:
                big_mask = self._max_radii2D > self.config.screen_size_prune_px
                n_big = int(big_mask.sum())
                if n_big > 0:
                    dead_mask = dead_mask | big_mask
                    print(f"[screen-size prune] step {global_step}: recycling {n_big} Gaussians with radius > {self.config.screen_size_prune_px}px (max {int(self._max_radii2D.max())}px)")
                # ⚠⚠ 2026-09-05 BUG 修正（第 7 個靜默 no-op，代價 = fpd_b12 的 9.6h）：
                # 這裡原本就把視窗清成 None，但 `add_new_gs`（下面第 ~716 行）才是
                # `cost_aware_densify` 讀 `_max_radii2D` 的地方 ⇒ 只要 `screen_size_prune_px > 0`
                # （現行最佳配方就有，300），cost_aware **永遠讀到 None、整段跳過**。
                # ⇒ `fpd_b12`（cost_aware -0.5）與更早的 `cad_b12` **都沒有量到任何東西**。
                # ⛔ 連帶推翻本檔 ~521 行的註解「cad_b12 only measured anything because its
                #    script also passed screen_size_prune_px 300」—— 方向相反，那正是讓它失效的原因。
                # 清除改到 `add_new_gs` 之後（見該處）。此時 N 尚未變（relocate 就地搬、
                # add_new_gs 先算 probs 再 append），所以 shape 檢查仍成立。
            # Observability: relocate_gs has no cap, and uncapped condemnation costs a churn
            # tax (gsplat-probe experience). Report the BREAKDOWN, not just the total —
            # measured 2026-07-27 that a 44% total is ~38% stock opacity deaths (normal in
            # early densify, A' had the same) + 5% v/c + <1% radius, so a bare total would
            # cry wolf. Only the non-stock channels are ours to tune.
            # Relocation gate: bound the churn volume, spending the budget on the most
            # expensive dead points first (see max_relocate_frac docstring).
            if self.config.max_relocate_frac > 0:
                n_pts = dead_mask.shape[0]
                budget = int(n_pts * self.config.max_relocate_frac)
                n_dead = int(dead_mask.sum())
                if n_dead > budget > 0:
                    dead_idx = dead_mask.nonzero(as_tuple=True)[0]
                    if self._max_radii2D is not None and self._max_radii2D.shape[0] == n_pts:
                        priority = self._max_radii2D[dead_idx].float()       # cost-first
                    else:
                        priority = -gaussian_model.get_opacities().squeeze(-1)[dead_idx]
                    keep = dead_idx[torch.topk(priority, budget).indices]
                    dead_mask = torch.zeros_like(dead_mask)
                    dead_mask[keep] = True
                    if global_step % 1500 == 0:
                        print(f"[reloc-gate] step {global_step}: {n_dead} dead -> relocating {budget} "
                              f"({self.config.max_relocate_frac:.0%} cap, most-expensive first)")
            if global_step % 1500 == 0:
                op_frac = float((gaussian_model.get_opacities().squeeze(-1) <= death_line).float().mean())
                print(f"[dead-mask] step {global_step}: total {float(dead_mask.float().mean()):.1%} "
                      f"= opacity {op_frac:.1%} (stock, pre-gate) + our channels")
            if self.config.densify_blind_report:
                self._report_densify_blindness(gaussian_model, global_step)
            self._err_triggered_unlock(gaussian_model, optimizers, global_step)
            self.relocate_gs(gaussian_model, optimizers, dead_mask)
            # ★ 2026-09-05：被 relocate 的粒子已經換位置換尺度了，但 `_max_radii2D` 還是
            #   **搬走前**的半徑。`add_new_gs`（下一行）的 `cost_aware_densify` 會讀它
            #   ⇒ 會用陳舊的大半徑去加權「剛剛被搬走、現在很小」的粒子。歸零才正確。
            #   （這是原本「screen_size_prune 與 cost_aware 不能同時開」的真正原因；
            #     修掉之後兩者可以並存 —— 而 v2 的 OOM 證明 prune 是承重的、不能拿掉。）
            if self._max_radii2D is not None and self._max_radii2D.shape[0] == dead_mask.shape[0]:
                self._max_radii2D[dead_mask] = 0
        if self._max_tiles2D is not None and self._max_tiles2D.shape[0] == dead_mask.shape[0]:
            self._max_tiles2D[dead_mask] = 0
            _n_before_add = gaussian_model.n_gaussians
            self._cur_step = global_step          # 相對預算的 rho(t) 與 Load_ref(t) 需要目前步數
            self.add_new_gs(gaussian_model, optimizers)
            if getattr(self.config, "churn_report", False):
                _n_after_add = gaussian_model.n_gaussians
                if not getattr(self, "_churn_announced", False):
                    self._churn_announced = True
                    print("[churn] ✅ 首次觸發：每個 densify 事件印一行（純打印，不改行為）", flush=True)
                print(f"[churn] step {global_step}: N={_n_before_add:,} dead->relocate={int(dead_mask.sum()):,} "
                      f"新增={_n_after_add - _n_before_add:,} => N={_n_after_add:,}", flush=True)
            # ★ 視窗在**所有**消費者用完之後才清（2026-09-05 修）。
            #   消費者順序：vpc -> screen_size_prune -> add_new_gs(cost_aware)。
            #   `add_new_gs` 會 append ⇒ 之後 shape 就對不上，所以必須在這裡清。
            self._max_radii2D = None
            self._max_tiles2D = None
            if self._tc_last is not None:
                print(f"[transparent-corrector] step {global_step}: {self._tc_last[0]}/{self._tc_last[1]} "
                      f"個高誤差高不透明父代的子代改為 opacity {self.config.transparent_corrector}")
                self._tc_last = None
            self._err_score = None                 # window restarts with the next interval

    _max_radii2D: torch.Tensor = None
    _max_tiles2D: torch.Tensor = None      # 2026-09-21：精確 binning 成本的同語意視窗
    _vpc_contrib: torch.Tensor = None   # cached multi-view contribution (value)
    _vpc_step: int = -1
    _ac_score: torch.Tensor = None
    _last_camera = None

    def _dar_report(self, global_step: int) -> None:
        """so a silent no-op is distinguishable from a mechanism that never ran: the 1700-step
        smoke printed nothing because %1500==0 landed on a densify-event step, where
        screen-size pruning has just cleared the radii window."""
        if global_step % 500 == 1:
            st = self._dar_stats
            print(f"[DAR-cost] step {global_step}: SKIPPED this step "
                  f"applied={st['applied']} skip(radii/v/lr)={st['no_radii']}/{st['no_v']}/{st['no_lr']}")


    _err_score: torch.Tensor = None
    _tc_last = None      # (n_hit, n_cand) of the last transparent-corrector event

    @torch.no_grad()
    @torch.no_grad()
    def _err_triggered_unlock(self, gaussian_model, optimizers, global_step: int) -> None:
        """把「持續高誤差 + 高不透明」的粒子壓回半透明，讓它們身後的幾何重新收到梯度。

        RTG-SLAM B4（stable->unstable reversion）的想法在 MCMC 上的對應物。見設定項
        `err_unlock_frac` 的 docstring 與 研究總覽 §11.7。
        """
        # ★ AC 導向的強制縮小（§11.82）：繞過取樣瓶頸，直接壓 footprint
        _sh = self.config.ac_shrink
        if _sh > 0 and self._ac_score is not None:
            a = self._ac_score
            if a.shape[0] == gaussian_model.n_gaussians and float(a.sum()) > 0:
                k = max(1, int(round(self.config.ac_shrink_frac * a.numel())))
                thr = torch.topk(a, k).values[-1]
                hit = a >= thr
                n_hit = int(hit.sum())
                # ⚠⚠ 2026-09-04 第二次修：**「反覆乘 0.5 + 下限」在結構上就是錯的**。
                # 第一版無下限：step 14999 時 15.19% 的粒子 scale < 1e-4。
                # 第二版下限 = median/50：**下限本身失控下滑**
                #   step 3000 -> 18000：3.64e-4 / 3.75e-4 / 3.06e-4 / 2.61e-4 / 2.14e-4 / 1.78e-4
                #   縮小的顆數同時 6,753 -> 41,660  ⇒ 13.96% 仍 < 1e-4。
                #   成因：下限綁在 median 上，而**縮小本身會壓低 median** ⇒ 正回饋。
                #   ⇒ **不可把安全限綁在這個機制自己會破壞的量上。**
                #
                # ✅ 第三版：**縮到絕對目標，不做乘法累積**。
                # 目標由機制本身推出：梯度抵消發生在「footprint 跨越多個像素、
                # 而殘差在其上正負相消」⇒ 要脫離盲區就是讓**投影半徑降到約 target_px**。
                # 於是 scale = target_px * z / (3 * fx)（3-sigma 半徑的反解），且**只縮不放**。
                # 這個運算是**冪等**的：套用兩次與一次相同 ⇒ 不會累積、不需要下限。
                cam0 = getattr(self, "_last_camera", None)
                if cam0 is not None:
                    with torch.no_grad():
                        pc = gaussian_model.get_xyz @ cam0.R.T + cam0.T
                        z = pc[:, 2].clamp_min(0.2)
                        tgt = self.config.ac_shrink_target_px * z / (3.0 * float(cam0.fx))
                        cur = gaussian_model.get_scales().max(dim=1).values
                        # 只對「命中且目前比目標大」的做，且直接設到目標（非乘法）
                        hit = hit & (cur > tgt)
                        n_hit = int(hit.sum())
                        if n_hit > 0:
                            ratio = (tgt[hit] / cur[hit]).clamp(max=1.0)
                            gaussian_model.scales.data[hit] += torch.log(ratio).unsqueeze(-1)
                            if global_step % 3000 == 0:
                                print(f"[ac-shrink] step {global_step}: {n_hit} 顆縮到 "
                                      f"{self.config.ac_shrink_target_px:.1f}px（AC 前 "
                                      f"{100 * self.config.ac_shrink_frac:.0f}%，冪等，不累積）")

        f = self.config.err_unlock_frac
        e = self._err_score
        if f <= 0 or e is None or e.numel() == 0:
            return
        o = gaussian_model.get_opacities().squeeze(-1)
        if o.shape[0] != e.shape[0]:
            return                                   # 拓樸剛變過，這個窗口不可用
        cand = o > self.config.err_unlock_min_opacity
        n_cand = int(cand.sum())
        if n_cand == 0:
            return
        k = max(1, int(round(f * n_cand)))
        # 只在候選之中挑誤差最高的 k 顆（不是全族群 —— 低 opacity 的沒擋住任何東西）
        thr = torch.topk(e[cand], k).values[-1]
        hit = cand & (e >= thr)
        n_hit = int(hit.sum())
        if n_hit == 0:
            return
        target = torch.logit(torch.tensor(
            self.config.err_unlock_to, device=o.device, dtype=torch.float32))
        raw = gaussian_model.get_property("opacities")
        raw[hit] = target.to(raw.dtype)
        # ⚠ 只改值不重置 Adam 狀態，動量會在幾步內把 opacity 推回去（機制會靜默失效）。
        # MCMC 自己的 relocation 也是這樣處理的（mcmc_density_controller.py:298）。
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=hit)
        print(f"[err-unlock] step {global_step}: {n_hit} 顆高誤差高不透明粒子壓回 "
              f"opacity {self.config.err_unlock_to} (候選 {n_cand}, 取前 {f:.0%})")

    def _accumulate_error_score(self, outputs, batch, gaussian_model) -> None:
        """Per-primitive tally of unexplained error at its screen position; see the
        err_guided_densify docstring. Cheap: one avg_pool over the frame plus an O(N) gather."""
        try:
            render = outputs.get("render", None)
            camera, image_info, _ = batch
            gt = image_info[1]
            if render is None or gt is None:
                return
            err = (render.detach() - gt).abs().mean(0, keepdim=True)[None]      # [1,1,H,W]
            tile = torch.nn.functional.avg_pool2d(err, 16, ceil_mode=True)[0, 0]  # [gy,gx]
            gy, gx = tile.shape
            means = gaussian_model.get_xyz.detach()
            pc = means @ camera.R.T + camera.T
            z = pc[:, 2].clamp_min(0.2)
            W, H = int(camera.width), int(camera.height)
            u = (float(camera.fx) * pc[:, 0] / z + W / 2).div(16).long().clamp(0, gx - 1)
            v = (float(camera.fy) * pc[:, 1] / z + H / 2).div(16).long().clamp(0, gy - 1)
            e = tile[v, u]
            e = torch.where(pc[:, 2] > 0.2, e, torch.zeros_like(e))             # behind camera: no claim
            n = gaussian_model.n_gaussians
            if self._err_score is None or self._err_score.shape[0] != n or self._err_score.device != e.device:
                self._err_score = torch.zeros_like(e)                           # self-heal on topology change
            self._err_score += e
        except Exception:
            pass                                    # a diagnostic must never take the run down

    @torch.no_grad()
    def _accumulate_ac_score(self, outputs, batch, gaussian_model) -> None:
        """逐顆累積「投影位置所在 tile 的**殘差高頻能量**」。

        與 `_accumulate_error_score` 的唯一差別是訊號的定義：
            DC（舊）  avg_pool(|render - gt|)          <- 整片偏移時大；天花板 1.22x
            AC（本）  std(render - gt) within tile     <- **殘差振盪**時大，即結構缺失
        用 `std = sqrt(E[e^2] - E[e]^2)` 兩次 avg_pool 求得，成本與舊版同量級。
        """
        try:
            render = outputs.get("render", None)
            camera, image_info, _ = batch
            gt = image_info[1]
            if render is None or gt is None:
                return
            self._last_camera = camera        # ac_shrink 需要它把目標像素半徑反解成 scale
            e = (render.detach() - gt).mean(0, keepdim=True)[None]        # 有號、灰階 [1,1,H,W]
            m1 = torch.nn.functional.avg_pool2d(e, 16, ceil_mode=True)
            m2 = torch.nn.functional.avg_pool2d(e * e, 16, ceil_mode=True)
            tile = (m2 - m1 * m1).clamp_min(0).sqrt()[0, 0]               # 逐 tile 的 AC 能量
            gy, gx = tile.shape
            pc = gaussian_model.get_xyz.detach() @ camera.R.T + camera.T
            z = pc[:, 2].clamp_min(0.2)
            W, H = int(camera.width), int(camera.height)
            u = (float(camera.fx) * pc[:, 0] / z + W / 2).div(16).long().clamp(0, gx - 1)
            v = (float(camera.fy) * pc[:, 1] / z + H / 2).div(16).long().clamp(0, gy - 1)
            a = torch.where(pc[:, 2] > 0.2, tile[v, u], torch.zeros_like(z))
            n = gaussian_model.n_gaussians
            if self._ac_score is None or self._ac_score.shape[0] != n \
                    or self._ac_score.device != a.device:
                self._ac_score = torch.zeros_like(a)                      # N 變動時自癒
            self._ac_score += a
        except Exception:
            pass                                    # 診斷/取樣訊號不得讓跑次死掉

    @torch.no_grad()
    def _report_densify_blindness(self, gaussian_model, global_step: int) -> None:
        """How much does opacity-sampling already correlate with where the error is?

        add_new_gs draws parents with probability proportional to opacity, so the error density a
        sampled parent sees is E[e] = sum(o*e)/sum(o). Compare against the plain population mean.
        ~1.0 means opacity is blind to error and steering has something to gain; >1 means opacity
        is already a decent proxy and the expected win is small. Computing the expectation in
        closed form avoids having to intercept the sampler.
        """
        e = self._err_score
        if e is None or e.numel() == 0 or float(e.sum()) <= 0:
            return
        o = gaussian_model.get_opacities().detach().squeeze(-1)
        if o.shape[0] != e.shape[0]:
            return
        base = float(e.mean())
        if base <= 0:
            return
        opa_w = float((o * e).sum() / o.sum().clamp_min(1e-12))
        top = e.topk(max(1, e.numel() // 20)).values.mean()                     # what a perfect sampler would see
        print(f"[densify-blind] step {global_step}: err@opacity-sampled/mean = {opa_w / base:.3f} "
              f"(1.00 = blind)  ceiling(top5%)/mean = {float(top) / base:.2f}  "
              f"err[mean/max]={base:.4f}/{float(e.max()):.4f}")

    @torch.no_grad()
    def _vpc_report(self, gaussian_model, global_step: int) -> None:
        """退化定理的前提檢驗：按 v/c 剪，比按 v 剪多省多少成本？（只讀，不剪枝）

        v = 多視角 Σ(T·α)/覆蓋像素（`_measure_multiview_contribution`，24 視角取樣）
        c = 區間內最大螢幕半徑^2（∝ tile 足跡 ∝ binning 成本）
        兩者都是既有機制在用的訊號，不需要 CUDA 改動。
        """
        if global_step % self.config.vpc_report != 0:
            return
        n = gaussian_model.n_gaussians
        radii = self._max_radii2D
        if radii is None or radii.shape[0] != n or float(radii.max()) <= 0:
            print(f"[vpc] step {global_step}: 沒有 radii 視窗 —— 需要 screen_size_prune_px > 0")
            return
        v = self._measure_multiview_contribution(gaussian_model)
        if v is None or v.shape[0] != n:
            print(f"[vpc] step {global_step}: 取不到 value")
            return
        if self.config.exact_tile_cost and self._max_tiles2D is not None \
                and self._max_tiles2D.shape[0] == n:
            c = self._max_tiles2D.float().clamp_min(1.0)          # 精確：就是 tile 數本身
        else:
            c = radii.float().clamp_min(1.0) ** 2                  # 代理：見 exact_tile_cost 的說明

        def rank(x):
            r = torch.empty_like(x)
            r[x.argsort()] = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
            return r

        # ⚠ 2026-08-26 修正：第一版在全體上比，但**零值粒子超過 10%**（`_measure_multiview_contribution`
        # 只取樣 24 個視角，加上 EXACT_SUPPORT 讓 o<=1/255 完全不進 binning）=> 兩種排序的
        # 最低 10% 都是同一批零值（0/c 還是 0）=> 比較是空的（實測兩邊都是 2.36% / 0.00%）。
        # 現在：零值另外報，所有統計只在 v > 0 的子集上做。
        nz = v > 0
        n_nz = int(nz.sum())
        print(f"[vpc] step {global_step}: N={n} 零值={100*(1-n_nz/max(n,1)):.1f}% "
              f"（零值的 v 與 v/c 排序相同 => 必須排除才有鑑別力）")
        if n_nz < 1000:
            print("      非零粒子太少，無法比較")
            return
        v, c = v[nz], c[nz]
        rv, rc = rank(v), rank(c)
        rho = float(((rv - rv.mean()) * (rc - rc.mean())).sum()
                    / (rv.std(unbiased=False) * rc.std(unbiased=False) * n_nz).clamp_min(1e-12))
        lv, lc = torch.log(v), torch.log(c)
        pear = float(((lv - lv.mean()) * (lc - lc.mean())).mean()
                     / (lv.std(unbiased=False) * lc.std(unbiased=False)).clamp_min(1e-12))
        vpc = v / c
        C, V = c.sum().clamp_min(1e-12), v.sum().clamp_min(1e-12)
        print(f"      非零 {n_nz} 顆：rho(v,c)={rho:+.3f} pearson(log v,log c)={pear:+.3f} "
              f"v/c 跨度={float(vpc.max()/vpc.min()):.2e}   （v=kappa*c 要求 rho≈1、跨度≈1）")
        for frac in (0.10, 0.30):
            k = max(1, int(n_nz * frac))
            m_v = torch.zeros(n_nz, dtype=torch.bool, device=v.device)
            m_v[v.topk(k, largest=False).indices] = True      # trim 現在做的事
            m_p = torch.zeros(n_nz, dtype=torch.bool, device=v.device)
            m_p[vpc.topk(k, largest=False).indices] = True    # 成本感知版
            cs_v, cs_p = float(c[m_v].sum() / C), float(c[m_p].sum() / C)
            vl_v, vl_p = float(v[m_v].sum() / V), float(v[m_p].sum() / V)
            ov = float((m_v & m_p).sum()) / k
            print(f"      剪最低 {100*frac:.0f}%：按 v 省成本 {100*cs_v:5.2f}% 損價值 {100*vl_v:5.2f}% ｜ "
                  f"按 v/c 省成本 {100*cs_p:5.2f}% 損價值 {100*vl_p:5.2f}% ｜ "
                  f"成本 {cs_p/max(cs_v,1e-12):5.2f}x 價值代價 {vl_p/max(vl_v,1e-12):5.2f}x 重疊 {100*ov:.0f}%")
        # ⚠ 2026-08-26：原本寫「成本倍率 >> 價值代價倍率 才算有自由度」，**那是錯的測法** ——
        # 兩種方法落在成本軸的完全不同位置（按 v 剪 10% 只砍 6.75% 成本，按 v/c 砍 49.6%），
        # 比「倍率的倍率」沒有意義。正確的問法是**等成本節省下誰損失的價值少**。
        print(f"      判讀：比較同一個成本節省水準下的價值損失（按 v 要剪更多顆才追得上 v/c 的成本節省）；"
              f"rho(v,c) 接近 1 才是退化")

    @torch.no_grad()
    def _absgrad_report(self, outputs, gaussian_model, global_step: int) -> None:
        """AbsGS 絕對值位置梯度 vs MCMC 現行的 opacity —— 排出來的名次有沒有實質差別？

        只讀，不改變任何行為。需要光柵器以 `ABSGRAD 1` 編譯（見 `absgrad_report` 的說明）。

        決策相關的問題不是「碰撞存不存在」而是「換了訊號會不會換掉增生的位置」：
        若 top-10% 幾乎重疊、rho 接近 1，這條線就沒有意義，不必再花 GPU。
        另外報 |g| 與螢幕足跡的相關 —— AbsGS 的主張是被它抓到的正是**大**粒子。
        """
        vp = outputs.get("viewspace_points")
        g = getattr(vp, "grad", None) if vp is not None else None
        if g is None or g.shape[-1] < 3:
            return
        a = g[:, 2].detach().abs()
        buf = getattr(self, "_absgrad_accum", None)
        if buf is None or buf.shape[0] != a.shape[0]:
            buf = torch.zeros_like(a)                 # N 變動時自癒（trim/relocate/add 都會變）
        self._absgrad_accum = buf + a
        # ⚠ 2026-08-27：`absgrad_densify > 0` 但 `absgrad_report == 0` 時，
        # 原本的 `global_step % 0` 直接 ZeroDivisionError（`agd_b12` 連死兩次）。
        # 累積發生在上一行、在這個檢查**之前** ⇒ 只要守住取模，densify 用途照常拿得到 `|g|`。
        if self.config.absgrad_report <= 0 or global_step % self.config.absgrad_report != 0:
            return
        e = self._absgrad_accum
        if float(e.sum()) <= 0:
            print(f"[absgrad] step {global_step}: 全為 0 —— 光柵器沒有以 ABSGRAD=1 重編，"
                  f"或 .so 沒有真的換掉（確認 mtime）")
            self._absgrad_accum = None
            return
        o = gaussian_model.get_opacities().detach().squeeze(-1)
        if o.shape[0] != e.shape[0]:
            self._absgrad_accum = None
            return

        def rank(v):
            r = torch.empty_like(v)
            r[v.argsort()] = torch.arange(v.numel(), dtype=v.dtype, device=v.device)
            return r

        ra, ro = rank(e), rank(o)
        rho = float(((ra - ra.mean()) * (ro - ro.mean())).sum()
                    / (ra.std(unbiased=False) * ro.std(unbiased=False) * ra.numel()).clamp_min(1e-12))
        k = max(1, e.numel() // 10)
        overlap = len(set(e.topk(k).indices.tolist()) & set(o.topk(k).indices.tolist())) / k
        zero = float((e <= 0).float().mean())
        # ★ 天花板（2026-08-26，`egd_b12` 的教訓）：`_report_densify_blindness` 量到
        # 我方誤差分數的 `ceiling(top5%)/mean` 只有 **1.22** —— 完美取樣器也只能看到
        # 1.22 倍的平均誤差 ⇒ 「往誤差高處增生」本來就沒什麼可賺，實測也確實淨負
        # （egd_b12 26.271 vs sched30 26.377）。但那是**我方誤差分數**的天花板：它把
        # |render-gt| 池化到 16px tile、在投影中心取值、**不乘足跡** ⇒ 分數天生就平。
        # 這裡對 |g| 算同一個量：若也 ~1.2，整個「誤差導向增生」家族就死了；
        # 若明顯更高，代表 1.22 只是我方訊號做得爛，AbsGS 那條線還活著。
        mean_e = e.mean().clamp_min(1e-30)
        k5 = max(1, e.numel() // 20)
        ceil5 = float(e.topk(k5).values.mean() / mean_e)
        opa_w = float((o * e).sum() / o.sum().clamp_min(1e-12) / mean_e)
        line = (f"[absgrad] step {global_step}: N={e.numel()} rho(|g|, opacity)={rho:+.3f} "
                f"top10%重疊={100*overlap:.1f}% 零值={100*zero:.1f}% "
                f"|g|@opacity取樣/平均={opa_w:.3f}(1.00=盲) "
                f"天花板(top5%)/平均={ceil5:.2f}  <- 誤差分數版只有 1.22 "
                f"|g|[中位/最大]={float(e.median()):.3e}/{float(e.max()):.3e}")
        r2 = getattr(self, "_max_radii2D", None)
        if r2 is not None and r2.shape[0] == e.shape[0] and float(r2.max()) > 0:
            rr = rank(r2.float())
            rho_r = float(((ra - ra.mean()) * (rr - rr.mean())).sum()
                          / (ra.std(unbiased=False) * rr.std(unbiased=False) * ra.numel()).clamp_min(1e-12))
            line += f" rho(|g|, 螢幕半徑)={rho_r:+.3f}"
        print(line)
        self._absgrad_accum = None

    @torch.no_grad()
    def _dar_cost_decay(self, outputs, batch, gaussian_model, global_step: int, pl_module) -> None:
        """DAR-style decoupled cost-aware opacity regularization; see dar_lambda docstring
        and `紀錄/現行方案與公式.md` §1. Runs on the on_train_batch_end hook,
        i.e. AFTER optimizer.step(), so Adam's moments stay photometric-only (the decoupling).
        """
        cfg = self.config
        if cfg.dar_lambda <= 0:
            return
        n = gaussian_model.n_gaussians
        self._dar_stats = getattr(self, "_dar_stats", {"applied": 0, "no_radii": 0, "no_v": 0, "no_lr": 0})
        # Cost from ANALYTIC projection, not the rasterizer's radii window. The window is a
        # running max that is reset at every densify event, so for the ~150 steps after an
        # event most points read 0 and ĉ collapses to the storage floor — measured on the
        # first DAR smoke: ĉ median 0.10 (= floor) at steps 1001/1501 vs 0.79 mid-window,
        # i.e. a 150-step sawtooth in the pressure. Our own §1 defines c_i = Σ_v c_iv, and
        # a per-view analytic footprint is both closer to that and free of the window's
        # topology fragility. Shares projected_radius() with the dynamic-K load estimator.
        from internal.utils.strip_cameras import projected_radius
        opa = gaussian_model.gaussians["opacities"]

        # 1/√v̂ from Adam's second moment for the opacity parameter = inverse marginal
        # utility (how much the photometric loss cares about this primitive).
        v_hat = None
        for opt in pl_module.gaussian_optimizers:
            st = opt.state.get(opa, None)
            if st is not None and "exp_avg_sq" in st:
                v_hat = st["exp_avg_sq"]
                break
        if v_hat is None:
            self._dar_stats["no_v"] += 1
            self._dar_report(global_step)
            return                                   # first step: no moments yet
        v = v_hat.squeeze(-1)
        st = opt.state.get(opa, {})
        beta2 = 0.999
        for g in opt.param_groups:
            if g.get("name") == "opacities" and "betas" in g:
                beta2 = g["betas"][1]
        t = float(st.get("step", 0) or 0)
        if t > 0:                                    # Adam bias correction
            v = v / (1.0 - beta2 ** t)
        # u_i ∈ (0,1]: 1/√(v̂+ε) normalised BY ITS OWN BOUND 1/√ε, so λ has a readable scale
        # and the term degrades gracefully. ε sits INSIDE the sqrt as in AdamW-GS Eq.8 — with
        # it outside, 1/√v̂ reached 1e8 here and λ·ĉ·(1/√v̂) saturated dar_clip for 100% of
        # points at every λ down to 5e-4, flattening ĉ's 7x spread to 1.00x (i.e. degenerating
        # into exactly the uniform decay this mechanism replaces). Measured b12@30k.
        eps = max(cfg.dar_eps, 1e-30)
        u = (eps ** 0.5) / (v + eps).sqrt()

        # cost term. detach()-by-construction: radii is a plain buffer, so the resource cost is
        # an externally given constant for this step. Two compressions, both measured on
        # b12@30k (1M pts):
        #  (a) physical cap. A splat cannot cost more than the whole frame — rasterizer work is
        #      bounded by the tile/pixel count — so clamp r² at the screen diagonal². Needed
        #      because the projection Jacobian is singular as z->0: a surfel at z~0.02 reports
        #      r²~1e16, an artefact rather than a cost. Raw r² spans 5.8e11x; clamped, 52x.
        #      19.4% of points saturate, and the large/normal cost ratio still reads 19.9x.
        #  (b) log1p + median normalisation, so ĉ is O(1) and dar_lambda is scene-independent.
        #
        # NOT used: the min(s)/max(s) "flatness discount" (a 3DGS-oriented idea resting on
        # 2DGS's finding that real surfaces flatten to discs). Our surfels carry scales [N,2] —
        # the third axis is identically zero, so flatness is not a degree of freedom here and
        # min/max only measures in-plane elongation. Measured on b12: r>2000px 0.852 vs
        # o>0.7 real surface 0.786 — no separation, and inverted, so the discount would tax
        # real surfaces harder than monsters.
        # from the camera, NOT outputs: on_train_batch_end receives Lightning's STEP_OUTPUT
        # (the loss dict), which has no "render" key — verified by a 1-minute smoke that
        # died on KeyError before any GPU time was spent.
        camera = batch[0]
        H, W = int(camera.height), int(camera.width)
        # same accessors as the dynamic-K caller: get_xyz is a property, get_scales() is
        # already activated (gaussian.py:147 scale_activation(self.scales))
        radii = projected_radius(gaussian_model.get_xyz.detach(),
                                 gaussian_model.get_scales().detach(), camera)
        area = (radii.float() ** 2).clamp(max=float(H * H + W * W))
        med = torch.median(area[area > 0]) if bool((area > 0).any()) else area.new_tensor(1.0)
        c_hat = cfg.dar_cost_storage + torch.log1p(area / med.clamp_min(1e-12))

        lr_o = None
        for opt in pl_module.gaussian_optimizers:
            for g in opt.param_groups:
                if g.get("name") == "opacities":
                    lr_o = g["lr"]
        if lr_o is None:
            self._dar_stats["no_lr"] += 1
            self._dar_report(global_step)
            return

        stride = (cfg.dar_lambda * c_hat * u).clamp(max=cfg.dar_clip)
        opa.sub_((lr_o * stride).unsqueeze(-1))       # opacities are logits: subtract = decay
        self._dar_stats["applied"] += 1
        if global_step % 500 == 1:                    # +1 offset: never a densify-event step
            o_now = torch.sigmoid(opa.detach().squeeze(-1))
            sat = float((cfg.dar_lambda * c_hat * u >= cfg.dar_clip).float().mean())
            print(f"[DAR-cost] step {global_step}: applied={self._dar_stats['applied']} "
                  f"skip(radii/v/lr)={self._dar_stats['no_radii']}/{self._dar_stats['no_v']}/{self._dar_stats['no_lr']} "
                  f"c_hat[min/med/max]={float(c_hat.min()):.2f}/{float(c_hat.median()):.2f}/{float(c_hat.max()):.2f} "
                  f"u[med]={float(u.median()):.3f} "
                  f"stride[med/max]={float(stride.median()):.4f}/{float(stride.max()):.4f} "
                  f"clipped={100*sat:.1f}% | o<0.005={100*float((o_now<0.005).float().mean()):.1f}% "
                  f"o<0.05={100*float((o_now<0.05).float().mean()):.1f}%")

    def _value_per_cost_mask(self, gaussian_model, global_step: int):
        """Bottom-`vpc_prune_frac` mask by v/c, or None when disabled/unavailable.

        value  = multi-view mean alpha-blending contribution Σ(T·α)/covered_px, refreshed
                 every `vpc_interval` steps via the rasterizer's record_transmittance path
                 (same signal the contribution-trim already uses, so no CUDA change).
        cost   = max screen radius^2 over the interval (∝ tile footprint ∝ binning bytes),
                 reusing the max-radii window kept for screen_size_prune.
        """
        cfg = self.config
        if cfg.vpc_prune_frac <= 0:
            return None
        n = gaussian_model.n_gaussians
        # CHEAP GUARD FIRST: cost comes from the live radii window (this runs before
        # screen_size_prune clears it). No snapshot — a snapshot goes stale the moment
        # add_new_gs appends points.
        radii = self._max_radii2D
        if radii is None or radii.shape[0] != n:
            return None
        # Value estimate: expensive (one transmittance pass over sampled views), so refresh
        # only on `vpc_interval`. Between refreshes, keep the cached value valid across
        # topology changes instead of forcing a re-measure: add_new_gs APPENDS (existing
        # indices stable) and relocate_gs moves dead points in place, so we pad new points
        # with the median (neutral -> a fresh point isn't condemned before it's measured).
        # The old `shape != n -> re-measure` guard fired at EVERY densify event (N grows 5%
        # each time), which made vpc_interval a no-op and dominated step time.
        need_refresh = self._vpc_contrib is None or (global_step - self._vpc_step) >= cfg.vpc_interval
        if not need_refresh and self._vpc_contrib.shape[0] != n:
            cur = self._vpc_contrib
            if cur.shape[0] < n:                       # padded for appended points
                pad = cur.median().expand(n - cur.shape[0])
                self._vpc_contrib = torch.cat([cur, pad])
            else:                                       # pruned elsewhere -> must re-measure
                need_refresh = True
        if need_refresh:
            t0 = time.time()
            contrib = self._measure_multiview_contribution(gaussian_model)
            if contrib is None:
                return None
            self._vpc_contrib, self._vpc_step = contrib, global_step
            print(f"[value-per-cost] step {global_step}: value refreshed in {time.time()-t0:.1f}s")
        if self.config.exact_tile_cost and self._max_tiles2D is not None \
                and self._max_tiles2D.shape[0] == n:
            cost = self._max_tiles2D.float().clamp_min(1.0)       # 精確：就是 tile 數本身
        else:
            cost = radii.float().clamp_min(1.0) ** 2              # ∝ tile footprint（代理）
        vpc = self._vpc_contrib / cost
        k = int(n * cfg.vpc_prune_frac)
        if k <= 0:
            return None
        thresh = torch.kthvalue(vpc, k).values
        return vpc <= thresh

    def _measure_multiview_contribution(self, gaussian_model):
        """One transmittance-only pass over the training cameras -> per-Gaussian value."""
        ref = getattr(self, "_pl_ref", None)
        pl = ref() if ref is not None else None
        if pl is None:
            return None
        renderer = pl.renderer
        if not hasattr(renderer, "forward"):
            return None
        cameras = pl.trainer.datamodule.dataparser_outputs.train_set.cameras
        device = gaussian_model.get_xyz.device
        bg = pl._fixed_background_color().to(device)
        acc = torch.zeros(gaussian_model.n_gaussians, device=device)
        stride = max(1, len(cameras) // 24)               # subsample views for speed
        used = 0
        with torch.no_grad():
            for i in range(0, len(cameras), stride):
                trans = renderer(cameras[i].to_device(device), gaussian_model,
                                 bg_color=bg, record_transmittance=True)
                if not torch.is_tensor(trans) or trans.shape[0] != acc.shape[0]:
                    return None
                acc += trans
                used += 1
        return acc / max(used, 1)

    def _current_min_opacity(self, step: int) -> float:
        cfg = self.config
        if cfg.min_opacity_final <= 0:
            return cfg.min_opacity
        t0 = cfg.min_opacity_anneal_start_iter if cfg.min_opacity_anneal_start_iter >= 0 else cfg.densify_from_iter
        t1 = cfg.min_opacity_anneal_end_iter
        frac = min(max((step - t0) / max(t1 - t0, 1), 0.0), 1.0)
        return cfg.min_opacity * (cfg.min_opacity_final / cfg.min_opacity) ** frac

    TILE_PX = 16

    def _update_load(self, outputs: dict) -> None:
        """每步累積**這個視角**的 binning 負載 `Load = Σ_i (2r_i/16)^2`，並保留區間內的最大值。

        這就是 `tools/cost_budget_probe.py` 量的 `B = max_view Load`，只是改成線上量：
        每步只渲染一台相機，所以逐步取 max 就是「已見視角中的最壞視角」。
        成本 = 每步一個 O(N) 的 sum，相對每步 460ms 可忽略。

        為什麼要這個而不是沿用 `_max_radii2D`（§11.60）：`_max_radii2D` 是**逐顆跨視角取 max**
        的半徑，把它加總得到的是「每顆都用自己最壞視角」的上界，比任何真實單一視角都大，
        與探針量到的 B 不是同一個量。預算若用錯的量標定，實驗就白跑。

        ⚠ 光柵器的 `radii` 與探針的 `3*fx*s/z*stretch` 未必逐位元相同（兩者都是 3-sigma
        意義下的螢幕半徑，但實作路徑不同）⇒ **不可直接把探針的 23.1M 當預算**，
        要先用 `cost_budget_report` 跑一次短標定讀出線上值。
        """
        radii = outputs.get("radii", None)
        if radii is None:
            return
        r = radii.float()
        vis = r > 0
        if not bool(vis.any()):
            return
        load_proxy = float(((2.0 * r[vis] / self.TILE_PX) ** 2).sum())
        load = load_proxy
        if self.config.exact_tile_cost:
            tiles = outputs.get("tiles", None)
            if tiles is None:
                if not getattr(self, "_warned_no_tiles", False):
                    self._warned_no_tiles = True
                    print("⚠⚠ [exact-tile-cost] 設了 exact_tile_cost 但光柵器沒回傳 `tiles` "
                          "=> **退回 radii^2 代理**（請重編光柵器；2026-09-21 起才有這個輸出）")
            else:
                load = float(tiles.double().sum())
                if not getattr(self, "_told_tile_units", False):
                    self._told_tile_units = True
                    print(f"[exact-tile-cost] ✅ 首次生效：本視角 精確 Load={load:,.0f} "
                          f"vs 代理={load_proxy:,.0f}（代理是 {load_proxy / max(load, 1):.3f} 倍）"
                          f" ⇒ **cost_budget 的單位已改變，先前標定的數值要除以這個比值重算**")
        if load > getattr(self, "_load_max", 0.0):
            self._load_max = load
        # 報告用的累積（2026-09-14）：平均需要 sum/count；不碰閘門的 `_load_max`
        self._rep_sum = getattr(self, "_rep_sum", 0.0) + load
        self._rep_cnt = getattr(self, "_rep_cnt", 0) + 1
        if load > getattr(self, "_rep_max", 0.0):
            self._rep_max = load
        # 閘門用的平均（2026-09-14）：與報告視窗分開，由 add_new_gs 在每個 densify 事件歸零
        self._gate_sum = getattr(self, "_gate_sum", 0.0) + load
        self._gate_cnt = getattr(self, "_gate_cnt", 0) + 1

    def _update_max_radii(self, outputs: dict, gaussian_model) -> None:
        radii = outputs.get("radii", None)
        if radii is None:
            return
        n = gaussian_model.n_gaussians
        if radii.shape[0] != n:
            return
        if self._max_radii2D is None or self._max_radii2D.shape[0] != n or self._max_radii2D.device != radii.device:
            self._max_radii2D = torch.zeros_like(radii)
        self._max_radii2D = torch.maximum(self._max_radii2D, radii)
        # ★ 2026-09-21：同樣窗口語意（逐顆跨視角取 max）的**精確** binning 成本。
        #   ⚠ 窗口語意本身是既有的設計債：DAR 當初刻意改用解析投影，理由就是
        #     「逐顆跨視角取 max 再加總，比任何真實單一視角都大」。換成 tiles 不會修好那件事，
        #     只是把「每一項」量準；窗口要不要改是**另一個**決定。
        if self.config.exact_tile_cost:
            tiles = outputs.get("tiles", None)
            if tiles is not None and tiles.shape[0] == n:
                if self._max_tiles2D is None or self._max_tiles2D.shape[0] != n \
                        or self._max_tiles2D.device != tiles.device:
                    self._max_tiles2D = torch.zeros_like(tiles)
                self._max_tiles2D = torch.maximum(self._max_tiles2D, tiles)

    def _prune_points(self, mask, gaussian_model, optimizers) -> None:
        # The Trim2DGS renderer (sep_depth_trim_2dgs_renderer.py) trims surfels mid-training
        # and calls this to remove them. MCMC keeps no per-Gaussian state buffers (only the
        # binoms table), so unlike Vanilla/RTGStable we just prune the model properties.
        # `mask`: True = prune.
        valid = ~mask
        gaussian_model.properties = Utils.prune_properties(valid, gaussian_model, optimizers)

    def compute_relocation(self, opacity_old, scale_old, N) -> Tuple[torch.Tensor, torch.Tensor]:
        # gsplat compute_relocation requires scales [N, 3]; pad 2D surfel scale with a zero
        # 3rd component and truncate the result. coeff depends only on opacity, so this is
        # exact for the real components.
        d = scale_old.shape[-1]
        if d < 3:
            pad = torch.zeros((scale_old.shape[0], 3 - d), dtype=scale_old.dtype, device=scale_old.device)
            scale_old = torch.cat([scale_old, pad], dim=-1)
        new_opacity, new_scaling = compute_relocation(opacity_old, scale_old, N, self.binoms)
        return new_opacity, new_scaling[:, :d]

    def _get_new_params(self, gaussian_model, idxs, ratio) -> Dict[str, torch.Tensor]:
        # Same as the parent but `new_scaling` is already the surfel's scale dim (2), so we
        # drop the parent's `.reshape(-1, 3)`.
        new_opacity, new_scaling = self.compute_relocation(
            opacity_old=gaussian_model.get_opacities()[idxs, 0],
            scale_old=gaussian_model.get_scales()[idxs],
            N=ratio[idxs, 0] + 1,
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)

        # RTG-SLAM 式的透明修正層：父代是「高誤差 + 高不透明」時，新粒子不走 Eq.9 的分裂，
        # 直接給一個固定的低 opacity（見 `transparent_corrector` 的 docstring / §11.9）。
        tc = self.config.transparent_corrector
        if tc > 0 and self._err_score is not None and self._err_score.shape[0] == gaussian_model.n_gaussians:
            with torch.no_grad():
                o_par = gaussian_model.get_opacities()[idxs, 0]
                e_par = self._err_score[idxs]
                cand = o_par > self.config.transparent_corrector_min_opacity
                n_cand = int(cand.sum())
                if n_cand > 0:
                    k = max(1, int(round(self.config.transparent_corrector_err_frac * n_cand)))
                    thr = torch.topk(e_par[cand], k).values[-1]
                    hit = cand & (e_par >= thr)
                    n_hit = int(hit.sum())
                    if n_hit > 0:
                        new_opacity[hit, 0] = tc
                        self._tc_last = (n_hit, n_cand)

        new_opacity = gaussian_model.opacity_inverse_activation(new_opacity)
        new_scaling = gaussian_model.scale_inverse_activation(new_scaling)

        new_params = {"opacities": new_opacity, "scales": new_scaling}
        for attr_name, value in gaussian_model.properties.items():
            if attr_name not in new_params:
                new_params[attr_name] = value[idxs]
        return new_params

    def _add_xyz_noise(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        if getattr(pl_module, "is_final_step", False) is True:
            return

        with torch.no_grad():
            # ★ 稀疏化（noise_gate_eps）：gate 對高 opacity 粒子本來就 ~0，
            #   先選出真正會動的那些，後面所有 O(N) 的工作都只對它們做。
            _eps = getattr(self.config, "noise_gate_eps", 0.0)
            _sel = None
            if _eps > 0:
                _g_all = self.op_sigmoid(1.0 - gaussian_model.get_opacities()).squeeze(-1)
                _sel = torch.nonzero(_g_all > _eps, as_tuple=True)[0]
                if _sel.numel() == 0:
                    return

            # Pad the 2D surfel scale with a zero normal component so the 3D covariance has
            # zero normal variance -> noise stays in the tangent plane (no off-surface drift).
            scales = gaussian_model.get_scales()
            d = scales.shape[-1]
            if d < 3:
                pad = torch.zeros((scales.shape[0], 3 - d), dtype=scales.dtype, device=scales.device)
                scales = torch.cat([scales, pad], dim=-1)

            _fast = getattr(self.config, "fast_noise", False)
            _q = gaussian_model.get_rotations()
            if _sel is not None:
                _q = _q[_sel]                       # ★ N x 3 x 3 只對會動的那些建
            if not _fast:
                cov_3d = compute_cov_3d(
                    scales=scales[_sel] if _sel is not None else scales,
                    scale_modifier=1.,
                    quaternions=_q,
                )
            else:
                from internal.utils.gaussian_projection import build_rotation_matrix
                _R = build_rotation_matrix(_q)
                cov_3d = None      # 走等價路徑，不 materialize（見 fast_noise docstring）

            xyz_lr = -1
            for opt in pl_module.gaussian_optimizers:
                for param_group in opt.param_groups:
                    if param_group["name"] == "means":
                        xyz_lr = param_group["lr"]
                if xyz_lr >= 0:
                    break
            assert xyz_lr >= 0

            _o = gaussian_model.get_opacities()
            if _sel is not None:
                _o = _o[_sel]
                scales = scales[_sel]
            noise = torch.randn((_o.shape[0], 3), dtype=scales.dtype, device=scales.device) \
                * self.op_sigmoid(1 - _o) * self.config.noise_lr * xyz_lr
            if cov_3d is not None:
                noise = torch.bmm(cov_3d, noise.unsqueeze(-1)).squeeze(-1)
            else:
                # R (s^2 * (R^T v))：與上面 `cov @ v` 數學等價，省掉 N x 3 x 3 的建構
                noise = torch.bmm(_R.transpose(1, 2), noise.unsqueeze(-1)).squeeze(-1)
                noise = noise * (scales ** 2)
                noise = torch.bmm(_R, noise.unsqueeze(-1)).squeeze(-1)
            if _sel is None:
                gaussian_model.means.add_(noise)
            else:
                gaussian_model.means.index_add_(0, _sel, noise)
