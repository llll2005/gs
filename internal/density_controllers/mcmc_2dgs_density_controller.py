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

    cost_aware_densify: float = 0.0

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

    cost_budget_report: int = 0
    """>0 時每 N 步印出線上量到的 `Load`（不改變任何行為），用來標定 `cost_budget`。"""
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
        if self.config.screen_size_prune_px > 0 or self.config.dar_lambda > 0 \
                or self.config.max_relocate_frac > 0 or self.config.vpc_prune_frac > 0 \
                or self.config.cost_aware_densify != 0:
            self._update_max_radii(outputs, gaussian_model)   # cost signal for DAR / gate / v-p-c
        # 成本預算（§11.60）：B 與 N 動態解耦（B/N 在軌跡上變動 2.2 倍；15k->30k 顆數 +98%
        # 而成本只 +9.4%）⇒ `cap_max` 按顆數計價是錯的貨幣。這裡量的是正確的貨幣。
        if self.config.cost_budget > 0 or self.config.cost_budget_report > 0:
            self._update_load(outputs)
            _cbr = self.config.cost_budget_report
            if _cbr > 0 and global_step % _cbr == 0:
                _n = gaussian_model.n_gaussians
                _l = getattr(self, "_load_max", 0.0)
                print(f"[cost-budget] step {global_step}: N={_n:,} "
                      f"Load(區間最壞視角)={_l:,.0f} Load/N={_l / max(_n, 1):.2f} "
                      f"預算={self.config.cost_budget:,.0f}"
                      f"{'（未設，純標定）' if self.config.cost_budget <= 0 else ''}")
        self._accumulate_error_score(outputs, batch, gaussian_model)
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
            if self.config.screen_size_prune_px > 0 and self._max_radii2D is not None \
                    and self._max_radii2D.shape[0] == dead_mask.shape[0]:
                big_mask = self._max_radii2D > self.config.screen_size_prune_px
                n_big = int(big_mask.sum())
                if n_big > 0:
                    dead_mask = dead_mask | big_mask
                    print(f"[screen-size prune] step {global_step}: recycling {n_big} Gaussians with radius > {self.config.screen_size_prune_px}px (max {int(self._max_radii2D.max())}px)")
                self._max_radii2D = None  # restart the window after each event
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
            self._report_densify_blindness(gaussian_model, global_step)
            self._err_triggered_unlock(gaussian_model, optimizers, global_step)
            self.relocate_gs(gaussian_model, optimizers, dead_mask)
            self.add_new_gs(gaussian_model, optimizers)
            if self._tc_last is not None:
                print(f"[transparent-corrector] step {global_step}: {self._tc_last[0]}/{self._tc_last[1]} "
                      f"個高誤差高不透明父代的子代改為 opacity {self.config.transparent_corrector}")
                self._tc_last = None
            self._err_score = None                 # window restarts with the next interval

    _max_radii2D: torch.Tensor = None
    _vpc_contrib: torch.Tensor = None   # cached multi-view contribution (value)
    _vpc_step: int = -1

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
        c = radii.float().clamp_min(1.0) ** 2

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
        cost = radii.float().clamp_min(1.0) ** 2          # ∝ tile footprint
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
        load = float(((2.0 * r[vis] / self.TILE_PX) ** 2).sum())
        if load > getattr(self, "_load_max", 0.0):
            self._load_max = load

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
            # Pad the 2D surfel scale with a zero normal component so the 3D covariance has
            # zero normal variance -> noise stays in the tangent plane (no off-surface drift).
            scales = gaussian_model.get_scales()
            d = scales.shape[-1]
            if d < 3:
                pad = torch.zeros((scales.shape[0], 3 - d), dtype=scales.dtype, device=scales.device)
                scales = torch.cat([scales, pad], dim=-1)

            cov_3d = compute_cov_3d(
                scales=scales,
                scale_modifier=1.,
                quaternions=gaussian_model.get_rotations(),
            )

            xyz_lr = -1
            for opt in pl_module.gaussian_optimizers:
                for param_group in opt.param_groups:
                    if param_group["name"] == "means":
                        xyz_lr = param_group["lr"]
                if xyz_lr >= 0:
                    break
            assert xyz_lr >= 0

            noise = torch.randn_like(gaussian_model.means) * (self.op_sigmoid(1 - gaussian_model.get_opacities())) * self.config.noise_lr * xyz_lr
            noise = torch.bmm(cov_3d, noise.unsqueeze(-1)).squeeze(-1)
            gaussian_model.means.add_(noise)
