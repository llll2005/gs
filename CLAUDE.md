# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reference Papers

Related papers are stored in `./參考論文/` (12 PDFs). When discussing theory, algorithms, loss design, or comparing methods, read the relevant PDF directly with the Read tool (use the `pages` parameter to target specific sections). The memory file `reference_papers.md` contains a full filename↔title↔module mapping table.

Key papers:
- `2411.00771v2.pdf` — CityGaussianV2 (main paper)
- `CityGaussian: Real-time...pdf` — CityGaussianV1 (main paper)
- `2dgs.pdf` — 2DGS (core representation)
- `GHAP.pdf` — related to `GHAP2coarse.py`
- `RaDe-GS.pdf` — depth regularization design

## Research Direction (Active Project)

This repo is being extended for an undergraduate research project (國科會/專題). Goal:
consumer-GPU (RTX 4050 Laptop, **6GB VRAM**, 46GB RAM) training of city-scale 2DGS at quality
competitive with CityGaussianV2.

**Records have a single entry (reorganised 2026-10-03): `紀錄/README.md` -> `紀錄/研究總覽.md` (the only body;
§0 = one-page current status, §10 = which old-era conclusions were re-measured) -> `紀錄/實驗分析/NN_*.md`, each paired with a
same-named `NN_*.png` (doc tables = the numbers on the figure; regenerate with `python tools/plot_analysis.py`).
The previous bodies are archived: `紀錄/new_archived/研究總覽_v4_至2026-10-19.md`, `_v3_至2026-10-15`, `_v2_至2026-10-03` and
`紀錄/new_archived/研究總覽_v1_至2026-09-13.md` — **code comments citing "研究總覽 §11.x/§13.x/§16.x" mean v1**.
**2026-10-19 reorganisation (user)**: the body keeps only what is currently valid; methodology lessons => `紀錄/方法論.md`;
every closed line + reopen condition => `紀錄/new_archived/已關閉的線.md` (cumulative); era table => `紀錄/new_archived/年代與舊結論處理狀態.md`;
retired manual sections => `紀錄/new_archived/完整指令手冊_退役段落_至2026-10-19.md`. New analyses 16 (steps/SH), 17 (out-of-block mask), 18 (road underground splats).
`_ctx.md`, the indexes, the survival list and the speed Q&A are also in `new_archived/` (one-line reason per file).
Read those before proposing mechanisms — this section is a summary and goes stale.**
**2026-11-13 reorganisation (user: "too many numbered docs; fold finished ones into the overview, merge similar ones; overview keeps only open + adopted"): open lines are now only 30 (frontier, permanent), 37 (depth cap & underground = main line; only `capfc` 只補前面 is open, full pre-11-13 text => `new_archived/實驗分析_過程/37_SfM三角形校正深度_至1113.md`), 38 (time & speed: step timing, slimming, speed combos, run-time ledger, two-round flow & window ratios moved from 37, steps merged from 36). 32 (sea, closed) / 33 (towers, trimask adopted) => `new_archived/實驗分析_定案/` with conclusions in `研究總覽.md` §5.4/§5.5; 36 => `實驗分析_過程/`. Earlier (11-12): 28/29/35 定案, 31/34 過程, specs 1–2 + handoff => `new_archived/規格與交接/`. Direction-level closures = `已關閉的線.md` §I; old-number lookup = `紀錄/實驗分析/README.md`.**
**2026-11-05 reorganisation (user) — filing rules:** `紀錄/實驗分析/` holds ONLY open lines, one doc per line (28 speed, 29 per-block
count, 30 current frontier [permanent], 31 underground, 32 sea floaters, 33 tall buildings, 34 next-recipe candidates); settled results go into
`研究總覽.md` §1–§8 (one-line conclusion + key numbers; index §8.1) with the original doc+figure moved to `new_archived/實驗分析_定案/`;
date-batch docs that were split into lines are in `new_archived/實驗分析_過程/`; old number -> new location table = `實驗分析/README.md`.
Unnumbered references: `參考_CityGS參考線.md` (was 09), `理論_ADMM統一框架.md` (was 14), `機制出處.md` (was 15 §4).
`已關閉的線.md` takes only direction-level closures with strong evidence; single-arm results stay in their line doc; ideas never run go to
§9 / `加速論文對照.md` as 延後／不做＋理由; a failed v1 with a planned v2 keeps the line open. Pre-reorg overview => `研究總覽_v5_至2026-11-05.md`.

**⛔⛔ 2026-09-13 third era boundary: images and poses were misaligned.** Every *local* training score in
`outputs/` — including the speed3 / SfM-init / sched30 / absgrad numbers further down this file — is
invalid; see `紀錄/研究總覽.md` §10 (item list: `紀錄/new_archived/倖存清單_2026-09-13.md`). New-era numbers live in `outputs/lab/` (the lab machine).
**Lab protocol, revised by the user 2026-10-04 (supersedes the 09-18 cap/parallel rules below): no VRAM cap — judge 6GB feasibility
by the recorded peak; training runs default to `[solo]` (exceptions: several recipes compared purely on scores, and short functional
tests), so every run's time and VRAM numbers are directly usable. Lab runs before 10-04 had cap 5.66 + max_split and mostly ran in
parallel => their wall-time / reserved-VRAM columns are not comparable across that boundary (quality and peak allocated are).**
**Grid protocol (user, 2026-10-10): all new runs use the 4x4 grid (`CITYGS_GRID=44 ... CITYGS_FAMILY=g44_ bash scripts/lab/task_cmp.sh <blk> <arm>`),
same grid cells as the official CityGSV2 4x4, so every block compares against the official per-block models. Single-variable arms run on
4x4 blocks 6 and 12; the 4x4 per-block reference without oreg0 is `outputs/lab/full44_best/blocks/block_<B>`. Base recipe since 10-14 =
`best1` = best0 (dup init + opacity_reg 0) + opacity binary-entropy reg 0.002 + trim criterion V/C + backward saturation-skip & in-block
pre-reduction (= every adopted speed-up; `task_cmp.sh` prefix `best1*`; reference run `g44_best1`, same recipe without the two speed flags = `g44_best0oeVC`): swap eval vs best0 -0.09/+0.01 dB, exact Load -34%/-35% (`紀錄/實驗分析/13` §2). Before 10-14 the base was `best0`.**
**Two-machine protocol (user, 2026-09-18): lab = relative comparison (which recipe is better, lab vs lab), kept at
3 parallel slots because throughput is ~2x and parallelism does NOT affect results — quality, offline Load and peak
VRAM are all neighbour-independent (only wall time and it/s are, and those have dedicated `[solo]` tools).
The laptop = absolute verification: re-run the stage winner there to claim the envelope.**

**★★★★★ 2026-11-12 current single-block base (user adopted after viewer check; no full scene yet):** `[solo] bash scripts/lab/task_tune25.sh <blk> s30g50+np16+tricap0p5+cmflat` = 11-10 new base (11-07~08 defaults with position lr back to x1 and SfM visibility v4 OFF — lr x2 caused 87% of the 11-09 full-scene −0.33 dB) + neighbourhood-plane every 16 steps + SfM-triangle-calibrated depth + per-contribution depth cap (`depth_cap_reg 0.5`, margin 1%) applied only on flat prior surfaces (`cmflat`, |n_z|>0.8 mask; passed b1/b4 the same evening). b1/b4 vs A+dist4: swap −0.012/−0.023 dB, F1 +.11/+.11, underground −73/−75%, time −4% (b4). Per-block caps follow SfM-point allocation (user rule). Differences from A+dist4 => `紀錄/完整Pipeline數學詳解.md` §14; status => `紀錄/研究總覽.md` §0. Validate candidates on b1/b4.
**11-13: base += `trimask`** (out-of-block mask judged by calibrated depth; user "進底"; b2 +0.058, b6 +0.019, b1 tie; masks for all 16 blocks generated) => arm prefix `s30g50+np16+tricap0p5+cmflat+trimask`; **11-13 evening: base 3 = `s20g70+np16+tricap0p5+cmflat+trimask+lowres+geo8+log0`** — pure training b1/b4 6.7/7.1 min (−46%), swap −0.004/+0.008, F1 −.005/−.007 (`紀錄/實驗分析/38` §3; base 2 = 30k version, 9.7/10.4 min). Also 11-13: rasterizer gained `depth_cap_grad_mode` (3 = 規格 3 §4-3 只補前面, component `capfc`; mode 0 bit-identical, `tools/check_depth_cap.py fc`).
**★★★★ Current recipe (2026-11-02) — read `紀錄/完整Pipeline數學詳解.md` (rewritten 11-02) before changing anything:**
A＋dist4 = 4x4, 30k steps/block, arm `best1maskdistrfed10b4` (out-of-block A/B/C mask + random background + cropped dup init,
dist 100 every 4 steps, opacity binary entropy one-sided until STEPS/3 then both ×4, accumulative V/C trim, FusedAdam) + flags
`record_reduce absgrad_gate fused_ssim sh_rest_interval 16 lambda_dist_interval 4`, cap 2.5M => full scene 28.058/.8824/.1142,
27.75M, 5.04 h (CityGS release 27.261). Light version: adaptive cap avg 1M 27.828, 3.19 h. Status/candidates: `紀錄/研究總覽.md` §0;
open lines as of 11-12 are listed above (30, 32, 33, 36, 37; settled/process docs archived 11-05 and 11-12). 11-04: strongest candidate = normal consistency every 8 steps
(`lambda_normal 0.0125 lambda_normal_interval 8`); main open problem = road surfaces live underground (5–9 coverage layers below SfM ground),
mono depth (±10%) too coarse to detect it => SfM-depth see-through is the next mechanism. Judge by goals, not resemblance to CityGS (user 11-04).
**11-05** (`紀錄/實驗分析/28`–`29`; SfM five arms in 31–33): lossless speed flags `trim_running_warp` (growth phase −8.5%/step) and `bwd_zero_grad_skip` (−1.7%) — rasterizer rebuilt
(commit `74b77ce`, local env + lab) => turn them on in new arms; `skip_unscored_pixels` / pair bit still need quality runs. SfM birth restriction
(`sfm_birth_radius 0.05 sfm_birth_eps 0.01`) cuts b3 underground layers 2.32 → 1.04 but thins SfM-sparse tall buildings. The 10-31 adaptive cap needs a
2.5M probe run (not general, user 11-05) => zero-cost `tools/sfm_cap_alloc.py` (r 0.69 with the probe weights; lab full scene queued). `price_*` v1 failed;
the sampling controller (`--model.cap_ctrl true`, held-out safe views) is under test. Next after count/speed: the tall-building problem (user 11-05).
The block below is older history.

**★★★ Current state of the proposition (2026-09-17, with 2026-10-03/10-09 additions; lab, blocks b6/b12/b13; authority = `紀錄/研究總覽.md` §0):**
```
2026-10-03       current best = cs60_sfmdup4 (conic exact AABB is the default + task_cmp.sh common flags + dup4 init):
                   60k val b6 30.75 / b13 30.32; lean_train (-20.9%/step, render bit-identical) installed on lab
                 official CityGSV2, unmodified source, current data: official 741-frame held-out 25.79 / .833 / .176 (24.48M)
                 v/c trim with conic on: 60k -0.17/-0.20, LPIPS better; on the 22k cost-quality frontier it beats cap-only
                   by ~+1.2 dB at equal Load; cost_budget (proxy or exact units) does NOT beat simply lowering cap
                 new-era rechecks: almost no floaters, failure tiles 1.3~2.3%, geometry improves with training
2026-10-09       ★ full scene 4x4 (cs60_sfmdup4, 16 blocks [solo]): merged 741-frame held-out 27.74/.879/.129 (17.63M, 35.6 FPS,
                   train peak <=4.17 GB) beats official release 27.26/.867/.154 and paper setting 27.28/.870/.147 -- SAME eval tool
                   (tools/eval_official_test.py). Tool gap 25.79 vs 27.26 RESOLVED: the official rasterizer renders 8~13% of test views
                   with wrong (dark/black) colours; swapping only the rasterizer fixes them => official release ≈ 27.2~27.6 (our rasterizer reads it 0.35 low
                   on normal frames) => our PSNR lead is +0.1~+0.5 dB; SSIM/LPIPS + resource leads are the robust claim.
                   Official does 2 backward passes/step (depth extra_loss) => 214 vs 121 ms/step at ~same N. => 紀錄/實驗分析/09 §0
pruning side     trim criterion v -> v/c (`renderer.init_args.trim_by_value_per_cost`)
                 short recipe (22k~27k, 3 blocks): same N, offline Load median -64~-67%, VRAM -7%, LPIPS better 3/3,
                   PSNR -0.002/-0.08/-0.17
                 ⚠ 60k full speed3 recipe (b6/b13): PSNR -0.27/-0.54 (3~7 sd), LPIPS slightly worse
                   => the quality cost GROWS with training length; 60k Load not measured yet
elongation       60k: hard prune -0.15/-0.23; relocate -0.02/-0.08 (tie, but first trigger hit 0~1 primitives);
                 trimvpc + hard prune -0.68/-0.95 (worse than additive) => not complementary
sampling side    loose budget = cost<->quality knob; B0/4 +0.66 dB on b6 but b12 replication degenerate => not citable
constraint side  `cost_budget` leaks (gates additions, not scale growth; per-primitive footprint 2.5~3.4x under budget);
                 max-view gate is the root cause of the b12 collapse (confirmed by the b12 reference trajectory:
                 healthy max-view Load crosses B0/4 at step ~1,500; b12 spikes p90 7.6x vs b6 2.7x);
                 relative budget, equal target cost (b6): loose->tight 27.26 @62.5B beats tight->loose 27.34 @91.2B
init             20k, mechanisms off (b6/b13): sfmfill 28.68/28.20 > sfm 28.52/28.02 > depth 27.19/26.84 > random 22.46/21.83
official CityGS  the official aerial "trim" stage never trims (renderer restored from the coarse ckpt);
                 fixed by CITYGS_KEEP_CFG_RENDERER=1 (verified on lab). User's global run (official config, our code,
                 no blocks, no trim): held-out test PSNR 25.62, N 11.44M. Unmodified-source reference run queued on lab.
```
**All GPU training runs on the lab machine (user, 2026-09-17); the laptop only does CPU analysis.**
Measure cost with `scripts/task_load_compare.sh` (offline, same tool, same cameras). The in-training report's
interval **mean** (added 2026-09-14) matches the offline mean (b6: 6.95M vs 6.96M); the older `Load(區間最壞視角)`
field is a spiky max and must not be compared across measurement paths.

**（歷史，已被上方 2026-09-14 現況修正）2026-09-07：命題的「取樣端」在寬鬆預算下被否證。** 三個變體（符號 ±／冪次+加法／
代理+精確 c／強度未校準+校準）全輸，29 小時 GPU；**基準（無成本項）仍是最佳**
（agd2 26.5602 vs cadd 26.3719 / fpd 26.4260 / cheap 26.2398）。
共同形狀「紋理比↑ 而 PSNR/SSIM/LPIPS↓」且**不是塵埃**（floater 方向不一致）⇒ **虛假細節**。
與 absgrad 互補：**往「誤差在的地方」增生有效（|g|），往「便宜/貴的地方」增生無效（c）**。
⚠ 且 `scale_reg` 是 **r=0.977** 的成本代理（空拍深度範圍窄）⇒
  「每個先前方法都是 `c_i ≡ 1` 的特例」**對 scale_reg 不成立**。
✅ 約束端已測（2026-09-13）：`cost_budget` 取代 `cap_max` —— **會漏**（只擋 add、不擋 scale 成長），見上方現況。
**引用以下任何成本感知的正面敘述前，先讀記憶 `cost_aware_sampling_refuted`。**

**Core contribution (原始提法，取樣端已否證): cost-aware density control.** Existing density control prices primitives by
COUNT (`cap_max`, opacity L1, scale L1) and is blind to what they actually cost to render. We
price them by render cost and solve a resource-constrained problem:

    max Q(θ_S)  s.t.  (1/K)·Σ_{i∈S} c_i ≤ B(hardware)

with `c_i` = screen-tile footprint — measured 2026-09-13 to be mainly a per-step **time** cost, not a VRAM cost
(same N: Load -63% cut forward+backward 27% but render-forward VRAM only ~6%; VRAM is dominated by N), `λ` = shadow price, `K` = strip tiling as a
supply-side lever. Every prior method surveyed is the `c_i ≡ 1` special case; see
`紀錄/研究總覽.md` §1 (theory) and §9 + `紀錄/論文核對表.md` (12 papers audited implementation-vs-text).
⚠ The claim "every prior method is the `c_i ≡ 1` special case" has **two counterexamples** —
RAIN-GS (arXiv 2403.09413), whose `s = HW/(9πN)` is a global screen budget, and **Taming 3DGS,
whose densify score contains `c^i_g` = "number of pixels covered by g in view i" per primitive**.
Taming's sign is **positive** (+0.1: large footprint -> blurry -> densify MORE), ours divides.
Their weight is 0.1 against `∇g`'s 50, i.e. ~0.2% of the score, so **they never really tested
that direction either**. The surviving distinction is `c` as **detector** vs `c` as **price**
(bound to `(1/K)·Σc_i ≤ B`), NOT "who uses `c`". See §3.1.

**★★ SfM-init（活線）—— ⚠ 2026-09-11：`speed3_sfminit_b12` 完賽但 60k 結果作廢**
`speed3_sfminit_b12` 是**手動啟動**的，resolved config 與 `speed3_b12` 差**四項**，不是單變數：
```
initialize_from  depth_init.ply -> null          <- 想測的變數
dynamic_strips   false -> **true**               <- 污染源
strip_vram_target_gb 5.4 -> 5.2 ／ strip_safety 0.6 -> 0.85 ／ skip_surf_normal false -> true
```
`dynk_K.log` 600 個採樣點：**K>1 佔 77.7%、K>=6 佔 69.3%**，而 `_strip_forward_backward`
的 docstring 自承逐條帶 SSIM 是 "boundary-window **approximation**"
⇒ **77.7% 的訓練步跑在被改過的 loss 上**，且慢 24%（1.85 vs 2.44 it/s）、VRAM 低 0.79 GB。
❌ 不可引用：25.82@60k、26.21@56.8k、「各項指標都變差」。
✅ 仍有效（K=1 期間，step < ~12,900）：**5,680 +0.72 ／ 11,360 +0.78 dB，三指標同向**。
⇒ 重做已排入：`scripts/task_sfminit2_b12.sh`（`sfminit2_b12`，唯一變數 init）。
⚠ K 是白付的（`tools/strip_k_audit.py`，純 CPU）：兩個 60k 模型在同參數下**都**被判 K=6，
  而 depth-init 同 N=2.34M 用 K=1 **實測**峰值 4.87 GB 沒 OOM ⇒ 預測器 budget 訂低約 1.1 GB。
詳見記憶 `sfm_init_beats_depth_init`、`dynamic_strips_confound` 與 §13。

**★ 現行最佳（2026-09-06）= `speed3`**：`sched30` + `absgrad_densify 2.0`
+ **`fast_noise` + `noise_gate_eps 1e-3`**（腳本 `scripts/task_speed3.sh` / `task_speed3_b7.sh`）
```
b12 26.6957（+4.9sd）  b7 25.1448（+0.3sd 平）   並且**快 b12 +6.83% / b7 +9.48%**
```
⚠ 定位是 **Pareto 改善（更快+不更差）**，不是分數突破 —— b7 四項全平 ⇒ **不可宣稱 +0.13 dB 是普遍增益**。
✅ 已採用且零風險：`densify_blind_report=False`（`_accumulate_error_score` 每步不再執行，
   三個消費者現行全為 0，**訓練行為完全不變**）。

**前一代 recipe (validated on BOTH b12 and b7, 2026-08-29):**
`sched30` + **`absgrad_densify 1.0`** — script `scripts/task_absgrad_densify.sh`
```
b12  26.29 -> 26.43   PSNR +6.3sd  SSIM +68.8sd  LPIPS +13.4sd
b7   24.99 -> 25.28   PSNR +10.0sd SSIM +99.6sd  LPIPS +16.7sd
```
- `sched30` = cap 2.6M / `densify_until_iter` 30k / `opacity_reg` **0.002** / `lambda_normal` 0 /
  `depth_loss_weight.init` 0. ⚠ **NOT `opacity_reg=0`** — that was an older result on older data.
- **`absgrad_densify`** steers MCMC's parent sampling by AbsGS's absolute positional gradient:
  `probs = opacity * (1 + w*|g|/mean|g|)`. Requires the rasterizer built with `ABSGRAD 1`
  (`cuda_rasterizer/auxiliary.h`); `|g|` is accumulated into the never-read `dL_dmean2D.z`,
  so renders and gradients are bit-identical. See v1 (`紀錄/new_archived/研究總覽_v1_至2026-09-13.md`) §11.45/§11.47.
- `EXACT_SUPPORT` in the trim rasterizer — drops primitives with `o <= 1/255` from binning.
  Byte-identical, -20.3% render VRAM.
- Zero-weight loss gating (`gs2d_metrics` / `citygsv2_metrics`): skip the normal/dist/depth
  losses when their weight is 0 — bit-identical, **-9%** wall clock. `skip_surf_normal` -1%.
  ⛔ `fused_ssim` was tried and **reverted**: -4.9% time but SSIM is an outlier vs same-config
  repeats (-4.06sd, sample sd +119% when included).

**⚫ Retired lines — do not restart without reading `紀錄/研究總覽.md` §8 first:**
DT-ADMM-GAT (the original proposal), reactive shadow-price λ, radius-based monster detection,
gradient checkpointing, Scaffold/latent-MLP variants. The dual mathematics from the ADMM line
survives, applied to the VRAM constraint.
⚠ **"error-guided densify" was on this list with "measured ceiling only 1.2×" — that was WRONG
and would have blocked the current best recipe.** The 1.2x ceiling belongs to OUR error score
(pooled |render-gt| at the projected centre, **not footprint-weighted**); AbsGS's `|g|` measures
**16.68x** on the same statistic. Signal quality, not the family, was the problem. See §11.30.
Also retired with measurements: `vpc_prune_frac` (-2.34 dB — relocating low-v/c primitives
destroys what they were doing; "change the set" works, "break the set" does not),
`scale_reg=0` (-18sd — the "dust" it makes is a by-product of it holding the scale ceiling),
pure time-compressed proxies (three time scales cannot all be preserved), `notrim2`
(b12 wins, b7 reverses).

**Hard constraints (these are 鐵律, violating them invalidates results):**
- **Pose input (decided by the user 2026-10-09): we use the same pose input as the official CityGaussian pipeline**
  (`sparse/0` = MatrixCity GT poses x1/100). The old "GT-free" wording is retired; claims must say "same pose input as official".
  Method otherwise uses images + those poses + SfM points + pseudo-depth.
  ⚠ **Two claims here were refuted (2026-08-01, memory `data_poses_are_gt`) and are kept only
  as history**: (a) the "~150x Sim3-misaligned official test set" — the official 741 frames are
  100% alignable; (b) "GT is not used at all" — `sparse/0` poses ARE the GT poses x1/100
  (alignment residual 0.0px), so we are already consuming GT poses. **"GT-free" needs re-deciding.**
  Comparison protocol is SfM held-out (`split_mode=experiment`) plus a self-run CityGS baseline;
  the paper's 27.23 is
  context, not a target to claim against. See memory `eval_protocol`.
- **Current data only** (SfM regenerated 2026-05-29; older coarse models are in a different
  coordinate system and must not be used as init).
- Everything in `outputs/` was trained on this one 6GB laptop.

Math in replies uses ASCII (`||x||`, `sum_i`, `theta_k`), never LaTeX — it renders in a terminal.

## Project Overview

CityGaussian is a large-scale scene reconstruction framework using Gaussian Splatting (3DGS/2DGS), built on top of [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning). It implements CityGaussianV2 (ICLR 2025) and CityGaussian (ECCV 2024) — both produce high-quality neural renderings of city-scale outdoor scenes via a divide-and-conquer approach.

## Environment Setup

```bash
conda create -yn gspl python=3.9 pip && conda activate gspl
pip install -r requirements/pyt201_cu118.txt   # match to your CUDA version
pip install -r requirements.txt                 # pulls lightning23.txt
pip install -r requirements/CityGS.txt          # modified Trim2DGS rasterizer
```

For VGGT-X joint pose optimization, additionally:
```bash
pip install -r requirements/gsplat.txt
```

## Commands

**⚠ These are OUR pipeline (depth-init). The upstream CityGS pipeline in the README uses a coarse
global model and different tools; mixing them silently fails.** Full recipes with all flags:
`紀錄/完整指令手冊.md` (rewritten 2026-09-11: §0 daily workflow, §3 the live training line,
§4 the diagnostic tools, §7 the dynamic-strips warning; the retired Mode A/B/C flows are in its
appendix H). The ones used daily are also in `紀錄/README.md` (操作速查).

**Data prep (once per dataset):**
```bash
python utils/estimate_dataset_depths.py data/matrix_city/aerial/train/block_all --image_dir input
python utils/partition_from_colmap.py data/matrix_city/aerial/train/block_all --block_dim 5 5 --content_threshold 0.08 --force
python utils/depth_init_blocks.py data/matrix_city/aerial/train/block_all --block_dim 5 5 --voxel_min 0.03 --voxel_max 0.7 --chunk_size 75
```
- `partition_from_colmap.py` takes `dataset_dir` **positionally** and needs **no coarse
  checkpoint**. Do NOT use `utils/partition_citygs.py` — that is the upstream coarse-init tool and
  will demand `--model_path`.
- `estimated_depths/` is per-image and partition-independent: changing `block_dim` does NOT
  require re-running it (it is the expensive step, 5621 images).
- `depth_init_blocks.py` defaults to `<dataset>/depth_init`; pass `--output_dir` when generating a
  different grid or it silently overwrites the PLYs every current experiment initialises from.

**Train one block** (what almost every experiment actually runs). The live config is
`configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml` — all 37 task scripts use it:
```bash
conda run -n gspl python -u main.py fit --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply --data.parser.block_id 12 --model.density.init_args.cap_max 2600000 -n <run_name>
```
Don't hand-type the recipe flags. Look commands up with `bash scripts/help.sh` / `<script> --help` (generated from the scripts themselves: arm/mode lists, per-arm overrides, env vars); `CITYGS_DRY=1 bash scripts/lab/task_cmp.sh <blk> <arm>` prints the full resolved training command. Base recipe: `CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=60000 CITYGS_FAMILY=g44_ bash scripts/lab/task_cmp.sh <blk> best1` (task_cmp saves only the final ckpt by default since 10-18; `CITYGS_SAVE_ITERS` to keep intermediates).
**Full-scene runs default to adaptive per-block caps (user, 2026-11-02)**: `task_fullscene.sh maskplan <STEPS> <avg cap>` sets each block's cap to
avg x `configs/cap_alloc_g44.json` weight (10-31 price allocation; same Σcap; family `g44s<k>ka<cap/1e5>_`); `MASKPLAN_UNIFORM=1` = old uniform `km`.
Single-block arms (`task_cmp.sh` alone) still use `CITYGS_CAP` as given, so they stay comparable to their baselines.
⚠ **`down_sample_factor: 1.2` 是全域設定，而且一直都是。** 影像以全解析度存在
`input/`，1.2 倍是 **dataparser 載入時**算的（`configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml`
的 `image_dir: input` + `down_sample_factor: 1.2`）；`images_1.2` 只是 `-> input` 的別名
（官方 test 集那份是 4 位數改名層）。⇒ **沒有「1.2 版」與「全解析度版」兩種跑次**，
跑次名不需要、也不該加 `1.2x` 這類後綴（2026-09-13 與使用者確認）。
SfM-init is the same command with `--model.initialize_from null`.
⚠ The previous example here was stale on three counts and is corrected above: it used the
**retired SB config** (`mcmc_2dgs_sb_...`), `cap_max 1000000` (now 2.6M), and
`opacity_reg 0.0` — which contradicts this file's own "**NOT `opacity_reg=0`**" note below.
All blocks: `python utils/train_citygs_partitions.py -n <name> --config <cfg> --init_mode depth
--depth_init_dir <dir> [--blocks 0 6 7]`

**Demo mode (2026-11-14, user):** prefix any training command with `CITYGS_DEMO=1` (handled in `scripts/lab/_common.sh` run_fit): fast SfM reading + cache,
no in-training validation, pre-resized images (`input_r1.2/`, needs `.complete`; `tools/make_resized_images.py`), no logs (no tfevents/lightning_logs/audit/chart pack) —
none of it changes training results. Every run (demo or not) writes `<out>/run_summary.json` (startup / pure training / save / task seconds, final N, peak VRAM);
aggregate with `python tools/run_summary_report.py <run>`. Scores still come from the offline evaluation. Details: `紀錄/實驗分析/38` §7.

**Merge / eval / deploy:**
```bash
python utils/merge_citygs_ckpts.py outputs/$NAME
python main.py test --config outputs/$NAME/config.yaml --save_val      # ⚠ `validate` overwrites results.txt
python tools/cull_dust.py <in.ckpt> <out.ckpt>                          # deployment-side, bit-identical
```

**⚠ Scheduling — use `scripts/runner.sh`, never write another watcher.** One GPU, so every run is
serial and something has to hold the queue. That something already exists:

```bash
setsid nohup bash scripts/runner.sh > /dev/null 2>&1 < /dev/null &   # start (usually already up)
$EDITOR scripts/queue.txt      # one task per line; edit/add/reorder any time, even mid-run
touch scripts/queue.stop       # stop after the current task
tail -f logs/runner.log        # status
```

It re-reads `scripts/queue.txt` before every task, so **only the line currently executing is
fixed**; everything below it can be changed while it works. Long commands go in `scripts/task_*.sh`
and the queue line just calls them (queue lines must not contain `$(...)` or backslashes). A `#`
comment block directly above a task names it in the ledger. Prefix a line with `[cpu]` for work
that never touches the GPU, so it does not queue behind training. `flock` prevents a second runner.

GPU availability is decided by `nvidia-smi` reported VRAM (<1500 MiB = free). **Do not use `pgrep`
for this** — `conda run` rewrites the cmdline so the pattern misses, a hung-but-unreaped process
still matches, and the checking shell's own command line matches itself. All three have cost this
project a stalled queue (2026-07-25, and again 2026-08-06..09 when nine ad-hoc `/tmp` watchers were
written instead of using this file; one sat blocked 2.5 h behind a 151 MiB zombie).

**Run ledger:** every training run appends structured START/DONE/DIED to `logs/quad_progress.log`
from `internal/callbacks.py` (step/N/it-s/VRAM/PSNR, and for DIED the exception plus where it
died). Wrapper scripts must not write those lines themselves.

**⚠⚠ Two-machine sync — lab pulls from GitHub, never file-by-file (2026-09-20, user's decision).**
Before starting any new training experiment, if the code changed at all: **commit + push locally,
then pull on lab.** One command does the whole thing, including deciding whether the rasterizer
needs rebuilding and verifying both sides byte-for-byte:

```bash
JTOK=<token> bash scripts/setup/lab_pull.sh          # 加 --no-build 可跳過重編
```

It refuses to run if the laptop has uncommitted changes or is ahead of `origin/main` — otherwise
lab would be running something that is not what you think you are testing.

Why this replaced `lab.py put`: lab was synced file-by-file for months, so its working tree had
drifted **573 files** from `origin/main` with 57 uncommitted edits and a HEAD three weeks stale —
"which version did lab actually run" had become unanswerable. Lab's pre-switch state is preserved
on the lab-local branch `lab_state_0920` (commit `6a43b2e`).
- lab's machine-local files (`.lab_machine`, `scripts/queue.txt`, `queue.stop`, `queue.bak_*`,
  `.ipynb_checkpoints/`) are now in `.gitignore`, so a reset never touches them.
- **lab's rebuild needs `CUDA_HOME` forced** to `/root/miniconda3/envs/gspl`: the container's
  default points at system nvcc 12.9 while the env's torch is cu118, and torch refuses to compile.
  Lab had in fact **never** been able to rebuild the rasterizer before this was found.
- glm is now vendored in the repo (1.0.3, MIT) and lab uses that same copy, so the two machines
  no longer differ by a glm version (measured: 1 ULP, tile counts identical).

**Rebuilding the trim rasterizer** (`submodules/diff-surfel-rasterization-trim-pp`):
```bash
rm -rf submodules/diff-surfel-rasterization-trim-pp/build          # distutils only checks .cu mtimes, not headers
conda run -n gspl env CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX \
  pip install --no-build-isolation --no-deps --force-reinstall submodules/diff-surfel-rasterization-trim-pp
```
`CUDA_HOME` must be forced: the shell exports `CUDA_PATH=/opt/cuda` (13.3) which torch prefers over
the matching 11.8 in the env, and the build then refuses. `--force-reinstall` is required because
pip silently skips a same-version local path. After rebuilding, verify the `.so` mtime changed —
"no measurable difference" from a rasterizer change almost always means it was not rebuilt.

**Tests:** `python -m unittest discover -s tests -p "*_test.py"` — the `-p` matters. The files are
named `<name>_test.py` (suffix), while unittest's default pattern is `test*.py` (prefix), so the
command without it discovers **zero tests and reports OK**. Current state (2026-09-11):
**86 collected, 81 pass, 5 error** on missing optional deps (`tinycudann`, `gsplat._torch_impl`,
the deformable-model tests) — all 5 are upstream tests, none are our code.
(The older "55 pass / 7 error" note was stale.)

## Architecture

### Training Entry Point
`main.py` → `internal/entrypoints/gspl.py:cli()` → PyTorch Lightning `CLI` wiring `GaussianSplatting` (LightningModule) with `DataModule`.

### Core Module: `internal/gaussian_splatting.py`
`GaussianSplatting(LightningModule)` composes four swappable components, all configured via YAML:
- **`gaussian`** (`internal/models/`) — Gaussian model (stores means, opacities, SH coefficients, scales, rotations)
- **`renderer`** (`internal/renderers/`) — Rasterization pipeline
- **`density`** (`internal/density_controllers/`) — Controls clone/split/prune (Adaptive Density Control)
- **`metric`** (`internal/metrics/`) — Loss functions (photometric + depth regularization)

### CityGSV2 Component Stack
- Model: `internal/models/gaussian_2d.py:Gaussian2D` (2D Gaussian Surfel)
- Renderer: `internal/renderers/sep_depth_trim_2dgs_renderer.py:SepDepthTrim2DGSRenderer`
- Density Controller: `internal/density_controllers/citygsv2_density_controller.py:CityGSV2DensityController`
- Metric: `internal/metrics/citygsv2_metrics.py:CityGSV2Metrics` (normal + depth loss)

For 3DGS (CityGSV1), swap to `VanillaGaussian`, `VanillaRenderer`, `VanillaDensityController`, `VanillaMetrics`.

SB-color variant (DBS port, 2026-07-17): swap model to `internal/models/gaussian_2d_sb.py:Gaussian2DSB` and renderer to `internal/renderers/sep_depth_trim_2dgs_sb_renderer.py:SepDepthTrim2DGSSBRenderer` (config: `configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml`). Color = SH0 DC + Spherical Beta lobes via `colors_precomp` — 25 floats/point vs 59 (SH3). Per-block cap calibration: `tools/calibrate_block_caps.py` (see `紀錄/完整指令手冊.md` §6;
the SB axis itself is retired — appendix S).

### Config System
All components are specified in YAML via `class_path` + `init_args` using jsonargparse. Example configs for each scene are in `configs/citygsv2_*.yaml`. CLI args can override any config key at runtime (e.g., `--data.path`, `--model.density.init_args.cap_max`).

### Data Pipeline
- `internal/dataparsers/` — Parsers for COLMAP, Blender, MatrixCity, etc. `EstimatedDepthBlockColmap` is the standard parser for CityGS scenes (reads depth maps generated by Depth Anything V2).
- `internal/dataset.py` — `DataModule` wraps the dataparser output.
- Data must be in COLMAP format under `data/your_scene/` with `images/` and `sparse/0/`.

### CityGaussian Partitioning
`internal/utils/citygs_partitioning_utils.py:CityGSPartitioning` divides the scene into an `N×M` grid of blocks. Each block is fine-tuned independently using `utils/train_citygs_partitions.py`, which spawns separate `main.py fit` processes. Results are merged by `utils/merge_citygs_ckpts.py`.

### Outputs
Training writes to `outputs/$NAME/`:
- `checkpoints/epoch=X-step=Y.ckpt` — PyTorch Lightning checkpoint
- `checkpoints/epoch=X-step=Y-xyz_rgb.ply` — Gaussian PLY export
- `lightning_logs/version_N/config.yaml` — Full resolved config
- `blocks/block_*/` — Per-block results when using partitioned training

## Key Facts for Development

- **SH degree**: Coarse models use `sh_degree: 0` (SH2 in naming = spherical harmonic degree 2 during fine-tuning). Full quality requires `sh_degree: 3` with `diable_trimming: true`.
- **Depth maps**: Must be pre-generated with `utils/estimate_dataset_depths.py` (uses Depth Anything V2 at `utils/Depth-Anything-V2/`).
- **Memory issues**: Reduce `max_cache_num` (default 1024 in train scripts), increase `prune_ratio`, or downsample images.
- **Block training failures**: If most blocks are skipped ("data too few"), the `aabb` setting in the partition config is likely wrong.
- **`depth_ratio`**: Controls precision vs. recall tradeoff for street-view mesh. `1.0` = better recall/complete roads; `0.0` = better precision.
- The `data` symlink points to `/home/LnoArch/Projects/專題/CityGS-X/data` (shared data directory).
- The `depth_anything_v2` symlink points to `/home/LnoArch/Projects/專題/Depth-Anything-V2/depth_anything_v2`.
