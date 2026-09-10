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

**Authority for everything below: `紀錄/研究總覽.md` — the single source of truth (proposition,
current numbers, theory, measurement rules, refuted hypotheses, traps). `紀錄/_ctx.md` is a compact
primer that costs far fewer tokens; read it first, then the relevant section of 研究總覽.
Everything else under `紀錄/` is in `archive/` and is history, not current state.
Read those before proposing mechanisms — this section is a summary and goes stale.**

**⛔⛔ 2026-09-07 定案：命題的「取樣端」已被否證。** 三個變體（符號 ±／冪次+加法／
代理+精確 c／強度未校準+校準）全輸，29 小時 GPU；**基準（無成本項）仍是最佳**
（agd2 26.5602 vs cadd 26.3719 / fpd 26.4260 / cheap 26.2398）。
共同形狀「紋理比↑ 而 PSNR/SSIM/LPIPS↓」且**不是塵埃**（floater 方向不一致）⇒ **虛假細節**。
與 absgrad 互補：**往「誤差在的地方」增生有效（|g|），往「便宜/貴的地方」增生無效（c）**。
⚠ 且 `scale_reg` 是 **r=0.977** 的成本代理（空拍深度範圍窄）⇒
  「每個先前方法都是 `c_i ≡ 1` 的特例」**對 scale_reg 不成立**。
❓ 只剩**約束端**未測：`cost_budget` 取代 `cap_max` ＝ `max Q s.t. (1/K)Σc_i ≤ B` 的字面形式。
**引用以下任何成本感知的正面敘述前，先讀記憶 `cost_aware_sampling_refuted`。**

**Core contribution (原始提法，取樣端已否證): cost-aware density control.** Existing density control prices primitives by
COUNT (`cap_max`, opacity L1, scale L1) and is blind to what they actually cost to render. We
price them by render cost and solve a resource-constrained problem:

    max Q(θ_S)  s.t.  (1/K)·Σ_{i∈S} c_i ≤ B(hardware)

with `c_i` = screen-tile footprint (VRAM-grounded), `λ` = shadow price, `K` = strip tiling as a
supply-side lever. Every prior method surveyed is the `c_i ≡ 1` special case; see
`紀錄/研究總覽.md` §3 (theory) and §8 (12 papers audited implementation-vs-text).
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
  so renders and gradients are bit-identical. See `紀錄/研究總覽.md` §11.45/§11.47.
- `EXACT_SUPPORT` in the trim rasterizer — drops primitives with `o <= 1/255` from binning.
  Byte-identical, -20.3% render VRAM.
- Zero-weight loss gating (`gs2d_metrics` / `citygsv2_metrics`): skip the normal/dist/depth
  losses when their weight is 0 — bit-identical, **-9%** wall clock. `skip_surf_normal` -1%.
  ⛔ `fused_ssim` was tried and **reverted**: -4.9% time but SSIM is an outlier vs same-config
  repeats (-4.06sd, sample sd +119% when included).

**⚫ Retired lines — do not restart without reading `紀錄/研究總覽.md` §7 first:**
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
- **GT-free**: method uses images + SfM + pseudo-depth only.
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
`紀錄/完整指令手冊.md`; the ones used daily are in `紀錄/_ctx.md`. Verified 2026-07-29.

**Data prep (once per dataset):**
```bash
python utils/estimate_dataset_depths.py data/matrix_city/aerial/train/block_all --encoder vitl
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

**Train one block** (what almost every experiment actually runs):
```bash
python -u main.py fit --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 1000000 \
  --model.metric.init_args.opacity_reg 0.0 \
  -n <run_name>
```
All blocks: `python utils/train_citygs_partitions.py -n <name> --config <cfg> --init_mode depth
--depth_init_dir <dir> [--blocks 0 6 7]`

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
command without it discovers **zero tests and reports OK**. Current state: 55 pass, 7 error on
missing optional deps (`tinycudann`, and `gsplat._torch_impl` which the installed gsplat no longer
exposes) — all 7 are upstream tests, none are our code.

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

SB-color variant (DBS port, 2026-07-17): swap model to `internal/models/gaussian_2d_sb.py:Gaussian2DSB` and renderer to `internal/renderers/sep_depth_trim_2dgs_sb_renderer.py:SepDepthTrim2DGSSBRenderer` (config: `configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml`). Color = SH0 DC + Spherical Beta lobes via `colors_precomp` — 25 floats/point vs 59 (SH3). Per-block cap calibration: `tools/calibrate_block_caps.py` (see `紀錄/完整指令手冊.md` §9).

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
