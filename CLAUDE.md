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

**Core contribution: cost-aware density control.** Existing density control prices primitives by
COUNT (`cap_max`, opacity L1, scale L1) and is blind to what they actually cost to render. We
price them by render cost and solve a resource-constrained problem:

    max Q(θ_S)  s.t.  (1/K)·Σ_{i∈S} c_i ≤ B(hardware)

with `c_i` = screen-tile footprint (VRAM-grounded), `λ` = shadow price, `K` = strip tiling as a
supply-side lever. Every prior method surveyed is the `c_i ≡ 1` special case; see
`紀錄/研究總覽.md` §3 (theory) and §8 (12 papers audited implementation-vs-text).
⚠ The claim "every prior method is the `c_i ≡ 1` special case" does NOT hold for RAIN-GS
(arXiv 2403.09413), whose `s = HW/(9πN)` is a per-primitive screen budget — see §3.1 for the
precise distinction that survives.

**Current best single-block recipe (b12, MatrixCity aerial):**
- `opacity_reg=0` — MCMC's opacity L1 exists to feed relocation with dead points, and our
  depth-init already solves the initialisation problem relocation is there to fix. Removing it:
  **+1.04 dB, 10.9× faster, 3G less VRAM** (22.117 → 23.158). It also turned out that the
  "two-phase dynamics" and the "87% zombie" economy were artifacts of that L1, not intrinsic.
- `EXACT_SUPPORT` in the trim rasterizer — drops primitives with `o ≤ 1/255` from binning.
  Byte-identical (the blend loop already discards `alpha < 1/255` in both passes), −20.3% render
  VRAM, training ceiling 2.0M → 2.5M primitives.
- SB color (`Gaussian2DSB`, F=25 vs SH3's 59). Costs −0.46 dB in this diffuse aerial content;
  see `紀錄/研究總覽.md` §8 for why (DBS never evaluates SB on a Gaussian kernel).

**⚫ Retired lines — do not restart without reading `紀錄/研究總覽.md` §7 first:**
DT-ADMM-GAT (the original proposal), reactive shadow-price λ, radius-based monster detection,
gradient checkpointing, Scaffold/latent-MLP variants, error-guided densify (measured ceiling only
1.2×). The dual mathematics from the ADMM line survives, applied to the VRAM constraint.

**Hard constraints (these are 鐵律, violating them invalidates results):**
- **GT-free**: method uses images + SfM + pseudo-depth only. MatrixCity GT is not used at all —
  the official test set is ~150× Sim3-misaligned with our COLMAP frame. Comparison protocol is
  SfM held-out (`split_mode=experiment`) plus a self-run CityGS baseline; the paper's 27.23 is
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
command without it discovers **zero tests and reports OK**. Current state: 10 pass, 5 error on
missing optional deps (`tinycudann`, and `gsplat._torch_impl` which the installed gsplat no longer
exposes) — none are our code.

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
