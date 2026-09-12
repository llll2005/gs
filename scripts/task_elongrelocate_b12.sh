#!/bin/bash
# ★★★★ Elongation Filter 路徑 (b)：**併入 MCMC 死亡判準、由 relocate 搬移** —— 使用者指定
#
# 與 (a) 唯一的差別：長寬比超標的粒子 OR 進 `dead_mask` => 走 relocation（搬到抽中的宿主）
# 而不是 `_prune_points`（刪除）。同一個門檻 20.0，方便與 (a) 對照。
#
# ⚠⚠ **這條路有既有反證**：`vpc_prune_frac` 用的就是同一個機制（OR 進 dead_mask），
#   實測 **-2.34 dB**（§11.34）。當時結論：「把低 v/c 的粒子搬走會**破壞它們原本在做的事**；
#   『改變集合』有效、『破壞集合』無效」。
#   => 先驗上不看好，但使用者 2026-09-12 決定兩路都測，報告時必須帶上這個先驗。
# ⚠ 目標與門檻的依據同 task_elongprune_b12.sh（binning 外接盒浪費 76.5%）。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/elongrelocate_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 1e-3 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --model.density.init_args.elongation_relocate 20.0 \
  -n elongrelocate_b12
