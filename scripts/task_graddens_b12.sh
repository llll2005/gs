#!/bin/bash
# ★★★★★ 乾淨的方法比較：grad-densify（CityGSV2 的密度控制）vs MCMC，同資料同硬體
#
# 記憶 `citygsv2_baseline_repro` 的「MCMC +1.457 勝 grad-densify」跑於 2026-06-26，
# 而 GT 錯位 2026-08-12 才修 => **作廢**（且該次 coarse 另被 viscrop 裁切、densify config 選錯）。
# ⇒ **當前資料上，我方從未有過任何有效的 grad-densify 數字。**
#
# ★ 為何這次不會 OOM：當初的 OOM 來自載入**全域 coarse 4.39M** 後 step-1 backward 爆掉。
#   本臂用**我方 depth-init PLY（約 33.8 萬顆）**起步，族群自己長大，不需一次吃下 4.39M。
#
# 只換兩個元件（且它們是一整包）：density + metric。其餘（sh3 / renderer+trim / 5x5 0.08 /
# down_sample 1.2 / 60k 步 / LR 排程 / depth-init）與現行最佳逐項相同。
#   ⚠ 拿掉 opacity_reg / scale_reg 是**必要的**：那兩個 L1 存在的目的是餵 MCMC 的死亡通道，
#     而 grad-densify 用 opacity reset，兩者是替代關係。
#   ⚠ 我方機制（absgrad_densify / cap_max / screen_size_prune）**本質上是對 MCMC 的改造**，
#     在 grad-densify 上沒有對應物 => 本臂是「兩個密度控制套件」的比較，不是逐旗標比較。
#
# ⚠ **無 cap**（官方如此）=> 族群大小由方法自己決定，這正是要量的東西之一；但有 OOM 風險。
# 判準：對照 agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%），另看終點顆數與 VRAM。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/graddens_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/graddensify_60k_sh3_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  -n graddens_b12
