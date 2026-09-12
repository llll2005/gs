#!/bin/bash
# 官方 CityGaussianV2 復現：**用 cityGS_origin 的未修改原始碼與原版 config**。
#
# 為什麼不用我方 repo 的 configs/citygsv2_*.yaml：記憶 citygs_official_configs 記著
# 「本機那些已被改過，不可當官方依據」。要復現就得用官方那份。
# 而記憶 citygsv2_baseline_repro 記著：**官方忠實 config 在 6GB 一定 OOM**
# => 當前資料上從來沒有過一次忠實的 V2 基準。3090 跑得起來，所以這裡**不鎖 6GB**
#    （這是唯一該例外的跑次：它的目的是「官方數字長什麼樣」，不是 6GB 可行性）。
#
# 官方流程（scripts/citygs/run_citygs_mc_aerial.sh）第 1 步：全域 coarse、sh2、30k 步。
set -u
# ★ 明確**不鎖** VRAM（使用者 2026-09-13 確認）：這支要答的是「官方數字長什麼樣」，
#   不是 6GB 可行性。萬一環境裡有殘留就清掉，並印出來確認。
unset CITYGS_VRAM_CAP_GB
echo "CITYGS_VRAM_CAP_GB=[${CITYGS_VRAM_CAP_GB:-未設 => 不限制}]"
ORIG=/workspace/data/hdd/11213/cityGS_origin
GS=/workspace/data/hdd/11213/gs
[ -d "$ORIG" ] || { echo "❌ 找不到 $ORIG"; exit 2; }
cd "$ORIG" || exit 1
# 資料共用（不複製 21GB）
[ -e data ] || ln -s "$GS/data" data
ls -la data/matrix_city/aerial/train/block_all/input >/dev/null 2>&1 || { echo "❌ 資料連結不到"; exit 2; }
echo "=== 官方 coarse（citygsv2_mc_aerial_coarse_sh2，全域 sh2 30k）$(date) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/citygsv2_mc_aerial_coarse_sh2.yaml \
  -n citygsv2_mc_aerial_coarse_sh2 \
  --data.train_max_num_images_to_cache 1024
RC=$?
echo "=== coarse 結束 rc=$RC $(date) ==="
[ "$RC" -ne 0 ] && exit "$RC"
echo "=== 官方 held-out 評測（test/block_all_test 全部）==="
conda run -n gspl --no-capture-output python -u main.py test \
  --config outputs/citygsv2_mc_aerial_coarse_sh2/config.yaml \
  --data.path data/matrix_city/aerial/test/block_all_test \
  --data.parser.eval_image_select_mode ratio \
  --data.parser.eval_ratio 1.0 \
  --save_val 2>&1 | tail -20
