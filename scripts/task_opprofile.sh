#!/bin/bash
# ★★★★★ 逐 op profile @高顆數：找「逐顆數成本」的浪費（約 5 分鐘）
#
# 為什麼是現在（2026-08-28）：wall-time 對照顯示我到目前為止砍的全是**逐像素固定成本**
#   step  4,000  fast/rep = 0.856   （顆數少 => 固定成本佔比大 => 省 14%）
#   step 23,700  fast/rep = 0.938   （顆數多 => 只省 6%）
# 成本模型扣掉已省的 67ms 後：固定 ~112ms(32%) / **逐顆數 ~235ms(68%)**
# => 下一個戰場是逐顆數那塊，而 Lightning 的 simple profiler 看不進 training_step。
#
# 預先讀碼點出的候選（等 profile 確認再動）：
#   1. `_add_xyz_noise` **每步**建 N x 3x3 共變異矩陣（2.34M 顆 => 單一張量就 84MB，
#      加上中間張量是每步數百 MB 的記憶體流量）。而 2DGS 的噪音**只該待在切平面**
#      （第三軸補 0）=> 建完整 3x3 再乘是繞遠路，可在 2D 切基底直接做。
#   2. Adam 狀態 M=4（params+grad+2 moments）x F=58 floats/pt => 模型狀態主導高顆數區。
#      候選：fused/foreach Adam、8-bit Adam、Adam-offload（7/24 報告估 M:4->2 => N_max 6.5M）。
#   3. SH degree 3 = 48 個色彩 float/pt。空拍內容漫反射為主，SH2=27 / SH1=12。
#      ⚠ SB(25 floats) 測過輸 0.46dB，但那是**換基底**；降 SH 階是**同基底減階**，不同事。
#
# ⚠ 2026-08-28：`--trainer.profiler pytorch` 跑完 40/40 步後死在 torch.save
#   （`TypeError: cannot pickle 'weakref'` —— profiler 掛的 hook 帶 weakref），
#   而 Lightning 是在 teardown 才印摘要 => **資料全丟**。改用 `advanced`（cProfile，
#   Python 層、不掛 CUDA hook）並關掉 checkpointing。`simple` 先前跑得好好的，同族。
# ⚠ 用 ckpt init 才能在 5 分鐘內到 2.34M 顆。已知副作用：renderer 會被 ckpt 的取代
#   （_ctx 陷阱 1）=> 本任務只看**逐顆數的 op**（optimizer/noise/rasterizer），
#   那些不受 renderer 旗標影響；逐像素的部分不從這裡讀。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
OUT=logs/opprofile_$(date +%m%d_%H%M).txt
rm -rf outputs/opprofile
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from "outputs/sched30_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt" \
  --data.parser.block_id 12 \
  --trainer.profiler advanced \
  --trainer.enable_checkpointing false \
  --trainer.max_steps 40 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 40 \
  --model.density.init_args.cap_max 2600000 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n opprofile 2>&1 | tee "$OUT"
echo; echo "===== 逐函式耗時（cProfile，取 training_step 段）====="
grep -a -A30 "Profile stats for: \[Strategy\]SingleDeviceStrategy.training_step" "$OUT" | head -34
