#!/bin/bash
# ★★★★★ 收割期 opacity 動力學的小實驗（兩臂各約 20 分鐘）—— 取代原本排的 9.6h 跑次
#
# 使用者質疑「這些不能先用小實驗驗嗎」——對，而且免費的 CPU 量測已經先削弱了我的假說：
#   收割期（30k->60k）**不是**「L1 把大家壓死」，而是**兩極化**：
#     o 中位 0.1083 -> 0.1539（升）／o>0.5 佔比 9.12% -> 23.22%（翻倍）
#     中間帶 0.005~0.05 從 22.26% 排乾到 8.99%／判死 3.77% -> 15.03%
#   ⇒ 往「確信」的遷移很可能就是收割期 +1 dB 的來源
#   ⇒ 原本排的 `freeze_opacity_after_densify`（凍結全部 opacity 梯度）會把它一起關掉，
#     大概率會輸，而且輸的理由會被誤讀成「回收缺口不重要」。已從佇列移除。
#
# 本實驗改測更外科的介入：**只移除 L1 下壓力，保留光度梯度** => 兩極化應該保留。
#   臂 A（對照）：opacity_reg 0.002 全程          = 現行行為
#   臂 B（介入）：opacity_reg 在 30k 之後歸零
# 都從 agd2_b12 的 **step=29999** ckpt 續跑 2,500 步（收割期的起點）。
#
# 判準（看**速率**，不看終點分數 —— 2,500 步不足以移動 PSNR）：
#   判死% 的上升速率：A 應約 +0.5%/1000 步（由 30k->42k 推得）；B 若明顯變緩 => L1 是主因
#   o>0.5 的上升速率：A 約 +0.86%/1000 步；**B 若也保持** => 兩極化沒被破壞（關鍵）
#   兩者同時成立 => 才值得投 9.6h 做完整跑次
#   若 B 的 o>0.5 也跟著變緩 => L1 其實是兩極化的驅動力之一 => 整條線收掉
#
# ✅ ckpt-init 只替換 gaussian_model 與 renderer，**metric 不受影響** => 旗標會生效。
#
# ⛔⛔ 2026-09-02 修正：第一版每臂跑了 **6.8 小時**而非預估的 20 分鐘（20 倍）。
# 真因：`--model.initialize_from <ckpt>` **只載權重、不接續 global_step**
#      （接續是 `--ckpt_path` 的事）⇒ `--trainer.max_steps 32500` 是「從 0 跑 32,500 步」。
# 連帶**階段邏輯全錯**：global_step 從 0 起算 ⇒ `densify_until_iter 30000` 讓 densify
#      重跑 30,000 步，族群一路長回 cap（也是速率從 2.1 掉到 1.3 it/s 的原因），
#      而三個旗標都只在最後 2,500 步才生效。
# ✅ 正確寫法（本版）：`max_steps 2500` + `densify_until_iter 0`
#      ⇒ 從第一步就在收割期，2,500 步全部是有效的介入期。
#      `opacity_reg_until_iter` 也跟著改成 0。
# ⚠ 一般化：**用 initialize_from 時，所有「按 step 排程」的參數都要按新的 0 起點重寫**。
#
# 逐項稽核（2026-09-02，使用者要求「要確保正確」）：
#   ✅ densify_until_iter  30000 -> **0**       從第一步就在收割期（trim 也隨之停，與真實收割期一致）
#   ✅ opacity_reg_until   30000 -> **0**       B 臂的介入從第一步生效
#   ⛔ means_lr            **真 bug，已修**
#        排程 lr = lr_init * (lr_final/lr_init)^(step/max_steps)（schedulers.py:82）
#        真實 LR(30000) = 6.4e-5 * 0.01^0.5 = **6.4e-6**
#        修正前 LR(0)   = 6.4e-5  => **10 倍**，means 會亂跑，漂移量測全毀
#        改為 lr_init 6.4e-6 + max_steps 30000（lr_final 不變）
#        => LR(s) = 6.4e-6 * 0.1^(s/30000) == 6.4e-5 * 0.01^((30000+s)/60000)  **逐點相等**
#   ✅ sh_degree           不用改：`_active_sh_degree` 是 persistent buffer，
#                          ckpt 內實測 = 3，載入即還原（歸零只發生在 setup_from_pcd/number）
#   ✅ depth_loss_weight   配方已設 init 0.0 => 排程無作用
#   ✅ scales/rotations/opacities lr  無排程，常數
#   ✅ 收尾 ckpt           Lightning 在 max_steps 結束時會存（A/B 臂實測有 step=32500.ckpt）
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
CKPT="outputs/agd2_b12/blocks/block_12/checkpoints/epoch=105-step=29999.ckpt"
[ -f "$CKPT" ] || { echo "找不到 30k ckpt: $CKPT"; exit 1; }

run_arm () {   # $1 = 臂名  $2 = opacity_reg_until_iter  $3 = harvest_relocate
  rm -rf "outputs/$1"
  conda run -n gspl --no-capture-output python -u main.py fit \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
    --model.initialize_from "$CKPT" \
    --data.parser.block_id 12 \
    --trainer.max_steps 2500 \
    --model.gaussian.init_args.optimization.means_lr_init 0.0000064 \
    --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 30000 \
    --model.density.init_args.cap_max 2600000 \
    --model.density.init_args.absgrad_densify 2.0 \
    --model.density.init_args.densify_until_iter 0 \
    --model.density.init_args.screen_size_prune_px 300 \
    --model.metric.init_args.opacity_reg 0.002 \
    --model.metric.init_args.opacity_reg_until_iter "$2" \
    --model.density.init_args.harvest_relocate "$3" \
    --model.metric.init_args.lambda_normal 0.0 \
    --model.metric.init_args.depth_loss_weight.init 0.0 \
    -n "$1"
}
run_arm oregprobe_a -1  false   # 對照：現行行為（L1 全程、收割期不回收）
run_arm oregprobe_b 0   false   # 介入1：只移除 L1 下壓力（step>=0 即生效）
run_arm oregprobe_c -1  true    # 介入2：L1 照舊，但收割期繼續回收死粒子（使用者提案）
#   ⚠ c 臂是「同一個操作、不同目的地」：notrim2 的純 churn 實測淨負，但那時 probs=opacity
#     （盲目）；現在是 o*(1+2|g|/mean|g|) => 送到梯度說缺細節的地方。
#   ⚠ c 臂的成本在目的地：relocate 會 reset 宿主 Adam 動量，而收割期宿主正在收斂外觀。
#     強度：每次事件約搬族群的 0.056%（1,350 顆），遠低於 notrim2 的 churn 密度。

echo; echo "===== 開獎 ====="
CUDA_VISIBLE_DEVICES="" python tools/opacity_drift.py \
  "agd2_b12@29999:起點30k" oregprobe_a:A對照 oregprobe_b:B移除L1 oregprobe_c:C收割回收
