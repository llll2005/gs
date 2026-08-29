#!/bin/bash
# ★★★★★ 退化定理的前提檢驗（~8 分鐘，只讀不剪枝）—— 使用者提問帶出來的
#
# 使用者問：「trim 剪掉最便宜的點所以沒省資源，那把成本感知演算法加回來呢？」
#
# ⚠ 我一度用「退化定理」擋掉這條線，**那是錯的引用**。原始檔案叫
#   `archive/S11_成本模型_bug下量測_作廢.md`，它自己寫著：
#     「解析論證（v_i = kappa*c_i => 選擇無自由度）本身不依賴資料，可能仍成立，
#       但支持它的**三個實測已作廢**（bug 期量的）」
#   => **前提 `v_i ≈ kappa*c_i` 從未在當前資料上驗證過。** 定理是條件句，antecedent 沒驗。
#
# 而 §11.23 反而指向相反方向：trim 按 v 剪最低 10%，卻**省不到 VRAM**
# （notrim2 多 11% 顆粒、VRAM 反而低 0.06G）=> 低 v 的正好是低 c 的
# => 按 `v/c` 剪才會剪到「拿得多、給得少」的那批（怪物/霧），
#    而省下的 VRAM 可以換更高的 cap（顆數值 +0.541 dB/加倍）。
#
# 在**現行最佳的成品模型**上量（notrim2_b12 的 60k ckpt），因為 v/c 的分布只有在
# 成熟族群上才有意義；從頭跑 600 步量到的是未成熟的族群。
# ⚠ ckpt 路徑 init 會把 renderer 整個換掉（_ctx 陷阱 1）—— 這裡**正好是我們要的**
#   （notrim2 的 renderer 就是 diable_trimming=true），且本任務只讀不訓練。
#
# 決定性數字：`bottom-10% by v/c` 相對 `bottom-10% by v` **多省下幾倍成本**
#   ~1x  => 退化成立，選子集沒有自由度 => 這條線收掉，出口只剩「改變集合」（densify 端）
#   >>1x => 有自由度 => 排 `notrim2 + vpc_prune_frac` 一臂；省下的 VRAM 再換 cap
# 同時看「價值代價」：若成本省 5x 但價值也多損 5x，那只是沿著同一條線移動，不是自由度。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CK=$(find outputs/notrim2_b12 -name "*.ckpt" | sed 's/.*step=\([0-9]*\)\.ckpt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
[ -z "$CK" ] && { echo "!! 找不到 notrim2_b12 的 ckpt"; exit 1; }
echo "用 ckpt: $CK"
rm -rf outputs/vpcreport
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from "$CK" \
  --data.parser.block_id 12 \
  --trainer.max_steps 400 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 400 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.vpc_report 200 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n vpcreport 2>&1 | tee logs/vpcreport.txt | grep -E "\[vpc\]|value-per-cost|DIED|Error"

echo
echo "================ 開獎 ================"
grep -a -A3 "\[vpc\] step" logs/vpcreport.txt || echo "(無輸出 —— 檢查 screen_size_prune_px 與 radii 視窗)"
