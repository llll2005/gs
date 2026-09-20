#!/bin/bash
# ★★★★★ 新年代的「時間花在哪／VRAM 花在哪」（真實訓練迴圈逐段；使用者 2026-09-17 要求在本機跑、不等 lab）
#
# 與 task_stepcost.sh（09-12，舊資料 b12）同一套量法（_StepProfiler，每個標記點同步），改成：
#   ① 新年代 b6（配對修正後資料）＋現行 speed3 旗標
#   ② 逐段 VRAM：段內峰值配置、段末保留（09-17 新增）=> 回答「乾淨起點 reserved 3.22 GB vs 訓練中 5.6~5.7 GB」
#      那約 2.4 GB 的缺口出在哪一段
#   ③ 補上原本沒被計時的「end_step 之後、trim 之前的 hook」（第 12 段）
#   ④ 高顆數段在 max_split_size_mb:128 關／開各跑一次 => 第一次在**真實迴圈**上量它的時間與 VRAM 代價
#      （09-11 只有微基準的 6.4%；09-12 使用者決定預設不開；09-13 lab 在 5.66 GB 上限 OOM 後 lab 腳本改成一律開）
# 兩段顆數（與 09-12 同設計）：
#   lo  SfM init 從頭 2,000 步（含起始 trim、step 1,000/1,500/2,000 的 trim、densify）
#   hi  從新年代 b6 ckpt 起跑 1,200 步（densify_until 30,000 仍在增生期 => step 1,000 的 trim 與 densify 都在視窗內）
# ⚠ 這三個跑次的台帳 VRAM 不可引用（profiler 每段 reset 峰值統計）；看 step_cost.txt
# ⚠ 不設 CITYGS_VRAM_CAP_GB：本機原生 6 GB；之後換 STEPCOST_CKPT=lab/cs_base 的 ckpt、STEPCOST_TAG=lab 在 lab 跑做跨機器對照
set -u
cd "$(dirname "$0")/.." || exit 1
export CITYGS_STEP_PROFILE=1
# ⚠ lab 執行期要鎖 arch：gsplat 第一次 densify 才 JIT 編譯，容器的 TORCH_CUDA_ARCH_LIST 含 10.0 會當場死
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
unset CITYGS_VRAM_CAP_GB   # 不鎖：量時間要避開上限干擾（時間平到撞牆為止）；兩台都不鎖才可比
# 命名：lab 產物收在 outputs/lab/（與 _common.sh 同判準）
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
CK=${STEPCOST_CKPT:-outputs/cs_refrep/blocks/block_6/checkpoints/epoch=40-step=21920.ckpt}
TAG=${STEPCOST_TAG:-local}
[ -f "$CK" ] || { echo "⛔ 找不到 $CK"; exit 2; }
R=logs/stepcost2_${TAG}_$(date +%m%d_%H%M).log
COMMON=(--config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6
  --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter 30000
  --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true
  --model.density.init_args.noise_gate_eps 0.001 --model.metric.init_args.opacity_reg 0.002
  --model.metric.init_args.lambda_normal 0.0 --model.metric.init_args.depth_loss_weight.init 0.0)
run1 () {   # run1 <跑次名> <alloc conf 或空字串> [參數...]
  local name=$1 ac=$2; shift 2
  [ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.aborted_$(date +%m%d_%H%M%S)"
  echo "##### $name（PYTORCH_CUDA_ALLOC_CONF=[${ac:-未設}]）$(date) #####"
  ( if [ -n "$ac" ]; then export PYTORCH_CUDA_ALLOC_CONF="$ac"; else unset PYTORCH_CUDA_ALLOC_CONF; fi
    conda run -n gspl --no-capture-output python -u main.py fit "${COMMON[@]}" -n "$name" "$@" ) 2>&1 \
    | grep -vE "^Epoch|it/s\]|^ *$|pkg_resources|declare_namespace|caching images|depth scale"
  return "${PIPESTATUS[0]}"
}
{ bad=0
  run1 "${PFX}stepcost2_${TAG}_lo" "" --model.initialize_from null --trainer.max_steps 2000 || bad=1
  run1 "${PFX}stepcost2_${TAG}_hi" "" --model.initialize_from "$CK" --trainer.max_steps 1200 || bad=1
  run1 "${PFX}stepcost2_${TAG}_hi_ms128" "max_split_size_mb:128" --model.initialize_from "$CK" --trainer.max_steps 1200 || bad=1
  exit "$bad"; } 2>&1 | tee "$R"
st=${PIPESTATUS[0]}
for n in lo hi hi_ms128; do
  echo; echo "===== stepcost2_${TAG}_$n ====="
  cat "outputs/${PFX}stepcost2_${TAG}_$n/blocks/block_6/step_cost.txt" 2>/dev/null || echo "⛔ 沒有 step_cost.txt"
done | tee -a "$R"
echo "報告：$R"
exit "$st"
