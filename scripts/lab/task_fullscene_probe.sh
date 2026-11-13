#!/bin/bash
# ★ 2026-11-13 使用者「我們是否有 CityGS full blk merge 的 F1 跟各項數據？如果沒有補量測」：
#   全場景版的各項量測＝16 塊逐塊跑 eval_arms_lab.sh 的 CPU 項（地面／地下層、立面、朝向帶、海面、路燈與車），每塊都帶 CityGS（自動）與我方的全場景跑次，
#   最後 tools/probe/fullscene_probe_agg.py 彙總成一張表（可加總的量加總、比例用逐塊加權）。全場景 F1 另見 task_geom_f1.sh（GPU、[solo]）。
# 用法：bash scripts/lab/task_fullscene_probe.sh [我方名稱=跑次 ...]   （預設 A＋dist4＝g44s30ksfm10fssh16di4_best1maskdistrfed10b4）
#   只用 CPU；與訓練同時跑時用 nice 19、執行緒 4，避免拖慢訓練的計時。輸出 logs/eval_arms_fullprobe_b<B>.log、logs/fullscene_probe_<日期>.log
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac
cd "$(dirname "$0")/../.." || exit 1
M=("$@"); [ ${#M[@]} -gt 0 ] || M=(A+dist4=g44s30ksfm10fssh16di4_best1maskdistrfed10b4)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for B in $(seq 0 15); do
  echo "== b$B $(date -u +%H:%M)"
  EVAL_ONLY=ground,facade,orient,sea,smallobj nice -n 19 bash scripts/lab/eval_arms_lab.sh "$B" fullprobe_b$B "${M[@]}" > /dev/null 2>&1
  tail -n 1 logs/eval_arms_fullprobe_b$B.log
done
conda run -n gspl --no-capture-output python -u tools/probe/fullscene_probe_agg.py logs/eval_arms_fullprobe_b{0..15}.log | tee logs/fullscene_probe_$(date -u +%m%d).log
echo DONE
