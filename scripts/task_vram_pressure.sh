#!/bin/bash
# ★★★★★ 逼近 VRAM 上限時變慢，是配置器在搬還是計算變多？（教授 2026-09-11 提的實驗，~35 分）
#
# 已知（tools/vram_gap.py 2026-09-04，N=2.34M）：
#   allocated 峰值 2.42 GB / reserved 4.87 GB => 缺口 2.45 GB **全是配置器碎片**
#   階段峰值 forward+backward 1.901 > trim 1.239 ~ forward 1.222 > optimizer 1.012
#   台帳的 VRAM=5.57/6.1G 是 memory_reserved()，含約 40% 碎片
# 沒量過的是**代價**：餘裕變小之後每步慢多少、慢在哪。
#
# 設計把「計算量」與「記憶體壓力」分開：
#   臂 A  餘裕充足、N 從小掃到大        => 純計算的基準線 t_compute(N)
#   臂 B  **N 固定 2.34M**、ballast 吃掉餘裕直到 OOM
#         => 計算量逐字相同 => 時間的變化**只能**來自配置器
#   臂 C  在壓力下比較三種 alloc conf（分三個 process，env 只能在啟動時設）
#
# 直接證據＝`torch.cuda.memory_stats()["num_alloc_retries"]`：
#   cudaMalloc 失敗 -> 釋放全部快取區塊 -> 重試（過程會同步裝置）。那就是「搬移」。
# ⚠ expandable_segments 在 torch 2.0.1 **不支援**（2.1 才有）。
# ⚠ 現行任務腳本都寫死 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
#   => 那是一個**已經被釘住但從沒對照過**的變數，臂 C 必須含「完全不設」。
set -u
cd "$(dirname "$0")/.." || exit 1
R=logs/vram_pressure_$(date +%m%d_%H%M).log
{
  echo "##### 臂 A：純計算基準線（retries 應該全 0）#####"
  env -u PYTORCH_CUDA_ALLOC_CONF conda run -n gspl --no-capture-output \
    python tools/vram_pressure.py --arm a --run speed3_b12 --step 60000

  echo
  echo "##### 臂 B：N 固定，只有餘裕在變（三種 alloc conf 各跑一次）#####"
  for CONF in "" "max_split_size_mb:128" "garbage_collection_threshold:0.8"; do
    echo
    echo "----- PYTORCH_CUDA_ALLOC_CONF = ${CONF:-（完全不設）} -----"
    if [ -z "$CONF" ]; then
      env -u PYTORCH_CUDA_ALLOC_CONF conda run -n gspl --no-capture-output \
        python tools/vram_pressure.py --arm b --n 2.34
    else
      PYTORCH_CUDA_ALLOC_CONF="$CONF" conda run -n gspl --no-capture-output \
        python tools/vram_pressure.py --arm b --n 2.34
    fi
  done
} 2>&1 | grep -vE "pkg_resources|declare_namespace" | tee "$R"
echo
echo "報告：$R"
echo "--- 一眼判讀：retries 有沒有出現過 ---"
grep -E "^ +[0-9]+\.[0-9]+G" "$R" | awk '{print $4}' | sort -u | tr '\n' ' '
echo "（全是 0 => 沒有搬移成本，慢是計算；有非 0 => 配置器在抖）"
