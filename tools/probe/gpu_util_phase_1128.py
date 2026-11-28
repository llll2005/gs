"""11-28 同卡同時訓練多塊的前提：單塊訓練時各解析度階段的 GPU 使用率（已關閉的線 K 的重開條件「單塊 GPU 使用率明顯降下來」）。

輸入：nvidia-smi 抽樣（timestamp, utilization.gpu, memory.used；-lms 500）與 task_concurrent.sh 加了 epoch 時間戳的訓練輸出。
階段切點＝訓練輸出的「[train-res] ✅ step N: ... （xS）」；第一個切點＝訓練開始（全解析度 -> 0.35 倍），
訓練結束＝開始 + run_summary.json 的 train_s（第一步 -> 最後一步）。
只用標準函式庫（lab 的系統 python3 沒有 numpy）。
用法：python3 tools/probe/gpu_util_phase_1128.py --util logs/gpuutil_x.csv --logs logs/conc_x_blk1.log [...] --t0 <epoch> --t1 <epoch>
"""
import argparse
import datetime
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--util", required=True)
ap.add_argument("--logs", nargs="+", required=True)
ap.add_argument("--t0", type=float, required=True)
ap.add_argument("--t1", type=float, required=True)
a = ap.parse_args()

samples = []
for line in open(a.util, encoding="utf-8", errors="replace"):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 3:
        continue
    try:
        t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        samples.append((t, float(p[1]), float(p[2])))
    except ValueError:
        continue


def mean_util(t0, t1):
    v = [u for t, u, _ in samples if t0 <= t < t1]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)


print(f"\n=== GPU 使用率（nvidia-smi utilization.gpu，每 0.5 s 抽樣 {len(samples)} 點）===")
wall = a.t1 - a.t0
u, n = mean_util(a.t0, a.t1)
print(f"整個任務牆鐘 {wall:.1f} s｜平均使用率 {u:.1f}%（{n} 點）")
RX = re.compile(r"^(\d+\.\d+) .*\[train-res\] ✅ step (\d+): 訓練解析度 .*（x([0-9.]+)）")
for lg in sorted(set(x for pat in a.logs for x in glob.glob(pat))):
    cuts = []
    for line in open(lg, encoding="utf-8", errors="replace"):
        m = RX.match(line)
        if m:
            cuts.append((float(m.group(1)), int(m.group(2)), float(m.group(3))))
    b = re.search(r"_blk(\d+)\.log$", lg)
    b = b.group(1) if b else "?"
    summ = sorted(glob.glob(f"outputs/lab/*/blocks/block_{b}/run_summary.json"), key=os.path.getmtime)
    tr = None
    for s in reversed(summ):
        if os.path.getmtime(s) >= a.t0:
            tr = json.load(open(s)).get("train_s"); break
    if not cuts:
        print(f"[{lg}] 找不到 [train-res] 切點")
        continue
    ts = cuts[0][0]
    te = ts + tr if tr else a.t1
    edges = [c[0] for c in cuts] + [te]
    print(f"[第 {b} 塊] 訓練開始 +{ts - a.t0:.1f} s、純訓練 {tr if tr else float('nan'):.1f} s")
    for k, (t, step, s) in enumerate(cuts):
        u, n = mean_util(t, edges[k + 1])
        print(f"   x{s:<5g} 從 step {step:>6}：{edges[k + 1] - t:7.1f} s｜使用率 {u:5.1f}%（{n} 點）")
