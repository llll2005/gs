#!/usr/bin/env python
"""proxy 前提驗證開獎：壓縮後族群軌跡回得來嗎？

比較基準＝全長跑次在**相同歸一化進度**的族群佔 cap 的比例。
壓縮的核心假設是「等比例縮放後軌跡在歸一化時間上重合」，這裡就是直接量那件事。
"""
import glob

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

CAP = 2_600_000
REF = ("noprior_b12", 60_000)          # 全長參考臂
ARMS = [("px_noprior_b12", 12_000, "s=0.2, trim 100（已知壞掉）"),
        ("probe_t500", 12_000, "s=0.2, trim 500（完全不縮）"),
        ("probe_t250", 12_000, "s=0.2, trim 250（縮一半）")]
CHECK = 0.25                            # 在 25% 進度處比較


def trace(run):
    evs = sorted(glob.glob(f"outputs/{run}/**/events.out.tfevents.*", recursive=True))
    if not evs:
        return None
    a = ea.EventAccumulator(evs[-1], size_guidance={ea.SCALARS: 0})
    a.Reload()
    if "train/gaussians_count" not in a.Tags()["scalars"]:
        return None
    return np.array([(s.step, s.value) for s in a.Scalars("train/gaussians_count")])


def at(t, step):
    return float(t[np.abs(t[:, 0] - step).argmin(), 1])


ref = trace(REF[0])
if ref is None:
    raise SystemExit(f"找不到參考臂 {REF[0]}")
ref_v = at(ref, CHECK * REF[1])
print(f"參考：{REF[0]} 在 {CHECK:.0%} 進度（step {int(CHECK*REF[1])}）= "
      f"{ref_v/1e6:.3f}M = cap 的 {100*ref_v/CAP:.0f}%\n")

print(f"{'臂':>16} {'顆數':>9} {'佔cap':>7} {'相對參考':>9} {'末段趨勢':>10}  判定")
for name, total, desc in ARMS:
    t = trace(name)
    if t is None:
        print(f"{name:>16} {'(未跑)':>9}")
        continue
    step = CHECK * total
    if t[-1, 0] < step * 0.9:
        print(f"{name:>16} {'(未到檢查點, 目前 step '+str(int(t[-1,0]))+')':>9}")
        continue
    v = at(t, step)
    w = max(8, len(t) // 4)
    slope = (np.log(t[-1, 1]) - np.log(t[-w, 1])) / max(t[-1, 0] - t[-w, 0], 1)
    frac = v / CAP
    ok = 0.40 <= frac <= 0.55 and slope > 0
    trend = "上升" if slope > 1e-5 else ("下跌" if slope < -1e-5 else "持平")
    verdict = "✅ 軌跡回來了" if ok else ("⛔ 在跌" if slope <= 0 else "🟨 偏離")
    print(f"{name:>16} {v/1e6:>8.3f}M {100*frac:>6.0f}% {v/ref_v:>8.2f}x {trend:>10}  {verdict}")
    print(f"{'':>16} {desc}")

print("""
判準：40~55% 且趨勢上升 => 該政策可用，可據此重排完整 proxy 驗證。
      全部都不行 => 縮 trim interval 不是主因，問題在壓縮本身（s 太小），
                    改用較大的 s（例如 0.33 = 20,000 步 ~4.7h）重驗前提。""")
