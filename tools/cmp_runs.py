#!/usr/bin/env python
"""跑次比較的標準入口：一律用「最後 4 個 val 點的平均」，不要用終點值。

為什麼（§11.40，2026-08-27）：三個功能完全相同的跑次上實測，
終點值的 PSNR sd = 0.0807，最後 4 點平均 = **0.0277**（緊 2.9 倍）；SSIM 緊 5 倍。
該估計量自我驗證通過：兩個同組態對照被正確判成噪音（+0.0 / −1.7 sigma），機制臂全部分離。

⚠ 用終點值曾讓我把 `notrim2`（真效應，+10.8 sigma）誤判成噪音（1.96 sigma），
   而且那是同一題的**第三次翻案**。估計量的選擇比指標的選擇更關鍵。

用法:  python tools/cmp_runs.py <base> <arm> [<arm> ...]
"""
import glob
import sys

from tensorboard.backend.event_processing import event_accumulator as ea

K = 4
SD = {"val/psnr": 0.0277, "val/ssim": 0.0001, "val/lpips": 0.0025 / 2.9,
      "val/texratio": 0.015 / 2.9}
NAMES = {"val/psnr": "PSNR", "val/ssim": "SSIM", "val/lpips": "LPIPS", "val/texratio": "紋理比"}
HIGHER = {"val/psnr": 1, "val/ssim": 1, "val/lpips": -1, "val/texratio": 1}


def avg(run, tag, k=K):
    evs = sorted(glob.glob(f"outputs/{run}/**/events.out.tfevents.*", recursive=True))
    if not evs:
        return None
    a = ea.EventAccumulator(evs[-1], size_guidance={ea.SCALARS: 0})
    a.Reload()
    if tag not in a.Tags()["scalars"]:
        return None
    v = [s.value for s in a.Scalars(tag)]
    return sum(v[-k:]) / min(k, len(v))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    base, arms = sys.argv[1], sys.argv[2:]
    print(f"基準 = {base}   估計量 = 最後 {K} 個 val 點的平均（§11.40）")
    print(f"{'臂':>18} " + " ".join(f"{NAMES[t]:>16}" for t in SD))
    for r in [base] + arms:
        row = []
        for t in SD:
            v = avg(r, t)
            row.append("(無)".rjust(16) if v is None else f"{v:16.4f}")
        print(f"{r:>18} " + " ".join(row))
    print()
    for r in arms:
        cells = []
        for t in SD:
            a, b = avg(r, t), avg(base, t)
            if a is None or b is None:
                cells.append(f"{NAMES[t]}: 無")
                continue
            d = (a - b) * HIGHER[t]
            n = d / SD[t]
            mk = "WIN" if n > 3 else ("LOSE" if n < -3 else "~平")
            cells.append(f"{NAMES[t]} {d:+.4f} ({n:+.1f}sd) {mk}")
        print(f"{r:>18} vs {base}:")
        for c in cells:
            print(f"{'':>20}{c}")
    print("""
判準：|sd| < 3 一律當「打平」，不可宣稱。
⚠ 建築低頻／最差10%／疊影% 來自 test 圖（單一快照）**無法平均** => 用 lowfreq_split /
  tail_analysis / veil_detect 另外看，且門檻要保守。""")


if __name__ == "__main__":
    main()
