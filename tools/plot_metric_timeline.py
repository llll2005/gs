"""Three metrics over the experiment history, to see which change moved which.

PSNR alone has been misleading: b12 is roughly half flat water, which is easy to fit, so PSNR stays
respectable while the buildings turn to mush. Plotting SSIM and LPIPS alongside it makes the
inflections attributable instead of reasoning from one number.

Only 當前資料 is drawn. The 2026-05-29 dataset regeneration moved the SfM frame AND the partition,
so old-data rows are not the same blocks and cannot share an axis -- CityGSV2's own faithful rerun
went 28.90 -> 21.78 PSNR / 0.169 -> 0.659 LPIPS across that boundary on "b7", which is the size of
the era effect and dwarfs every mechanism we have tested.

The remaining break inside 當前資料 is the rasterizer: EXACT_SUPPORT landed 2026-07-28 19:42. It is
bit-identical on a FIXED model (verified maxdiff 0), but during training it withholds gradient from
primitives dropped at binning, so runs either side of it are not the same experiment.
"""
import csv, sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("紀錄/實驗總表.csv", encoding="utf-8")))
def f(r, k):
    try: return float(r[k])
    except: return None

EXACT = datetime(2026, 7, 28)
# self-run CityGSV2 on the CURRENT data -- the only baseline this axis may be compared against
BASE = {"7": (21.78, 0.560, 0.659)}
blocks = sys.argv[1:] or ["12", "7"]
fig, axes = plt.subplots(len(blocks), 1, figsize=(16, 5.4 * len(blocks)), squeeze=False)

for ax, blk in zip(axes[:, 0], blocks):
    d = [r for r in rows if r["block"] == blk and r["era"] == "當前資料"
         and f(r, "psnr") and f(r, "ssim") and f(r, "lpips")]
    d.sort(key=lambda r: (r["date"], r["run"]))
    if not d:
        continue
    x = list(range(len(d)))
    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in d]
    ax.plot(x, [f(r, "psnr") for r in d], "o-", color="#1f77b4", lw=1.7, ms=4.5, label="PSNR ↑")
    ax.set_ylabel("PSNR", color="#1f77b4"); ax.tick_params(axis="y", labelcolor="#1f77b4")
    a2 = ax.twinx()
    a2.plot(x, [f(r, "ssim") for r in d], "s-", color="#2ca02c", lw=1.5, ms=4, label="SSIM ↑")
    a2.plot(x, [f(r, "lpips") for r in d], "^-", color="#d62728", lw=1.8, ms=4.5, label="LPIPS ↓")
    a2.set_ylabel("SSIM ↑ / LPIPS ↓"); a2.set_ylim(0, 1)

    pos = next((i for i, dt in enumerate(dates) if dt >= EXACT), None)
    if pos is not None:
        ax.axvline(pos - 0.5, color="k", ls="--", lw=1.3)
        ax.axvspan(-0.5, pos - 0.5, color="0.5", alpha=.10)
        ax.text(pos - 0.4, ax.get_ylim()[1], " EXACT_SUPPORT\n 之後（新 kernel）",
                fontsize=8.5, va="top")
        ax.text((pos - 1) / 2, ax.get_ylim()[0], "舊 kernel", fontsize=8.5, ha="center", va="bottom")
    if blk in BASE:
        a2.axhline(BASE[blk][2], color="#d62728", ls=":", lw=1.4)
        a2.text(len(d) - .5, BASE[blk][2], " 自跑 CityGSV2 LPIPS", fontsize=7.5,
                color="#d62728", va="bottom", ha="right")
        ax.axhline(BASE[blk][0], color="#1f77b4", ls=":", lw=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['date'][5:]} {r['run'][:24]}" for r in d], rotation=90, fontsize=6.2)
    ax.set_title(f"block {blk}（當前資料）— {len(d)} 次實驗，依時間序", fontsize=12)
    ax.grid(alpha=.25)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("紀錄/metric_timeline.png", dpi=125, bbox_inches="tight")
print("[out] 紀錄/metric_timeline.png")
