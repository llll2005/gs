#!/usr/bin/env python
"""紀錄/實驗分析/ 的圖（2026-10-03）：每份分析文件一張**同名** PNG，多個面板以 (a)(b)(c)… 標號。

做法：
  1. 舊的三支畫圖腳本（plot_cost_frontier／plot_resource_report／plot_commonconf_ab）以 CITYGS_FIG_OUT 指向暫存目錄畫面板
  2. 本檔畫新的面板（速度、tile 上限、新年代重測、init、成本感知取樣、官方參考線）
  3. 依 DOCS 的對應把面板直向拼成一張，左上角加 (a)(b)… => 紀錄/實驗分析/<文件名>.png
數字全部寫在本檔與那三支腳本裡；出處寫在各分析文件的「資料」欄。
用法：python tools/plot_analysis.py
"""
import os
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as _fm
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "紀錄", "實驗分析")
try:
    FONT_FILE = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK TC"],
                               capture_output=True, text=True).stdout.strip()
    _fm.fontManager.addfont(FONT_FILE); _fn = _fm.FontProperties(fname=FONT_FILE).get_name()
except Exception as _e:
    FONT_FILE = None; _fn = "DejaVu Sans"; print("⚠⚠ 找不到中文字型：", _e)
plt.rcParams.update({"font.sans-serif": [_fn, "DejaVu Sans"], "axes.unicode_minus": False,
                     "figure.dpi": 150, "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#898781", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8e6bd1"
MUTED, INK2 = "#898781", "#52514e"


def save(fig, d, name):
    p = os.path.join(d, name)
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)


def barlabels(ax, bars, fmt="{:.2f}", dy=2, size=7.5):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, dy),
                    textcoords="offset points", ha="center", fontsize=size, color=INK2)


# ══════════════ 新面板 ══════════════

def n05_cap60k(d):
    caps = [0.7, 1.2, 1.7, 2.6]
    b6 = [28.61, 29.39, 29.90, 30.48]
    b13 = [28.37, 29.06, np.nan, 30.00]          # cap 1.7M b13 跑中（10-03）
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(caps, b6, "-o", color=C1, lw=2, label="b6：只調 cap（conic 開、預設 SfM）")
    ax.plot([c for c, v in zip(caps, b13) if v == v], [v for v in b13 if v == v], "-o", color=C2, lw=2,
            label="b13：只調 cap（1.7M 跑中）")
    arms = [("v/c trim", 30.31, 29.80, "v"), ("fastgrow", 30.59, 30.10, "^"),
            ("dup2", 30.58, 30.13, "s"), ("dup4", 30.75, 30.32, "D")]
    for i, (nm, a, b, m) in enumerate(arms):
        x = 2.75 + 0.13 * i
        ax.plot(x, a, m, color=C1, ms=7); ax.plot(x, b, m, color=C2, ms=7)
        ax.annotate(nm, (x, max(a, b)), xytext=(0, 8), textcoords="offset points", fontsize=7.5, color=INK2, ha="center")
    ax.axvline(2.68, color=MUTED, lw=0.8, ls=":")
    ax.text(2.69, 28.45, "同 cap 2.6M 的各臂\n（藍 b6／橙 b13）", fontsize=7.5, color=INK2)
    ax.set_xlabel("cap_max（百萬顆；終點 N 約為 0.9 倍）"); ax.set_ylabel("val PSNR（dB，val⊂train）")
    ax.set_xlim(0.55, 3.25); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("60k：只調 cap 的品質曲線，與同 cap 2.6M 下各臂（右側點）", fontsize=10.5)
    save(fig, d, "n05_cap60k.png")


def n07_sampling(d):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axs[0]
    w = [-2.8, 0, 2.278, 4.0]
    ps = [29.141, 29.006, 28.821, 28.833]
    ld = [7.69, 6.99, 6.22, 5.91]
    ax.plot(w, ps, "-o", color=C1, lw=2)
    ax.set_xlabel("cost_add_densify 權重 w（負＝偏好貴，Taming 方向）"); ax.set_ylabel("PSNR（dB）", color=C1)
    ax2 = ax.twinx(); ax2.plot(w, ld, "--s", color=C2, lw=1.6); ax2.set_ylabel("Load 中位（百萬，代理單位）", color=C2)
    ax2.grid(False)
    ax.set_title("寬鬆預算（cap 2.6M）：w 是成本↔品質旋鈕", fontsize=10)
    ax = axs[1]
    pts = [("refc 固定 0.5", 97.6, 26.94, C2), ("refh1 前鬆後緊", 62.5, 27.26, C1),
           ("refh2 前緊後鬆", 91.2, 27.34, C3), ("無預算基準", 145.9, 29.00, MUTED)]
    for nm, x, y, c in pts:
        ax.plot(x, y, "o", color=c, ms=9)
        ax.annotate(f"{nm}\n{y:.2f}", (x, y), xytext=(6, -4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("整趟累積渲染成本（十億，區間平均 Load × 150 步）"); ax.set_ylabel("PSNR（dB）")
    ax.set_xlim(50, 175)
    ax.set_title("相對預算三種排程（目標總成本相同）", fontsize=10)
    ax = axs[2]
    frac = [10, 25, 50, 75]
    for nm, ys, c in [("b12@15k", [57.30, 26.61, 8.89, 2.26], C1), ("b13@15k", [57.36, 22.34, 6.12, 1.12], C2),
                      ("b6@60k", [48.70, 21.50, 5.42, 0.78], C3)]:
        ax.plot(frac, ys, "-o", color=c, lw=2, label=nm)
    ax.set_xlabel("預算佔總成本（%）"); ax.set_ylabel("按 v/c 選比按 v 選多拿的總價值（%）")
    ax.legend(fontsize=8); ax.set_title("背包的空間：預算越緊越大（離線上界）", fontsize=10)
    save(fig, d, "n07_sampling.png")


def n08_init20k(d):
    arms = ["random", "depth", "sfm", "fill01", "sfmfill", "vox015", "fill10", "pct05", "dup2", "dup4"]
    b6 = [22.46, 27.19, 28.52, 28.58, 28.68, 28.77, 28.85, 28.89, 29.54, 30.06]
    b13 = [21.83, 26.84, 28.02, 28.09, 28.20, 28.32, 28.41, 28.25, 29.01, 29.48]
    x = np.arange(len(arms)); wd = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.4))
    barlabels(ax, ax.bar(x - wd / 2, b6, wd, color=C1, label="b6"), size=7)
    barlabels(ax, ax.bar(x + wd / 2, b13, wd, color=C2, label="b13"), size=7)
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylim(20, 31)
    ax.set_ylabel("PSNR（dB，20k、機制全關）"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("20k 機制全關：init 類型與 sfmfill 參數（⚠ dup2/dup4 起始顆數多 2/4 倍，與顆數混淆）", fontsize=10)
    save(fig, d, "n08_init20k.png")


def n08_init60k(d):
    arms = ["sfmfill", "fastgrow\n(add_ratio 1.2)", "dup2", "dup4"]
    b6 = [0.04, 0.11, 0.10, 0.27]
    b13 = [0.01, 0.10, 0.13, 0.32]
    x = np.arange(len(arms)); wd = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    barlabels(ax, ax.bar(x - wd / 2, b6, wd, color=C1, label="b6"), fmt="{:+.2f}")
    barlabels(ax, ax.bar(x + wd / 2, b13, wd, color=C2, label="b13"), fmt="{:+.2f}")
    ax.axhspan(-0.24, 0.24, color="#f1e3a6", alpha=0.35, lw=0, label="22k 噪音底 3sd（60k 未量）")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylabel("ΔPSNR vs 預設 SfM（dB）")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("60k 完整配方（conic 開、同 N 2.34M）：相對預設 SfM init 的差", fontsize=10.5)
    save(fig, d, "n08_init60k.png")


def n09_official_blocks(d):
    lines = ["官方原版\n（trim 從未執行）", "＋一行修正\n（trim 執行）", "論文設定\n（＋ω0.9、prune 0.025）"]
    n3 = [5.23, 0.91, 3.05]; v3 = [31.69, 30.86, 31.78]
    n6 = [7.11, np.nan, 5.31]; v6 = [30.14, np.nan, 30.35]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(3); wd = 0.38
    ax = axs[0]
    barlabels(ax, ax.bar(x - wd / 2, n3, wd, color=C1, label="官方 block 3"), fmt="{:.2f}M")
    b = ax.bar(x + wd / 2, [0 if v != v else v for v in n6], wd, color=C2, label="官方 block 6")
    for bb, v in zip(b, n6):
        if v == v:
            ax.annotate(f"{v:.2f}M", (bb.get_x() + bb.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
        else:
            ax.annotate("未跑", (bb.get_x() + bb.get_width() / 2, 0.2), ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(lines, fontsize=8.5); ax.set_ylabel("終點顆數 N（百萬）"); ax.legend(fontsize=8)
    ax.set_title("官方程式碼、官方 4x4 分區：終點顆數", fontsize=10)
    ax = axs[1]
    barlabels(ax, ax.bar(x - wd / 2, v3, wd, color=C1, label="官方 block 3"))
    b = ax.bar(x + wd / 2, [0 if v != v else v for v in v6], wd, color=C2, label="官方 block 6")
    for bb, v in zip(b, v6):
        if v == v:
            ax.annotate(f"{v:.2f}", (bb.get_x() + bb.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    ax.set_ylim(28, 32.5); ax.set_xticks(x); ax.set_xticklabels(lines, fontsize=8.5)
    ax.set_ylabel("val PSNR（dB，val⊂train）"); ax.legend(fontsize=8)
    ax.set_title("同上：val PSNR（論文設定：顆數少 25~42%、val 反而高）", fontsize=10)
    save(fig, d, "n09_official_blocks.png")


def n10_lean_loop(d):
    segs = ["backward", "forward", "週期 trim\n(1 次/1000 步窗)", "optimizer", "迴圈外", "loss", "其他"]
    off = [84.85, 36.69, 27.87, 15.15, 8.00, 5.02, 3.33]
    on = [68.64, 21.13, 27.59, 15.15, 8.03, 5.05, 3.32]
    cols = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD", "#CCB974", "#B0B0B0"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for i, (row, nm, real) in enumerate([(off, "lean 關", 152.30), (on, "lean 開", 120.54)]):
        left = 0
        for v, c, s in zip(row, cols, segs):
            ax.barh(i, v, left=left, color=c, edgecolor="white", label=s if i == 0 else None)
            if v > 8:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
            left += v
        ax.text(left + 2, i, f"段落和 {left:.0f} ms／真實每步 {real:.1f} ms", va="center", fontsize=8.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["lean 關", "lean 開"]); ax.invert_yaxis()
    ax.set_xlim(0, 230); ax.set_xlabel("每步 wall ms（逐段同步計時；b6、N 2.29M、增生期 1,200 步含 1 次 trim）")
    ax.legend(ncol=7, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("lean_render 真實訓練迴圈：每步 152.3 -> 120.5 ms（-20.9%）；trim 幾乎沒變", fontsize=10.5)
    save(fig, d, "n10_lean_loop.png")


def n10_kernels(d):
    cats = ["光柵 backward\n（逐配對）", "光柵 forward＋record\n（逐配對）", "Adam", "SSIM", "SH cat＋排序\n＋前處理", "其他"]
    # 「其他」＝profiler 的「其他小運算」扣掉 autograd 包裝列（_RasterizeGaussians*，與光柵 kernel 重複計時）與 SH cat
    off = [80.55, 74.29, 14.81, 10.02, 3.73 + 3.17 + 2.18, 59.60 - 3.73]
    on = [62.63, 58.47, 14.82, 9.57, 3.72 + 3.18 + 2.17, 61.89 - 3.72]
    x = np.arange(len(cats)); wd = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axs[0]
    barlabels(ax, ax.bar(x - wd / 2, off, wd, color=MUTED, label="lean 關"), fmt="{:.1f}")
    barlabels(ax, ax.bar(x + wd / 2, on, wd, color=C3, label="lean 開"), fmt="{:.1f}")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5); ax.set_ylabel("kernel 時間（ms/步，含 1 台相機的 record）")
    ax.legend(fontsize=8)
    ax.set_title("kernel 拆解（profiler，b6 @14,999，N 2.6M；已扣掉 autograd 包裝列的重複計時）", fontsize=10)
    ax = axs[1]
    lab_ = ["1 步", "50 步"]
    fe = [20.90, 13.50]; fu = [48.69, 43.32]
    xx = np.arange(2)
    barlabels(ax, ax.bar(xx - wd / 2, fe, wd, color=C1, label="foreach（現行）"), fmt="{:.1f}")
    barlabels(ax, ax.bar(xx + wd / 2, fu, wd, color=C2, label="fused"), fmt="{:.1f}")
    ax.set_xticks(xx); ax.set_xticklabels(lab_); ax.set_ylabel("optimizer.step（ms，中位）"); ax.legend(fontsize=8)
    ax.set_title("fused Adam（torch 2.0.1）反而慢", fontsize=10)
    save(fig, d, "n10_kernels.png")


def n11_tiles(d):
    steps = [15, 30, 60]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
    ax = axs[0]
    for nm, a, b, c in [("b6", [41.5, 36.3, 34.4], [30.0, 30.1, 20.2], C1), ("b13", [35.9, 31.3, 30.3], [36.1, 39.9, 30.6], C2)]:
        ax.plot(steps, a, "-o", color=c, lw=2, label=f"{nm} ① tile 內沒有任何像素 alpha≥1/255")
        ax.plot(steps, b, "--s", color=c, lw=1.5, label=f"{nm} ② tile 已飽和後才輪到")
    ax.set_xlabel("訓練步數（千）"); ax.set_ylabel("佔 binning 配對（%）"); ax.set_ylim(0, 50); ax.legend(fontsize=7.5)
    ax.set_title("可剃掉的配對（渲染逐位元不變）", fontsize=10)
    ax = axs[1]
    thr = [0.005, 0.01, 0.02, 0.03, 0.05]
    for nm, t, w, c in [("b6", [2.4, 15.3, 61.7, 85.5, 95.5], [0.8, 12.3, 62.4, 88.2, 97.5], C1),
                        ("b13", [0.5, 18.7, 61.3, 81.1, 92.4], [0.3, 17.9, 62.5, 83.3, 94.3], C2)]:
        ax.plot(thr, t, "-o", color=c, lw=2, label=f"{nm} tile 比例")
        ax.plot(thr, w, "--s", color=c, lw=1.5, label=f"{nm} 工作量比例")
    ax.set_xscale("log"); ax.set_xlabel("絕對 L1 門檻（tile 平均）"); ax.set_ylabel("低於門檻的比例（%）")
    ax.legend(fontsize=7.5); ax.set_title("60k：低誤差 tile 並不比較便宜（兩線重疊）", fontsize=10)
    ax = axs[2]
    for nm, still, worse, c in [("b6", [90.1, 80.2, 94.2, 98.4, 99.6], [7.5, 11.4, 4.7, 3.1, 2.5], C1),
                                ("b13", [46.3, 78.1, 94.7, 98.3, 99.5], [22.6, 10.8, 4.8, 3.4, 2.8], C2)]:
        ax.plot(thr, still, "-o", color=c, lw=2, label=f"{nm} 仍低")
        ax.plot(thr, worse, "--s", color=c, lw=1.5, label=f"{nm} 誤差變 1.5 倍以上")
    ax.set_xscale("log"); ax.set_xlabel("絕對 L1 門檻"); ax.set_ylabel("30k 時低誤差的 (視角,tile) 到 60k（%）")
    ax.legend(fontsize=7.5); ax.set_title("「驗過不再驗」：多數仍低，但 2~23% 會變差", fontsize=10)
    save(fig, d, "n11_tiles.png")


def n12_recheck(d):
    runs = ["speed3", "speed3_trimvpc", "speed3_trimvpc_elong", "elong_prune", "elong_relocate",
            "cs60_base", "cs60_conic", "cs60_fastgrow", "cs60_sfmdup4", "cs60_sfmfill"]
    g = {6: {"early": [(2.673, -0.403), (1.798, -0.266), (1.721, -0.247), (2.437, -0.316), (2.485, -0.361),
                       (2.292, -0.152), (2.174, -0.121), (1.429, 0.150), (1.203, 0.380), (2.262, -0.184)],
             "late": [(1.059, 0.798), (1.050, 0.861), (1.017, 0.959), (1.041, 0.830), (1.066, 0.784),
                      (1.037, 0.887), (1.054, 0.769), (1.048, 0.814), (1.035, 0.910), (1.055, 0.820)],
             "off": (1.003, 0.993)},
         13: {"early": [(2.350, 0.066), (1.879, 0.158), (1.937, 0.098), (2.423, 0.113), (2.274, 0.101),
                        (2.242, 0.087), (2.180, 0.123), (1.749, 0.258), (1.322, 0.237), (1.885, 0.156)],
              "late": [(1.076, 0.690), (1.068, 0.765), (1.023, 0.875), (1.015, 0.901), (1.075, 0.683),
                       (1.119, 0.668), (1.075, 0.743), (1.056, 0.840), (1.074, 0.725), (1.075, 0.843)],
              "off": (1.006, 0.948)}}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axs[0]
    for blk, c in [(6, C1), (13, C2)]:
        e = np.array(g[blk]["early"]); l = np.array(g[blk]["late"])
        ax.scatter(e[:, 0], e[:, 1], color=c, alpha=0.45, s=22, label=f"b{blk} @1,499")
        ax.scatter(l[:, 0], l[:, 1], color=c, s=28, marker="D", label=f"b{blk} @60k")
        for a, b in zip(e, l):
            ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=c, alpha=0.25, lw=0.8))
        ax.plot(*g[blk]["off"], "*", color=c, ms=15, mec="k", mew=0.6, label=f"官方合併 b{blk} 視角")
    ax.axvline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("slope（渲染深度 / SfM 深度；1 = 正確）"); ax.set_ylabel("corr"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("幾何隨訓練變好（10 個跑次，箭頭 1,499 -> 60k）", fontsize=10)
    ax = axs[1]
    k = ["k≥1", "k≥3", "k≥10"]; x = np.arange(3); wd = 0.2
    for i, (nm, v, c) in enumerate([("b6 顆數%", [0.49, 0.29, 0.12], C1), ("b6 不透明度質量%", [0.12, 0.03, 0.00], "#86b6ee"),
                                    ("b13 顆數%", [0.93, 0.44, 0.22], C2), ("b13 不透明度質量%", [0.57, 0.09, 0.01], "#f4a77f")]):
        barlabels(ax, ax.bar(x + (i - 1.5) * wd, v, wd, color=c, label=nm), size=6.5)
    ax.set_xticks(x); ax.set_xticklabels([f"空區票數 {s}" for s in k]); ax.set_ylabel("落在 SfM 光線空區的比例（%）")
    ax.legend(fontsize=7)
    ax.set_title("空區雕刻：空區裡幾乎沒有顆粒；拿掉 ≈ 隨機拿掉", fontsize=10)
    ax = axs[2]
    arms = ["cs60_sfmdup4", "cs60_conic", "speed3"]
    b6 = [1.28, 1.49, 1.79]; b13 = [1.60, 2.30, 2.27]
    xx = np.arange(3); wd2 = 0.38
    barlabels(ax, ax.bar(xx - wd2 / 2, b6, wd2, color=C1, label="b6（11,952 tile）"), fmt="{:.2f}%")
    barlabels(ax, ax.bar(xx + wd2 / 2, b13, wd2, color=C2, label="b13（16,853 tile）"), fmt="{:.2f}%")
    ax.set_xticks(xx); ax.set_xticklabels(arms, fontsize=8.5); ax.set_ylabel("失敗 tile（%）"); ax.legend(fontsize=8)
    ax.set_title("失敗區（GT std≥0.1、corr<0.6）：1.3~2.3%（舊年代報 46%）", fontsize=10)
    save(fig, d, "n12_recheck.png")


# ══════════════ 組合 ══════════════
DOCS = {
    "01_訓練時間組成": ["fig1_time_breakdown", "fig9_scaling_phase", "fig7_wall_time"],
    "02_VRAM與配置器": ["fig2_maxsplit_ab", "fig3_vram_segments"],
    "03_儲存與RAM": ["fig4_ckpt_storage", "fig6_run_dir", "fig5_ram_cache"],
    "04_外接盒與binning成本": ["cc1_quality_delta", "cc2_time", "cc3_load_vram", "cc4_proxy_decoupled"],
    "05_成本品質前緣與vc_trim": ["f1_frontier_22k", "f2_60k_vpc_init", "n05_cap60k", "fig8_trimvpc_time"],
    "06_事後剪枝與fine-tune": ["f3_tile_topk_posthoc", "f6_prune_finetune"],
    "07_成本感知取樣與預算": ["n07_sampling"],
    "08_初始化": ["n08_init20k", "f4_init60k", "n08_init60k"],
    "09_官方參考線": ["n09_official_blocks", "f7_official_trim", "f5_official_prune"],
    "10_速度優化_lean與kernel拆解": ["n10_lean_loop", "n10_kernels"],
    "11_逐配對工作量上限": ["n11_tiles"],
    "12_新年代重測_幾何floater失敗區": ["n12_recheck"],
}


def compose(panels, out, width=2200):
    ims = []
    for p in panels:
        im = Image.open(p).convert("RGB")
        r = width / im.width
        ims.append(im.resize((width, int(im.height * r)), Image.LANCZOS))
    pad = 30
    H = sum(i.height for i in ims) + pad * (len(ims) + 1)
    canvas = Image.new("RGB", (width + 2 * pad, H), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(FONT_FILE, 44) if FONT_FILE else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    y = pad
    for k, im in enumerate(ims):
        canvas.paste(im, (pad, y))
        if len(ims) > 1:
            dr.text((pad + 6, y + 4), f"({chr(97 + k)})", fill=(11, 11, 11), font=font)
        y += im.height + pad
    canvas.save(out, optimize=True)


def main():
    os.makedirs(DST, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="citygs_panels_")
    env = dict(os.environ, CITYGS_FIG_OUT=tmp)
    for sc in ("plot_cost_frontier.py", "plot_resource_report.py", "plot_commonconf_ab.py"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", sc)], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"⛔ {sc} 失敗")
    for f in (n05_cap60k, n07_sampling, n08_init20k, n08_init60k, n09_official_blocks,
              n10_lean_loop, n10_kernels, n11_tiles, n12_recheck):
        f(tmp)
    for doc, panels in DOCS.items():
        paths = [os.path.join(tmp, p + ".png") for p in panels]
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            raise SystemExit(f"⛔ {doc} 缺面板：{miss}")
        compose(paths, os.path.join(DST, doc + ".png"))
        print(f"  -> 紀錄/實驗分析/{doc}.png（{len(panels)} 個面板）")


if __name__ == "__main__":
    if any(x in ("-h", "--help") for x in sys.argv[1:]):
        print(__doc__)
    else:
        main()
