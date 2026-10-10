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
    b13 = [28.37, 29.06, 29.46, 30.00]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.plot(caps, b6, "-o", color=C1, lw=2, label="b6：只調 cap（conic 開、預設 SfM）")
    ax.plot([c for c, v in zip(caps, b13) if v == v], [v for v in b13 if v == v], "-o", color=C2, lw=2,
            label="b13：只調 cap（conic 開、預設 SfM）")
    # 10-10 修正：各臂都是 cap 2.6M，放回 x=2.6（只做左右小錯開避免重疊；舊版放在 2.75~3.15 看起來像別的 cap）
    arms = [("v/c trim", 30.31, 29.80, "v"), ("dup4 ★現行", 30.75, 30.32, "*"), ("oreg0", 31.38, 31.32, "D")]
    for i, (nm, a, b, m) in enumerate(arms):
        x = 2.6 + 0.045 * (i - 1)
        ax.plot(x, a, m, color=C1, ms=9 if m == "*" else 7, mec="k", mew=0.4); ax.plot(x, b, m, color=C2, ms=9 if m == "*" else 7, mec="k", mew=0.4)
        ax.annotate(f"{nm}（b6 {a:.2f}／b13 {b:.2f}）", (x, min(a, b) if m == "v" else max(a, b)), xytext=(-14 if m == "v" else 14, -14 if m == "v" else 0), textcoords="offset points", fontsize=8, color=INK2, va="center", ha="right" if m == "v" else "left")
    ax.set_xlabel("cap_max（百萬顆；終點 N 約為 0.9 倍）"); ax.set_ylabel("val PSNR（dB，val⊂train）")
    ax.set_xlim(0.55, 3.25); ax.set_ylim(28.3, 31.6); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("60k：只調 cap 的品質曲線；cap 2.6M 處的點＝同顆數下的各臂（藍 b6／橙 b13）", fontsize=10.5)
    save(fig, d, "n05_cap60k.png")


def n02_current(d):
    """現行配方的 VRAM：峰值隨 N 的斜率與 6GB 上限、lean 的峰值配置。"""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    ax = axs[0]
    n = np.array([1.85, 2.34]); pk = np.array([3.26, 4.48])          # lab 同塊（09-13）：cs_base 22k、speed3 60k
    k = (pk[1] - pk[0]) / (n[1] - n[0])
    xx = np.linspace(1.6, 3.15, 50)
    ax.plot(xx, pk[0] + k * (xx - n[0]), "--", color=MUTED, lw=1.2, label=f"線性外推（{k:.2f} GB／百萬顆）")
    ax.plot(n, pk, "o", color=C1, ms=8, label="lab 實測峰值實佔（cs_base 22k／speed3 60k）")
    ax.errorbar([2.34], [4.335], yerr=[[0.065], [0.065]], fmt="D", color=C3, ms=8, capsize=5,
                label="★ 現行最佳 cs60_sfmdup4 等 60k 臂（4.27~4.40）")
    ax.axhline(6.1, color="k", lw=1, ls="--"); ax.text(1.62, 6.17, "本機 6.1 GiB 牆", fontsize=8)
    ax.axhline(5.66, color=C2, lw=1, ls=":"); ax.text(1.62, 5.45, "lab 上限 5.66 GiB（只管 PyTorch 配置器，光柵器另約 +0.28）", fontsize=7.5, color=C2)
    ax.axvspan(2.9, 3.0, color=C2, alpha=0.12); ax.text(2.92, 3.0, "6GB 內\n可行上限\n約 2.9~3.0M", fontsize=8, color=INK2)
    ax.axvline(2.34, color=MUTED, lw=0.6, ls=":"); ax.text(2.36, 2.75, "現行 cap 2.6M\n交付 2.34M", fontsize=7.5, color=INK2)
    ax.set_xlabel("終點顆數 N（百萬，sh3）"); ax.set_ylabel("峰值實佔（GiB，台帳）"); ax.set_ylim(2.5, 6.6)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.01, 0.52))
    ax.set_title("峰值隨 N：VRAM 由顆數主導；cap 再往上加就出 6GB 信封", fontsize=10)
    ax = axs[1]
    barlabels(ax, ax.bar(["lean 關", "lean 開", "★ lean＋\nrecord_reduce\n（現行）", "＋tile_cull\n（不採用）"],
                         [4022, 3821, 3821, 3730], color=[MUTED, "#86d0b4", C3, MUTED], width=0.6), fmt="{:,.0f}")
    ax.set_ylim(0, 4700); ax.set_ylabel("峰值配置（MiB）"); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("峰值配置：lean -5%；record_reduce 不變；tile_cull 再 -2%（但淨變慢）", fontsize=10)
    fig.suptitle("現行配方的 VRAM（lab 鎖 5.66 GiB）", fontsize=11)
    save(fig, d, "n02_current.png")


def n07_sampling(d):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axs[0]
    w = [-2.8, 0, 2.278, 4.0]
    ps = [29.141, 29.006, 28.821, 28.833]
    ld = [7.69, 6.99, 6.22, 5.91]
    ax.plot(w, ps, "-o", color=C1, lw=2)
    ax.annotate("★ 現行 w=0", (0, ps[list(w).index(0)]), xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("cost_add_densify 權重 w（負＝偏好貴，Taming 方向）"); ax.set_ylabel("PSNR（dB）", color=C1)
    ax2 = ax.twinx(); ax2.plot(w, ld, "--s", color=C2, lw=1.6); ax2.set_ylabel("Load 中位（百萬，代理單位）", color=C2)
    ax2.grid(False)
    ax.set_title("寬鬆預算（cap 2.6M）：w 是成本↔品質旋鈕", fontsize=10)
    ax = axs[1]
    pts = [("refc 固定 0.5", 97.6, 26.94, C2), ("refh1 前鬆後緊", 62.5, 27.26, C1),
           ("refh2 前緊後鬆", 91.2, 27.34, C3), ("★ 現行：無預算", 145.9, 29.00, MUTED)]
    for nm, x, y, c in pts:
        ax.plot(x, y, "o", color=c, ms=9)
        ax.annotate(f"{nm}\nPSNR {y:.2f}｜成本 {x:.1f}B", (x, y), xytext=(6, -4), textcoords="offset points", fontsize=8, color=INK2)
    ax.annotate("", (145.9, 28.6), (97.6, 28.6), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text(121, 28.65, "無預算多花 +50~133% 成本", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("整趟累積渲染成本（十億，區間平均 Load × 150 步）"); ax.set_ylabel("PSNR（dB）")
    ax.set_xlim(50, 185)
    ax.set_title("相對預算三種排程（目標總成本相同）：前鬆後緊成本最低；無預算較高分是因為成本多一半以上", fontsize=9)
    ax = axs[2]
    frac = [10, 25, 50, 75]
    for nm, ys, c in [("b12@15k", [57.30, 26.61, 8.89, 2.26], C1), ("b13@15k", [57.36, 22.34, 6.12, 1.12], C2),
                      ("b6@60k", [48.70, 21.50, 5.42, 0.78], C3)]:
        ax.plot(frac, ys, "-o", color=c, lw=2, label=nm)
    ax.set_xlabel("預算佔總成本（%）"); ax.set_ylabel("按 v/c 選比按 v 選多拿的總價值（%）")
    ax.legend(fontsize=8); ax.set_title("背包的空間：預算越緊越大\n（離線上界：在訓練好的模型上用貪婪背包算，不是訓練結果）", fontsize=9.5)
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
    # 10-08：dup5（起始 > cap）與 dup4 抖動 1.0／0.25 的 b13 出來了（vs cs60_conic b13 30.00）
    arms = ["sfmfill", "fastgrow\n(add_ratio 1.2)", "dup2", "★ dup4\n（現行最佳）", "dup5\n（起始 > cap）", "dup4\n抖動 1.0", "dup4\n抖動 0.25"]
    b6 = [0.04, 0.11, 0.10, 0.27, 0.28, 0.29, 0.18]
    b13 = [0.01, 0.10, 0.13, 0.32, 0.30, 0.33, 0.19]
    x = np.arange(len(arms)); wd = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.4))
    barlabels(ax, ax.bar(x - wd / 2, b6, wd, color=C1, label="b6"), fmt="{:+.2f}")
    bb = ax.bar(x + wd / 2, [v if v == v else 0 for v in b13], wd, color=C2, label="b13")
    barlabels(ax, [r for r, v in zip(bb, b13) if v == v], fmt="{:+.2f}")
    for xi, v in zip(x, b13):
        if v != v:
            ax.text(xi + wd / 2, 0.01, "b13\n排隊中", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axhspan(-0.24, 0.24, color="#f1e3a6", alpha=0.35, lw=0, label="22k 噪音底 3sd（60k 未量）")
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(3.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5); ax.set_ylabel("ΔPSNR vs 預設 SfM（dB）")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("60k 完整配方（conic 開、同 N 2.34M）：相對預設 SfM init 的差；dup4 之後再加份數或改抖動都持平（兩塊都確認）", fontsize=10.5)
    save(fig, d, "n08_init60k.png")


def n09_official_blocks(d):
    """官方 block 3／6：release（trim 從未執行）vs 論文設定（trim 執行＋ω0.9、prune 0.025）。10-10：拿掉「一行修正」那欄（使用者）。"""
    lines = ["官方 release\n（trim 從未執行）", "官方論文設定\n（trim 執行＋ω0.9、prune 0.025）"]
    n3 = [5.23, 3.05]; v3 = [31.69, 31.78]; n6 = [7.11, 5.31]; v6 = [30.14, 30.35]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.4))
    x = np.arange(2); wd = 0.36
    for ax, (a3, a6, fmt, yl, tt) in zip(axs, [(n3, n6, "{:.2f}M", "終點顆數 N（百萬）", "終點顆數：論文設定少 25~42%"),
                                                (v3, v6, "{:.2f}", "val PSNR（dB，val⊂train）", "val：論文設定反而略高")]):
        barlabels(ax, ax.bar(x - wd / 2, a3, wd, color=C1, label="官方 block 3"), fmt=fmt)
        barlabels(ax, ax.bar(x + wd / 2, a6, wd, color=C2, label="官方 block 6"), fmt=fmt)
        ax.set_xticks(x); ax.set_xticklabels(lines, fontsize=8.5); ax.set_ylabel(yl); ax.legend(fontsize=8); ax.set_title(tt, fontsize=10)
    axs[1].set_ylim(29.5, 32.2)
    save(fig, d, "n09_official_blocks.png")


def n09_trim(d):
    """官方 block 3：trim 有沒有執行（同官方程式碼與 config，只差 trim 是否真的跑）。"""
    met = [("終點顆數（百萬）", 5.23, 0.91, "{:.2f}"), ("val PSNR（dB）", 31.69, 30.87, "{:.2f}"),
           ("訓練峰值實佔（GB）", 7.30, 2.75, "{:.2f}"), ("訓練時間（h）", 3.31, 2.29, "{:.2f}")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (nm, a, b, fmt) in zip(axs, met):
        barlabels(ax, ax.bar([0, 1], [a, b], color=[MUTED, C1], width=0.6), fmt=fmt, size=8.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["trim 未執行\n（release 的實際行為）", "trim 執行"], fontsize=8)
        ax.set_title(nm, fontsize=10)
        if "PSNR" in nm:
            ax.set_ylim(29.5, 32.2)
    axs[3].text(1, 2.29 * 0.5, "⚠ 期間外部\n佔卡約 2 GB", ha="center", fontsize=7.5, color="white")
    fig.suptitle("官方 block 3：trim 真的執行時顆數 -83%、峰值 -62%、val -0.8 dB —— release 的 aerial trim 階段其實從沒剪過", fontsize=10.5)
    save(fig, d, "n09_trim.png")


# 10-09：我方最佳解 4x4（16 塊 [solo]、lean＋record_reduce、dup4、cap 2.6M）vs 官方 release vs 官方論文設定
#   資料：lab outputs/lab/full44_best（chart_data、台帳）、cityGS_origin(_trimfix)/outputs/*/blocks、logs/citygs_official_resources.tsv
#   峰值實佔一律 GB（1e9）：官方 MiB x 1.048576 / 1000；時間 = 分鐘（我方台帳 START->DONE；官方 wall_s）
F44 = {
    'oh': [20.239, 19.675, 22.011, 20.452, 19.714, 17.9, 19.316, 19.017, 21.96, 19.767, 20.406, 21.354, 22.545, 22.073, 21.824, 22.738],
    'rh': [20.114, 19.947, 21.322, 17.15, 20.025, 18.357, 19.171, 16.871, 21.124, 17.491, 18.43, 18.961, 26.387, 22.869, 22.206, 23.212],
    'ph': [19.811, 19.81, 20.99, 16.638, 19.569, 18.22, 18.898, 16.401, 20.665, 17.074, 18.112, 18.613, 26.191, 22.521, 21.809, 23.076],
    'ohl': [0.5392, 0.5408, 0.3863, 0.4693, 0.567, 0.6447, 0.4906, 0.5315, 0.4726, 0.5475, 0.4944, 0.4427, 0.3642, 0.4181, 0.4438, 0.3911],
    'rhl': [0.4457, 0.4051, 0.3079, 0.4466, 0.4524, 0.4634, 0.3987, 0.4723, 0.3772, 0.4976, 0.4537, 0.4212, 0.1671, 0.2942, 0.346, 0.3043],
    'phl': [0.4894, 0.4324, 0.332, 0.5285, 0.5038, 0.4947, 0.4269, 0.5356, 0.3979, 0.5355, 0.4894, 0.4687, 0.1631, 0.3136, 0.3647, 0.324],
    'opk': [4.12, 4.1, 4.09, 4.11, 4.07, 4.09, 4.05, 4.14, 4.11, 4.17, 4.05, 4.08, 4.05, 4.13, 4.07, 4.11],
    'rpk': [10.81, 9.5, 8.39, 7.65, 12.43, 10.23, 9.7, 8.59, 14.56, 12.78, 11.16, 10.21, 13.35, 14.03, 14.17, 9.33],
    'ppk': [7.34, 7.06, 6.89, 5.24, 9.57, 8.4, 8.25, 6.19, 10.14, 9.65, 8.47, 6.47, 9.79, 11.23, 9.74, 6.79],
    'omin': [108, 120, 118, 119, 122, 112, 109, 121, 117, 114, 117, 122, 112, 111, 111, 107],
    'rmin': [245, 219, 206, 198, 271, 244, 240, 224, 291, 280, 269, 352, 291, 317, 300, 358],
    'pmin': [259, 248, 237, 215, 340, 278, 275, 255, 351, 323, 308, 273, 360, 396, 351, 459],
    'rext': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5302, 16251, 6174, 7128, 8914, 18161],
    'pext': [8290, 5722, 3674, 4610, 7358, 5108, 7712, 6448, 7698, 8278, 14854, 11654, 14704, 8696, 12228, 17351],
}
LBL3 = ["★ 我方 4x4", "官方 release", "官方論文設定"]


def n09_full44_merged(d):
    """合併模型、官方 741 幀 held-out，三者同一支工具（tools/eval_official_test.py）。"""
    q = {"PSNR": [27.739, 27.261, 27.280], "SSIM": [0.8790, 0.8671, 0.8697], "LPIPS（越低越好）": [0.1293, 0.1540, 0.1467]}
    r = {"N（百萬）": [17.63, 24.48, 24.72], "渲染 FPS": [35.6, 21.9, 21.9],
         "16 塊總 N（百萬）": [37.44, 129.55, 87.89]}
    fig, axs = plt.subplots(1, 6, figsize=(17, 3.6))
    cols = [C1, MUTED, C2]
    for ax, (k, v) in zip(axs, list(q.items()) + list(r.items())):
        bars = ax.bar(range(3), v, color=cols, width=0.65)
        fmt = "{:.3f}" if max(v) < 1 else ("{:.2f}" if max(v) < 50 else "{:.1f}")
        barlabels(ax, bars, fmt=fmt, size=8)
        lo = min(v); hi = max(v)
        if k in ("PSNR", "SSIM", "LPIPS（越低越好）"):
            ax.set_ylim(lo - (hi - lo) * 1.2, hi + (hi - lo) * 0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels(["我方", "release", "論文\n設定"], fontsize=8.5)
        ax.set_title(k, fontsize=10)
    fig.suptitle("合併模型・官方 741 幀 held-out（三者同一支評分工具）：我方三項全贏、顆數少 28%、FPS 1.6 倍"
                 "　（PSNR 領先保守估計 +0.1~+0.5 dB：見面板 d 的光柵器偏差）", fontsize=10.5)
    save(fig, d, "n09_full44_merged.png")


def n09_full44_blocks(d):
    x = np.arange(16); wd = 0.27
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.4))
    ax = axs[0]
    for k, c, dx, lb in [("oh", C1, -wd, LBL3[0]), ("rh", MUTED, 0, LBL3[1]), ("ph", C2, wd, LBL3[2])]:
        ax.bar(x + dx, F44[k], wd, color=c, label=lb)
    ax.set_ylim(15, 27.5); ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.set_xlabel("4x4 block"); ax.set_ylabel("同塊 held-out PSNR（dB，下界）"); ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.set_title("同塊 held-out（我方塊視角）：平均 20.69／20.23／19.90；LPIPS 官方 16 塊全較好\n"
                 "⚠ 偏向官方：官方單塊從全場景 coarse 起步、看得到塊外", fontsize=9.5)
    ax = axs[1]
    for k, c, mk, lb in [("opk", C1, "o", LBL3[0]), ("rpk", MUTED, "s", LBL3[1]), ("ppk", C2, "^", LBL3[2])]:
        ax.plot(x, F44[k], marker=mk, color=c, lw=0, ms=6, label=lb)
    ax.axhline(6.44, color="#c0392b", lw=1, ls="--"); ax.text(15.4, 6.6, "6 GiB 卡", color="#c0392b", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xlabel("4x4 block"); ax.set_ylabel("訓練峰值實佔（GB）"); ax.legend(fontsize=7.5)
    ax.set_title("峰值實佔：我方 4.05~4.17；release 7.65~14.56；論文設定 5.24~11.23（只有 b3、b7 低於 6 GiB）", fontsize=9.5)
    ax = axs[2]
    for k, c, dx, lb in [("omin", C1, -wd, LBL3[0]), ("rmin", MUTED, 0, LBL3[1]), ("pmin", C2, wd, LBL3[2])]:
        bars = ax.bar(x + dx, [m / 60 for m in F44[k]], wd, color=c, label=lb)
        if k != "omin":
            ext = F44["rext" if k == "rmin" else "pext"]
            for b_, e in zip(bars, ext):
                if e > 2000:
                    b_.set_hatch("///"); b_.set_alpha(0.45)
    ax.set_xticks(x); ax.set_xlabel("4x4 block"); ax.set_ylabel("訓練時間（h）"); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("訓練時間（斜線＝期間外部佔卡 > 2 GB，不可引用）\nblock 0~9 乾淨：我方 19.3 h vs release 40.3 h（0.48 倍）", fontsize=9.5)
    save(fig, d, "n09_full44_blocks.png")


def n09_stepprof(d):
    """同 N 附近的每步逐段時間（每標記點同步）：官方 stepprof b6（開頭 1,500 步）vs 我方 b6 @14,999 起 1,200 步。"""
    segs = ["forward", "backward", "optimizer", "loss", "週期 trim\n（÷500 攤平）", "其他"]
    ours = [21.15, 68.75, 15.14, 5.06, 35.95, 10.81]
    rel = [47.45, 137.96, 15.66, 6.11, 0.0, 6.55]
    pap = [41.61, 127.95, 10.74, 6.13, 58.20, 0.0]   # 官方 stepprof 的 trim 列已是每步攤平（1,500 步窗）；其他欄為負值（-14.8）記 0
    names = ["★ 我方 b6\nN 2.29M（lean＋rr）", "官方 release b6\nN 2.50M（trim 不跑）", "官方論文設定 b6\nN 1.75M（含 trim）"]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    cols = [C1, C2, C3, C4, "#c9b458", MUTED]
    for i, row in enumerate([ours, rel, pap]):
        left = 0
        for j, v in enumerate(row):
            if v > 0:
                ax.barh(i, v, left=left, color=cols[j], label=segs[j] if i == 0 else None)
                if v > 9:
                    ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
                left += v
        ax.text(left + 3, i, f"{left:.0f} ms", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("每步 ms（逐段同步計時）"); ax.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("同 N 附近：官方每步做 2 次 backward（loss 與 depth 的 extra_loss 各一次，各約 64~69 ms）、forward 帶幾何通道；"
                 "我方 depth／normal 權重 0 => 1 次 backward＋lean", fontsize=9.5)
    save(fig, d, "n09_stepprof.png")


def n09_evalgap(d):
    """評分工具落差 1.47 dB 的二分（10-09）。資料＝紀錄/實驗分析/data/evalgap/（lab logs/evalgap/ 的副本）。"""
    import csv
    E = os.path.join(DST, "data", "evalgap")
    off = {r["name"]: (float(r["psnr"]), float(r["bright_ratio"])) for r in csv.DictReader(open(os.path.join(E, "official_release_brightness.tsv")), delimiter="\t")}
    our = {r["name"]: float(r["psnr"]) for r in csv.DictReader(open(os.path.join(E, "ours_release.csv")))}
    n = sorted(set(off) & set(our))
    xo = np.array([off[k][0] for k in n]); yo = np.array([our[k] for k in n]); br = np.array([off[k][1] for k in n])
    rd = lambda f: list(csv.DictReader(open(os.path.join(E, f)), delimiter="\t"))
    A, B = rd("official_env_direct.tsv"), rd("official_code_our_env.tsv")
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1.05, 1.45, 1]})
    ax = axs[0]
    sc = ax.scatter(xo, yo, c=np.clip(br, 0, 1.1), cmap="viridis", s=9, vmin=0, vmax=1.1)
    ax.plot([8, 36], [8, 36], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("官方工具（官方光柵器）逐幀 PSNR"); ax.set_ylabel("我方工具（我方光柵器）逐幀 PSNR")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046); cb.set_label("官方渲染亮度／GT 亮度", fontsize=8)
    ax.set_title("　　同一個官方 release 模型、741 幀\n　　左上角那群＝官方渲染變暗／全黑的幀", fontsize=9.5)
    ax = axs[1]
    x = np.arange(len(A)); wd = 0.4
    ax.bar(x - wd / 2, [float(r["psnr"]) for r in A], wd, color=MUTED, label="官方程式碼＋官方光柵器（直接渲染）")
    ax.bar(x + wd / 2, [float(r["psnr"]) for r in B], wd, color=C1, label="官方程式碼＋只換我方光柵器")
    ax.axvline(15.5, color=INK2, lw=0.8, ls=":")
    ax.text(7.5, 33, "官方 test 最暗的 16 幀", ha="center", fontsize=8.5, color=INK2)
    ax.text(18.5, 33, "正常幀 6 張", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([r["name"][:4] for r in A], rotation=60, fontsize=7.5); ax.set_ylim(0, 35)
    ax.set_ylabel("PSNR（dB）"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title("二分：繞過 test 迴圈仍重現（0184 10.31，逐位數相同）；只換光柵器即恢復\n暗幀 alpha 0.9997（畫面是滿的）=> 壞在顏色路徑", fontsize=9.5)
    ax = axs[2]
    est = [("官方工具（原數字）", 25.79, MUTED), ("官方工具・只算正常幀\n（亮度比 ≥0.85，645 幀）", 27.18, MUTED),
           ("我方工具（暗幀已正常）", 27.26, MUTED), ("我方工具＋正常幀光柵器偏差補回\n（+0.35，上界）", 27.61, MUTED),
           ("★ 我方 4x4（我方工具）", 27.74, C1)]
    for i, (lb, v, c) in enumerate(est):
        ax.barh(i, v, color=c, height=0.6); ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(est))); ax.set_yticklabels([e[0] for e in est], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(25.5, 28.2); ax.set_xlabel("官方 741 幀 PSNR（dB）")
    ax.set_title("官方 release 的合理分數 ≈ 27.2~27.6\n=> 我方領先 +0.1~+0.5 dB（論文報告 27.23）", fontsize=9.5)
    save(fig, d, "n09_evalgap.png")


def n05_arms60k(d):
    """60k 單變數臂 vs cs60_conic（同 N 2.34M）：val、塊內 held-out、精確 Load 中位。"""
    arms = ["vpctilek", "oent", "cag4（absgrad 4）", "cag0（absgrad 關）", "★ dup4"]
    dv = {6: [-0.03, -0.62, 0.02, -0.07, 0.27], 13: [-0.01, -0.13, -0.03, -0.05, 0.32]}
    dh = {6: [-0.336, 0.220, 0.149, -0.134, -0.024], 13: [-0.124, 0.422, 0.157, 0.025, 0.246]}
    dl = {6: [-37.3, -47.9, -2.2, -3.9, -5.6], 13: [-37.7, -52.0, -0.7, -12.5, -8.3]}
    dlp = {6: [-0.0032, -0.0243, -0.0030, 0.0037, -0.0137], 13: [-0.0088, -0.0316, -0.0031, -0.0016, -0.0230]}
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
    mk = ["o", "s", "^", "v", "*"]
    for ax, D, yl, tt in [(axs[0], dv, "Δ val PSNR（dB）", "val⊂train"), (axs[1], dh, "Δ 塊內 held-out PSNR（dB）", "塊內 held-out")]:
        for blk, c in [(6, C1), (13, C2)]:
            for j, a in enumerate(arms):
                if D[blk][j] == D[blk][j]:
                    ax.scatter(dl[blk][j], D[blk][j], color=c, marker=mk[j], s=70 if mk[j] == "*" else 40,
                               label=f"{a}" if blk == 6 else None)
                    ax.annotate(f"b{blk}", (dl[blk][j], D[blk][j]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=c)
        ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=0.8)
        if D is dh:
            ax.axhspan(-0.2, 0.2, color="#f1e3a6", alpha=0.35, lw=0, label="held-out 噪音約 ±0.2（dup 變體散布）")
        ax.set_xlabel("Δ 精確 Load 中位（%，越左越便宜）"); ax.set_ylabel(yl)
        ax.set_title(f"{tt}：vs cs60_conic（同 N）", fontsize=10)
    h_, l_ = axs[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.36, 0.0), ncol=6, fontsize=8)
    ax = axs[2]; x = np.arange(len(arms)); wd = 0.38
    ax.bar(x - wd / 2, dlp[6], wd, color=C1, label="b6"); ax.bar(x + wd / 2, [0 if v != v else v for v in dlp[13]], wd, color=C2, label="b13")
    ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, fontsize=8)
    ax.set_ylabel("Δ 塊內 held-out LPIPS（越負越好）"); ax.legend(fontsize=8)
    ax.set_title("held-out LPIPS：oent 兩塊最好、vpctilek 也較好", fontsize=10)
    fig.suptitle("60k 單變數臂：oent 的 val 輸但 held-out 贏且 Load 減半（val 與 held-out 反向）；vpctilek 同 val、Load -37%；absgrad 強度在噪音內",
                 fontsize=10.5)
    save(fig, d, "n05_arms60k.png")


def n10_lean_loop(d):
    segs = ["backward", "forward", "週期 trim\n(1 次/1000 步窗)", "optimizer", "迴圈外", "loss", "其他"]
    rows = [  # (段落, 名稱, 真實步 ms（不含 trim）, 每次 trim s)
        ([84.85, 36.69, 27.87, 15.15, 8.00, 5.02, 3.33], "不開 lean（10-02）", 152.30, 27.87),
        ([68.64, 21.13, 27.59, 15.15, 8.03, 5.05, 3.32], "開 lean（10-02）", 120.54, 27.59),
        ([68.75, 21.15, 17.97, 15.14, 8.02, 5.06, 2.79], "★ 開 lean＋record_reduce（現行，10-03）", 120.69, 17.97),
        ([66.62, 27.59, 18.08, 15.14, 8.02, 5.05, 2.88], "＋tile_cull（不採用，10-03）", 124.96, 18.08)]
    cols = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD", "#CCB974", "#B0B0B0"]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for i, (row, nm, real, trim_s) in enumerate(rows):
        left = 0
        for v, c, s in zip(row, cols, segs):
            ax.barh(i, v, left=left, color=c, edgecolor="white", label=s if i == 0 else None)
            if v > 8:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
            left += v
        ax.text(left + 2, i, f"段落和 {left:.0f}／真實步 {real:.1f}（不含 trim）／增生期每步 {real + trim_s * 2:.0f} ms", va="center", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0, 285); ax.set_xlabel("每步 wall ms（逐段同步計時；b6、N 2.29M、@14,999 起 1,200 步含 1 次 trim；增生期每步＝真實步＋每次 trim÷500）")
    ax.legend(ncol=7, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("真實訓練迴圈：lean 每步 -20.9%；record_reduce 每次 trim -35%（增生期每步 -10.9%）；tile_cull 讓 forward +30% => 不採用", fontsize=10)
    save(fig, d, "n10_lean_loop.png")


def n10_kernels(d):
    cats = ["光柵 backward\n（逐配對）", "光柵 forward＋record\n（逐配對）", "Adam", "SSIM", "SH cat＋排序\n＋前處理", "其他"]
    # 「其他」＝profiler 的「其他小運算」扣掉 autograd 包裝列（_RasterizeGaussians*，與光柵 kernel 重複計時）與 SH cat
    off = [80.55, 74.29, 14.81, 10.02, 3.73 + 3.17 + 2.18, 59.60 - 3.73]
    on = [62.63, 58.47, 14.82, 9.57, 3.72 + 3.18 + 2.17, 61.89 - 3.72]
    x = np.arange(len(cats)); wd = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axs[0]
    barlabels(ax, ax.bar(x - wd / 2, off, wd, color=MUTED, label="不開 lean"), fmt="{:.1f}")
    barlabels(ax, ax.bar(x + wd / 2, on, wd, color=C3, label="開 lean（未含 record_reduce）"), fmt="{:.1f}")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5); ax.set_ylabel("kernel 時間（ms/步，含 1 台相機的 record）")
    ax.legend(fontsize=8)
    ax.set_title("kernel 拆解（10-02 資料：profiler，b6 @14,999，N 2.6M；之後的 record_reduce／飽和跳過＋先加總見 01(a)）", fontsize=10)
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
    ax.plot([60], [35.7], "*", color=C3, ms=15, mec="k", mew=0.6, label="b6 tile_cull 實測 -35.7%（6 視角）")
    ax.set_xlabel("訓練步數（千）"); ax.set_ylabel("佔 binning 配對（%）"); ax.set_ylim(0, 50); ax.legend(fontsize=7)
    ax.set_title("可剃掉的配對：tile_cull 實測與上限相符（但淨變慢，`10` §6）", fontsize=10)
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


REP12 = ["預設 conic", "★ dup4（現行）", "oreg0", "vpctilek", "oent", "v/c trim"]


def n12_recheck(d):
    """10-10 改版：只留代表性跑次（使用者）；幾何改成 60k 的 corr 長條＋官方合併模型參考線。"""
    corr = {6: [0.769, 0.910, 0.876, 0.893, 0.877, 0.939], 13: [0.743, 0.725, 0.916, 0.742, 0.849, 0.718]}
    fail = {6: [1.49, 1.28, 0.02, 1.42, 0.75, 1.49], 13: [2.30, 1.60, 0.12, 1.92, 1.01, 2.10]}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    x = np.arange(len(REP12)); wd = 0.38
    ax = axs[0]
    barlabels(ax, ax.bar(x - wd / 2, corr[6], wd, color=C1, label="b6"), fmt="{:.2f}", size=7)
    barlabels(ax, ax.bar(x + wd / 2, corr[13], wd, color=C2, label="b13"), fmt="{:.2f}", size=7)
    ax.axhline(0.993, color=C1, ls="--", lw=1); ax.axhline(0.948, color=C2, ls="--", lw=1)
    ax.text(5.6, 0.996, "官方合併模型（b6 視角）0.99", ha="right", fontsize=7.5, color=C1)
    ax.text(5.6, 0.951, "官方合併（b13）0.95", ha="right", fontsize=7.5, color=C2)
    ax.set_ylim(0.6, 1.05); ax.set_xticks(x); ax.set_xticklabels(REP12, rotation=15, fontsize=8)
    ax.set_ylabel("渲染深度 vs SfM 可見點深度的 corr（60k）"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("幾何：越接近 1 越準；oreg0 兩塊都明顯較好\n（官方是合併模型、塊外有鄰塊，單塊模型天生吃虧）", fontsize=9.5)
    ax = axs[1]
    k = ["k≥1", "k≥3", "k≥10"]; xk = np.arange(3); w4 = 0.2
    for i, (nm, v, c) in enumerate([("b6 顆數%", [0.49, 0.29, 0.12], C1), ("b6 不透明度質量%", [0.12, 0.03, 0.00], "#86b6ee"),
                                    ("b13 顆數%", [0.93, 0.44, 0.22], C2), ("b13 不透明度質量%", [0.57, 0.09, 0.01], "#f4a77f")]):
        barlabels(ax, ax.bar(xk + (i - 1.5) * w4, v, w4, color=c, label=nm), size=6.5)
    ax.set_xticks(xk); ax.set_xticklabels([f"空區票數 {s}" for s in k]); ax.set_ylabel("落在 SfM 光線空區的比例（%）")
    ax.legend(fontsize=7); ax.set_title("空區雕刻（dup4）：空區裡幾乎沒有顆粒 => floater 很少", fontsize=9.5)
    ax = axs[2]
    barlabels(ax, ax.bar(x - wd / 2, fail[6], wd, color=C1, label="b6（11,952 tile）"), fmt="{:.2f}%", size=7)
    barlabels(ax, ax.bar(x + wd / 2, fail[13], wd, color=C2, label="b13（16,853 tile）"), fmt="{:.2f}%", size=7)
    ax.set_xticks(x); ax.set_xticklabels(REP12, rotation=15, fontsize=8); ax.set_ylabel("失敗 tile（%）"); ax.legend(fontsize=8)
    ax.set_title("失敗區（GT std≥0.1、corr<0.6）：oreg0 幾乎消失（0.02／0.12%）\n（官方逐塊模型的同一統計：待量）", fontsize=9.5)
    fig.suptitle("60k 代表性跑次（5x5 b6／b13）", fontsize=10.5)
    save(fig, d, "n12_recheck.png")


def n12_heldout(d):
    """塊內 held-out（代表性跑次；10-10 拿掉 @1,499 對照與舊跑次）。"""
    val = {6: [30.48, 30.75, 31.38, 30.45, 29.86, 30.31], 13: [30.00, 30.32, 31.32, 29.99, 29.87, 29.80]}
    ho = {6: [18.21, 18.19, 18.18, 17.87, 18.43, 17.19], 13: [19.92, 20.17, 20.96, 19.80, 20.35, 19.30]}
    lp = {6: [.6507, .6370, .5674, .6475, .6264, .6596], 13: [.4555, .4325, .3639, .4467, .4239, .4570]}
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    for blk, c in [(6, C1), (13, C2)]:
        ax.scatter(val[blk], ho[blk], color=c, s=34, label=f"b{blk}")
        for nm, a, b in zip(REP12, val[blk], ho[blk]):
            ax.annotate(nm, (a, b), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel("val⊂train PSNR（dB，60k）"); ax.set_ylabel("塊內 held-out PSNR（dB，60k，下界）"); ax.legend(fontsize=8)
    ax.set_title("val vs 塊內 held-out：v/c 類 val 小輸、held-out 大輸\n（`09`／mergeview：損失大半在塊外，合併會丟掉）", fontsize=9.5)
    ax = axs[1]
    x = np.arange(len(REP12)); wd = 0.38
    barlabels(ax, ax.bar(x - wd / 2, lp[6], wd, color=C1, label="b6"), fmt="{:.3f}", size=7)
    barlabels(ax, ax.bar(x + wd / 2, lp[13], wd, color=C2, label="b13"), fmt="{:.3f}", size=7)
    ax.set_xticks(x); ax.set_xticklabels(REP12, rotation=15, fontsize=8.5)
    ax.set_ylim(0.3, 0.72); ax.set_ylabel("塊內 held-out LPIPS（越低越好）"); ax.legend(fontsize=8)
    ax.set_title("held-out LPIPS：oreg0 兩塊都最好", fontsize=10)
    fig.suptitle("塊內官方 held-out（b6 210 幀／b13 276 幀）—— 單塊模型看不到鄰塊內容，絕對值是下界；全場景對官方見 `09`", fontsize=10.5)
    save(fig, d, "n12_heldout.png")


# ══════════════ 10-10 圖表改版（使用者逐張意見）══════════════
SEGC = {"backward": "#4C72B0", "forward": "#55A868", "trim": "#C44E52", "optimizer": "#8172B3", "loss": "#CCB974",
        "loop": "#64B5CD", "other": "#AAAAAA"}
SEGN = {"backward": "backward", "forward": "forward（光柵化）", "trim": "週期 trim", "optimizer": "optimizer.step",
        "loss": "loss（L1+SSIM）", "loop": "迴圈外（dataloader／Lightning）", "other": "其他"}


def n01_steps(d):
    """每步逐段時間（增生期；lab 3090 [solo]；每標記點同步）：官方兩條線＋我方近期改動。"""
    rows = [
        ("官方 release b6\nN 2.50M（trim 不跑）｜10-06", dict(forward=47.45, backward=137.96, optimizer=15.66, loss=6.11, trim=0, loop=6.55, other=0), "＝真實步（無 trim）"),
        ("官方論文設定 b6\nN 1.75M（trim 執行）｜10-06", dict(forward=41.61, backward=127.95, optimizer=10.74, loss=6.13, trim=58.20, loop=0, other=0), "trim 為取樣窗攤平"),
        ("我方 lean 關\nb6 N 2.29M｜10-02", dict(forward=36.69, backward=84.85, optimizer=15.15, loss=5.02, trim=27875 / 500, loop=8.00, other=3.33), ""),
        ("＋lean_render\n（-20.9%／步）｜10-02", dict(forward=21.13, backward=68.64, optimizer=15.15, loss=5.05, trim=27589 / 500, loop=8.03, other=3.32), ""),
        ("＋record_reduce（現行）\ntrim 每次 -35%｜10-03", dict(forward=21.15, backward=68.75, optimizer=15.14, loss=5.06, trim=17975 / 500, loop=8.02, other=2.79), ""),
        ("＋飽和跳過＋block 內先加總\nN 2.58M｜10-10（60k 判定中）", dict(forward=19.07, backward=40.86, optimizer=14.79, loss=5.05, trim=17104 / 500, loop=7.36, other=4.60), "（60k 分數判定已排）"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.4))
    order = ["backward", "forward", "trim", "optimizer", "loss", "loop", "other"]
    for i, (lb, seg, note) in enumerate(rows):
        left = 0
        for k in order:
            v = seg[k]
            if v <= 0:
                continue
            ax.barh(i, v, left=left, color=SEGC[k], label=SEGN[k] if i == 2 else None)
            if v >= 12:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
            left += v
        ax.text(left + 3, i, f"{left:.0f} ms" + (f"　{note}" if note else ""), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0, 330); ax.set_xlabel("每步 ms（增生期；我方 trim 按真實週期 500 步攤平；全部 lab 3090 獨佔）")
    ax.legend(fontsize=8, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.set_title("同 N 附近的每步時間：官方每步 2 次 backward（深度 extra_loss）；我方 lean／record_reduce／飽和跳過＋先加總逐步降下來", fontsize=10.5)
    save(fig, d, "n01_steps.png")


def n01_scaling(d):
    """每步時間隨顆數：lab [solo] step_breakdown（單台相機、同 ckpt），60k 只調 cap 的跑次＋預設（conic、lean 關）。"""
    N = np.array([0.63, 1.08, 1.53, 2.34])
    data = {6: ([12.49, 14.30, 16.00, 20.56], [39.98, 46.56, 48.88, 55.99]),
            13: ([15.86, 20.09, 23.10, 28.13], [47.31, 57.49, 64.13, 76.55])}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, (blk, (fw, fb)) in zip(axs, data.items()):
        for ys, c, lb in [(fw, C3, "光柵化 forward"), (fb, C1, "forward＋backward")]:
            ys = np.array(ys); k, b0 = np.polyfit(N, ys, 1)
            ax.plot(N, ys, "o", color=c, ms=7, label=f"{lb}（{k:.1f} ms／百萬顆，截距 {b0:.1f}）")
            xx = np.linspace(0, 2.6, 20); ax.plot(xx, k * xx + b0, "--", color=c, lw=1)
        ax.set_xlim(0, 2.6); ax.set_ylim(0, None)
        ax.set_xlabel("終點顆數 N（百萬；60k、只調 cap）"); ax.set_ylabel("ms（單台相機，中位）")
        ax.legend(fontsize=8, loc="upper left"); ax.set_title(f"b{blk}：每步時間 ≈ 固定 + 斜率 × N（截距＝逐像素與與顆數無關的部分）", fontsize=9.5)
    fig.suptitle("時間隨顆數線性成長（lab 3090 [solo]、5x5 b6／b13、lean 關；N 加倍 forward+backward 約 +20~30%，不是 +100%）", fontsize=10.5)
    save(fig, d, "n01_scaling.png")


def n01_maxsplit(d):
    rows = [("本機 4050 b6 N 1.54M\n不開 max_split（09-11）", 340.8, MUTED), ("本機 4050 b6 N 1.54M\n開 max_split_size_mb:128（09-11）", 367.9, C2),
            ("★ 現行：lab 3090 b6 N 2.29M\n不鎖 VRAM、不開 max_split（lean＋rr，10-03）", 120.7, C1)]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i, (lb, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.6); ax.text(v + 4, i, f"{v:.1f} ms", va="center", fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("真實每步 ms（不含週期 trim）")
    ax.set_title("max_split_size_mb:128：同一台、同 N 下慢 8.0%、實際配置不變（只在鎖 VRAM 上限時防碎片 OOM 才開）；\n"
                 "10-04 起 lab 不鎖、不開 —— ⚠ 本機兩列與現行列機器與 N 都不同，只看 max_split 的相對差", fontsize=9.5)
    save(fig, d, "n01_maxsplit.png")


def n02_segments(d):
    """現行（lab、b6 @14,999 起、N 2.58M、lean＋rr＋飽和跳過＋先加總）每段峰值配置＋常駐／渲染分項。"""
    seg = [("迴圈外（常駐）", 2506, "#64B5CD"), ("forward", 3091, "#55A868"), ("loss", 3387, "#CCB974"),
           ("optimizer", 3468, "#8172B3"), ("trim", 3562, "#C44E52"), ("backward", 3868, "#4C72B0")]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    b = ax.bar(range(len(seg)), [s[1] for s in seg], color=[s[2] for s in seg])
    barlabels(ax, b, fmt="{:,.0f}", size=8)
    ax.axhline(6706, color=C2, ls=":", lw=1); ax.text(5.4, 6750, "段末保留 6,706 MiB（配置器快取，不鎖時會長）", ha="right", fontsize=8, color=C2)
    ax.axhline(6144, color="k", ls="--", lw=1); ax.text(5.4, 6190, "6 GiB", ha="right", fontsize=8)
    ax.set_xticks(range(len(seg))); ax.set_xticklabels([s[0] for s in seg], fontsize=8.5); ax.set_ylabel("段內峰值配置（MiB）")
    ax.set_ylim(0, 7300); ax.set_title("現行逐段 VRAM（lab b6 @14,999、N 2.58M；10-10）：峰值在 backward 3.87 GiB", fontsize=9.5)
    ax = axs[1]
    lbl = ["預設 conic\nb6 N 2.34M", "dup4（現行）\nb6", "預設 conic\nb13", "dup4（現行）\nb13"]
    res = [2.022, 2.022, 2.022, 2.022]; ren = [1.820, 1.824, 1.893, 1.877]
    ax.bar(range(4), res, color="#64B5CD", label="常駐：參數＋梯度＋Adam（只看 N：864 B／顆）")
    ax.bar(range(4), ren, bottom=res, color="#4C72B0", label="forward＋backward 峰值增量（渲染暫存）")
    for i in range(4):
        ax.text(i, res[i] + ren[i] + 0.05, f"{res[i] + ren[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(lbl, fontsize=8); ax.set_ylabel("GB（step_breakdown，60k 終點）"); ax.set_ylim(0, 5.0); ax.legend(fontsize=7.5, loc="upper center")
    ax.set_title("分項：常駐約 53%、渲染暫存約 47%", fontsize=9.5)
    save(fig, d, "n02_segments.png")


def n02_value(d):
    """性價比：品質 vs 訓練峰值 VRAM（60k、只調 cap、b6／b13 平均），擬合 PSNR = a + b·ln N、VRAM = c + d·cap，邊際 dB／GB 與膝點。"""
    cap = np.array([0.7, 1.2, 1.7, 2.6])
    psnr = np.array([(28.61 + 28.37) / 2, (29.39 + 29.06) / 2, (29.90 + 29.46) / 2, (30.48 + 30.00) / 2])
    ho = np.array([(17.93 + 19.75) / 2, (17.81 + 19.86) / 2, (17.82 + 19.97) / 2, (18.21 + 19.92) / 2])
    vram = np.array([(1.50 + 1.53) / 2, (2.25 + 2.24) / 2, (3.01 + 3.04) / 2, (4.35 + 4.37) / 2])
    b, a = np.polyfit(np.log(cap), psnr, 1)
    dv, cv = np.polyfit(cap, vram, 1)
    nmax = (6.44 - cv) / dv
    nn = np.linspace(0.3, nmax, 200); pp = a + b * np.log(nn); vv = cv + dv * nn
    xn = (vv - vv[0]) / (vv[-1] - vv[0]); yn = (pp - pp[0]) / (pp[-1] - pp[0])
    kn = int(np.argmax(yn - xn))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axs[0]
    ax.plot(vv, pp, "-", color=C1, lw=1.5, label=f"擬合 val PSNR = {a:.2f} + {b:.2f}·ln N（N 加倍 +{b * np.log(2):.2f} dB）")
    ax.plot(vram, psnr, "o", color=C1, ms=7, label="實測 val（b6／b13 平均）")
    ax2 = ax.twinx(); ax2.plot(vram, ho, "s--", color=C2, ms=6, lw=1, label="實測塊內 held-out（右軸；幾乎不隨 N 變）")
    ax2.set_ylim(ho.mean() - 1.5, ho.mean() + 1.5); ax2.set_ylabel("塊內 held-out PSNR（dB）", color=C2); ax2.legend(fontsize=7.5, loc="upper left")
    for x, y, c in zip(vram, psnr, cap):
        ax.annotate(f"cap {c}M", (x, y), xytext=(4, -12), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axvline(6.44, color="k", ls="--", lw=1); ax.text(6.4, 28.0, f"6 GiB\nN≈{nmax:.1f}M", ha="right", fontsize=8)
    ax.plot(vv[kn], pp[kn], "*", color="#c0392b", ms=15); ax.annotate(f"膝點 N≈{nn[kn]:.2f}M\n（{vv[kn]:.2f} GB）", (vv[kn], pp[kn]), xytext=(-95, 8), textcoords="offset points", fontsize=8.5, color="#c0392b")
    ax.set_xlabel("訓練峰值實佔 VRAM（GB；在 cap 時）"); ax.set_ylabel("PSNR（dB）"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(f"VRAM ≈ {cv:.2f} + {dv:.2f}·cap（GB）；品質對數成長", fontsize=9.5)
    ax = axs[1]
    marg = b / (nn * dv)
    ax.plot(nn, marg, "-", color=C1, lw=2)
    for n0 in (1.2, 2.6):
        ax.plot(n0, b / (n0 * dv), "o", color=C2); ax.annotate(f"N {n0}M：{b / (n0 * dv):.2f} dB／GB", (n0, b / (n0 * dv)), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axvline(nn[kn], color="#c0392b", ls=":", lw=1); ax.axvline(nmax, color="k", ls="--", lw=1)
    ax.set_xlabel("cap N（百萬）"); ax.set_ylabel("邊際收益：多 1 GB VRAM 換到的 dB")
    ax.set_title("性價比函數（邊際 dB／GB = b／(N·d)）隨 N 衰減：膝點之後每 GB 換到的品質不到一半", fontsize=9.5)
    fig.suptitle("顆數的性價比（60k、預設 conic、opacity_reg 0.002、lean 關；⚠ 現行 lean 峰值約 -0.3 GB、oreg0 約 -0.4 GB，曲線會左移）", fontsize=10.5)
    save(fig, d, "n02_value.png")


def n05_main(d):
    """主要跑次 vs 預設基準 cs60_conic，b6／b13 平均（60k、同 N 2.34M）。"""
    arms = ["★ dup4（現行配方）", "oreg0（opacity_reg 0）", "vpctilek", "oent", "v/c trim"]
    dval = [0.295, 1.11, -0.02, -0.375, -0.185]
    dho = [0.111, 0.505, -0.23, 0.32, -0.82]
    dho_m = [np.nan, np.nan, -0.16, -0.163, np.nan]
    dload = [-6.95, -36.1, -37.5, -49.95, -45.4]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]; y = np.arange(len(arms)); h = 0.27
    ax.barh(y - h, dval, h, color=C1, label="val PSNR")
    ax.barh(y, dho, h, color=C2, label="塊內 held-out（單塊模型）")
    ax.barh(y + h, [0 if v != v else v for v in dho_m], h, color=C3, label="只算合併後會留下的部分（mergeview）")
    for i in range(len(arms)):
        for off, v in ((-h, dval[i]), (0, dho[i]), (h, dho_m[i])):
            if v == v:
                ax.text(v + (0.02 if v >= 0 else -0.02), i + off, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(y); ax.set_yticklabels(arms); ax.invert_yaxis()
    ax.set_xlabel("Δ dB vs 預設 cs60_conic（b6／b13 平均）"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("品質：oreg0 最大；v/c 類在單塊 held-out 的損失大半在塊外（合併會丟掉）", fontsize=9.5)
    ax = axs[1]
    b_ = ax.barh(y, dload, color=MUTED)
    for i, v in enumerate(dload):
        ax.text(v - 1, i, f"{v:.0f}%", va="center", ha="right", fontsize=8)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlim(-60, 5)
    ax.set_xlabel("Δ 精確 Load 中位（%，越左越便宜）"); ax.set_title("渲染成本", fontsize=9.5)
    fig.suptitle("60k 主要跑次（5x5 b6／b13 平均、同 N 2.34M、單次）；10-10 起改在 4x4 上以 oreg0 為基底續測", fontsize=10.5)
    save(fig, d, "n05_main.png")


def n05_official(d):
    """全場景（合併模型）對官方：渲染成本 vs 官方 741 幀 held-out（同一支評分工具、同 600 台相機量 Load）。"""
    pts = [("★ 我方 4x4", 2.52, 27.739, 0.129, 17.63, C1), ("官方 release", 5.92, 27.261, 0.154, 24.48, MUTED),
           ("官方論文設定", 5.92, 27.280, 0.147, 24.72, C2)]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for nm, x, y, lp, n, c in pts:
        ax.plot(x, y, "o", color=c, ms=12)
        ax.annotate(f"{nm}\nPSNR {y:.2f}／LPIPS {lp:.3f}\nN {n:.1f}M", (x, y), xytext=(14, -42) if "release" in nm else ((14, 4) if "論文" in nm else (14, -10)), textcoords="offset points", fontsize=8.5)
    ax.errorbar(5.92, 27.4, yerr=0.2, fmt="none", ecolor=MUTED, capsize=6, lw=1)
    ax.text(5.98, 27.62, "release 合理區間 27.2~27.6\n（光柵器偏差，09 §0.1）", fontsize=7.5, color=INK2)
    ax.set_xlim(1.5, 7.5); ax.set_ylim(27.0, 27.95)
    ax.set_xlabel("合併模型渲染成本：精確 Σtiles 中位（百萬／視角）"); ax.set_ylabel("官方 741 幀 held-out PSNR")
    ax.set_title("全場景對官方 CityGSV2：我方渲染成本 -57%、品質同級或略好", fontsize=10)
    save(fig, d, "n05_official.png")


def n05_frontier60k(d):
    """60k 實測成本－品質前緣：只調 cap 的曲線（實測，取代推估）＋各臂；同 Load 下比只調 cap 高多少。"""
    cap = {6: ([1.419, 1.883, 2.195, 2.773], [28.61, 29.39, 29.90, 30.48]), 13: ([1.637, 2.151, 2.601, 3.337], [28.37, 29.06, 29.46, 30.00])}
    arms = {6: [("★ dup4", 2.616, 30.75), ("oreg0", 1.773, 31.38), ("vpctilek", 1.739, 30.45), ("oent", 1.444, 29.86),
                ("v/c trim", 1.536, 30.31), ("conic 關", 4.666, 30.42)],
            13: [("★ dup4", 3.059, 30.32), ("oreg0", 2.131, 31.32), ("vpctilek", 2.078, 29.99), ("oent", 1.603, 29.87),
                 ("v/c trim", 1.795, 29.80), ("conic 關", 6.180, 29.84)]}
    mk = {"★ dup4": ("*", C1), "oreg0": ("D", "#c0392b"), "vpctilek": ("s", C3), "oent": ("^", C4), "v/c trim": ("v", C2), "conic 關": ("x", MUTED)}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, blk in zip(axs, (6, 13)):
        lx, ly = cap[blk]
        ax.plot(lx, ly, "-o", color=INK2, lw=1.6, label="只調 cap（0.7／1.2／1.7／2.6M，實測）")
        for x, y, c_ in zip(lx, ly, ("0.7M", "1.2M", "1.7M", "2.6M")):
            ax.annotate(c_, (x, y), xytext=(4, -11), textcoords="offset points", fontsize=7, color=MUTED)
        for nm, x, y in arms[blk]:
            m, c = mk[nm]
            ax.plot(x, y, m, color=c, ms=10 if m == "*" else 7, label=nm)
            ref = np.interp(np.log(x), np.log(lx), ly) if lx[0] <= x <= lx[-1] else np.nan
            if ref == ref:
                ax.annotate(f"{y - ref:+.2f}", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=c)
        ax.set_xscale("log"); ax.set_xlabel("渲染成本：精確 Load 中位（百萬／視角，對數）"); ax.set_ylabel("val PSNR（60k）")
        ax.legend(fontsize=7.5, loc="lower right"); ax.set_title(f"b{blk}：數字＝同 Load 下比只調 cap 高多少（dB）", fontsize=9.5)
    fig.suptitle("60k 成本－品質前緣（實測；取代舊版用 22k 斜率推估的灰色欄）：oreg0 同成本高 +2 dB 以上、v/c +1.2~1.5", fontsize=10.5)
    save(fig, d, "n05_frontier60k.png")


def n04_time(d):
    """conic 開關的時間（60k 終點模型、同 N 2.34M、lab [solo] step_breakdown 單台相機）—— 取代 22k 三槽平行版。"""
    v = {6: ([24.49, 64.90], [20.56, 55.99]), 13: ([37.59, 92.21], [28.13, 76.55])}
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, blk in zip(axs, (6, 13)):
        off, on = v[blk]; x = np.arange(2); wd = 0.38
        barlabels(ax, ax.bar(x - wd / 2, off, wd, color=MUTED, label="conic 關"), fmt="{:.1f}")
        b = ax.bar(x + wd / 2, on, wd, color=C3, label="conic 開（現行預設）")
        for bb, a_, o_ in zip(b, off, on):
            ax.annotate(f"{o_:.1f}\n{100 * (o_ / a_ - 1):+.1f}%", (bb.get_x() + bb.get_width() / 2, o_), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["光柵化 forward", "forward＋backward"]); ax.set_ylabel("ms（單台相機中位）"); ax.legend(fontsize=8)
        ax.set_title(f"b{blk}（60k、N 2.34M、[solo]）", fontsize=10)
    fig.suptitle("conic 精確外接盒的時間：forward -16~-25%、forward＋backward -14~-17%（lab [solo] 實測，取代舊版 22k 數字）", fontsize=10.5)
    save(fig, d, "n04_time.png")


def n04_load(d):
    """conic 開關的渲染成本與 VRAM（60k 終點、同 N 2.34M）。"""
    L = {"b6 中位": (4.666, 2.773), "b6 max": (7.586, 5.190), "b13 中位": (6.180, 3.337), "b13 max": (9.821, 5.712)}
    V = {"b6": (4.58, 4.35), "b13": (4.68, 4.37)}
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, D_, yl, tt in [(axs[0], L, "精確 Σtiles（百萬／視角）", "渲染成本：-32~-46%"), (axs[1], V, "訓練峰值實佔（GB）", "VRAM：只 -5~-7%（VRAM 由 N 主導）")]:
        x = np.arange(len(D_)); wd = 0.38
        a_ = [v[0] for v in D_.values()]; o_ = [v[1] for v in D_.values()]
        barlabels(ax, ax.bar(x - wd / 2, a_, wd, color=MUTED, label="conic 關"), fmt="{:.2f}")
        b = ax.bar(x + wd / 2, o_, wd, color=C3, label="conic 開（現行）")
        for bb, aa, oo in zip(b, a_, o_):
            ax.annotate(f"{oo:.2f}\n{100 * (oo / aa - 1):+.0f}%", (bb.get_x() + bb.get_width() / 2, oo), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(list(D_.keys())); ax.set_ylabel(yl); ax.legend(fontsize=8); ax.set_title(tt, fontsize=10)
    fig.suptitle("conic 精確外接盒（60k、同 N 2.34M）：省的是渲染成本與時間，不是 VRAM", fontsize=10.5)
    save(fig, d, "n04_load.png")


# ══════════════ 組合 ══════════════
DOCS = {
    "01_訓練時間組成": ["n01_steps", "n01_scaling", "n01_maxsplit"],
    "02_VRAM與配置器": ["n02_segments", "n02_value"],
    "03_儲存與RAM": ["fig4_ckpt_storage", "fig6_run_dir", "fig5_ram_cache"],
    "04_外接盒與binning成本": ["cc1_quality_delta", "n04_time", "n04_load", "cc4_proxy_decoupled"],
    "05_成本品質前緣與vc_trim": ["n05_main", "n05_official", "n05_frontier60k", "n05_cap60k", "fig8_trimvpc_time"],
    "06_事後剪枝與fine-tune": ["f3_tile_topk_posthoc", "f6_prune_finetune"],
    "07_成本感知取樣與預算": ["n07_sampling"],
    "08_初始化": ["n08_init20k", "f4_init60k", "n08_init60k"],
    "09_官方參考線": ["n09_full44_merged", "n09_full44_blocks", "n09_stepprof", "n09_evalgap", "n09_official_blocks", "n09_trim", "f5_official_prune"],
    "10_速度優化_lean與kernel拆解": ["n10_kernels"],
    "11_逐配對工作量上限": ["n11_tiles"],
    "12_新年代重測_幾何floater失敗區": ["n12_recheck", "n12_heldout"],
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
    for f in (n02_current, n05_cap60k, n07_sampling, n08_init20k, n08_init60k, n09_official_blocks, n09_full44_merged, n09_full44_blocks, n09_stepprof, n09_evalgap, n05_arms60k, n09_trim,
              n01_steps, n01_scaling, n01_maxsplit, n02_segments, n02_value, n05_main, n05_official, n05_frontier60k, n04_time, n04_load,
              n10_lean_loop, n10_kernels, n11_tiles, n12_recheck, n12_heldout):
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
