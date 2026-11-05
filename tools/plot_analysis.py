#!/usr/bin/env python
"""紀錄/實驗分析/ 的圖（2026-10-03）：每份分析文件一張**同名** PNG，多個面板以 (a)(b)(c)… 標號。

做法：
  1. 舊的三支畫圖腳本（plot_cost_frontier／plot_resource_report／plot_commonconf_ab）以 CITYGS_FIG_OUT 指向暫存目錄畫面板
  2. 本檔畫新的面板（速度、tile 上限、新年代重測、init、成本感知取樣、CityGS 參考線）
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
# ★ 2026-11-05 紀錄整理（使用者）：實驗分析/ 只留開啟的線；定案與過程文件移到 new_archived/ 下 => 每張圖寫到「同名 .md 所在的資料夾」
DOC_DIRS = [DST, os.path.join(ROOT, "紀錄"), os.path.join(ROOT, "紀錄", "new_archived", "實驗分析_定案"), os.path.join(ROOT, "紀錄", "new_archived", "實驗分析_過程")]


def doc_dir(doc):
    for d in DOC_DIRS:
        if os.path.exists(os.path.join(d, doc + ".md")):
            return d
    raise SystemExit(f"⛔ 找不到 {doc}.md（找過 {DOC_DIRS}）")
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
    """CityGS block 3／6：release（trim 從未執行）vs 論文設定（trim 執行＋ω0.9、prune 0.025）。10-10：拿掉「一行修正」那欄（使用者）。"""
    lines = ["CityGS release\n（trim 從未執行）", "CityGS 論文設定\n（trim 執行＋ω0.9、prune 0.025）"]
    n3 = [5.23, 3.05]; v3 = [31.69, 31.78]; n6 = [7.11, 5.31]; v6 = [30.14, 30.35]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.4))
    x = np.arange(2); wd = 0.36
    for ax, (a3, a6, fmt, yl, tt) in zip(axs, [(n3, n6, "{:.2f}M", "終點顆數 N（百萬）", "終點顆數：論文設定少 25~42%"),
                                                (v3, v6, "{:.2f}", "val PSNR（dB，val⊂train）", "val：論文設定反而略高")]):
        barlabels(ax, ax.bar(x - wd / 2, a3, wd, color=C1, label="CityGS block 3"), fmt=fmt)
        barlabels(ax, ax.bar(x + wd / 2, a6, wd, color=C2, label="CityGS block 6"), fmt=fmt)
        ax.set_xticks(x); ax.set_xticklabels(lines, fontsize=8.5); ax.set_ylabel(yl); ax.legend(fontsize=8); ax.set_title(tt, fontsize=10)
    axs[1].set_ylim(29.5, 32.2)
    save(fig, d, "n09_official_blocks.png")


def n09_trim(d):
    """CityGS block 3：trim 有沒有執行（同 CityGS 程式碼與 config，只差 trim 是否真的跑）。"""
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
    fig.suptitle("CityGS block 3：trim 真的執行時顆數 -83%、峰值 -62%、val -0.8 dB —— release 的 aerial trim 階段其實從沒剪過", fontsize=10.5)
    save(fig, d, "n09_trim.png")


# 10-09：我方最佳解 4x4（16 塊 [solo]、lean＋record_reduce、dup4、cap 2.6M）vs CityGS release vs CityGS 論文設定
#   資料：lab outputs/lab/full44_best（chart_data、台帳）、cityGS_origin(_trimfix)/outputs/*/blocks、logs/citygs_official_resources.tsv
#   峰值實佔一律 GB（1e9）：CityGS MiB x 1.048576 / 1000；時間 = 分鐘（我方台帳 START->DONE；CityGS wall_s）
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
# 10-10：合併模型在各 4x4 塊 held-out 視角上的逐塊平均（tools/merged_heldout_by_block.py；我方 PNG 量化版、CityGS release 用我方評分工具的逐幀 CSV）
M44 = {"n": [172, 213, 228, 193, 240, 306, 324, 271, 237, 359, 382, 271, 116, 181, 219, 171],
       "o": [28.101, 27.454, 26.762, 26.711, 28.033, 27.617, 27.212, 27.054, 28.039, 27.572, 27.458, 27.344, 27.823, 27.314, 27.977, 28.396],
       "r": [27.340, 27.180, 26.707, 26.687, 27.385, 27.312, 27.067, 26.976, 27.321, 27.190, 27.201, 27.209, 27.098, 26.905, 27.493, 27.901]}

LBL3 = ["★ 我方 4x4", "CityGS release", "CityGS 論文設定"]


def n09_full44_merged(d):
    """合併模型、MatrixCity test 741 幀 held-out，同一支工具（tools/eval_official_test.py）。"""
    # ★ 10-30 換成現行最佳 A＋dist4（g44s30km25fssh16di4_best1maskdistrfed10b4）與輕量版 cap 1.5M（實驗分析 21 §1）；
    #   16 塊總 N＝16 x 每塊終點 N（cap 的 0.9：2.25M／1.35M）
    lab = ["★A＋\ndist4", "A cap\n1.5M", "full44\n舊最佳", "release", "論文\n設定"]
    q = {"PSNR": [28.058, 27.931, 27.739, 27.261, 27.280], "SSIM": [0.8824, 0.8774, 0.8790, 0.8671, 0.8697],
         "LPIPS（越低越好）": [0.1142, 0.1286, 0.1293, 0.1540, 0.1467]}
    r = {"合併 N（百萬）": [27.75, 17.12, 17.63, 24.48, 24.72], "渲染 FPS": [40.6, 50.3, 35.6, 21.9, 21.9],
         "16 塊總 N（百萬）": [36.0, 21.6, 37.44, 129.55, 87.89]}
    fig, axs = plt.subplots(1, 6, figsize=(18, 3.8))
    cols = ["#123f7a", C3, C1, MUTED, C2]
    for ax, (k, v) in zip(axs, list(q.items()) + list(r.items())):
        bars = ax.bar(range(5), v, color=cols, width=0.72)
        fmt = "{:.3f}" if max(v) < 1 else ("{:.2f}" if max(v) < 50 else "{:.1f}")
        barlabels(ax, bars, fmt=fmt, size=7.5)
        lo = min(v); hi = max(v)
        if k in ("PSNR", "SSIM", "LPIPS（越低越好）"):
            ax.set_ylim(lo - (hi - lo) * 1.2, hi + (hi - lo) * 0.6)
        ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=7.5)
        ax.set_title(k, fontsize=10)
    fig.suptitle("合併模型・MatrixCity test 741 幀 held-out（同一支評分工具）：★ 現行 A＋dist4 28.058／.8824／.1142、FPS 1.85 倍、16 塊純訓練 5.04 h；"
                 "輕量版 cap 1.5M 用比舊最佳少 3% 的顆數贏它\n（PSNR 領先保守估計：見面板 d 的光柵器偏差；A＋dist4 合併顆數比 release 多 13%，半精度部署 3.15 vs 3.37 GiB）", fontsize=10)
    save(fig, d, "n09_full44_merged.png")


def n09_full44_blocks(d):
    x = np.arange(16); wd = 0.27
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.4))
    ax = axs[0]
    for k, c, dx, lb in [("oh", C1, -wd, LBL3[0]), ("rh", MUTED, 0, LBL3[1]), ("ph", C2, wd, LBL3[2])]:
        ax.bar(x + dx, F44[k], wd, color=c, label=lb)
    ax.set_ylim(15, 27.5); ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.set_xlabel("4x4 block"); ax.set_ylabel("同塊 held-out PSNR（dB，下界）"); ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.set_title("同塊 held-out（我方塊視角）：平均 20.69／20.23／19.90；LPIPS CityGS 16 塊全較好\n"
                 "⚠ 偏向 CityGS：CityGS 單塊從全場景 coarse 起步、看得到塊外", fontsize=9.5)
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


def n09_merged_blocks(d):
    """同一批塊視角：單塊模型 vs 合併模型（我方 vs CityGS release）。b12 單塊輸 3.8 dB，合併後贏 0.73。"""
    x = np.arange(16); wd = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(16, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    ds = np.array(F44["oh"]) - np.array(F44["rh"]); dm = np.array(M44["o"]) - np.array(M44["r"])
    ax.bar(x - wd / 2, ds, wd, color=MUTED, label="單塊模型（圖 b 左）")
    ax.bar(x + wd / 2, dm, wd, color=C1, label="合併模型（同一批視角）")
    ax.axhline(0, color="k", lw=0.8)
    for i in (12,):
        ax.annotate(f"b12 單塊 {ds[i]:+.2f}\n合併 {dm[i]:+.2f}", (i, ds[i]), xytext=(-70, -6), textcoords="offset points", fontsize=8.5, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    ax.set_xticks(x); ax.set_xlabel("4x4 block（held-out 視角 = 相機落在該塊訓練相機 AABB 內）"); ax.set_ylabel("我方 − CityGS release（dB）")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("同一批視角：單塊模型 我方 9/16 塊贏；合併模型 我方 16/16 塊贏（+0.02~+0.76）", fontsize=9.5)
    ax = axs[1]
    ax.bar([0, 1], [F44["oh"][12], F44["rh"][12]], 0.35, color=[MUTED, MUTED], alpha=0.6)
    ax.bar([0.4, 1.4], [M44["o"][12], M44["r"][12]], 0.35, color=[C1, C1])
    for xx, v in zip([0, 1, 0.4, 1.4], [F44["oh"][12], F44["rh"][12], M44["o"][12], M44["r"][12]]):
        ax.text(xx, v + 0.2, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks([0.2, 1.2]); ax.set_xticklabels(["我方", "CityGS release"]); ax.set_ylim(15, 30); ax.set_ylabel("held-out PSNR（b12 的 116 幀）")
    ax.set_title("b12（角落塊 000_003）：灰＝單塊、藍＝合併\n我方單塊缺 5.3 dB 的塊外內容；CityGS 單塊（9.71M，從全場景 coarse 微調）只缺 0.7", fontsize=9.5)
    save(fig, d, "n09_merged_blocks.png")


def n09_stepprof(d):
    """同 N 附近的每步逐段時間（每標記點同步）：CityGS stepprof b6（開頭 1,500 步）vs 我方 b6 @14,999 起 1,200 步。"""
    segs = ["forward", "backward", "optimizer", "loss", "週期 trim\n（÷500 攤平）", "其他"]
    ours = [21.15, 68.75, 15.14, 5.06, 35.95, 10.81]
    rel = [47.45, 137.96, 15.66, 6.11, 0.0, 6.55]
    pap = [41.61, 127.95, 10.74, 6.13, 58.20, 0.0]   # CityGS stepprof 的 trim 列已是每步攤平（1,500 步窗）；其他欄為負值（-14.8）記 0
    names = ["★ 我方 b6\nN 2.29M（lean＋rr）", "CityGS release b6\nN 2.50M（trim 不跑）", "CityGS 論文設定 b6\nN 1.75M（含 trim）"]
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
    ax.set_title("同 N 附近：CityGS 每步做 2 次 backward（loss 與 depth 的 extra_loss 各一次，各約 64~69 ms）、forward 帶幾何通道；"
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
    ax.set_xlabel("CityGS 工具（CityGS 光柵器）逐幀 PSNR"); ax.set_ylabel("我方工具（我方光柵器）逐幀 PSNR")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046); cb.set_label("CityGS 渲染亮度／GT 亮度", fontsize=8)
    ax.set_title("　　同一個 CityGS release 模型、741 幀\n　　左上角那群＝CityGS 渲染變暗／全黑的幀", fontsize=9.5)
    ax = axs[1]
    x = np.arange(len(A)); wd = 0.4
    ax.bar(x - wd / 2, [float(r["psnr"]) for r in A], wd, color=MUTED, label="CityGS 程式碼＋CityGS 光柵器（直接渲染）")
    ax.bar(x + wd / 2, [float(r["psnr"]) for r in B], wd, color=C1, label="CityGS 程式碼＋只換我方光柵器")
    ax.axvline(15.5, color=INK2, lw=0.8, ls=":")
    ax.text(7.5, 33, "MatrixCity test 最暗的 16 幀", ha="center", fontsize=8.5, color=INK2)
    ax.text(18.5, 33, "正常幀 6 張", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([r["name"][:4] for r in A], rotation=60, fontsize=7.5); ax.set_ylim(0, 35)
    ax.set_ylabel("PSNR（dB）"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title("二分：繞過 test 迴圈仍重現（0184 10.31，逐位數相同）；只換光柵器即恢復\n暗幀 alpha 0.9997（畫面是滿的）=> 壞在顏色路徑", fontsize=9.5)
    ax = axs[2]
    est = [("CityGS 工具（原數字）", 25.79, MUTED), ("CityGS 工具・只算正常幀\n（亮度比 ≥0.85，645 幀）", 27.18, MUTED),
           ("我方工具（暗幀已正常）", 27.26, MUTED), ("我方工具＋正常幀光柵器偏差補回\n（+0.35，上界）", 27.61, MUTED),
           ("★ 我方 4x4（我方工具）", 27.74, C1)]
    for i, (lb, v, c) in enumerate(est):
        ax.barh(i, v, color=c, height=0.6); ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(est))); ax.set_yticklabels([e[0] for e in est], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(25.5, 28.2); ax.set_xlabel("MatrixCity test 741 幀 PSNR（dB）")
    ax.set_title("CityGS release 的合理分數 ≈ 27.2~27.6\n=> 我方領先 +0.1~+0.5 dB（論文報告 27.23）", fontsize=9.5)
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
    """10-10 改版：只留代表性跑次（使用者）；幾何改成 60k 的 corr 長條＋CityGS 合併模型參考線。"""
    corr = {6: [0.769, 0.910, 0.876, 0.893, 0.877, 0.939], 13: [0.743, 0.725, 0.916, 0.742, 0.849, 0.718]}
    fail = {6: [1.49, 1.28, 0.02, 1.42, 0.75, 1.49], 13: [2.30, 1.60, 0.12, 1.92, 1.01, 2.10]}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    x = np.arange(len(REP12)); wd = 0.38
    ax = axs[0]
    barlabels(ax, ax.bar(x - wd / 2, corr[6], wd, color=C1, label="b6"), fmt="{:.2f}", size=7)
    barlabels(ax, ax.bar(x + wd / 2, corr[13], wd, color=C2, label="b13"), fmt="{:.2f}", size=7)
    ax.axhline(0.993, color=C1, ls="--", lw=1); ax.axhline(0.948, color=C2, ls="--", lw=1)
    ax.text(5.6, 0.996, "CityGS 合併模型（b6 視角）0.99", ha="right", fontsize=7.5, color=C1)
    ax.text(5.6, 0.951, "CityGS 合併（b13）0.95", ha="right", fontsize=7.5, color=C2)
    ax.set_ylim(0.6, 1.05); ax.set_xticks(x); ax.set_xticklabels(REP12, rotation=15, fontsize=8)
    ax.set_ylabel("渲染深度 vs SfM 可見點深度的 corr（60k）"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("幾何：越接近 1 越準；oreg0 兩塊都明顯較好\n（CityGS 是合併模型、塊外有鄰塊，單塊模型天生吃虧）", fontsize=9.5)
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
    ax.set_title("失敗區（GT std≥0.1、corr<0.6）：oreg0 幾乎消失（0.02／0.12%）\n（CityGS 逐塊模型的同一統計：待量）", fontsize=9.5)
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
    fig.suptitle("塊內 MatrixCity test held-out（b6 210 幀／b13 276 幀）—— 單塊模型看不到鄰塊內容，絕對值是下界；全場景對 CityGS 見 `09`", fontsize=10.5)
    save(fig, d, "n12_heldout.png")


# ══════════════ 10-10 圖表改版（使用者逐張意見）══════════════
SEGC = {"backward": "#4C72B0", "forward": "#55A868", "trim": "#C44E52", "optimizer": "#8172B3", "loss": "#CCB974",
        "loop": "#64B5CD", "other": "#AAAAAA"}
SEGN = {"backward": "backward", "forward": "forward（光柵化）", "trim": "週期 trim", "optimizer": "optimizer.step",
        "loss": "loss（L1+SSIM）", "loop": "迴圈外（dataloader／Lightning）", "other": "其他"}


def n01_steps(d):
    """每步逐段時間（增生期；lab 3090 [solo]；每標記點同步）：CityGS 兩條線＋我方近期改動。"""
    rows = [
        ("CityGS release b6\nN 2.50M（trim 不跑）｜10-06", dict(forward=47.45, backward=137.96, optimizer=15.66, loss=6.11, trim=0, loop=6.55, other=0), "＝真實步（無 trim）"),
        ("CityGS 論文設定 b6\nN 1.75M（trim 執行）｜10-06", dict(forward=41.61, backward=127.95, optimizer=10.74, loss=6.13, trim=58.20, loop=0, other=0), "trim 為取樣窗攤平"),
        ("我方 lean 關\nb6 N 2.29M｜10-02", dict(forward=36.69, backward=84.85, optimizer=15.15, loss=5.02, trim=27875 / 500, loop=8.00, other=3.33), ""),
        ("＋lean_render\n（-20.9%／步）｜10-02", dict(forward=21.13, backward=68.64, optimizer=15.15, loss=5.05, trim=27589 / 500, loop=8.03, other=3.32), ""),
        ("＋record_reduce（現行）\ntrim 每次 -35%｜10-03", dict(forward=21.15, backward=68.75, optimizer=15.14, loss=5.06, trim=17975 / 500, loop=8.02, other=2.79), ""),
        ("＋飽和跳過＋block 內先加總\nN 2.58M｜10-10", dict(forward=19.07, backward=40.86, optimizer=14.79, loss=5.05, trim=17104 / 500, loop=7.36, other=4.60), "（60k 換塊評測：與 best0 平手 +0.01／+0.03 => 可開）"),
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
    ax.set_title("同 N 附近的每步時間：CityGS 每步 2 次 backward（深度 extra_loss）；我方 lean／record_reduce／飽和跳過＋先加總逐步降下來", fontsize=10.5)
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
    """全場景（合併模型）對 CityGS：渲染成本 vs MatrixCity test 741 幀 held-out（同一支評分工具、同 600 台相機量 Load）。"""
    pts = [("★ 我方 4x4", 2.52, 27.739, 0.129, 17.63, C1), ("CityGS release", 5.92, 27.261, 0.154, 24.48, MUTED),
           ("CityGS 論文設定", 5.92, 27.280, 0.147, 24.72, C2)]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for nm, x, y, lp, n, c in pts:
        ax.plot(x, y, "o", color=c, ms=12)
        ax.annotate(f"{nm}\nPSNR {y:.2f}／LPIPS {lp:.3f}\nN {n:.1f}M", (x, y), xytext=(14, -42) if "release" in nm else ((14, 4) if "論文" in nm else (14, -10)), textcoords="offset points", fontsize=8.5)
    ax.errorbar(5.92, 27.4, yerr=0.2, fmt="none", ecolor=MUTED, capsize=6, lw=1)
    ax.text(5.98, 27.62, "release 合理區間 27.2~27.6\n（光柵器偏差，09 §0.1）", fontsize=7.5, color=INK2)
    ax.set_xlim(1.5, 7.5); ax.set_ylim(27.0, 27.95)
    ax.set_xlabel("合併模型渲染成本：精確 Σtiles 中位（百萬／視角）"); ax.set_ylabel("MatrixCity test 741 幀 held-out PSNR")
    ax.set_title("全場景對 CityGSV2：我方渲染成本 -57%、品質同級或略好", fontsize=10)
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
def n13_struct(d):
    """13 §2.2：60k 4x4 結構（數字＝labrun/struct_cmp.py、speck.py 2026-10-12）。"""
    arms = ["基底 best0", "v/c × tile 名次 trim", "opacity 二元熵正則", "兩者合開"]
    tiny = {"b6": [10.5, 40.0, 12.9, 64.1], "b12": [9.8, 31.9, 22.8, 59.7]}
    eff = {"b6": [61.2, 27.4, 61.8, 11.3], "b12": [77.5, 47.0, 72.3, 26.6]}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(len(arms)); w = 0.38
    for ax, dat, yl, tt in ((axs[0], tiny, "%", "微小顆粒（最大軸 < 1e-4）佔全部顆粒"), (axs[1], eff, "萬顆", "盒內「有效」顆粒（非微小）")):
        for k, (b, c) in enumerate((("b6", C1), ("b12", C2))):
            barlabels(ax, ax.bar(x + (k - 0.5) * w, dat[b], w, color=c, label=f"block {b[1:]}"), "{:.1f}")
        ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=9); ax.set_ylabel(yl); ax.set_title(tt, fontsize=10); ax.legend(fontsize=8)
    save(fig, d, "n13_struct.png")


_RACE_SHOW = [("現行 trim：v（每像素平均貢獻，最好視角）", "現行 trim v（最好視角）", C1, "o"),
              ("v/c（訓練期判準）", "v/c（訓練期判準）", C2, "v"),
              ("A：V / 覆蓋像素", "V/C（總貢獻÷覆蓋像素）", C3, "s"),
              ("便宜成本：V / (β像素 + 覆蓋像素)", "V/(β+C)，β 像素單位", C4, "D"),
              ("A：總貢獻 V", "總貢獻 V", "#b8860b", "^"),
              ("隨機（下界）", "隨機（下界）", MUTED, "x")]


def _race(name):
    import pandas as pd
    R = os.path.join(DST, "data", "discriminant_race")
    df = pd.read_csv(os.path.join(R, name + ".csv"))
    ch = os.path.join(R, name + "_cheap.csv")
    if os.path.exists(ch):
        c = pd.read_csv(ch); df = pd.concat([df, c[c["判別式"] != "不剪"]])
    return df


def n13_race(d):
    """13 §1：離線判別式比賽（tools/discriminant_race.py），同顆數預算，held-out 本塊區域 PSNR 對 Load。"""
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (nm, tt) in zip(axs, (("g44_best0_b12", "b12 基底 best0（微小顆粒 4.5%）"), ("full44_b10", "b10 大樓塊（full44）"),
                                   ("g44_best0vo_b12", "壓力測試：v/c×名次＋二元熵合開 b12（微小 26%）"))):
        df = _race(nm); b = df[df["判別式"] == "不剪"].iloc[0]
        for key, lab, c, m in _RACE_SHOW:
            x = df[(df["判別式"] == key) & (df["預算"] == "顆數")].sort_values("p")
            if len(x) == 0:
                continue
            ax.plot(100 * (x["Load中位M"] / b["Load中位M"] - 1), x["區域PSNR"] - b["區域PSNR"], marker=m, color=c, label=lab, lw=1.4, ms=5)
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_title(tt, fontsize=10)
        ax.set_xlabel("精確 Load 中位變化 %（三點＝保留 90/70/50% 顆數）"); ax.set_ylabel("Δ 本塊區域 PSNR（dB）")
        lo = min(-4.0, ax.get_ylim()[0]); ax.set_ylim(max(lo, -4.0), 0.3)
    axs[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("單次剪枝不重訓、MatrixCity test held-out、只剪本塊盒內顆粒（v/c 掉到 −10 dB，超出畫面）", fontsize=10)
    save(fig, d, "n13_race.png")


def n13_tiny(d):
    """13 §1：壓力測試模型上，同顆數保留 90% 時，被剪掉的顆粒中有多少是微小顆粒（盒內微小佔 26%）。"""
    df = _race("g44_best0vo_b12")
    x = df[(df["預算"] == "顆數") & (df["p"] == 0.9)]
    rows = [(lab, float(x[x["判別式"] == key]["剪掉中微小顆粒pct"].iloc[0]), c) for key, lab, c, _ in _RACE_SHOW if (x["判別式"] == key).any()]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    for bb, r in zip(bars, rows):
        ax.annotate(f"{r[1]:.1f}%", (bb.get_width(), bb.get_y() + bb.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.axvline(26.2, color=MUTED, ls="--", lw=1); ax.annotate("盒內微小佔比 26.2%（隨機剪的期望）", (26.2, len(rows) - 0.6), xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED)
    ax.invert_yaxis(); ax.set_xlabel("被剪掉的顆粒中，微小顆粒的比例 %（保留 90% 顆數）"); ax.set_xlim(0, 105)
    save(fig, d, "n13_tiny.png")


def n13_swap(d):
    """13 §2：4x4 b6／b12 訓練＋換塊評測（full44 合併模型只換一塊，本塊幾何區域 PSNR），相對 best0；Load＝單塊精確 Σtiles 中位（訓練相機）。
    數字：lab logs/swap_b6_1014_0110.log、swap_b12_1014_0129.log、各跑次 chart_data/load.txt（2026-10-14）。"""
    R = {  # 臂: (Δ幾何 b6, b12, ΔLoad% b6, b12)
        "best0 重複（fast，渲染逐位元同）": (0.005, 0.025, -2.5, 1.3),
        "週期 trim 全關": (-0.395, -0.353, 7.7, 6.2),
        "三次門檻剪（trim3）": (-0.291, -0.176, -6.1, -11.8),
        "v/c × tile 名次": (-1.124, -0.192, -53.5, -46.4),
        "opacity 二元熵正則": (-0.031, 0.026, -12.3, -22.9),
        "低解析度起步": (-0.037, -0.086, -2.9, -5.0),
        "拉長顆粒硬剪": (-0.180, 0.028, 0.6, 2.0),
        "V/C（總貢獻÷覆蓋像素）": (-0.062, -0.244, -17.2, -9.7),
        "V/(β+C)，β=55": (-0.042, -0.038, 18.5, 14.9),
        "★ 二元熵＋V/C（10-14 採用＝best1）": (-0.089, 0.005, -34.2, -34.9),
    }
    off = {"v/c（訓練期判準）": (-3.817, -0.895, -63.4, -54.4), "v/c×名次＋二元熵": (-3.350, -0.927, -65.1, -61.1)}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    cols = {"V/C（總貢獻÷覆蓋像素）": C3, "V/(β+C)，β=55": C4, "opacity 二元熵正則": C2, "best0 重複（fast，渲染逐位元同）": MUTED,
            "★ 二元熵＋V/C（10-14 採用＝best1）": "#123f7a"}
    for k, (a6, a12, l6, l12) in R.items():
        x, y = (l6 + l12) / 2, (a6 + a12) / 2
        c = cols.get(k, C1)
        ax.errorbar(x, y, xerr=abs(l6 - l12) / 2, yerr=abs(a6 - a12) / 2, fmt="o", color=c, ms=6, lw=0.8, capsize=2)
        ax.annotate(k, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhspan(-0.03, 0.03, color="#e1e0d9", alpha=0.6, lw=0)
    ax.annotate("±0.03：兩次 best0 重複的差", (-58, 0.035), fontsize=8, color=MUTED)
    ax.set_xlim(-60, 25); ax.set_ylim(-0.8, 0.12)
    ax.set_xlabel("精確 Load 中位變化 %（相對 best0；b6／b12 平均，橫線＝兩塊範圍）")
    ax.set_ylabel("Δ 本塊幾何區域 PSNR（dB，換塊評測）")
    ax.set_title("4x4 b6／b12 訓練 60k 後換進合併模型（相對 best0）；超出畫面：" + "、".join(f"{k} {(v[0]+v[1])/2:+.2f} dB @ {(v[2]+v[3])/2:+.0f}%" for k, v in off.items()), fontsize=9)
    save(fig, d, "n13_swap.png")


# ══════════════ 10-14 使用者：CityGS 數據放進對應的我方圖表＋現行最佳（best0）══════════════
OFFD = os.path.join(DST, "data", "official")      # lab logs/ 的副本（2026-10-14 下載；citygs_official_*、full44_*）
# 我方 full44（cs60_sfmdup4 配方、opacity_reg 0.002）逐塊精確 Load 中位（chart_data/load.txt ③）；CityGS 取自 citygs_official_blockload*.log 60k
LOAD44 = {
    "ours": [2.576, 2.433, 2.789, 2.975, 2.305, 2.517, 2.834, 3.008, 2.338, 2.335, 2.555, 2.536, 2.870, 2.367, 2.211, 2.625],
    "rel": [6.329, 5.297, 6.737, 5.398, 5.694, 7.041, 8.229, 6.409, 6.530, 7.717, 8.093, 6.900, 7.372, 8.685, 8.813, 7.163],
    "pap": [6.167, 5.117, 6.596, 5.273, 5.741, 6.653, 7.938, 6.273, 6.481, 7.697, 8.131, 6.909, 7.450, 8.776, 8.479, 6.643],
}
BEST0 = {"peak": {6: 3.95, 12: 3.92}, "load": {6: 1.893, 12: 1.859}}   # g44_best0 60k（台帳峰值實佔 GB；chart_data/load.txt ③ 中位，百萬）
BEST1 = {"peak": {6: 3.92, 12: 3.92}, "load": {6: 1.259, 12: 1.212}}   # g44_best1 60k（10-14 台帳峰值實佔；chart_data/load.txt 精確中位）
# ★ 10-24 現行最佳 ed10b4 全場景（g44s30km25_best1maskdistrfed10b4：遮罩＋dist100＋30k＋cap 2.5M＋融合 Adam＋累加式 trim＋熵只往 0 推、10k 後切回 ×4）
#   min＝純訓練（train_status val step0 → 30k，labrun/time_1024.py；與台帳 START→DONE 差 ≤1 分）；pk＝台帳峰值實佔 GB
#   fs＝再加融合 SSIM（現行最快；只有 b0／b6 實測）
ED = {"min": [1712 / 60, 1689 / 60, 1545 / 60, 1705 / 60, 1612 / 60, 1728 / 60, 1586 / 60, 1717 / 60,
              1606 / 60, 1600 / 60, 1582 / 60, 1668 / 60, 1767 / 60, 1493 / 60, 1549 / 60, 1785 / 60],
      "pk": [3.87, 3.92, 3.90, 3.93, 3.86, 3.90, 3.93, 3.93, 3.87, 3.89, 3.90, 3.90, 3.92, 3.87, 3.87, 3.89],
      "fs_min": {0: 1501 / 60, 6: 1380 / 60}}


# ★ 10-26 現行最佳 A＋dist4 全場景（g44s30km25fssh16di4_best1maskdistrfed10b4＝ed10b4＋融合 SSIM＋高階 SH 每 16 步＋dist 每 4 步）與 cap 1.5M（g44s30km15fssh16_…）
#   s＝純訓練秒（labrun/time_1024.py）；pk＝台帳峰值實佔 GB（labrun/peak_1027.py）——⚠ 修正前的程式（SH Adam 快取回歸，實驗分析 21 §5）
#   pk_fix＝修正後（0b1f0e9）第一批台帳：A＋dist4＋均勻撒點全場景 b0~b6（10-28；撒點跑次，其餘同配方；該全場景在 b6 後撤下）
AD4 = {"s": [1203, 1187, 1079, 1195, 1083, 1202, 1084, 1186, 1087, 1098, 1083, 1149, 1215, 1008, 1052, 1234],
       "pk": [4.68, 4.71, 4.63, 4.72, 4.61, 4.68, 4.68, 4.70, 4.66, 4.69, 4.66, 4.67, 4.72, 4.62, 4.63, 4.69],
       "pk_fix": [4.14, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15]}
A15 = {"s": [1013, 1097, 971, 1066, 1039, 1042, 920, 1040, 977, 938, 958, 1036, 1074, 910, 958, 1093],
       "pk": [3.00, 3.03, 2.96, 3.02, 2.97, 2.98, 2.99, 3.00, 2.99, 2.99, 2.96, 3.00, 3.02, 2.94, 2.95, 3.00]}


def n01_blocktime(d):
    """逐塊訓練時間（4x4、lab 3090 [solo]）：★ 現行 A＋dist4（30k）vs 我方 full44（60k）vs CityGS release／論文設定（60k）。"""
    x = np.arange(16); wd = 0.21
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.bar(x - 1.5 * wd, [t / 3600 for t in AD4["s"]], wd, color="#123f7a", label="★ 我方 A＋dist4（現行最佳，30k）")
    for k, c, dx, lb in [("omin", C1, -0.5 * wd, "我方 full44（10-05 舊最佳，60k）"), ("rmin", MUTED, 0.5 * wd, LBL3[1]), ("pmin", C2, 1.5 * wd, LBL3[2])]:
        bars = ax.bar(x + dx, [m / 60 for m in F44[k]], wd, color=c, label=lb)
        if k != "omin":
            for b_, e in zip(bars, F44["rext" if k == "rmin" else "pext"]):
                if e > 2000:
                    b_.set_hatch("///"); b_.set_alpha(0.45)
    ax.plot(x - 1.5 * wd, [m / 60 for m in ED["min"]], marker="_", color="#e8a33d", ms=11, mew=2, lw=0, label="ed10b4（10-24，30k）", zorder=5)
    ax.set_xticks(x); ax.set_xlabel("4x4 block"); ax.set_ylabel("訓練時間（h）"); ax.legend(fontsize=8, loc="upper left", ncol=3)
    e09 = sum(AD4["s"][:10]) / 3600
    ax.set_title(f"逐塊訓練時間（斜線＝期間同卡外部佔用 > 2 GB，不可引用）：block 0~9 乾淨的合計 ★ A＋dist4 {e09:.2f} h／full44 19.3 h／release 40.3 h"
                 f"（A＋dist4 是 release 的 {e09 / 40.3:.2f} 倍）\n"
                 f"16 塊合計 A＋dist4 {sum(AD4['s']) / 3600:.2f} h（每塊 17~21 分）、ed10b4 7.32 h、cap 1.5M {sum(A15['s']) / 3600:.2f} h", fontsize=9.5)
    save(fig, d, "n01_blocktime.png")


def n02_peak3(d):
    """訓練峰值（GB）：現行 A＋dist4（修正前／後）、cap 1.5M、full44 舊配方、CityGS release／論文設定（同塊）；16 塊全貌。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1, 1.8]})
    ax = axs[0]
    grp = [("★ A＋dist4 修正後（b6；10-28 撒點跑次實測，b12 未量）", [AD4["pk_fix"][6], np.nan], "#123f7a"),
           ("A＋dist4 修正前（台帳）", [AD4["pk"][6], AD4["pk"][12]], "#5a8fd0"),
           ("A cap 1.5M 修正前", [A15["pk"][6], A15["pk"][12]], C3),
           ("我方 full44（舊最佳）", [F44["opk"][6], F44["opk"][12]], "#a9c6ec"),
           ("CityGS release", [F44["rpk"][6], F44["rpk"][12]], MUTED), ("CityGS 論文設定", [F44["ppk"][6], F44["ppk"][12]], C2)]
    x = np.arange(2); wd = 0.13
    for k, (lb, v, c) in enumerate(grp):
        bars = ax.bar(x + (k - 2.5) * wd, v, wd, color=c, label=lb)
        barlabels(ax, [b_ for b_, vv in zip(bars, v) if not np.isnan(vv)], "{:.2f}", size=6.5)
    ax.axhline(6.44, color="#c0392b", lw=1, ls="--"); ax.text(1.45, 6.6, "6 GiB 卡", color="#c0392b", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["block 6", "block 12"]); ax.set_ylabel("訓練峰值實佔（GB）"); ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, 15.5); ax.set_title("同塊訓練峰值：我方 A＋dist4 4.15 GB（修正後）、1.5M 3.0 GB vs CityGS 6.2~12.7 GB", fontsize=9.5)
    ax = axs[1]
    xx = np.arange(16)
    for k, c, mk, lb in [("opk", "#a9c6ec", "o", "我方 full44"), ("rpk", MUTED, "s", LBL3[1]), ("ppk", C2, "^", LBL3[2])]:
        ax.plot(xx, F44[k], marker=mk, color=c, lw=0, ms=6, label=lb)
    ax.plot(xx, AD4["pk"], marker="*", color="#5a8fd0", lw=0, ms=9, label="A＋dist4 修正前")
    ax.plot(xx[:7], AD4["pk_fix"], marker="*", color="#123f7a", lw=0, ms=12, label="★ A＋dist4 修正後（b0~b6）")
    ax.plot(xx, A15["pk"], marker="D", color=C3, lw=0, ms=5, label="A cap 1.5M 修正前")
    ax.axhline(6.44, color="#c0392b", lw=1, ls="--")
    ax.set_xticks(xx); ax.set_xlabel("4x4 block"); ax.set_ylabel("訓練峰值實佔（GB）"); ax.legend(fontsize=7.5, ncol=3)
    ax.set_title("16 塊：★ A＋dist4 修正後 4.14~4.15（修正前 4.61~4.72）；1.5M 2.94~3.03；full44 4.05~4.17；release 7.65~14.56；論文設定 5.24~11.23", fontsize=9.5)
    save(fig, d, "n02_peak3.png")


def n03_official(d):
    """成果檔與部署大小：單塊 ckpt（b6）、合併 ckpt、部署——我方 vs CityGS（`citygs_official_storage.log`）。"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.3), gridspec_kw={"width_ratios": [0.9, 1.1, 1.1]})
    lab = ["★ 我方\ned10b4", "我方\nfull44", "CityGS\nrelease", "CityGS\n論文設定"]; col = ["#123f7a", C1, MUTED, C2]
    ax = axs[0]
    # GiB（CityGS storage log 的「GB」是 2^30；full44 1,628,658,586 B；ed10b4 b6 30k ckpt 2.25M 顆 x 696 B，chart_data/storage.txt「1.5 GB」）
    v = [2.25e6 * 696 / 2**30, 1.52, 3.0, 2.3]; n = [2.25, 2.34, 7.11, 5.31]
    b = ax.bar(lab, v, color=col); barlabels(ax, b, "{:.2f} GiB", size=8)
    for i in range(4):
        ax.text(i, v[i] / 2, f"N {n[i]:.2f}M\n{2**30 * v[i] / (n[i] * 1e6):.0f} B／顆", ha="center", color="white", fontsize=7.5)
    ax.set_ylabel("GiB"); ax.set_title("單塊 ckpt（block 6；含 Adam；ed10b4 30k、其餘 60k）\n我方 SH3 696 B／顆；CityGS SH2 456 B／顆（參數 148＋Adam 296＋緩衝 12）", fontsize=9)
    lab5 = ["★ A＋\ndist4", "A cap\n1.5M", "full44\n舊最佳", "CityGS\nrelease", "CityGS\n論文設定"]; col5 = ["#123f7a", C3, C1, MUTED, C2]
    n5 = [27.75, 17.12, 17.63, 24.48, 24.72]
    ax = axs[1]
    # GiB：A＋dist4／1.5M 取自 half_ckpt 的輸出（17.99／11.10 GiB）；full44 11,703 MiB；CityGS storage log
    v = [17.99, 11.10, 11.43, 40.5, 29.0]
    b = ax.bar(lab5, v, color=col5); barlabels(ax, b, "{:.1f} GiB", size=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("GiB"); ax.set_title("合併 ckpt（含優化器狀態；我方只留各塊盒內顆粒）\nCityGS 合併時把 16 塊的優化器狀態一起疊進去（1,566 B／顆）", fontsize=9)
    ax = axs[2]
    v = [3.15, 1.95, 17.63e6 * 232 / 2**30, 24.48e6 * 148 / 2**30, 24.72e6 * 148 / 2**30]
    note = ["半精度", "半精度", "float32", "float32\nSH2", "float32\nSH2"]
    b = ax.bar(lab5, v, color=col5); barlabels(ax, b, "{:.2f} GiB", size=8)
    for i in range(5):
        ax.text(i, v[i] / 2, f"{n5[i]:.2f}M\n{note[i]}", ha="center", color="white", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("GiB"); ax.set_title("部署（只存參數；我方 tools/half_ckpt.py：means 以外 float16、741 幀 PSNR 不變）\n"
                                       "★ A＋dist4 是 release 的 0.94 倍、cap 1.5M 0.58 倍", fontsize=9)
    fig.suptitle("儲存對照 CityGS（lab `citygs_official_storage.log`；我方 ckpt 實測檔案大小；A＋dist4＝10-26 現行最佳）；CityGS CPU RAM 快取未量", fontsize=10)
    save(fig, d, "n03_official.png")


def n04_load3(d):
    """逐塊精確 Load 中位（同一支工具、各模型自己那塊的訓練相機）：我方 full44／best0 vs CityGS。"""
    x = np.arange(16); wd = 0.27
    fig, ax = plt.subplots(figsize=(14, 3.9))
    for k, c, dx, lb in [("ours", C1, -wd, "我方 full44（opacity_reg 0.002）"), ("rel", MUTED, 0, LBL3[1]), ("pap", C2, wd, LBL3[2])]:
        ax.bar(x + dx, LOAD44[k], wd, color=c, label=lb)
    ax.plot([6 - wd, 12 - wd], [BEST0["load"][6], BEST0["load"][12]], marker="o", mfc="none", color="#123f7a", lw=0, ms=8, label="我方 best0（b6／b12）")
    ax.plot([6 - wd, 12 - wd], [BEST1["load"][6], BEST1["load"][12]], marker="*", color="#123f7a", lw=0, ms=12, label="★ 我方 best1（現行，b6／b12）")
    r = np.mean(np.array(LOAD44["ours"]) / np.array(LOAD44["rel"]))
    ax.set_xticks(x); ax.set_xlabel("4x4 block"); ax.set_ylabel("精確 Load 中位（百萬 tile／視角）"); ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.set_ylim(0, 10.5)
    ax.set_title(f"逐塊渲染成本：我方 full44 平均是 release 的 {100 * r:.0f}%；best0 再低約 1/3、現行 best1 又再低約 1/3（b6 1.26M／b12 1.21M）\n"
                 "（60k 終點、各模型自己那塊的全部訓練相機；CityGS lab `citygs_official_blockload*.log`、我方 `chart_data/load.txt`）", fontsize=9.5)
    save(fig, d, "n04_load3.png")


def n05_main4x4(d):
    """4x4 主要跑次（相對現行 best0）：換塊評測的本塊幾何區域 PSNR、精確 Load（取代 5x5 舊版 05(a)）。"""
    R = [("full44 舊配方（opacity_reg 0.002）", -0.054, 0.093, 49.7, 54.4),
         ("best0 重複（只差訓練隨機性）", 0.005, 0.025, -2.5, 1.3),
         ("★ 二元熵＋V/C（10-14 採用＝best1）", -0.089, 0.005, -34.2, -34.9),
         ("opacity 二元熵正則（best1 的組成）", -0.031, 0.026, -12.3, -22.9),
         ("trim 判準 V/C（best1 的組成）", -0.062, -0.244, -17.2, -9.7)]
    fig, axs = plt.subplots(1, 2, figsize=(16, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(R)); h = 0.38
    ax = axs[0]
    ax.barh(y - h / 2, [r[1] for r in R], h, color=C1, label="block 6", zorder=2)
    ax.barh(y + h / 2, [r[2] for r in R], h, color=C2, label="block 12", zorder=2)
    for i, r in enumerate(R):
        for off, v in ((-h / 2, r[1]), (h / 2, r[2])):
            if v < -0.95:      # 截斷：標在長條內
                ax.text(-0.98, i + off, f"{v:+.2f}（截斷）", va="center", ha="left", fontsize=7, color="white")
            else:
                ax.text(v + (0.01 if v >= 0 else -0.01), i + off, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.axvspan(-0.03, 0.03, color="#e1e0d9", alpha=0.7, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in R], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(-0.4, 0.2); ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Δ 本塊幾何區域 PSNR（dB，換塊評測，相對 best0；灰帶 ±0.03＝重複的差）")
    ax.set_title("品質：二元熵單開落在雜訊內；與 V/C 合開 b6 −0.09／b12 +0.01（採用，換 Load −1/3）", fontsize=9.5)
    ax = axs[1]
    ax.barh(y - h / 2, [r[3] for r in R], h, color=C1); ax.barh(y + h / 2, [r[4] for r in R], h, color=C2)
    for i, r in enumerate(R):
        for off, v in ((-h / 2, r[3]), (h / 2, r[4])):
            ax.text(v + (1 if v >= 0 else -1), i + off, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlim(-75, 65)
    ax.set_xlabel("Δ 精確 Load 中位（%，相對 best0；越左越便宜）"); ax.set_title("渲染成本：舊配方比 best0 貴約 50%", fontsize=9.5)
    fig.suptitle("★ 現行 best1 與它的組成（4x4 b6／b12 60k，換進合併模型；相對 best0＝dup init＋opacity_reg 0）；"
                 "其他被關掉的臂見 new_archived／已關閉的線", fontsize=10)
    save(fig, d, "n05_main4x4.png")


def _prune(name):
    import csv
    with open(os.path.join(OFFD, f"full44_prunecurve_{name}.tsv")) as f:
        r = list(csv.DictReader(f, delimiter="\t"))
    return [float(x["keep"]) for x in r], [float(x["N"]) / 1e6 for x in r], [float(x["psnr"]) for x in r], [float(x["lpips"]) for x in r]


def n06_prune3(d):
    """合併模型依 opacity 剪枝（不重訓）、MatrixCity test 741 幀 held-out、同一支評分工具：我方 vs release vs 論文設定（lab full44_prunecurve_*.tsv，10-10）。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.4))
    for nm, c, mk, lb in [("ours", C1, "o", "★ 我方 4x4"), ("release", MUTED, "s", LBL3[1]), ("paper", C2, "^", LBL3[2])]:
        k, n, p, l = _prune(nm)
        axs[0].plot(n, p, marker=mk, color=c, label=lb)
        for kk, nn, pp in zip(k, n, p):
            axs[0].annotate(f"{int(round(kk * 100))}%", (nn, pp), xytext=(3, -10), textcoords="offset points", fontsize=7, color=c)
        axs[1].plot([100 * x for x in k], [x - p[0] for x in p], marker=mk, color=c, label=lb)
    axs[0].set_xlabel("保留顆數 N（百萬）"); axs[0].set_ylabel("held-out PSNR（dB）"); axs[0].invert_xaxis(); axs[0].legend(fontsize=8)
    axs[0].set_title("絕對值：同 N 時我方都較高（我方 8.8M 26.21 vs CityGS 12.2~12.4M 25.07~25.80）", fontsize=9.5)
    axs[1].set_xlabel("保留比例（%，依 opacity 由高到低）"); axs[1].set_ylabel("Δ PSNR vs 不剪（dB）"); axs[1].invert_xaxis(); axs[1].legend(fontsize=8)
    axs[1].set_title("相對值：剪 10% CityGS 兩條線都 0.000（≈ 245 萬顆白付）、25% release −0.04／論文設定 −0.26；\n我方 10% −0.04、25% −0.38 => 我方的顆粒用得較滿", fontsize=9.5)
    fig.suptitle("合併模型依 opacity 剪枝（不重訓；MatrixCity test 741 幀 held-out、同一支評分工具）", fontsize=10.5)
    save(fig, d, "n06_prune3.png")


def n12_fail3(d):
    """合併模型逐塊（各塊 held-out 視角）：PSNR 與失敗 tile 比例，我方 vs CityGS（lab full44_failheld.txt）。"""
    import re
    rows = []
    for l in open(os.path.join(OFFD, "full44_failheld.txt"), encoding="utf-8"):
        m = re.match(r"\s*(\d+|全)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+\S+\s+ours ([\d.]+)／([\d.]+)\s+release ([\d.]+)／([\d.]+)\s+paper ([\d.]+)／([\d.]+)", l)
        if m:
            rows.append(m.groups())
    blk = [r for r in rows if r[0] != "全"]; allr = [r for r in rows if r[0] == "全"][0]
    x = np.arange(len(blk)); wd = 0.27
    fig, axs = plt.subplots(1, 2, figsize=(17, 4.3))
    ax = axs[0]
    for j, c, lb in [(2, C1, "★ 我方 4x4"), (3, MUTED, LBL3[1]), (4, C2, LBL3[2])]:
        ax.bar(x + (j - 3) * wd, [float(r[j]) for r in blk], wd, color=c, label=lb)
    ax.set_ylim(26.2, 28.6); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in blk]); ax.set_xlabel("4x4 block（該塊 held-out 視角）")
    ax.set_ylabel("合併模型 PSNR（dB）"); ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_title(f"PSNR：我方 16/16 塊都高（全 741 幀 {float(allr[2]):.2f} vs {float(allr[3]):.2f}／{float(allr[4]):.2f}）", fontsize=9.5)
    ax = axs[1]
    for j, c, lb in [(6, C1, "★ 我方 4x4"), (8, MUTED, LBL3[1]), (10, C2, LBL3[2])]:
        ax.bar(x + ((j - 6) / 2 - 1) * wd, [float(r[j]) for r in blk], wd, color=c, label=lb)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in blk]); ax.set_xlabel("4x4 block")
    ax.set_ylabel("失敗 tile %（過對比門檻的 tile 中）"); ax.legend(fontsize=8)
    ax.set_title(f"⚠ 失敗 tile：我方反而較多（全 {float(allr[6]):.2f}% vs {float(allr[8]):.2f}%／{float(allr[10]):.2f}%）——多在每格離接縫遠處（10-12 接縫檢查）；\n"
                 "區塊 10 換成 best0 後該塊 0.40% -> 0.17%（10-12），全場景 best0 未重跑", fontsize=9.5)
    fig.suptitle("合併模型逐塊對照 CityGS（MatrixCity test 741 幀 held-out、同一支評分工具；失敗 tile＝48px、GT 灰階 std >= 0.10 且 corr < 0.6）", fontsize=10.5)
    save(fig, d, "n12_fail3.png")


# ══════════════ 10-15 使用者：圖表更新到現行最佳 best1（最快）＋報告用的組成與取捨（`實驗分析/15`）══════════════
F1B = {   # 區塊幾何 F1（tools/block_f1.py，不配準；CityGS＝全場景網格裁到本塊、全部相機；我方＝換塊、本塊相機；`09` §6.1）
    "CityGS release": {6: (.423, .803, .554), 12: (.566, .847, .678)},
    "我方 full44（舊配方）": {6: (.272, .667, .386), 12: (.423, .680, .521)},
    "我方 best0": {6: (.372, .750, .498), 12: (.588, .766, .665)},
    "★ 我方 best1（現行）": {6: (.430, .801, .559), 12: (.654, .819, .727)},
}


def n09_blockf1(d):
    """區塊級幾何 F1（10-15）：CityGS release vs 我方三代配方，b6／b12；長條＝F1，標 P／R。"""
    fig, ax = plt.subplots(figsize=(13, 3.9))
    keys = list(F1B); cols = [MUTED, "#7fa9e0", C3, C1]; wd = 0.2; x = np.arange(2)
    for k, (nm, c) in enumerate(zip(keys, cols)):
        v = [F1B[nm][b][2] for b in (6, 12)]
        bars = ax.bar(x + (k - 1.5) * wd, v, wd, color=c, label=nm)
        for b_, b in zip(bars, (6, 12)):
            P, R, F = F1B[nm][b]
            ax.text(b_.get_x() + b_.get_width() / 2, F + 0.01, f"{F:.3f}\nP {P:.2f}／R {R:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(["block 6", "block 12"]); ax.set_ylim(0, 0.9); ax.set_ylabel("F1（dTau 0.0065）")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.set_title("區塊幾何 F1（MatrixCity 空拍 GT、CityGS 抽網格參數、不配準）：舊配方落後 CityGS 0.17；opacity_reg 0 補回大半，best1 追平（b6）／超過（b12）CityGS\n"
                 "（全場景不配準：我方 full44 0.425 vs CityGS 0.590；best1 全場景未訓練）", fontsize=9.5)
    save(fig, d, "n09_blockf1.png")


def _time_logs():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from parse_best1_time import parse
    base = os.path.join(DST, "data", "best1_time")
    T = {6: parse(os.path.join(base, "best1_time_b6_1015_0655.log")), 12: parse(os.path.join(base, "best1_time_b12_1015_0602.log"))}
    # ★ 10-19：累加式 trim／＋融合 Adam／＋稀疏 Adam（b6，同 ckpt 同工具；累加停止條件修正後的那次）
    T[6].update(parse(os.path.join(base, "best1_time_b6_1019_1559.log")))
    return T


def n15_time(d):
    """best1 的時間（lab 3090 [solo]、真實訓練迴圈，`task_best1_time.sh loop`）：best0 → ＋加速旗標 → best1 → ＋累加式 trim → ＋融合／稀疏 Adam。
    ⚠ 10-19 修正換算：真實步時間本來就不含 trim => 每步 = 真實步 ＋ 第 12 段 ＋ trim 單次/500（舊版少算約 16 ms）。"""
    T = _time_logs()
    groups = [(6, [("best0", "best0"), ("best0fast", "＋加速"), ("best1", "★best1"), ("best1run", "＋累加\ntrim"),
                   ("best1runfa", "＋融合\nAdam"), ("best1runsa", "＋稀疏\nAdam")]),
              (12, [("best0", "best0"), ("best0fast", "＋加速"), ("best1", "★best1")])]
    segs = [("backward", C1), ("forward", C3), ("trim_per_step", C2), ("optimizer", C4), ("loss", "#c9b458")]
    cols = {"best0": MUTED, "best0fast": "#7fa9e0", "best1": C1, "best1run": "#4c9a6a", "best1runfa": "#d08a2c", "best1runsa": "#b0b0b0"}
    pos = []
    x = 0
    for b, arms in groups:
        for a, nm in arms:
            pos.append((x, b, a, nm)); x += 1
        x += 1
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axs[0]
    for k, (x, b, a, nm) in enumerate(pos):
        g = T[b][(a, "grow")]; bot = 0
        for sk, c in segs:
            v = g.get(sk, 0.0); ax.bar(x, v, 0.7, bottom=bot, color=c, label=sk if k == 0 else None); bot += v
        ax.plot([x - 0.4, x + 0.4], [g["step"]] * 2, color="k", lw=2, label="真實每步" if k == 0 else None)
        ax.text(x, max(bot, g["step"]) + 1.5, f"{g['step']:.0f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks([p[0] for p in pos]); ax.set_xticklabels([f"b{p[1]}\n{p[3]}" for p in pos], fontsize=7.5)
    ax.set_ylabel("ms／步（增生期；trim 按真實週期 500 步攤平）"); ax.legend(fontsize=7.5, ncol=3, loc="upper right"); ax.set_ylim(0, 130)
    ax.set_title("增生期每步（@14,999 起 1,200 步；堆疊＝逐段同步量、黑線與數字＝真實每步）", fontsize=9)
    ax = axs[1]
    for x, b, a, nm in pos:
        h = T[b][(a, "harvest")]
        ax.bar(x, h["backward"], 0.7, color=C1); ax.bar(x, h.get("optimizer", 0), 0.7, bottom=h["backward"], color=C4)
        ax.bar(x, h["step"] - h["backward"] - h.get("optimizer", 0), 0.7, bottom=h["backward"] + h.get("optimizer", 0), color="#d8d6cf")
        ax.text(x, h["step"] + 1, f"{h['step']:.0f}", ha="center", fontsize=8)
    ax.set_xticks([p[0] for p in pos]); ax.set_xticklabels([f"b{p[1]}\n{p[3]}" for p in pos], fontsize=7)
    ax.set_ylim(0, 80); ax.set_ylabel("ms／步（收割期，不增生）")
    ax.set_title("收割期每步（@41,999 起）：藍＝backward、紫＝optimizer（Adam）", fontsize=9)
    ax = axs[2]
    for x, b, a, nm in pos:
        g, h = T[b][(a, "grow")], T[b][(a, "harvest")]; m = 30000 * (g["step"] + h["step"]) / 1000 / 60
        ax.bar(x, m, 0.7, color=cols[a]); ax.text(x, m + 1, f"{m:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks([p[0] for p in pos]); ax.set_xticklabels([f"b{p[1]}\n{p[3]}" for p in pos], fontsize=7)
    ax.set_ylabel("分鐘／60k（估計）"); ax.set_ylim(0, 100)
    e = lambda b, a: 30000 * (T[b][(a, 'grow')]['step'] + T[b][(a, 'harvest')]['step']) / 6e4
    ax.set_title(f"一塊 60k ≈ 30k×增生＋30k×收割（不含驗證／存檔）\nb6：best1 {e(6,'best1'):.0f} → 累加 trim {e(6,'best1run'):.0f} → 融合 Adam {e(6,'best1runfa'):.0f} 分"
                 f"（實測 68 → 54 → 45）", fontsize=9)
    fig.suptitle("現行 best1 的訓練時間（4x4、lab RTX 3090 [solo]、量測期間卡上無其他行程；b6 後三組 10-19 量、其餘 10-15；CityGS release 同 N 約 214 ms／步，`09` §0.3）", fontsize=10)
    save(fig, d, "n15_time.png")


def n15_tradeoff(d):
    """採用中的機制：品質變化 vs 成本變化（品質＝換塊評測本塊幾何區域 PSNR 兩塊平均，或逐位元相同＝0）。"""
    T = _time_logs()
    tm = lambda a, b: np.mean([30000 * (T[k][(b, 'grow')]['step'] + T[k][(b, 'harvest')]['step']) / (30000 * (T[k][(a, 'grow')]['step'] + T[k][(a, 'harvest')]['step'])) - 1 for k in (6, 12)]) * 100
    tm6 = lambda a, b: ((T[6][(b, 'grow')]['step'] + T[6][(b, 'harvest')]['step']) / (T[6][(a, 'grow')]['step'] + T[6][(a, 'harvest')]['step']) - 1) * 100
    P = [("conic 精確外接盒（09-21）", 0.00, -40.1, "Load", "22k b6 29.05 vs 29.05；forward −26%", (6, 8)),
         ("lean_render（10-02）", 0.00, -20.9, "每步", "渲染逐位元相同", (6, 8)),
         ("record_reduce（10-03）", 0.00, -10.9, "每步（增生期）", "每次 trim −35%；逐位元相同", (-150, -34)),
         ("飽和跳過＋先加總（10-10）", 0.02, tm("best0", "best0fast"), "60k 時間", "增生期 backward −26~−29%（4x4）", (8, 10)),
         ("opacity_reg 0（best0，10-10）", -0.02, -33.0, "Load", "F1 +0.11／+0.14；高樓塊 b10 +0.96 dB", (6, -26)),
         ("★ 二元熵＋V/C（best1，10-14）", -0.04, tm("best0fast", "best1"), "60k 時間", "Load −34／−35%；F1 再 +0.06～0.07", (8, -26)),
         # ★ 10-19 候選（下一個基底；時間只有 b6 的同工具量測）
         ("候選：累加式 trim（10-16）", 0.02, tm6("best1", "best1run"), "60k 時間（b6）", "F1 平；60k 實測 68 → 54 分", (-230, 22)),
         ("候選：融合 Adam（10-19）", 0.00, tm6("best1run", "best1runfa"), "60k 時間（b6）", "與 torch 逐位元相同；實測 54 → 45 分", (-40, 66))]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axhspan(-0.03, 0.03, color="#e1e0d9", alpha=0.7, lw=0)
    for nm, dq, dc, unit, note, off in P:
        mk = "^" if unit == "Load" else "o"; c = C1 if nm.startswith("★") else (C2 if unit == "Load" else C3)
        cand = nm.startswith("候選")
        ax.plot(dc, dq, marker=mk, color=c, ms=11 if nm.startswith("★") else 9, lw=0 if not cand else 1.5,
                markerfacecolor="none" if cand else c)
        ax.annotate(f"{nm}\n{unit} {dc:+.0f}%｜{note}", (dc, dq), xytext=off, textcoords="offset points", fontsize=7.5)
    ax.axvline(0, color=MUTED, lw=1); ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(-45, 5); ax.set_ylim(-0.15, 0.1)
    ax.set_xlabel("成本變化 %（▲＝精確 Load、●＝每步／60k 訓練時間；越左越便宜）")
    ax.set_ylabel("Δ 品質（dB；換塊評測兩塊平均，灰帶 ±0.03＝雜訊）")
    ax.set_title(f"best1 的每個組成：品質都在雜訊附近，成本一路往左（逐項相對上一個基底；best0 → best1 合計 60k 時間 {tm('best0','best1'):.0f}%、Load −34%／−35%；空心＝候選）", fontsize=9.5)
    save(fig, d, "n15_tradeoff.png")


# ══════════════ 10-16～10-19（實驗分析 16／17／18） ══════════════
# 數字＝lab logs：swap_b{6,12}_1017_16*／1018_*（換塊評測，本塊幾何區域 PSNR）、geom_f1/blocks/arms_b*.log（區塊 F1）、
#   inbox_diag_b*_1018_*（tools/inbox_diag.py）、quad_progress.log（START→DONE）。Δ 一律相對 best1（g44_best1：b6 −0.049／b12 −0.108 對 full44 基準）。

def n16_steps(d):
    """總步數（累加式 trim、SH3）：換塊 Δ、LPIPS、區塊 F1 對每塊訓練時間。"""
    st = [60, 30, 20, 15, 10]
    t = {6: [54, 29, 20, 15, 10], 12: [55, 28, 20, 15, 12]}
    dq = {6: [-0.01, 0.27, 0.14, 0.32, 0.01], 12: [0.05, 0.05, 0.18, 0.35, 0.18]}
    f1 = {6: [0.561, 0.520, 0.474, 0.443, 0.396], 12: [0.723, 0.715, 0.699, 0.670, 0.613]}
    lp = {6: [.1173, .1175, .1182, .1191, .1215], 12: [.1168, .1163, .1166, .1165, .1180]}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for b, c in ((6, C1), (12, C2)):
        ax[0].plot(t[b], dq[b], "o-", color=c, label=f"b{b}")
        ax[1].plot(t[b], lp[b], "o-", color=c, label=f"b{b}")
        ax[2].plot(t[b], f1[b], "o-", color=c, label=f"b{b}")
        for i, k in enumerate(st):
            ax[0].annotate(f"{k}k", (t[b][i], dq[b][i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax[0].axhspan(-0.03, 0.03, color="#e1e0d9", lw=0); ax[0].set_ylabel("Δ 本塊新視角 PSNR（dB，相對 best1）")
    ax[1].set_ylabel("LPIPS（換塊評測，越低越好）"); ax[2].set_ylabel("區塊幾何 F1")
    for b, c, v in ((6, C1, 0.554), (12, C2, 0.678)):
        ax[2].axhline(v, color=c, ls="--", lw=1); ax[2].annotate(f"CityGS b{b} {v}", (60, v), fontsize=7.5, color=c, ha="right", va="bottom")
    for a in ax:
        a.set_xlabel("每塊訓練時間（分；lab 3090 [solo]）"); a.legend()
    ax[0].set_title("短訓練在 PSNR 上反而贏（15k 最高）", fontsize=9.5)
    ax[1].set_title("但 LPIPS 隨步數變差（細節少）", fontsize=9.5)
    ax[2].set_title("幾何 F1 隨步數單調下降（15k 兩塊都低於 CityGS）", fontsize=9.5)
    save(fig, d, "n16_steps.png")


def n16_sh(d):
    """SH 階數（60k）＋15k＋SH1：換塊 Δ、ckpt、峰值。"""
    nm = ["SH3\n(best1)", "SH2", "SH1", "SH0", "15k＋SH1"]
    q6 = [0, 0.30, 0.53, 0.38, 0.52]; q12 = [0, 0.06, 0.23, 0.22, 0.52]
    ck = [1.60, 0.99, 0.59, 0.35, 0.59]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.0))
    x = np.arange(len(nm)); w = 0.38
    barlabels(ax[0], ax[0].bar(x - w / 2, q6, w, color=C1, label="b6"), "{:+.2f}")
    barlabels(ax[0], ax[0].bar(x + w / 2, q12, w, color=C2, label="b12"), "{:+.2f}")
    ax[0].axhspan(-0.03, 0.03, color="#e1e0d9", lw=0); ax[0].set_xticks(x, nm); ax[0].legend()
    ax[0].set_ylabel("Δ 本塊新視角 PSNR（dB，相對 best1）")
    ax[0].set_title("SH1 兩塊都最好（高階 SH 把逐幀曝光背進視角相依的顏色）；LPIPS 都在 .1176~.1198", fontsize=9)
    barlabels(ax[1], ax[1].bar(x, ck, 0.6, color=C3), "{:.2f}")
    ax[1].set_xticks(x, nm); ax[1].set_ylabel("單塊 ckpt（GB，含 Adam；2.34M 顆）")
    ax[1].set_title("模型大小：SH1 是 SH3 的 37%；峰值 VRAM 3.92 -> 2.09 GB", fontsize=9)
    save(fig, d, "n16_sh.png")


def n16_official(d):
    """同塊對 CityGS：CityGS release 合併模型只換第 6／12 塊（同光柵器、同 held-out 視角）＋資源。"""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.0))
    nm = ["15k＋SH1", "SH1 60k"]; x = np.arange(2); w = 0.38
    barlabels(ax[0], ax[0].bar(x - w / 2, [0.53, 0.61], w, color=C1, label="b6"), "{:+.2f}")
    barlabels(ax[0], ax[0].bar(x + w / 2, [0.54, 0.24], w, color=C2, label="b12"), "{:+.2f}")
    ax[0].axhline(0.35, color=MUTED, ls=":", lw=1); ax[0].annotate("我方光柵器讀 CityGS 約低 0.35（10-09）", (1.4, 0.36), fontsize=7.5, color=INK2, ha="right")
    ax[0].set_xticks(x, nm); ax[0].legend(); ax[0].set_ylabel("Δ 本塊新視角 PSNR（dB，相對 CityGS release）")
    ax[0].set_title("換進 CityGS 模型：兩塊都贏（扣掉光柵器落差後約 +0.2）", fontsize=9.5)
    lab = ["訓練時間\n(分)", "峰值 VRAM\n(GB)", "盒內顆數\n(萬)"]
    ours = [12.5, 2.08, 66.0]; off = [315.0, 8.42, 99.5]
    xx = np.arange(3)
    b1 = ax[1].bar(xx - w / 2, np.array(ours) / np.array(off), w, color=C1, label="我方 15k＋SH1")
    ax[1].bar(xx + w / 2, [1, 1, 1], w, color=MUTED, label="CityGS（=1）")
    for i, (o, f) in enumerate(zip(ours, off)):
        ax[1].annotate(f"{o:g} vs {f:g}", (i - w / 2, o / f), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
    ax[1].set_xticks(xx, lab); ax[1].set_ylabel("相對 CityGS（b6／b12 平均）"); ax[1].legend()
    ax[1].set_title("資源：時間 4%、VRAM 25%、盒內顆數 66%；幾何 F1 輸（0.441／0.653 vs 0.554／0.678）", fontsize=9)
    save(fig, d, "n16_official.png")


def n17_mask(d):
    """塊外遮罩（20k、累加式 trim），相對對照 g44s20k_best1run。「只開隨機背景」臂只留在文件（使用者 10-30：圖上不比）。"""
    nm = ["對照\n20k", "遮罩\ncap 2.6M", "遮罩\ncap 1.2M"]
    f6 = [0.474, 0.545, 0.532]; f12 = [0.699, 0.753, 0.741]
    q6 = [0, -0.11, -0.15]; q12 = [0, 0.02, -0.01]
    lp6 = [.1182, .1165, .1176]; lp12 = [.1166, .1133, .1153]
    tm = [20, 15, 11.5]; pk = [3.92, 3.56, 1.95]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    x = np.arange(len(nm)); w = 0.38
    barlabels(ax[0], ax[0].bar(x - w / 2, f6, w, color=C1, label="b6"), "{:.3f}")
    barlabels(ax[0], ax[0].bar(x + w / 2, f12, w, color=C2, label="b12"), "{:.3f}")
    ax[0].axhline(0.554, color=C1, ls="--", lw=1); ax[0].axhline(0.678, color=C2, ls="--", lw=1)
    ax[0].set_ylim(0.35, 0.8); ax[0].set_xticks(x, nm); ax[0].legend(); ax[0].set_ylabel("區塊幾何 F1（虛線＝CityGS）")
    ax[0].set_title("幾何：遮罩 +0.07／+0.05", fontsize=9.5)
    for off, vals, c, lb in ((-w / 2, q6, C1, "b6"), (w / 2, q12, C2, "b12")):
        ax[1].bar(x + off, vals, w, color=c, label=lb)
        for i, v in enumerate(vals):
            ax[1].annotate(f"{v:+.2f}" if v else "0", (i + off, v), xytext=(0, 2 if v >= 0 else -2), textcoords="offset points",
                           ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    ax[1].axhline(0, color=MUTED, lw=0.8); ax[1].set_ylim(-0.2, 0.08)
    ax[1].set_xticks(x, nm); ax[1].legend(loc="lower left"); ax[1].set_ylabel("Δ 本塊新視角 PSNR（dB，相對對照）")
    ax[1].set_title("PSNR：−0.15～+0.02 dB", fontsize=9.5)
    barlabels(ax[2], ax[2].bar(x - w / 2, lp6, w, color=C1, label="b6"), "{:.4f}")
    barlabels(ax[2], ax[2].bar(x + w / 2, lp12, w, color=C2, label="b12"), "{:.4f}")
    ax[2].set_ylim(0.110, 0.120); ax[2].set_xticks(x, nm); ax[2].legend(loc="upper right"); ax[2].set_ylabel("LPIPS（越低越好；y 軸從 0.110 起）")
    ax[2].set_title("LPIPS：遮罩變好（b12 .1166 -> .1133）", fontsize=9.5)
    barlabels(ax[3], ax[3].bar(x - w / 2, tm, w, color=C3, label="時間（分）"), "{:.1f}")
    a3 = ax[3].twinx(); a3.bar(x + w / 2, pk, w, color=C4, label="峰值 VRAM（GB）")
    for i, v in enumerate(pk):
        a3.annotate(f"{v:.2f}", (i + w / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax[3].set_xticks(x, nm); ax[3].set_ylabel("每塊訓練時間（分）"); a3.set_ylabel("峰值 VRAM（GB）"); a3.grid(False)
    h1, l1 = ax[3].get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7.5, framealpha=0.9, frameon=True)
    ax[3].set_title("資源：cap 1.2M 版 N 1.08M、時間 −42%、VRAM 減半", fontsize=9.5)
    save(fig, d, "n17_mask.png")


def n18_road(d):
    """馬路靠地底面元撐的比例（暗柏油像素中，渲染第一表面在本塊地面下 1 m 以上），與地底顆粒佔實際混合貢獻。"""
    nm = ["15k＋SH1", "best1\n60k", "SH1\n60k", "單目深度\n60k", "normal＋深度\n60k", "dist100\n60k", "SH1＋dist100\n60k", "SH1＋dist100\n30k", "15k＋SH1\n＋dist100", "CityGS\nrelease"]
    r6 = [52.9, 37.7, 44.5, 34.0, 30.5, 26.3, 33.9, 40.3, 50.0, 16.3]
    r12 = [54.7, 52.4, 67.7, 54.9, 25.8, 40.2, 56.5, 54.4, 53.4, 38.5]
    v6 = [5.7, 4.6, 4.4, 3.3, 4.5, 1.6, 2.3, 4.1, 5.9, 2.0]; v12 = [12.0, 6.4, 10.3, 5.0, 4.7, 1.9, 2.2, 4.5, 7.1, 1.9]
    fig, ax = plt.subplots(2, 1, figsize=(15, 7.2))
    x = np.arange(len(nm)); w = 0.38
    barlabels(ax[0], ax[0].bar(x - w / 2, r6, w, color=C1, label="b6"), "{:.0f}")
    barlabels(ax[0], ax[0].bar(x + w / 2, r12, w, color=C2, label="b12"), "{:.0f}")
    ax[0].set_xticks(x, nm, fontsize=8); ax[0].legend(); ax[0].set_ylabel("暗柏油中渲染表面在地底的 %")
    ax[0].set_title("馬路有一半靠地底面元撐（拿掉地底顆粒就變透明＝使用者看到的「特定視角馬路透明」）；b12 地面估計受水面影響，看相對值", fontsize=9.5)
    barlabels(ax[1], ax[1].bar(x - w / 2, v6, w, color=C1, label="b6"), "{:.1f}")
    barlabels(ax[1], ax[1].bar(x + w / 2, v12, w, color=C2, label="b12"), "{:.1f}")
    ax[1].set_xticks(x, nm, fontsize=8); ax[1].legend(); ax[1].set_ylabel("地面下 10 m 以上顆粒佔實際混合貢獻 %")
    ax[1].set_title("深度失真正則 100（60k）把地底貢獻壓到 CityGS 水準；15k 下無效（沒有收割期可以拉正）", fontsize=9.5)
    save(fig, d, "n18_road.png")


# ══════════════ 19 全因子消融（10-18～10-22） ══════════════
# (臂, cap 百萬, 步數千, 換塊 Δ（相對 full44 基準合併模型）, LPIPS, 區塊 F1, 每塊分鐘（台帳 START→DONE）, 暗柏油表面在地底 %, 地面下 10 m 顆粒的實際貢獻 %)
# 來源：lab logs/swap_b{6,12}_102*.log、outputs/lab/geom_f1/blocks/arms_b{6,12}.log、logs/inbox_diag_b{6,12}_102*.log、logs/quad_progress.log
N19 = {
    6: [('無遮罩', 1.0, 30, -0.196, 0.1232, 0.4185, 22, 48.9, 5.63),
         ('無遮罩＋dist', 1.0, 30, -0.109, 0.1228, 0.4409, 25, 42.7, 4.77),
         ('遮罩＋dist', 1.0, 30, 0.025, 0.1174, 0.5901, 24, 17.4, 0.81),
         ('遮罩', 1.0, 30, -0.005, 0.1174, 0.5728, 18, 18.9, 1.98),
         ('無遮罩', 2.5, 30, 0.237, 0.1176, 0.5207, 34, 42.2, 4.79),
         ('無遮罩＋dist', 2.5, 30, 0.26, 0.1176, 0.541, 38, 35.7, 3.23),
         ('遮罩＋dist', 2.5, 30, 0.072, 0.1159, 0.6436, 34, 8.5, 0.33),
         ('遮罩', 2.5, 30, 0.031, 0.1159, 0.6294, 31, 11.0, 1.28),
         ('無遮罩', 5.0, 30, 0.339, 0.1155, 0.5866, 53, 25.9, 1.56),
         ('無遮罩＋dist', 5.0, 30, 0.353, 0.1156, 0.6118, 58, 21.7, 0.52),
         ('遮罩＋dist', 5.0, 30, -0.001, 0.1159, 0.6185, 57, 6.4, 0.55),
         ('遮罩', 5.0, 30, 0.002, 0.1158, 0.6203, 53, 9.3, 1.22),
         ('無遮罩', 1.0, 40, -0.352, 0.1226, 0.4408, 28, 42.9, 5.01),
         ('無遮罩＋dist', 1.0, 40, -0.325, 0.1222, 0.4592, 32, 37.2, 3.9),
         ('遮罩＋dist', 1.0, 40, -0.224, 0.1174, 0.6122, 28, 15.4, 0.37),
         ('遮罩', 1.0, 40, -0.248, 0.1174, 0.5898, 24, 17.3, 1.47),
         ('無遮罩', 2.5, 40, 0.006, 0.1175, 0.5417, 45, 39.2, 4.56),
         ('無遮罩＋dist', 2.5, 40, 0.036, 0.1175, 0.5616, 51, 33.5, 2.57),
         ('遮罩＋dist', 2.5, 40, -0.176, 0.1161, 0.6567, 78, 8.3, 0.13),
         ('遮罩', 2.5, 40, -0.184, 0.116, 0.6462, 41, 10.1, 0.82),
         ('無遮罩', 5.0, 40, 0.114, 0.1154, 0.6161, 70, 22.7, 1.66),
         ('無遮罩＋dist', 5.0, 40, 0.122, 0.1156, 0.636, 77, 18.6, 0.08),
         ('遮罩＋dist', 5.0, 40, -0.237, 0.1163, 0.6232, 74, 6.3, 0.21),
         ('遮罩', 5.0, 40, -0.24, 0.1161, 0.6263, 69, 9.3, 0.94),
         ('無遮罩', 1.0, 50, -0.357, 0.1223, 0.4494, 35, 39.1, 5.24),
         ('無遮罩＋dist', 1.0, 50, -0.399, 0.1221, 0.4738, 42, 31.3, 2.07),
         ('遮罩＋dist', 1.0, 50, -0.305, 0.1175, 0.622, 35, 14.6, 0.28),
         ('遮罩', 1.0, 50, -0.301, 0.1174, 0.6006, 31, 16.2, 0.96),
         ('無遮罩', 2.5, 50, -0.022, 0.1175, 0.5576, 56, 36.5, 4.12),
         ('無遮罩＋dist', 2.5, 50, -0.015, 0.1175, 0.5785, 66, 30.3, 2.33),
         ('遮罩＋dist', 2.5, 50, -0.296, 0.1164, 0.66, 58, 8.7, 0.03),
         ('遮罩', 2.5, 50, -0.296, 0.1163, 0.6505, 52, 10.0, 0.75),
         ('無遮罩', 5.0, 50, 0.054, 0.1155, 0.6291, 88, 22.4, 0.68),
         ('無遮罩＋dist', 5.0, 50, 0.026, 0.1158, 0.6444, 96, 16.9, 0.11),
         ('遮罩＋dist', 5.0, 50, -0.366, 0.1168, 0.6161, 93, 6.1, 0.44),
         ('遮罩', 5.0, 50, -0.351, 0.1166, 0.6183, 86, 8.6, 1.1)],
    12: [('無遮罩', 1.0, 30, -0.456, 0.1235, 0.585, 21, 64.8, 14.89),
         ('無遮罩＋dist', 1.0, 30, -0.416, 0.1241, 0.6103, 24, 49.1, 20.6),
         ('遮罩＋dist', 1.0, 30, -0.117, 0.1165, 0.735, 22, 46.0, 6.68),
         ('遮罩', 1.0, 30, -0.137, 0.116, 0.7195, 18, 52.9, 6.45),
         ('無遮罩', 2.5, 30, -0.065, 0.1161, 0.7171, 31, 54.0, 9.97),
         ('無遮罩＋dist', 2.5, 30, -0.126, 0.1173, 0.7408, 35, 39.1, 0.61),
         ('遮罩＋dist', 2.5, 30, -0.032, 0.1129, 0.7977, 35, 31.4, 0.86),
         ('遮罩', 2.5, 30, -0.068, 0.1127, 0.7946, 45, 45.9, 3.57),
         ('遮罩＋dist', 1.0, 50, -0.128, 0.1165, 0.7579, 34, 41.0, 3.78),
         ('遮罩', 1.0, 50, -0.082, 0.1164, 0.7416, 29, 47.5, 3.86),
         ('無遮罩', 2.5, 50, -0.093, 0.1165, 0.7225, 52, 54.4, 8.01),
         ('無遮罩＋dist', 2.5, 50, -0.121, 0.1177, 0.7531, 66, 41.2, 1.43),
         ('遮罩＋dist', 2.5, 50, -0.073, 0.1133, 0.7931, 63, 26.2, 0.66),
         ('遮罩', 2.5, 50, -0.078, 0.113, 0.7815, 51, 39.7, 2.05)],
}
N19_REF = {6: dict(best1=(-0.049, 0.1175, 0.559, 68, 37.7, 4.6), off_f1=0.554, off_road=16.3),
           12: dict(best1=(-0.108, 0.1168, 0.727, 66, 52.4, 6.4), off_f1=0.678, off_road=38.5)}
N19_COL = {"無遮罩": MUTED, "無遮罩＋dist": C2, "遮罩": C3, "遮罩＋dist": C1}
N19_MK = {1.0: "o", 2.5: "s", 5.0: "^"}
N19_SUSPECT = {(6, "遮罩＋dist", 2.5, 40), (12, "遮罩", 2.5, 30)}   # 時間可能被外部佔卡拖慢（品質可用）


def n19_fact(d):
    """全因子：區塊 F1 對每塊訓練時間（顏色＝臂、形狀＝cap、旁註＝步數）；水平線＝best1 60k 與 CityGS release。"""
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.6))
    for ax, b in zip(axs, (6, 12)):
        ref = N19_REF[b]
        for arm, cap, st, dg, lp, f1, mi, road, ug in N19[b]:
            win = arm == "遮罩＋dist" and st == 30 and cap in (1.0, 2.5)
            sus = (b, arm, cap, st) in N19_SUSPECT
            ax.scatter(mi, f1, s=110 if win else 60, marker=N19_MK[cap], color=N19_COL[arm], edgecolor="k" if win else "none", lw=1.6, zorder=5 if win else 3)
            ax.annotate(f"{st}k" + ("⚠" if sus else ""), (mi, f1), xytext=(5, 5) if win else (5, -3), textcoords="offset points",
                        fontsize=7.5 if win else 7, color="k" if win else INK2, fontweight="bold" if win else None)
        b1 = ref["best1"]
        ax.scatter(b1[3], b1[2], marker="*", s=260, color="k", zorder=4)
        ax.annotate("best1 60k", (b1[3], b1[2]), xytext=(6, 4), textcoords="offset points", fontsize=8.5, fontweight="bold")
        ax.axhline(ref["off_f1"], color="k", ls="--", lw=1)
        ax.text(98, ref["off_f1"] + 0.003, f"CityGS release {ref['off_f1']:.3f}（每塊 4~6 小時）", ha="right", fontsize=8)
        ax.set_xlim(10, 100); ax.set_xlabel("每塊訓練時間（分；台帳 START→DONE，lab 3090 [solo]）"); ax.set_ylabel("區塊幾何 F1")
        ax.set_title(f"b{b}：粗框＝遮罩＋dist100 30k（2.5M／1M）；⚠＝時間可能被外部佔卡拖慢", fontsize=9.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=a, ms=8) for a, c in N19_COL.items()] + \
        [Line2D([], [], marker=m, ls="", color="#888", label=f"cap {c:g}M", ms=8) for c, m in N19_MK.items()]
    axs[0].legend(handles=h, fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("全因子消融（SH3、週期 trim、init 跟 cap 走）：遮罩是主槓桿；遮罩＋dist100＋2.5M＋30k 兩塊 F1 最高、時間約 best1 的一半（b12 只跑縮減後的 14 格）", fontsize=10)
    save(fig, d, "n19_fact.png")


def n19_road(d):
    """30k 的馬路地底比例與部署大小（盒內顆數）：遮罩把容量留在本塊。"""
    fig, axs = plt.subplots(1, 2, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    sel = [("無遮罩", 2.5), ("無遮罩＋dist", 2.5), ("遮罩", 1.0), ("遮罩＋dist", 1.0), ("遮罩", 2.5), ("遮罩＋dist", 2.5)]
    nm = [f"{a}\n{c:g}M 30k" for a, c in sel] + ["best1\n60k", "CityGS\nrelease"]
    def road(b, a, c):
        return next(r[7] for r in N19[b] if r[0] == a and r[1] == c and r[2] == 30)
    r6 = [road(6, a, c) for a, c in sel] + [N19_REF[6]["best1"][4], N19_REF[6]["off_road"]]
    r12 = [road(12, a, c) for a, c in sel] + [N19_REF[12]["best1"][4], N19_REF[12]["off_road"]]
    x = np.arange(len(nm)); w = 0.38; ax = axs[0]
    barlabels(ax, ax.bar(x - w / 2, r6, w, color=C1, label="b6"), "{:.0f}")
    barlabels(ax, ax.bar(x + w / 2, r12, w, color=C2, label="b12"), "{:.0f}")
    ax.set_xticks(x, nm, fontsize=8); ax.legend(fontsize=8); ax.set_ylabel("暗柏油中渲染表面在地底的 %")
    ax.set_title("馬路靠地底面元撐的比例（`18` 的量法）：遮罩＋dist 2.5M 兩塊都低於 CityGS", fontsize=9.5)
    ax = axs[1]
    # 盒內顆數（百萬）＝合併時保留、部署的大小；CityGS release 盒內 105.3 萬／93.6 萬（`16` §3）
    inb = {"best1 60k": (0.435, 0.648), "遮罩＋dist 1M 30k": (0.654, 0.757), "遮罩＋dist 2.5M 30k": (1.566, 1.885), "CityGS release": (1.053, 0.936)}
    x = np.arange(len(inb))
    barlabels(ax, ax.bar(x - w / 2, [v[0] for v in inb.values()], w, color=C1, label="b6"), "{:.2f}")
    barlabels(ax, ax.bar(x + w / 2, [v[1] for v in inb.values()], w, color=C2, label="b12"), "{:.2f}")
    ax.set_xticks(x, [k.replace(" ", "\n", 1) for k in inb], fontsize=8); ax.set_ylabel("盒內顆數（百萬）＝部署大小")
    ax.set_title("代價：遮罩把顆粒留在本塊 => 合併後模型變大（2.5M 版是 best1 的 3.6／2.9 倍）", fontsize=9.5)
    save(fig, d, "n19_road.png")


# 20：遮罩臂全場景回退（10-23）。逐塊＝MatrixCity test 741 幀依相機位置歸塊（labrun/perblock_1023.py），Δ 相對 full44_best
N20_BLK = {"遮罩 2.5M（現行 opacity）": [-0.533, -0.206, -0.129, -0.142, -0.299, -0.046, -0.079, -0.094, -0.103, -0.044, -0.063, -0.033, -0.120, +0.043, -0.032, -0.106],
           "遮罩 1M（現行 opacity）": [-0.586, -0.318, -0.218, -0.177, -0.421, -0.234, -0.190, -0.189, -0.339, -0.238, -0.217, -0.165, -0.324, -0.200, -0.294, -0.377],
           # 10-24 ed10b4 全場景（熵只往 0 推、10k 後切回雙向 ×4；labrun/perblock_1024.py）
           "★ 遮罩 2.5M ed10b4（10-24）": [+0.156, +0.260, +0.099, +0.083, +0.134, +0.206, +0.149, +0.087, +0.149, +0.199, +0.183, +0.134, +0.043, +0.148, +0.148, +0.035]}
# b0 換塊評測（logs/swap_b0_1023_1132.log、_1238.log；172 幀）：整張 Δ、本塊區域 Δ、幾何區域 Δ、接縫帶 Δ（0~.05、.05~.1、.1~.2、.2~.3、>.3）
N20_SWAP = [("2.5M 現行", -0.169, -0.503, -0.676, [-1.46, -0.89, -0.46, -0.32, -0.16]),
            ("2.5M 現行（重跑）", -0.180, -0.544, -0.734, [-1.48, -1.01, -0.48, -0.32, -0.20]),
            ("2.5M o（L1 放回）", -0.065, -0.177, -0.322, [-0.28, -0.21, -0.17, -0.16, -0.13]),
            ("2.5M ne（熵關）", -0.055, -0.239, -0.366, [-0.33, -0.17, -0.02, -0.05, -0.36]),
            ("★ 2.5M oe（兩者）", +0.004, -0.023, -0.044, [-0.02, +0.01, -0.03, -0.05, +0.01]),
            ("1M 現行", -0.086, -0.225, -0.421, [-0.24, -0.23, -0.30, -0.10, -0.23]),
            ("1M oe", -0.046, -0.144, -0.276, [-0.12, -0.15, -0.18, -0.02, -0.18])]
# b0 漂浮位置（現行 2.5M 漂浮物 xy 取 3000 點、半徑 0.01）的中層（高度 0.25~0.7、o>0.1）中位、盒內 opacity 中位（labrun/layers_v.py）
N20_LAYER = [("2.5M 現行", 69, 1.00), ("2.5M o", 0, 1.00), ("2.5M ne", 0, 0.91), ("2.5M oe", 0, 0.18), ("舊最佳 60k", 0, 0.34)]


def n20_blocks(d):
    """全場景逐塊 Δ（MatrixCity test 741 幀依相機位置歸塊）：2.5M 的損失集中在 b0／b4／b1，1M 是全面小輸。"""
    fig, ax = plt.subplots(figsize=(17, 4.4))
    x = np.arange(16); k = len(N20_BLK); w = 0.8 / k
    for j, (nm, v) in enumerate(N20_BLK.items()):
        barlabels(ax, ax.bar(x + (j - (k - 1) / 2) * w, v, w, color=(C2, C4, C1, C3)[j], label=nm), "{:+.2f}", dy=-10, size=6.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"b{b}" for b in range(16)]); ax.set_ylabel("PSNR Δ（dB，相對 full44_best 60k）")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("4x4 全場景、MatrixCity test 741 幀依相機位置歸塊：遮罩 2.5M 輸在 b0（−0.53，水面）／b4／b1；1M 每塊都小輸（容量）；10-24 ed10b4 16 塊全贏", fontsize=9.5)
    save(fig, d, "n20_blocks.png")


def n20_swap(d):
    """b0 換塊評測：opacity 變體、重跑對照、1M；接縫帶；漂浮位置的中層。"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1.5, 1.2, 0.9]})
    ax = axs[0]; y = np.arange(len(N20_SWAP)); h = 0.38
    b1 = ax.barh(y - h / 2, [r[1] for r in N20_SWAP], h, color=C1, label="整張畫面 Δ")
    b2 = ax.barh(y + h / 2, [r[2] for r in N20_SWAP], h, color=C2, label="本塊區域 Δ（alpha 定義）")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_width(); ax.annotate(f"{v:+.3f}", (v, b.get_y() + b.get_height() / 2), xytext=(-3 if v < 0 else 3, 0),
                                           textcoords="offset points", ha="right" if v < 0 else "left", va="center", fontsize=7, color=INK2)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y, [r[0] for r in N20_SWAP], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("PSNR Δ（dB，相對 full44_best；b0 視角 172 幀）"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("只換 b0：opacity 改回舊值（oe）追平；重跑對照與原跑次差 0.01", fontsize=9.5)
    ax = axs[1]; xs = ["0~.05", ".05~.1", ".1~.2", ".2~.3", ">.3"]
    for (nm, *_, band), c, ls in zip(N20_SWAP, (C2, C2, C4, C3, C1, MUTED, INK2), ("-", ":", "-", "-", "-", "-", "--")):
        ax.plot(xs, band, ls, marker="o", color=c, lw=2 if "★" in nm else 1.3, label=nm)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("離接縫距離（內部格距）"); ax.set_ylabel("本塊區域 PSNR Δ")
    ax.legend(fontsize=7, ncol=2, loc="lower right"); ax.set_title("損失集中在接縫旁（水面在 b0 邊緣）", fontsize=9.5)
    ax = axs[2]; x = np.arange(len(N20_LAYER))
    barlabels(ax, ax.bar(x, [r[1] for r in N20_LAYER], 0.6, color=C2), "{:.0f}")
    ax.set_xticks(x, [r[0].replace(" ", "\n", 1) for r in N20_LAYER], fontsize=8); ax.set_ylabel("漂浮位置的中層顆數（中位）")
    a2 = ax.twinx(); a2.plot(x, [r[2] for r in N20_LAYER], "D", color=C1, ms=7); a2.set_ylim(0, 1.1); a2.set_ylabel("盒內 opacity 中位", color=C1)
    a2.grid(False); ax.set_title("中層＝多出來的不透明層（三個變體都沒有）", fontsize=9.5)
    save(fig, d, "n20_swap.png")


# 21：配方 A 全場景（10-24～10-26）。全場景＝MatrixCity test 741 幀；逐塊＝labrun/perblock_1027.py（相機位置歸塊、可重疊）；單塊＝換塊評測＋區塊 F1
# (名稱, PSNR, LPIPS, 合併 N（百萬）, 16 塊純訓練 h, 顏色)
N21_FULL = [("full44 舊最佳（60k）", 27.739, .1293, 17.63, 30.7, MUTED), ("ed10b4", 27.868, .1153, 27.83, 7.32, C4),
            ("★ 配方 A", 28.048, .1143, 27.93, 5.90, C1), ("★ A＋dist 每 4 步", 28.058, .1142, 27.75, 5.04, "#123f7a"),
            ("★ A cap 1.5M", 27.931, .1286, 17.12, 4.48, C3), ("A cap 1M", 27.776, .1448, 11.50, 3.86, C2)]
N21_BLK = {"配方 A − ed10b4": [+0.171, +0.137, +0.137, +0.150, +0.206, +0.177, +0.155, +0.161, +0.218, +0.203, +0.173, +0.164, +0.236, +0.233, +0.178, +0.149],
           "A＋dist4 − ed10b4": [+0.179, +0.127, +0.131, +0.139, +0.203, +0.168, +0.171, +0.179, +0.253, +0.198, +0.184, +0.179, +0.238, +0.263, +0.204, +0.178],
           "A 平行重跑 − ed10b4": [+0.178, +0.144, +0.155, +0.170, +0.196, +0.167, +0.168, +0.179, +0.209, +0.195, +0.178, +0.170, +0.222, +0.230, +0.183, +0.153],
           "A cap 1.5M − ed10b4": [-0.022, -0.006, +0.056, +0.078, +0.071, +0.088, +0.091, +0.100, +0.110, +0.105, +0.077, +0.089, +0.157, +0.167, +0.070, +0.034],
           "A cap 1M − ed10b4": [-0.241, -0.206, -0.038, +0.021, -0.087, -0.083, -0.038, +0.003, -0.056, -0.061, -0.072, -0.026, +0.028, +0.018, -0.108, -0.136]}
# 單塊 (名稱, b6 本塊區域 Δ, b6 F1, b0 本塊區域 Δ, b0 F1)；None＝沒量
N21_SINGLE = [("ed10b4", 0.057, 0.521, -0.040, 0.758), ("配方 A", 0.215, 0.529, 0.123, 0.737),
              ("A＋dist 每 4 步（全場景的塊）", 0.206, 0.522, -0.022, 0.740), ("A＋dist 每 4 步（單塊）", 0.199, 0.525, 0.092, 0.741),
              ("＋撒點 依誤差", 0.210, 0.559, 0.061, 0.743), ("＋撒點 均勻", 0.210, 0.570, None, None),
              ("cap 1.5M", 0.203, 0.514, 0.040, 0.733), ("cap 1M", 0.172, 0.485, 0.022, 0.705)]


def n21_full(d):
    """全場景：PSNR 對 16 塊純訓練時間（點大小＝合併 N），CityGS release 畫成水平線。"""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for nm, p, lp, n, h, c in N21_FULL:
        ax.scatter(h, p, s=n * 22, color=c, alpha=0.85, edgecolor="white", lw=1, zorder=3)
        dx, dy, ha = {"★ A＋dist 每 4 步": (-16, 10, "right"), "★ 配方 A": (16, 4, "left"), "ed10b4": (18, -10, "left"),
                      "★ A cap 1.5M": (-14, -20, "right")}.get(nm, (12, -4, "left"))
        ax.annotate(f"{nm}\n{p:.3f}／LPIPS {lp:.4f}／{n:.1f}M", (h, p), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axhline(27.261, color=MUTED, ls="--", lw=1); ax.text(12, 27.275, "CityGS release 27.261（24.48M；b0~9 就要 40.3 h）", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_xticks([4, 5, 6, 8, 12, 20, 30]); ax.set_xticklabels(["4", "5", "6", "8", "12", "20", "30"])
    ax.set_xlim(2.6, 60); ax.set_ylim(27.2, 28.2)
    ax.set_xlabel("16 塊純訓練時間（h，對數；[solo]）"); ax.set_ylabel("MatrixCity test 741 幀 PSNR")
    ax.set_title("4x4 全場景：配方 A 28.048（5.90 h）、＋dist 每 4 步 28.058（5.04 h）；cap 1.5M 用舊最佳的顆數贏它", fontsize=9.5)
    save(fig, d, "n21_full.png")


def n21_blocks(d):
    """逐塊 Δ（相對 ed10b4）：配方 A 16 塊全贏；dist 每 4 步與平行重跑同級；1.5M 只在 b0／b1 輸。"""
    fig, ax = plt.subplots(figsize=(17, 4.4))
    x = np.arange(16); k = len(N21_BLK); w = 0.84 / k
    for j, (nm, v) in enumerate(N21_BLK.items()):
        ax.bar(x + (j - (k - 1) / 2) * w, v, w, color=(C1, "#123f7a", "#9cc3ee", C3, C2)[j], label=nm)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"b{b}" for b in range(16)]); ax.set_ylabel("PSNR Δ（dB，相對 ed10b4）")
    ax.legend(fontsize=8, ncol=5, loc="lower center")
    ax.set_title("MatrixCity test 741 幀逐塊：配方 A 全贏 ed10b4（+0.14～+0.24）；＋dist 每 4 步 與 A 差 −0.01～+0.04＝平行重跑同級（−0.01～+0.02）；1M 輸在 b0／b1／b15", fontsize=9.5)
    save(fig, d, "n21_blocks.png")


def n21_single(d):
    """單塊：本塊區域 Δ（換塊評測）對區塊 F1；b6 圓、b0 方。撒點把 b6 的 F1 往上推；b0 的區域 Δ 重跑就差 0.11。"""
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    cols = [C4, C1, "#123f7a", "#5a8fd0", C3, "#0e7a52", "#d9a400", C2]
    for ax, (bi, ci, title) in zip(axs, ((1, 2, "b6（324 幀）：撒點 F1 +0.03／+0.04，PSNR 不變"), (3, 4, "b0（172 幀）：dist 每 4 步兩次訓練本塊區域差 0.11（重跑噪音）"))):
        for r, c in zip(N21_SINGLE, cols):
            if r[bi] is None: continue
            ax.scatter(r[bi], r[ci], s=70, color=c, marker="o" if bi == 1 else "s", zorder=3)
            left = bi == 1 and "單塊" in r[0]
            ax.annotate(r[0], (r[bi], r[ci]), xytext=(-6 if left else 6, 3), ha="right" if left else "left",
                        textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xlabel("本塊區域 PSNR Δ（換塊評測，相對 full44_best）"); ax.set_ylabel("區塊幾何 F1")
        ax.set_title(title, fontsize=9.5)
    axs[1].annotate("", (0.092, 0.7415), (-0.022, 0.7415), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    save(fig, d, "n21_single.png")


# 22：viewer 看出的新視角缺陷（10-28～10-29）。浮塊＝盒內 o>0.2 且高出 SfM 包絡面（2 cm 格、5x5 鄰域最大；無 SfM 的格＝海）0.1 以上
#   v2（b12）＝面元頂端（中心＋3 sigma 垂直延伸，labrun/float2_blk_1029.py）；v1（b0）＝只看中心（labrun/float_blk_1029.py）
# (名稱, 日期, 海上, 陸上)
N22_B12 = [("CityGS release", "", 1651, 222), ("full44_best 舊最佳（opacity L1 0.002，60k）", "10-05", 1126, 185),
           ("best0（拿掉 opacity L1）", "10-10", 22143, 3954), ("best0＋二元熵＋V/C", "10-14", 99733, 16363),
           ("best1（＋兩個加速＝10-14 基底）", "10-14", 105028, 18048), ("best1＋dist100", "10-16", 84718, 12845),
           ("best1＋法向＋深度正則", "10-16", 95478, 15165), ("15k＋SH1", "10-17", 26795, 3878),
           ("遮罩＋dist＋30k（10-22 全場景臂）", "10-22", 215172, 36151), ("ed10b4（熵只往 0 推、10k 後雙向 ×4）", "10-24", 106012, 16032),
           ("★ A＋dist4（現行最佳）", "10-26", 128793, 21701)]
# (名稱, 浮塊顆數, 組)；b0、30k cap 2.5M 遮罩臂（opacity 變體，20 §6）與配方 A 家族
N22_B0 = [("full44_best 舊最佳（60k）", 829, 0), ("oe：L1 0.002＋熵關", 8530, 0), ("ed：熵只往 0 推", 28700, 0), ("ed15：15k 後雙向", 33806, 0),
          ("o：L1 0.002＋熵雙向", 43600, 0), ("ed10", 53700, 0), ("ed10b4（全場景採用）", 61700, 0), ("ne：L1 0、熵關", 76783, 0),
          ("o05：L1 0.0005＋熵", 153000, 0), ("遮罩臂 rf：L1 0＋熵雙向", 214644, 0),
          ("配方 A", 64000, 1), ("A＋深度補點初始化", 24000, 1), ("A＋撒點（依誤差）", 91000, 1), ("A＋撒點（均勻）", 108000, 1),
          ("★ A＋dist4", 50000, 1), ("A＋dist4＋撒點（均勻）", 102000, 1)]
# 面元朝向一致度（labrun/orient_blk_1029.py）：盒內 o>0.5、8 近鄰 |n_i . n_j| 中位；亂向＝<0.7 的比例；(名稱, 一致度, 亂向 %, 顆數 M, 長軸中位 cm)
N22_OR = {3: [("CityGS release", 0.928, 11.8, 0.594, 0.72), ("full44_best 舊最佳", 0.802, 30.1, 0.380, 0.69),
              ("遮罩＋dist＋30k（10-22）", 0.708, 48.4, 1.776, 0.15), ("ed10b4", 0.770, 36.4, 1.710, 0.26),
              ("配方 A", 0.764, 37.3, 1.734, 0.28), ("★ A＋dist4", 0.765, 37.1, 1.756, 0.27), ("A cap 1M", 0.773, 35.3, 0.723, 0.51)],
          12: [("CityGS release", 0.908, 14.6, 0.352, 0.63), ("full44_best 舊最佳", 0.797, 31.1, 0.275, 0.46),
               ("best1", 0.801, 30.0, 0.631, 0.42), ("best1＋dist100", 0.803, 29.9, 0.608, 0.43),
               ("best1＋法向（λ 0.0125）", 0.951, 14.1, 0.643, 0.43), ("best1＋法向＋深度", 0.957, 12.0, 0.633, 0.44),
               ("15k＋SH1", 0.688, 52.4, 0.680, 0.33), ("ed10b4", 0.786, 32.7, 1.561, 0.23), ("★ A＋dist4", 0.782, 33.4, 1.597, 0.24)]}


def n22_b12hist(d):
    """b12 歷代跑次的海上／陸上浮塊（v2，面元頂端）：退步從 10-10 拿掉 opacity L1 開始。"""
    fig, ax = plt.subplots(figsize=(14, 4.9))
    y = np.arange(len(N22_B12)); h = 0.4
    sea = [r[2] for r in N22_B12]; land = [r[3] for r in N22_B12]
    cs = [MUTED if r[0].startswith("CityGS") else ("#123f7a" if "★" in r[0] else C1) for r in N22_B12]
    ax.barh(y - h / 2, sea, h, color=cs, label="海上（無 SfM 的格）")
    ax.barh(y + h / 2, land, h, color=C2, alpha=0.8, label="陸上")
    for i, (v1, v2) in enumerate(zip(sea, land)):
        ax.text(v1 * 1.08, i - h / 2, f"{v1 / 1000:.1f}k", va="center", fontsize=7.5, color=INK2)
        ax.text(v2 * 1.08, i + h / 2, f"{v2 / 1000:.2f}k", va="center", fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(100, 6e5)
    ax.set_yticks(y, [f"{r[0]}{'（' + r[1] + '）' if r[1] else ''}" for r in N22_B12], fontsize=8.5); ax.invert_yaxis()
    ax.annotate("×20：拿掉 opacity L1", xy=(22143, 2 - h / 2), xytext=(1.8e5, 1.6), fontsize=8.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9))
    ax.annotate("×4.5：＋二元熵＋V/C", xy=(99733, 3 - h / 2), xytext=(2.6e5, 3.4), fontsize=8.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C1, label="海上（無 SfM 的格；灰＝CityGS、深藍＝現行）"), Patch(color=C2, alpha=0.8, label="陸上")],
              fontsize=8, loc="lower right")
    ax.set_xlabel("浮塊顆數（對數；盒內 o>0.2、面元頂端高出 SfM 包絡面 0.1 以上）")
    ax.set_title("b12 浮塊歷代：舊最佳與 CityGS 都只有 1～2 千顆；10-10 拿掉 opacity L1 後 ×20、熵＋V/C 再 ×4.5；現行 A＋dist4 海上 12.9 萬顆\n"
                 "（MatrixCity test 741 幀全是空拍高度 => test PSNR 量不到；法向＋深度正則只減 9%）", fontsize=9.5)
    save(fig, d, "n22_b12hist.png")


def n22_b0var(d):
    """b0 浮塊（v1，只看中心）：opacity 變體（遮罩臂 30k 2.5M）與配方 A 家族（深度補點、撒點、dist4）。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, g, title in zip(axs, (0, 1), ("opacity 變體（b0、遮罩＋dist＋30k 2.5M；20 §6 的臂）：只有 L1 能把浮塊壓到 1 萬以下，但 [solo] 訓練 +49%",
                                           "配方 A 家族（b0）：深度補點初始化 −62%；撒點讓浮塊加倍")):
        R = [r for r in N22_B0 if r[2] == g]
        y = np.arange(len(R))
        cs = [MUTED if "舊最佳" in r[0] else ("#123f7a" if "★" in r[0] or "採用" in r[0] else (C3 if "補點" in r[0] else (C2 if "撒點" in r[0] else C1))) for r in R]
        ax.barh(y, [r[1] for r in R], 0.62, color=cs)
        for i, r in enumerate(R):
            ax.text(r[1] * 1.08, i, f"{r[1] / 1000:.1f}k" if r[1] >= 1000 else f"{r[1]}", va="center", fontsize=8, color=INK2)
        ax.set_xscale("log"); ax.set_xlim(300 if g == 0 else 8000, 8e5)
        ax.set_yticks(y, [r[0] for r in R], fontsize=8.5); ax.invert_yaxis()
        ax.set_xlabel("浮塊顆數（對數；v1＝只看中心高出 SfM 包絡 0.1）"); ax.set_title(title, fontsize=9.2)
    save(fig, d, "n22_b0var.png")


def n22_orient(d):
    """面元朝向一致度：立面「毛邊」＝朝向亂＋小面元多；法向正則把 b12 拉到 0.95（勝 CityGS），但每塊 +63% 時間。"""
    fig, axs = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, b in zip(axs, (3, 12)):
        R = N22_OR[b]; x = np.arange(len(R))
        cs = [MUTED if r[0].startswith("CityGS") else ("#123f7a" if "★" in r[0] else (C3 if "法向" in r[0] else C1)) for r in R]
        bars = ax.bar(x, [r[1] for r in R], 0.62, color=cs)
        for b_, r in zip(bars, R):
            ax.text(b_.get_x() + b_.get_width() / 2, r[1] + 0.004, f"{r[1]:.3f}\n亂向 {r[2]:.0f}%\n{r[3]:.2f}M／{r[4]:.2f} cm", ha="center", fontsize=6.8, color=INK2)
        ax.set_ylim(0.6, 1.02); ax.set_xticks(x, [r[0].replace("（", "\n（").replace("＋", "\n＋", 1) for r in R], fontsize=7.3)
        ax.set_ylabel("8 近鄰朝向一致度中位（|n_i . n_j|）")
        ax.set_title({3: "b3（使用者看到毛邊的塊）：現行 0.765 vs 舊最佳 0.802、CityGS 0.928；不透明面元是舊最佳的 4.6 倍、長軸一半",
                      12: "b12：渲染式法向正則 0.80 → 0.95（勝 CityGS 0.908）、F1 0.727 → 0.739；但不減浮塊（沿法向平移不變）"}[b], fontsize=9)
    fig.suptitle("盒內 o>0.5 的面元：標註＝亂向比例（一致度 < 0.7）、顆數／長軸中位", fontsize=9.5)
    save(fig, d, "n22_orient.png")


def n22_road_roof(d):
    """左：馬路靠地底面元的比例（18 的同一支工具）從 ed10b4 起退步；右：高樓區在 MatrixCity test 視角的 PSNR 與 CityGS 同級。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    nm = ["best1\n60k（10-14）", "遮罩＋dist100\n30k（10-22）", "ed10b4\n（10-24）", "★ A＋dist4\n（10-26）", "CityGS\nrelease"]
    r6 = [37.7, 8.5, 21.3, 24.5, 16.3]; r12 = [52.4, 31.4, 46.2, 55.2, 38.5]
    x = np.arange(len(nm)); w = 0.38
    barlabels(ax, ax.bar(x - w / 2, r6, w, color=C1, label="b6"), "{:.1f}")
    barlabels(ax, ax.bar(x + w / 2, r12, w, color=C2, label="b12"), "{:.1f}")
    ax.set_xticks(x, nm, fontsize=8); ax.legend(fontsize=8); ax.set_ylabel("暗柏油中渲染表面在地底的 %")
    ax.set_title("馬路地底面元（新視角變透明的原因，`18`）：10-22 遮罩＋dist100 壓到 CityGS 以下，ed10b4／A＋dist4 又退回\n（b12 的地面估計受水面影響，看相對值）", fontsize=9.2)
    ax = axs[1]
    labs = ["b3（高樓 SfM 點最多的 8 幀）", "b2（同）"]; v = {"★ A＋dist4": [24.54, 23.61], "A＋dist4＋撒點（均勻）": [24.68, 23.82], "CityGS release": [24.92, 23.63]}
    x = np.arange(2); w = 0.26
    for k, (lb, vv) in enumerate(v.items()):
        barlabels(ax, ax.bar(x + (k - 1) * w, vv, w, color=("#123f7a", C2, MUTED)[k], label=lb), "{:.2f}")
    ax.set_ylim(20, 27); ax.set_xticks(x, labs); ax.set_ylabel("高樓區 PSNR（dB）")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.set_title("高樓頂樓在 MatrixCity test 視角：與 CityGS 同級（b3 −0.38、b2 −0.02）\n=> viewer 看到的「整個糊掉」是新視角（低角度）問題，三個模型都有", fontsize=9.2)
    save(fig, d, "n22_road_roof.png")


# ───────────── 23（10-30～10-31）：cap 劑量、熵拆開、塔頂、幾何約束臂
N23_DOSE = {  # cap(M): (b3 本塊, b3 幾何, b12 本塊, b12 幾何)；換塊 Δ 相對 full44；b3 貼牆度；每塊分鐘
    2.5: (0.22, 0.41, 0.07, 0.18, 0.76, 20), 1.0: (0.26, 0.48, 0.03, 0.10, 0.88, 13),
    0.5: (0.12, 0.21, -0.08, -0.09, 0.93, 9.5), 0.2: (-0.36, -0.62, -0.37, -0.59, 0.96, 7.5)}
N23_ARMS = [  # 臂, 時間 %（b3／b12）, b12 F1, 說明
    ("現行 A＋dist4", 0, 0, 0.601, ""), ("深度支撐閘門（ds）", 8, 8, 0.60, "海上浮塊 −85～95%"),
    ("單側不能看穿（st4）", 9, 0, 0.694, "F1 +0.09；b12 低角度 3.3 → 7.3%"), ("CityGS 式單目深度每 4 步（dl4）", 22, 22, 0.61, "PSNR +0.03"),
    ("逐顆共面（pl）", 43, 43, None, "朝向 0.987"), ("前期 L1（l1e）", 36, 36, 0.43, "浮塊 −90% 但塊內顆粒 ÷5")]
N23_TOWER = [(99, 0.50), (999, 0.28), (2999, 0.04), (4999, 0.007)]
N23_TOWER30K = [("現行", 0.0), ("拿掉熵", 0.0), ("trim 最大貢獻保護", 0.0), ("MCMC |g| 保護", 0.0), ("L1 混 L2（w2）", 0.94)]


def n23_dose(d):
    """cap 劑量：b3 在 1M 最好、b12 要 2.5M；貼牆度隨 cap 降而變好；每塊時間 20 → 7.5 分。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.3))
    caps = sorted(N23_DOSE)
    ax = axs[0]
    for k, (lab, c, ls) in enumerate((("b3 本塊", C1, "-"), ("b3 幾何區域", C1, "--"), ("b12 本塊", C2, "-"), ("b12 幾何區域", C2, "--"))):
        ax.plot(caps, [N23_DOSE[c_][k] for c_ in caps], ls, marker="o", color=c, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xscale("log"); ax.set_xticks(caps, [f"{c}M" for c in caps])
    ax.set_xlabel("每塊 cap"); ax.set_ylabel("換塊評測 Δ PSNR（相對 full44，dB）"); ax.legend(fontsize=8)
    ax.set_title("cap 劑量（A＋dist4、30k）：b3 在 1M 最好、b12 要 2.5M => 兩塊最佳顆數不同（自適應 cap 的依據）", fontsize=9.2)
    ax = axs[1]
    adh = [N23_DOSE[c][4] for c in caps]; tm = [N23_DOSE[c][5] for c in caps]
    ax.plot(caps, adh, marker="o", color=C3, label="b3 立面貼牆度（CityGS 0.97）")
    ax.set_xscale("log"); ax.set_xticks(caps, [f"{c}M" for c in caps]); ax.set_ylabel("貼牆度中位"); ax.set_ylim(0.7, 1.0)
    ax2 = ax.twinx(); ax2.plot(caps, tm, marker="s", color=C4, label="每塊訓練分鐘"); ax2.set_ylabel("分鐘"); ax2.grid(False)
    for c, t in zip(caps, tm):
        ax2.text(c, t + 0.6, f"{t:g} 分", ha="center", fontsize=8, color=C4)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("顆數少 => 貼牆度變好、每塊時間 20 → 7.5 分；但 LPIPS 與 F1 單調變差（b3 F1 0.378 → 0.270、b12 0.601 → 0.478）", fontsize=9.2)
    save(fig, d, "n23_dose.png")


def n23_arms(d):
    """幾何約束臂（10-30～31，A＋dist4 上，b3／b12）：時間代價與 b12 F1。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    R = N23_ARMS; y = np.arange(len(R)); h = 0.38
    ax = axs[0]
    ax.barh(y - h / 2, [r[1] for r in R], h, color=C1, label="b3"); ax.barh(y + h / 2, [r[2] for r in R], h, color=C2, label="b12")
    for i, r in enumerate(R):
        ax.text(max(r[1], r[2]) + 1, i, r[4], va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(y, [r[0] for r in R], fontsize=8.5); ax.invert_yaxis(); ax.set_xlim(0, 75)
    ax.set_xlabel("每塊純訓練時間增加 %（相對現行）"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("時間代價：閘門 +8%、單側不能看穿 +9%（b3）；單目深度 loss、逐顆共面、前期 L1 都太貴", fontsize=9.2)
    ax = axs[1]
    RR = [r for r in R if r[3] is not None]
    bars = ax.bar(np.arange(len(RR)), [r[3] for r in RR], 0.6, color=["#123f7a"] + [C1] * (len(RR) - 1))
    for b_, r in zip(bars, RR):       # ds／dl4 當時只記「不變」「+0.01」=> 標 ≈
        ax.text(b_.get_x() + b_.get_width() / 2, r[3] + 0.004, (f"≈{r[3]:.2f}" if r[0].endswith(("（ds）", "（dl4）")) else f"{r[3]:.3f}"),
                ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(np.arange(len(RR)), [r[0].replace("（", "\n（") for r in RR], fontsize=7.5); ax.set_ylim(0.3, 0.75)
    ax.axhline(0.727, color=MUTED, ls="--", lw=0.8); ax.text(len(RR) - 0.5, 0.732, "g44_best1 60k 0.727", fontsize=7.5, color=INK2, ha="right")
    ax.set_ylabel("b12 區塊 F1"); ax.set_title("b12 幾何 F1：單側不能看穿 0.601 → 0.694（之後拆項：全部來自補洞項）", fontsize=9.2)
    save(fig, d, "n23_arms.png")


def n23_tower(d):
    """b2 超高塔頂（0244 幀）：opacity 一路被推到 0、5k～8k 步消失；只有 L1 混 L2 救得回來。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    st = [s for s, _ in N23_TOWER]; ov = [o for _, o in N23_TOWER]
    ax.plot(st, ov, marker="o", color=C2); ax.set_yscale("log"); ax.set_xscale("log")
    for s, o in N23_TOWER:
        ax.text(s, o * 1.25, f"{o:g}", ha="center", fontsize=8, color=INK2)
    ax.axvspan(5000, 8000, color="#f1e3a6", alpha=0.5, lw=0); ax.text(6200, 0.012, "5k～8k 步\n塔頂消失", ha="center", fontsize=8)
    ax.set_xlabel("訓練步（對數）"); ax.set_ylabel("塔頂附近顆粒 opacity 中位（對數）")
    ax.set_title("現行 A＋dist4 的 b2 超高塔頂：opacity 從第 99 步起一路被往下推（少數視角才看得到 => L1 每像素只投 ±1 票）", fontsize=9.2)
    ax = axs[1]
    bars = ax.bar(np.arange(len(N23_TOWER30K)), [v for _, v in N23_TOWER30K], 0.6, color=[MUTED] * 4 + ["#123f7a"])
    barlabels(ax, bars, "{:.2f}")
    ax.set_xticks(np.arange(len(N23_TOWER30K)), [n for n, _ in N23_TOWER30K], fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("30k 時塔頂 SfM 點 5 cm 內有 o>0.5 顆粒的比例")
    ax.set_title("對策（b2，30k）：拿掉熵、trim 保護、|g| 保護都在 8k～12k 步歸零；只有 L1 混 L2 留住（opacity 中位 1.0）", fontsize=9.2)
    save(fig, d, "n23_tower.png")


# ───────────── 24（11-01）：法向／SfM 豁免／鄰域平面／合併／單目深度診斷／全場景 1M
N24_ARMS = [  # 臂, (b3 本塊 Δ, b3 F1, b3 海上浮塊k, b3 朝向, b3 低角度%, b3 秒), (b12 同)
    ("現行 A＋dist4", (0.223, .378, 51.6, .765, 5.2, 1195), (0.065, .601, 126.3, .782, 3.3, 1215)),
    ("L1 混 L2（w2）", (0.277, .388, 48.0, None, 3.9, 1170), (0.062, .611, 93.0, .767, 5.8, 1206)),
    ("法向每 2 步", (0.247, .404, 65.2, .935, 2.9, 1380), (0.130, .614, 128.2, .946, 6.2, 1440)),
    ("法向每 4 步", (0.239, .395, 66.0, .931, 2.7, 1260), (0.126, .619, 140.0, .943, 4.7, 1320)),
    ("法向每 8 步", (0.208, .391, 68.9, .930, 2.7, 1200), (0.162, .620, 136.1, .941, 4.2, 1260)),
    ("閘門＋不能看穿（dsst4）", (0.19, .416, 9.9, .748, 3.8, 1381), (0.15, .684, 8.5, .752, 6.4, 1352)),
    ("＋SfM 豁免（dsxst4）", (0.218, .420, 10.8, .747, 4.7, 1368), (0.137, .694, 8.6, .753, 7.8, 1350)),
    ("＋L2＋法向每 4 步（完整組合）", (0.080, .442, 12.3, .935, 2.4, 1382), (0.194, .691, 9.1, .942, 5.8, 1373)),
    ("鄰域平面：位置項", (0.258, .365, 47.6, .753, 4.7, 1318), (0.075, .600, 99.7, .767, 2.6, 1318)),
    ("鄰域平面：位置＋朝向", (0.208, .362, 40.2, .907, 2.4, 1314), (0.096, .605, 96.1, .872, 3.4, 1277)),
    ("訓練中合併（搬走、只併同色）", (0.242, .374, 49.2, .757, 3.9, 1219), (0.066, .602, 114.6, .770, 4.4, 1230)),
]
N24_MONO = {  # 高度帶: (①現行, ②每張真深度, ③每 60 px 真深度, ④每 120 px SfM) 判前方 %
    "全部": (14.6, 9.7, 0.4, 5.8), "z<0.05 地面": (9.3, 2.7, 0.1, 4.5), "z 0.05~1": (12.8, 10.4, 0.5, 3.6),
    "z 1~2": (37.2, 27.2, 1.3, 18.5), "z 2~3": (46.6, 37.7, 1.6, 21.3), "z≥3 超高塔頂": (68.3, 50.7, 1.9, 32.1)}
N24_FULL = [  # 名稱, PSNR, SSIM, LPIPS, N(M), 16 塊純訓練 h, FPS
    ("CityGS release", 27.261, .8671, .1540, 24.48, None, 21.9), ("★ A＋dist4 每塊 2.5M", 28.058, .8824, .1142, 27.75, 5.04, 40.6),
    ("A＋dist4 每塊 1M", 27.805, .8709, .1443, 11.57, 3.23, 57.7), ("A＋dist4 自適應 平均 1M", 27.828, .8725, .1406, 11.64, 3.19, 57.7)]


def n24_swap(d):
    """11-01 單塊臂（A＋dist4 上）：換塊本塊 Δ 與 F1。"""
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.4))
    R = N24_ARMS; y = np.arange(len(R))
    for ax, k, b in zip(axs, (1, 2), (3, 12)):
        v = [r[k][0] for r in R]; f1 = [r[k][1] for r in R]
        cs = ["#123f7a" if i == 0 else (C3 if "法向" in r[0] or "平面" in r[0] else (C2 if "閘門" in r[0] or "SfM" in r[0] or "組合" in r[0] else C1)) for i, r in enumerate(R)]
        ax.barh(y, v, 0.62, color=cs)
        for i, (a, f) in enumerate(zip(v, f1)):
            ax.text(max(a, 0) + 0.005, i, f"{a:+.3f}｜F1 {f:.3f}｜{R[i][k][5] / 60:.0f} 分", va="center", fontsize=7.6, color=INK2)
        ax.axvline(R[0][k][0], color=MUTED, ls="--", lw=0.8)
        ax.set_yticks(y, [r[0] for r in R], fontsize=8.3); ax.invert_yaxis(); ax.set_xlim(0, 0.42)
        ax.set_xlabel("換塊評測 本塊區域 Δ PSNR（相對 full44，dB）")
        ax.set_title({3: "b3：完整組合本塊掉到 +0.080（塊內 >0.3 帶 +0.13 → −0.13）；其餘在雜訊內（單塊約 ±0.1）",
                      12: "b12：閘門類 +0.07～0.13、F1 0.68～0.69；法向每 8 步 +0.16"}[b], fontsize=9.2)
    fig.suptitle("11-01 單塊臂（A＋dist4 30k；標註＝F1｜純訓練分鐘；現行 b3 20 分、b12 20 分）", fontsize=9.6)
    save(fig, d, "n24_swap.png")


def n24_geom(d):
    """同一批臂的幾何：海上浮塊、朝向一致度、低角度透空。"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    R = N24_ARMS; y = np.arange(len(R)); h = 0.38
    spec = [(2, "海上浮塊（千顆，盒內 o>0.2 中心高出 SfM 包絡 0.1）", "{:.1f}k"), (3, "朝向一致度（8 近鄰 |n_i·n_j| 中位）", "{:.3f}"),
            (4, "合成低角度透空率 %（CityGS 實心像素上）", "{:.1f}%")]
    for ax, (j, lab, fmt) in zip(axs, spec):
        for k, (b, c) in enumerate(((1, C1), (2, C2))):
            vals = [r[b][j] if r[b][j] is not None else np.nan for r in R]
            bars = ax.barh(y + (k - 0.5) * h, vals, h, color=c, label=f"b{3 if b == 1 else 12}")
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(v * 1.02 if j != 3 else v + 0.003, i + (k - 0.5) * h, fmt.format(v), va="center", fontsize=6.6, color=INK2)
        ax.set_yticks(y, [r[0] for r in R] if j == 2 else [""] * len(R), fontsize=8.3); ax.invert_yaxis(); ax.set_xlabel(lab, fontsize=8.5)
        if j == 3:
            ax.set_xlim(0.7, 1.0)
        ax.legend(fontsize=8, loc="lower right")
    axs[0].set_title("閘門類壓到 1 萬左右；法向反而 +26～33%", fontsize=9.2)
    axs[1].set_title("法向與「位置＋朝向」把朝向拉到 0.87～0.95", fontsize=9.2)
    axs[2].set_title("b12 低角度：閘門＋不能看穿 7.8% 最差；位置項 2.6% 最好", fontsize=9.2)
    save(fig, d, "n24_geom.png")


def n24_mono(d):
    """單目深度誤差診斷（5,606 張、MatrixCity 真深度只診斷）：真表面被判「在前方」的比例，依高度、四種對齊。"""
    fig, ax = plt.subplots(figsize=(15, 4.4))
    keys = list(N24_MONO); x = np.arange(len(keys)); w = 0.2
    labs = ["① 現行（每張一組、SfM 擬合）", "② 每張一組（真深度擬合）", "③ 每 60 px 一組（真深度擬合）", "④ 每 120 px 一組（SfM 擬合）"]
    for k, (lb, c) in enumerate(zip(labs, (C2, MUTED, C3, C1))):
        barlabels(ax, ax.bar(x + (k - 1.5) * w, [N24_MONO[kk][k] for kk in keys], w, color=c, label=lb), "{:.1f}", size=7)
    ax.set_xticks(x, keys); ax.set_ylabel("判前方 %（e < −5%）"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("單目深度把高樓「壓扁」：越高越被說成更遠（塔頂 68%）；每張重擬合救不了、小範圍擬合幾乎全救 => 大範圍低頻扭曲；"
                 "SfM 豁免已蓋住 92.7% 的表面", fontsize=9.2)
    save(fig, d, "n24_mono.png")


def n24_full(d):
    """全場景（MatrixCity test 741 幀）：每塊 1M 兩版 vs 2.5M vs CityGS。"""
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.9))
    R = N24_FULL; x = np.arange(len(R)); cs = [MUTED, "#123f7a", C1, C3]
    for ax, j, lab, fmt in zip(axs, (1, 3, 4, 5), ("PSNR（dB）", "LPIPS（越低越好）", "合併顆數（M）", "16 塊純訓練（h）"), ("{:.3f}", "{:.4f}", "{:.2f}", "{:.2f}")):
        vals = [r[j] if r[j] is not None else 0 for r in R]
        bars = ax.bar(x, vals, 0.62, color=cs)
        for b_, r in zip(bars, R):
            ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), fmt.format(r[j]) if r[j] is not None else "未量",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(x, [r[0].replace("A＋dist4 ", "A＋dist4\n") for r in R], fontsize=7.2); ax.set_title(lab, fontsize=9.2)
        if j == 1:
            ax.set_ylim(27, 28.2)
    fig.suptitle("全場景：自適應 cap 平均 1M 以同顆數贏均勻 1M（+0.023 dB、LPIPS −2.6%；全場景雜訊 0.002）；1M 對 2.5M 時間 −36%、顆數 −58%、FPS +42%，PSNR −0.23", fontsize=9.6)
    save(fig, d, "n24_full.png")


N26_ARMS = [  # 臂, b3 (held-out PSNR, LPIPS, 牆鐘 Δ%（fit 起訖秒數算）, 離線 Load M), b12 同（None＝沒跑）；本機筆電 1M、對同機基準
    ("基準 A＋dist4 1M", (11.642, .7671, 0.0, 1.742), (11.782, .8137, 0.0, 1.388)),
    ("反向 V/C", (11.296, .8397, 28.6, 8.243), None),
    ("trim 只看 V", (11.640, .7682, 20.1, 3.387), None),
    ("拿掉二元熵", (11.652, .7657, -0.1, 1.638), (11.805, .8112, -1.4, 1.315)),
    ("scale 正則 0", (11.667, .7660, 4.6, 2.123), (11.802, .8120, 5.9, 1.966)),
    ("全部（熵關＋sr0＋只看 V＋合併）", (11.671, .7705, 21.8, 4.133), (11.824, .8138, 17.0, 2.989)),
    ("補洞加嚴 A（margin 0.02）", (11.640, .7672, 5.4, 1.857), None),
    ("補洞加嚴 B（margin 0.01）", (11.633, .7666, -2.7, 1.718), None),
    ("★ 法向每 8 步", (11.672, .7658, 5.4, 1.940), (11.894, .8065, 6.9, 1.409)),
]
N26_GROUND = {  # 模型: (貼地覆蓋, 蓋滿格 %, 貼地躺平 %, 地下覆蓋)
    "b3": [("CityGS", 1.12, 43.4, 90.9, 1.08), ("lab 現行 2.5M", 0.58, 19.2, 35.3, 8.37), ("lab 完整組合", 0.91, 32.7, 89.9, 7.96),
           ("lab 鄰域平面", 0.65, 21.4, 88.2, 7.67), ("本機基準 1M", 0.54, 18.4, 44.7, 4.92), ("拿掉熵", 0.55, 18.3, 45.3, 4.58),
           ("scale 正則 0", 0.67, 23.3, 67.3, 4.52), ("全部", 0.60, 18.4, 78.0, 6.36), ("補洞加嚴 A", 0.54, 18.2, 47.2, 4.64),
           ("補洞加嚴 B", 0.55, 18.3, 47.6, 5.16), ("★ 法向每 8 步", 0.62, 21.9, 79.2, 5.25)],
    "b12": [("CityGS", 1.25, 41.6, 83.0, 1.14), ("本機基準 1M", 0.73, 26.3, 57.7, 9.00), ("拿掉熵", 0.70, 26.2, 52.4, 7.33),
            ("scale 正則 0", 0.80, 29.7, 68.9, 8.96), ("全部", 0.70, 25.4, 71.7, 10.51), ("★ 法向每 8 步", 0.79, 30.8, 87.8, 8.94)],
}
N26_THICK = {  # 模型: [(厚度 cm, 低角度透空 %, 前方遮擋 %, 空拍改動 >2/255 %)]
    "CityGS": [(0, 0.00, 0.00, 0.0), (2, 0.08, 0.07, 0.67), (5, 0.20, 0.29, 5.02), (10, 0.39, 0.75, 15.98), (20, 0.81, 1.72, 34.08), (50, 2.20, 4.64, 56.06)],
    "本機 1M": [(0, 5.84, 34.97, 0.0), (2, 5.83, 35.01, 1.37), (5, 5.63, 35.08, 3.59), (10, 5.23, 35.30, 10.83), (20, 4.61, 35.96, 25.01), (50, 4.03, 38.21, 47.11)],
    "lab 2.5M": [(0, 6.18, 35.86, 0.0), (2, 6.37, 37.95, 4.27), (5, 6.03, 38.30, 7.74), (10, 5.46, 38.68, 15.83), (20, 4.62, 39.46, 29.57), (50, 4.23, 41.81, 51.53)],
}
N26_MONO = {  # 模型: (深 >1/2/5/10 %), (淺 >1/2/5/10 %)
    "CityGS": ((32.1, 26.4, 15.8, 9.8), (44.6, 35.6, 25.0, 15.3)),
    "本機 1M": ((40.2, 33.9, 20.9, 11.0), (42.9, 38.4, 29.9, 19.2)),
    "法向每 8 步": ((41.5, 35.6, 21.7, 11.2), (42.5, 38.0, 29.6, 19.2)),
}
N26_UNDER = [("SfM（track≥3）", 0.9, 8.8), ("SfM（全部）", 3.1, 12.5), ("初始化 PLY", 3.8, 21.3), ("本機 1M 訓練後", 10.0, 27.1), ("CityGS", 1.1, 16.5)]  # 低於地面 3 m／0.5 m 的 %


def n26_arms(d):
    """11-03～04 本機 1M 各臂：held-out Δ、牆鐘 Δ%、離線 Load Δ%（對同機同塊基準）。"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4))
    R = N26_ARMS; y = np.arange(len(R)); h = 0.38
    spec = [(lambda r, b: r[0] - b[0], "held-out PSNR Δ（dB；整張畫面，只比差）", "{:+.3f}"),
            (lambda r, b: r[2], "訓練牆鐘 Δ%（本機；反向 V/C、只看 V 那兩次有干擾）", "{:+.1f}%"),
            (lambda r, b: 100 * (r[3] / b[3] - 1), "離線渲染 Load Δ%（精確 Σtiles 每視角平均）", "{:+.1f}%")]
    for ax, (fn, lab, fmt) in zip(axs, spec):
        for k, (bi, c) in enumerate(((1, C1), (2, C2))):
            base = R[0][bi]
            vals = [fn(r[bi], base) if r[bi] is not None else np.nan for r in R]
            ax.barh(y + (k - 0.5) * h, vals, h, color=c, label="b3" if bi == 1 else "b12")
            for i, v in enumerate(vals):
                if not np.isnan(v) and i > 0:
                    ax.text(v, i + (k - 0.5) * h, " " + fmt.format(v), va="center", fontsize=6.8, color=INK2,
                            ha="left" if v >= 0 else "right")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [r[0] for r in R] if ax is axs[0] else [""] * len(R), fontsize=8.3); ax.invert_yaxis()
        ax.set_xlabel(lab, fontsize=8.5); ax.legend(fontsize=8, loc="lower right")
    axs[2].set_xscale("symlog", linthresh=50)
    axs[0].set_title("法向每 8 步兩塊都升（b3 +0.03、b12 +0.11）；反向 V/C −0.35", fontsize=9.2)
    axs[1].set_title("「全部」+17～22%；拿掉熵 ±1%；法向 +5～7%", fontsize=9.2)
    axs[2].set_title("Load：反向 V/C ×4.7、只看 V ×1.9、全部 ×2.2～2.4；拿掉熵 −5～6%", fontsize=9.2)
    fig.suptitle("11-03～04 本機筆電 1M（A＋dist4、無閘門；對同機同塊基準 b3 37.6 分／1.74M、b12 35.0 分／1.39M）", fontsize=9.6)
    save(fig, d, "n26_arms.png")


def n26_ground(d):
    """地面格（SfM）上：貼地覆蓋、蓋滿格、地下覆蓋（Σ o·2π·su·sv ÷ 地面格面積）。"""
    fig, axs = plt.subplots(1, 2, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, blk in zip(axs, ("b3", "b12")):
        R = N26_GROUND[blk]; y = np.arange(len(R)); h = 0.38
        cs_near = [MUTED if r[0] == "CityGS" else C1 for r in R]
        ax.barh(y - h / 2, [r[1] for r in R], h, color=cs_near)
        ax.barh([np.nan], [np.nan], h, color=C1, label="貼地覆蓋（±0.5 m，層；灰＝CityGS）")
        ax.barh(y + h / 2, [r[4] for r in R], h, color=C2, label="地面下覆蓋（層）")
        for i, r in enumerate(R):
            ax.text(r[1] + 0.05, i - h / 2, f"{r[1]:.2f}｜蓋滿 {r[2]:.0f}%｜躺平 {r[3]:.0f}%", va="center", fontsize=6.8, color=INK2)
            ax.text(r[4] + 0.05, i + h / 2, f"{r[4]:.2f}", va="center", fontsize=6.8, color=INK2)
        ax.set_yticks(y, [r[0] for r in R], fontsize=8.3); ax.invert_yaxis(); ax.set_xlim(0, 11.5)
        ax.set_xlabel("覆蓋（層；不算遮擋）", fontsize=8.5); ax.legend(fontsize=8, loc="lower right")
    axs[0].set_title("b3：我們的路面主要在地下（4.5～8.4 層，93% 不透明）、正確高度只有 0.5～0.9 層；CityGS 地下 1.1 層。熵／scale 正則／補洞／法向都沒碰到它", fontsize=9.0)
    axs[1].set_title("b12：地下 7.3～10.5 層（拿掉熵最少）；法向把貼地躺平 58 → 88%", fontsize=9.0)
    save(fig, d, "n26_ground.png")


def n26_thick(d):
    """固定厚度（渲染端）：低角度透空、前方遮擋、空拍被改動的像素 vs 厚度。"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
    cs = {"CityGS": MUTED, "本機 1M": C1, "lab 2.5M": C2}
    for ax, j, lab in zip(axs, (1, 2, 3), ("低角度透空率 %", "前方遮擋 %（比 CityGS 表面近 20% 以上）", "空拍改動 >2/255 的像素 %")):
        for m, rows in N26_THICK.items():
            x = [r[0] for r in rows]; v = [r[j] for r in rows]
            ax.plot(x, v, "o-", color=cs[m], label=m, lw=1.6, ms=4)
            ax.text(x[-1], v[-1], f" {v[-1]:.1f}", fontsize=7, color=cs[m], va="center")
        ax.set_xscale("symlog", linthresh=2); ax.set_xticks([0, 2, 5, 10, 20, 50], ["0", "2", "5", "10", "20", "50"])
        ax.set_xlabel("固定厚度（cm）", fontsize=8.5); ax.set_title(lab, fontsize=9.2); ax.legend(fontsize=8)
    axs[1].axhline(27.37, color=C3, ls="--", lw=0.9); axs[1].text(0.2, 28.2, "法向每 8 步（厚度 0）27.4%", fontsize=7.4, color=C3)
    fig.suptitle("固定厚度（渲染端，不重訓）：10 cm 透空 5.84 → 5.23%，但空拍 11～16% 像素被改動、前方遮擋不降 => 沒有奇效；低角度最大的缺陷是巨大不透明面元擋在前面（35%）", fontsize=9.4)
    save(fig, d, "n26_thick.png")


def n26_mono(d):
    """單目看穿判準校準（CityGS 當地面在正確高度的對照）＋地下層從哪來。"""
    fig, axs = plt.subplots(1, 2, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]; ms = ("1%", "2%", "5%", "10%"); x = np.arange(len(ms)); w = 0.13
    cs = {"CityGS": MUTED, "本機 1M": C1, "法向每 8 步": C3}
    for k, (m, (deep, shal)) in enumerate(N26_MONO.items()):
        barlabels(ax, ax.bar(x + (k - 2.5) * w, deep, w, color=cs[m], label=f"{m} 比單目深"), "{:.1f}", size=6.5)
        barlabels(ax, ax.bar(x + (k + 0.5) * w, shal, w, color=cs[m], alpha=0.45, label=f"{m} 比單目淺"), "{:.1f}", size=6.5)
    ax.set_xticks(x, [f"門檻 {m}" for m in ms]); ax.set_ylabel("像素 %（三個模型都實心、單目有效）"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("單目誤差約 ±10%：連 CityGS 都有 26% 像素「比單目深 >2%」；我們只多 5～8 個百分點 => 1～2% 的看穿判準大多是雜訊", fontsize=9.0)
    ax = axs[1]; R = N26_UNDER; y = np.arange(len(R)); h = 0.38
    barlabels_h = lambda bars, vals: [ax.text(v + 0.3, b_.get_y() + b_.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7, color=INK2) for b_, v in zip(bars, vals)]
    barlabels_h(ax.barh(y - h / 2, [r[1] for r in R], h, color=C2, label="低於地面 3 m"), [r[1] for r in R])
    barlabels_h(ax.barh(y + h / 2, [r[2] for r in R], h, color=C1, alpha=0.6, label="低於地面 0.5 m"), [r[2] for r in R])
    ax.set_yticks(y, [r[0] for r in R], fontsize=8.3); ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("b3 地面格上的點 %（相對該格 SfM 中位高度）", fontsize=8.5)
    ax.set_title("SfM 是對的（±1 m）；初始化 3.8% => 訓練後 10%：是訓練推下去的", fontsize=9.0)
    save(fig, d, "n26_mono.png")


# ══════════════ 27／28／29（11-05 本機 b3 1M、lab 計時跑次）══════════════
N27_ARMS = [  # 臂: (PSNR, LPIPS, 不透明離 SfM >5 m %, 地下覆蓋 層, 海面上方浮塊, 低角度破洞 %)
    ("組合臂", 11.662, .7667, 10.7, 2.32, 121, 2.6), ("只追蹤（重跑）", 11.663, .7670, 10.7, 2.27, 114, 1.7),
    ("只鎖", 11.666, .7693, 8.8, 1.88, 80, 2.6), ("★ 只限增生", 11.669, .7694, 4.0, 1.04, 35, 1.9),
    ("全開", 11.662, .7712, 3.4, 0.81, 18, 1.5), ("全開、增生 ×0.1", 11.667, .7699, 4.4, 1.07, 19, 1.5),
    ("CityGS", np.nan, np.nan, 9.7, 1.08, 10, 0.0)]
N27_FATE = [  # step: (存活, trim, 搬家, 補洞) 錨點 229,695
    (1500, 211314, 17906, 206, 269), (3000, 157764, 62164, 8212, 1555), (4500, 107575, 101004, 18821, 2248),
    (6000, 72077, 128826, 25843, 2841), (9000, 40913, 154467, 30629, 3447), (12000, 24728, 168817, 32101, 3748),
    (15000, 15621, 177841, 32122, 3791), (30000, 14500, 178887, 32122, 3791)]
N27_TIME = [  # step: (離 SfM >5 m %, 地下覆蓋 層, 海面上方浮塊, 不透明 千顆)
    (3000, 11.6, 3.08, 73, 362), (6000, 15.7, 3.68, 187, 423), (9000, 15.4, 3.81, 199, 508), (12000, 14.3, 4.25, 336, 705),
    (15000, 13.8, 4.35, 409, 716), (20000, 11.7, 2.78, 221, 584), (30000, 10.7, 2.27, 114, 517)]
N27_TOWER = {  # 棟: 各臂 100～150 m 對 CityGS 表面的破洞 %（組合臂, 只鎖, 只限增生, 全開）
    "棟 1（246 m）": (15, 25, 25, 34), "棟 2（340 m）": (23, 31, 22, 28), "棟 3 玻璃塔（347 m）": (11, 13, 10, 11),
    "棟 4（352 m）": (15, 21, 23, 29), "棟 5（221 m）": (21, 33, 23, 30), "棟 6（209 m）": (10, 14, 26, 29)}


def n27_arms(d):
    """SfM 五臂：幾何（地下覆蓋、離 SfM 遠、海面浮塊、低角度破洞）與畫質。"""
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8))
    R = N27_ARMS; y = np.arange(len(R)); names = [r[0] for r in R]
    cols = [MUTED if r[0] == "CityGS" else (C3 if r[0].startswith("★") else C1) for r in R]
    for ax, j, lab, fmt in zip(axs, (4, 3, 5, 6), ("地下覆蓋（層）", "不透明顆粒離可靠 SfM > 5 m（%）", "海面上方浮塊（顆）", "低角度玻璃塔破洞（%）"),
                               ("{:.2f}", "{:.1f}%", "{:.0f}", "{:.1f}%")):
        v = [r[j] for r in R]
        ax.barh(y, v, color=cols)
        for i, x in enumerate(v):
            ax.text(x, i, " " + fmt.format(x), va="center", fontsize=7.4, color=INK2)
        ax.set_yticks(y, names if ax is axs[0] else [""] * len(R), fontsize=8.3); ax.invert_yaxis(); ax.set_title(lab, fontsize=9.2)
    fig.suptitle("SfM 五臂（本機 b3 1M、底＝組合臂）：只限增生把地下覆蓋 2.32 → 1.04 層（CityGS 1.08）、離 SfM 遠 10.7 → 4.0%；"
                 "held-out PSNR 11.662～11.669 持平、LPIPS .7667 → .7694～.7712（小幅變差）", fontsize=9.6)
    save(fig, d, "n27_arms.png")


def n27_fate(d):
    """錨點（與可靠 SfM 重合的初始顆粒）去向＋時間軸。"""
    fig, axs = plt.subplots(1, 2, figsize=(18, 4.8))
    ax = axs[0]; F = N27_FATE; x = [r[0] / 1000 for r in F]
    ax.stackplot(x, [r[1] for r in F], [r[2] for r in F], [r[3] for r in F], [r[4] for r in F],
                 labels=["存活", "trim 剪掉", "死掉被搬家", "補洞拿去"], colors=[C3, C2, C4, MUTED], alpha=0.85)
    ax.set_xlabel("step（千）"); ax.set_ylabel("顆（共 229,695）"); ax.legend(fontsize=8, loc="center right")
    ax.set_title("SfM 點是被 trim 剪掉（77.9%），不是漂走：活到最後 6.3%、位移中位 1.47 m；鎖住的話 53% 變近乎透明", fontsize=9.2)
    ax = axs[1]; T = N27_TIME; x = [r[0] / 1000 for r in T]
    ax.plot(x, [r[2] for r in T], "o-", color=C2, label="地下覆蓋（層）"); ax.set_ylabel("地下覆蓋（層）", color=C2)
    ax2 = ax.twinx(); ax2.plot(x, [r[1] for r in T], "s--", color=C1, label="離 SfM > 5 m 的不透明 %"); ax2.set_ylabel("離 SfM > 5 m（%）", color=C1)
    ax2.grid(False)
    for xi, r in zip(x, T):
        ax.text(xi, r[2] + 0.06, f"{r[2]:.2f}", fontsize=7, color=C2, ha="center")
    ax.axvline(15, color=MUTED, ls=":", lw=0.9); ax.text(15.2, 2.4, "增生結束", fontsize=7.5, color=MUTED)
    ax.set_xlabel("step（千）"); ax.set_title("地下層與離 SfM 遠的顆粒都在增生期長出來、收割期才減 => 要擋就擋在增生", fontsize=9.2)
    save(fig, d, "n27_fate.png")


def n27_tower(d):
    """高樓 100～150 m 對 CityGS 表面的破洞：SfM 限制的代價在一般高樓，玻璃塔沒變。"""
    fig, ax = plt.subplots(figsize=(18, 4.2))
    ks = list(N27_TOWER); x = np.arange(len(ks)); w = 0.2
    for k, (lab, c) in enumerate(zip(("組合臂", "只鎖", "只限增生", "全開"), (C1, C4, C3, C2))):
        barlabels(ax, ax.bar(x + (k - 1.5) * w, [N27_TOWER[b][k] for b in ks], w, color=c, label=lab), "{:.0f}", size=7)
    ax.set_xticks(x, ks); ax.set_ylabel("100～150 m 破洞 %"); ax.legend(fontsize=8, ncol=4)
    ax.set_title("代價：SfM 很稀的一般高樓上半部變薄（全開 10～23% → 28～34%）；玻璃塔（棟 3）各臂都沒變", fontsize=9.4)
    save(fig, d, "n27_tower.png")


N28_STEP = [("backward", 19.66), ("forward", 11.92), ("迴圈外（載入／Lightning）", 2.61), ("loss（L1＋融合 SSIM）", 1.81),
            ("Adam（融合）", 1.05), ("log_metrics", 0.81), ("其他 hook／scheduler", 0.85), ("MCMC 事件", 0.45), ("週期 trim", 0.05)]
N28_FLAGS = [  # 設定: 一般步 %, dist 步 %, 增生期（累加 V/C）%
    ("零梯度略過", -1.7, 0.2, np.nan), ("＋C 區 forward 略過", -2.4, -3.4, np.nan), ("V/C warp 先加總", np.nan, np.nan, -8.5),
    ("V/C warp＋C 區", np.nan, np.nan, -10.9), ("配對 bit", -2.7, np.nan, np.nan), ("配對 bit＋C 區", -5.0, np.nan, np.nan)]


def n28_steps(d):
    """lab 3090 現行配方每步逐段（b3 2.25M）。"""
    fig, ax = plt.subplots(figsize=(18, 4.2))
    R = N28_STEP; y = np.arange(len(R))
    ax.barh(y, [r[1] for r in R], color=[C2, C2, MUTED, C1, C3, MUTED, MUTED, C4, C4])
    for i, r in enumerate(R):
        ax.text(r[1], i, f" {r[1]:.2f} ms（{100 * r[1] / 39.25:.1f}%）", va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(y, [r[0] for r in R], fontsize=8.5); ax.invert_yaxis(); ax.set_xlabel("ms／步（一步 39.25 ms＝25.5 it/s；30k＝19.6 分）")
    ax.set_title("lab 3090 現行配方（A＋dist4、b3 2.25M）：光柵化 80%、CPU 側約 11%（GPU 空等）、Adam 1.05 ms 已在頻寬極限", fontsize=9.4)
    save(fig, d, "n28_steps.png")


def n28_flags(d):
    """本機 b3 組合臂 ckpt：各旗標每步光柵化＋loss 時間變化。"""
    fig, ax = plt.subplots(figsize=(18, 4.2))
    R = N28_FLAGS; x = np.arange(len(R)); w = 0.26
    for k, (lab, c) in enumerate(zip(("一般步", "dist 步（每 4 步一次）", "增生期（累加 V/C 開）"), (C1, C4, C3))):
        v = [r[1 + k] for r in R]
        bars = ax.bar(x + (k - 1) * w, v, w, color=c, label=lab)
        for b_, vv in zip(bars, v):
            if not np.isnan(vv):
                ax.text(b_.get_x() + b_.get_width() / 2, vv, f"{vv:+.1f}%", ha="center", va="top" if vv < 0 else "bottom", fontsize=7.4, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x, [r[0] for r in R], fontsize=8.5); ax.set_ylabel("每步時間變化 %"); ax.legend(fontsize=8)
    ax.set_title("無損：V/C warp 先加總 −8.5%（forward −17.6%）、零梯度略過 −1.7%；C 區 forward 略過、配對 bit 會改梯度 => 待品質", fontsize=9.4)
    save(fig, d, "n28_flags.png")


def n28_sync(d):
    """tile 同步損失（稽核 kernel 的逐像素迴圈次數當代理）＋C 區代理 vs 實測。"""
    fig, axs = plt.subplots(1, 2, figsize=(18, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    parts = [("真的要跑", 0.755, C3), ("warp 內閒置（SIMT）", 0.155, MUTED), ("整塊同步損失", 0.089, C2)]
    left = 0
    for lab, v, c in parts:
        ax.barh([0], [v], left=left, color=c, label=lab); ax.text(left + v / 2, 0, f"{lab}\n{v:.3f}", ha="center", va="center", fontsize=8, color="white")
        left += v
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("以「整塊一起走」的工作量為 1（24 個訓練視角）")
    ax.set_title("改成 warp 各走各的，上限只省 8.9%（逐視角 6～12%）；逐像素迴圈只有 27% 真的混合", fontsize=9.2)
    ax = axs[1]
    barlabels(ax, ax.bar([0, 1], [8.7, 2.4], color=[MUTED, C1]), "{:.1f}%", size=8)
    ax.set_xticks([0, 1], ["工作量代理估的", "實測（一般步）"]); ax.set_ylabel("C 區略過省下 %")
    ax.set_title("C 區略過：代理估 8.7%、實測 2.4% => C 區的 tile 本來就便宜", fontsize=9.2)
    save(fig, d, "n28_sync.png")


N29_W = {  # 塊: (探針權重, SfM 權重)
    0: (0.902, 0.619), 1: (1.043, 1.196), 2: (1.324, 1.146), 3: (1.022, 1.041), 4: (1.329, 1.530), 5: (0.828, 0.832), 6: (0.962, 0.644),
    7: (0.831, 1.008), 8: (0.922, 1.064), 9: (0.801, 0.907), 10: (1.035, 0.928), 11: (0.866, 1.145), 12: (0.686, 0.682),
    13: (1.336, 1.061), 14: (1.328, 1.331), 15: (0.786, 0.866)}
N29_PROBE = [(3, 471, 281, 129), (6, 551, 357, 177), (9, 663, 429, 194), (12, 831, 555, 230), (15, 900, 629, 226), (20, 760, 554, 207), (30, 658, 471, 180)]
N29_N = [(3, 0.82), (6, 1.00), (9, 0.97), (12, 0.93), (15, 0.90), (20, 0.74), (30, 0.67)]   # b3 時間軸實際 N（百萬）
N29_HOLD = {"b3": (1.47, 0.63), "b12": (3.02, 0.73)}   # 留 16 台：從 >=3 掉到 <3 的點 %（隨機, 挑安全相機）


def n29_pred(d):
    """訓練前訊號 vs 10-31 探針權重。"""
    fig, axs = plt.subplots(1, 2, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]; xs = [v[1] for v in N29_W.values()]; ys = [v[0] for v in N29_W.values()]
    ax.scatter(xs, ys, color=C1, s=28)
    for b, (p, s) in N29_W.items():
        ax.text(s, p, f" b{b}", fontsize=7.5, color=INK2)
    ax.plot([0.55, 1.6], [0.55, 1.6], ls=":", color=MUTED, lw=0.9)
    ax.set_xlabel("SfM 權重（塊內可靠 SfM 點數 / 平均；訓練前就有）"); ax.set_ylabel("10-31 探針權重（要先跑 2.5M 全場景）")
    ax.set_title("相關 0.69（Spearman 0.68）；b3／b12 幾乎一樣（1.04／0.68 vs 1.02／0.69）", fontsize=9.2)
    ax = axs[1]
    R = [("可靠 SfM 點數", 0.69, 0.68), ("全部 SfM 點數", 0.71, 0.67), ("塊內視野和", 0.31, 0.26), ("相機數", 0.15, 0.09), ("高度差", 0.14, 0.11), ("平均 track", -0.22, -0.17)]
    x = np.arange(len(R)); w = 0.38
    barlabels(ax, ax.bar(x - w / 2, [r[1] for r in R], w, color=C1, label="Pearson"), "{:+.2f}", size=7.2)
    barlabels(ax, ax.bar(x + w / 2, [r[2] for r in R], w, color=C4, label="Spearman"), "{:+.2f}", size=7.2)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x, [r[0] for r in R], fontsize=8.3); ax.legend(fontsize=8)
    ax.set_title("訓練前的逐塊訊號與探針權重的相關", fontsize=9.2)
    save(fig, d, "n29_pred.png")


def n29_price(d):
    """定價自決 v1：離線各 λ 給的容量 vs 實際 N；訓練結果。"""
    fig, axs = plt.subplots(1, 2, figsize=(18, 4.6))
    ax = axs[0]; x = [r[0] for r in N29_PROBE]
    for k, (lab, c) in enumerate(zip(("λ 0.0021", "λ 0.00415（v1 用的）", "λ 0.0083"), (C3, C2, C4))):
        ax.plot(x, [r[1 + k] / 1000 for r in N29_PROBE], "o-", color=c, label=lab)
    ax.plot([r[0] for r in N29_N], [r[1] for r in N29_N], "s--", color=MUTED, label="組合臂實際 N")
    ax.set_xlabel("step（千）"); ax.set_ylabel("百萬顆"); ax.legend(fontsize=8)
    ax.set_title("V/C 早期偏低 => 固定 λ 在 3k 只給 0.28M；各時點變動 8～29%、沒有收斂點", fontsize=9.2)
    ax = axs[1]
    R = [("b3 組合臂", 0.673, .7667), ("b3 價格 v1", 0.151, .7858), ("b3 價格＋提早結束", 0.150, .7867), ("b12 組合臂", 0.665, .8078), ("b12 價格＋提早結束", 0.184, .8247)]
    y = np.arange(len(R))
    ax.barh(y, [r[1] for r in R], color=[C1, C2, C2, C1, C2])
    for i, r in enumerate(R):
        ax.text(r[1], i, f" {r[1]:.2f}M｜LPIPS {r[2]:.4f}", va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(y, [r[0] for r in R], fontsize=8.5); ax.invert_yaxis(); ax.set_xlim(0, 0.95); ax.set_xlabel("終點顆數（百萬）")
    ax.set_title("v1 失敗：顆數壓垮到 0.15～0.18M、LPIPS +0.017～0.019（被剪的不算合格 => cap 只降不升）", fontsize=9.2)
    save(fig, d, "n29_price.png")


def n29_holdout(d):
    """採樣控制器：留出 16 台驗證視角的觀測損失（SfM 觀測紀錄模擬）。"""
    fig, ax = plt.subplots(figsize=(18, 3.8))
    x = np.arange(2); w = 0.36
    barlabels(ax, ax.bar(x - w / 2, [N29_HOLD[b][0] for b in ("b3", "b12")], w, color=MUTED, label="隨機挑"), "{:.2f}%", size=8)
    barlabels(ax, ax.bar(x + w / 2, [N29_HOLD[b][1] for b in ("b3", "b12")], w, color=C3, label="挑安全相機（看到的點別人也看得到）"), "{:.2f}%", size=8)
    ax.set_xticks(x, ["b3（899 台）", "b12（516 台）"]); ax.set_ylabel("SfM 點從 ≥3 台觀測掉到 <3 台（%）"); ax.legend(fontsize=8)
    ax.set_title("留出驗證視角的代價：每點中位只被 5 台塊內相機看到；挑安全相機把損失降到 1/2～1/4，控制期結束（15k）後再放回訓練", fontsize=9.2)
    save(fig, d, "n29_holdout.png")


N25_ARMS = [  # 原 25 §1（lab 2.5M、對 A＋dist4）：臂, 換塊 Δ (b3, b12), 海上浮塊 千顆 (b3, b12), 低角度透空 % (b3, b12), 時間 % (b3, b12)
    ("基準 A＋dist4", (0, 0), (51.6, 126), (5.2, 3.3), (0, 0)),
    ("閘門 平均、每步", (0.03, 0.02), (6.4, 5.7), (4.8, 4.3), (16, 14)),
    ("閘門 投票、每步", (0.03, 0.04), (6.3, 4.9), (4.8, 3.5), (12, 14)),
    ("★ 閘門 投票、每 8 步", (0.12, 0.01), (8.7, 7.9), (4.5, 2.6), (0, 4)),
    ("閘門＋MCMC 補洞放點", (-0.06, 0.05), (6.6, 8.6), (3.7, 3.0), (18, 14)),
    ("補洞＋遮擋回收", (0.08, 0.05), (6.8, 10.7), (4.5, 3.6), (6, 15)),
    ("法向每 8 步", (-0.02, 0.13), (68.9, 136), (2.7, 4.2), (0, 1)),
    ("★ 法向 8＋投票 8＋補洞", (0.07, 0.18), (9.5, 13.0), (2.5, 6.1), (19, 14)),
    ("合併後刪除 前半", (0.05, 0.03), (41.2, 108), (5.4, 4.2), (-1, -1)),
    ("合併後刪除 後半", (0.08, -0.04), (30.2, 96.1), (4.2, 4.4), (-7, -3)),
    ("★ 合併後刪除 全程", (0.02, -0.01), (26.0, 82.1), (4.4, 4.1), (-4, -2)),
    ("合併後刪除 後半更積極", (0.03, -0.03), (28.4, 88.9), (4.5, 4.8), (-8, -4)),
    ("合併優先的 trim", (0.06, 0.01), (46.5, 112), (3.9, 4.5), (3, 2)),
    ("★ 增生方向限制", (-0.05, -0.02), (58.3, 130), (4.9, 2.1), (-2, -3)),
]


def n25_arms(d):
    """lab 11-02～03 批次（2.5M、A＋dist4 底）：換塊、海上浮塊、低角度透空、時間（原 25 §1）。"""
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.6))
    R = N25_ARMS; y = np.arange(len(R)); h = 0.38
    spec = [(1, "換塊 Δ（本塊幾何區域 PSNR，對 A＋dist4；雜訊 ±0.03）", "{:+.2f}"), (2, "海上浮塊（千顆）", "{:.1f}"),
            (3, "低角度透空（%）", "{:.1f}"), (4, "訓練時間 Δ%", "{:+.0f}%")]
    for ax, (j, lab, fmt) in zip(axs, spec):
        for k, c in enumerate((C1, C2)):
            v = [r[j][k] for r in R]
            ax.barh(y + (k - 0.5) * h, v, h, color=c, label="b3" if k == 0 else "b12")
            for i, x in enumerate(v):
                ax.text(x, i + (k - 0.5) * h, " " + fmt.format(x), va="center", fontsize=6.6, color=INK2, ha="left" if x >= 0 else "right")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [r[0] for r in R] if ax is axs[0] else [""] * len(R), fontsize=8.2); ax.invert_yaxis()
        ax.set_title(lab, fontsize=9.0); ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("lab 11-02～03（2.5M、對 A＋dist4）：閘門投票每 8 步浮塊 −83～94%、時間 +0～4%（最划算）；法向 8＋投票 8＋補洞換塊最好但 +14～19%；合併後刪除全程 N −21～24% 品質平", fontsize=9.6)
    save(fig, d, "n25_arms.png")


DOCS = {
    "01_訓練時間組成":["n01_steps", "n01_scaling", "n01_maxsplit", "n01_blocktime"],
    "02_VRAM與配置器": ["n02_segments", "n02_value", "n02_peak3"],
    "03_儲存與RAM": ["fig4_ckpt_storage", "fig6_run_dir", "fig5_ram_cache", "n03_official"],
    "04_外接盒與binning成本": ["cc1_quality_delta", "n04_time", "n04_load", "cc4_proxy_decoupled", "n04_load3"],
    "05_成本品質前緣": ["n05_main4x4", "n05_official"],
    "08_初始化": ["n08_init20k", "f4_init60k", "n08_init60k"],
    "參考_CityGS參考線": ["n09_full44_merged", "n09_full44_blocks", "n09_merged_blocks", "n09_stepprof", "n09_evalgap", "n09_official_blocks", "n09_trim", "n06_prune3", "n09_blockf1"],
    "10_速度優化_lean與kernel拆解": ["n10_kernels"],
    "12_新年代重測_幾何floater失敗區": ["n12_recheck", "n12_heldout", "n12_fail3"],
    "13_trim判準VC與opacity二元熵": ["n13_race", "n13_swap"],
    "15_現行最佳best1_組成與取捨": ["n15_time", "n15_tradeoff", "n09_blockf1"],
    "16_步數與SH_輕量化": ["n16_steps", "n16_sh", "n16_official"],
    "17_塊外遮罩訓練": ["n17_mask"],
    "18_馬路地底面元與幾何正則": ["n18_road"],
    "19_全因子消融_遮罩dist_cap步數": ["n19_fact", "n19_road"],
    "20_全場景遮罩回退與opacity": ["n20_blocks", "n20_swap"],
    "21_配方A全場景_加速cap與撒點": ["n21_full", "n21_blocks", "n21_single"],
    "22_新視角缺陷_海面浮塊立面毛邊與幾何閘門": ["n22_b12hist", "n22_b0var", "n22_orient", "n22_road_roof"],
    "23_cap劑量熵拆開塔頂與幾何約束臂": ["n23_dose", "n23_arms", "n23_tower"],
    "24_法向SfM豁免鄰域平面合併與單目深度診斷": ["n24_swap", "n24_geom", "n24_mono", "n24_full"],
    "26_本機1M_小圓片拆解地下層厚度與法向": ["n26_arms", "n26_ground", "n26_thick", "n26_mono"],
    "27_b3海面高樓SfM距離與組合臂": ["n27_arms", "n27_fate", "n27_tower"],
    "28_時間組成硬體上限與光柵器加速旗標": ["n28_steps", "n28_flags", "n28_sync"],
    "29_每塊顆數_探針分配SfM分配定價自決與採樣控制": ["n29_pred", "n29_price", "n29_holdout"],
    "30_現行前緣": ["n21_full", "n24_full", "n23_dose"],
    "25_lab1102批次_閘門補洞合併增生方向與拆項": ["n25_arms"],
    "31_地下層_路面在地底": ["n18_road", "n26_ground", "n26_mono", "n27_fate", "n27_arms"],
    "32_海面浮塊": ["n22_b12hist", "n22_b0var", "n24_geom", "n25_arms", "n27_arms"],
    "33_高樓_塔頂玻璃塔與上半部": ["n23_tower", "n24_mono", "n27_tower", "n22_orient"],
    "34_下一版配方候選元件": ["n25_arms", "n24_swap", "n26_arms", "n27_arms"],
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
    for f in (n02_current, n05_cap60k, n07_sampling, n08_init20k, n08_init60k, n09_official_blocks, n09_full44_merged, n09_full44_blocks, n09_merged_blocks, n09_stepprof, n09_evalgap, n05_arms60k, n09_trim,
              n01_steps, n01_scaling, n01_maxsplit, n02_segments, n02_value, n05_main, n05_official, n05_frontier60k, n04_time, n04_load,
              n10_lean_loop, n10_kernels, n11_tiles, n12_recheck, n12_heldout, n13_struct, n13_race, n13_tiny, n13_swap,
              n01_blocktime, n02_peak3, n03_official, n04_load3, n05_main4x4, n06_prune3, n12_fail3, n09_blockf1, n15_time, n15_tradeoff, n16_steps, n16_sh, n16_official, n17_mask, n18_road, n19_fact, n19_road, n20_blocks, n20_swap, n21_full, n21_blocks, n21_single, n22_b12hist, n22_b0var, n22_orient, n22_road_roof,
              n23_dose, n23_arms, n23_tower, n24_swap, n24_geom, n24_mono, n24_full, n26_arms, n26_ground, n26_thick, n26_mono,
              n27_arms, n27_fate, n27_tower, n28_steps, n28_flags, n28_sync, n29_pred, n29_price, n29_holdout, n25_arms):
        f(tmp)
    for doc, panels in DOCS.items():
        paths = [os.path.join(tmp, p + ".png") for p in panels]
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            raise SystemExit(f"⛔ {doc} 缺面板：{miss}")
        out = doc_dir(doc)
        compose(paths, os.path.join(out, doc + ".png"))
        print(f"  -> {os.path.relpath(out, ROOT)}/{doc}.png（{len(panels)} 個面板）")


if __name__ == "__main__":
    if any(x in ("-h", "--help") for x in sys.argv[1:]):
        print(__doc__)
    else:
        main()
