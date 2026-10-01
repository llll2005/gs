"""2026-10-01 成本-品質前緣與相關比較圖 -> 紀錄/new_figures_costfrontier/

數字全部來自 lab 的離線量測與訓練紀錄（同工具、同相機；來源寫在 README.md）。
配色：參考調色盤前 3 個類別色（藍/橙/青綠，all-pairs 驗證通過）；參考基準用中性灰虛線，不佔類別色。
青綠對背景對比偏低 => 每個系列都直接標文字，README 另附數據表。
"""
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as _fm

try:
    _ff = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK TC"],
                         capture_output=True, text=True).stdout.strip()
    _fm.fontManager.addfont(_ff); _fn = _fm.FontProperties(fname=_ff).get_name()
except Exception as _e:
    _fn = "DejaVu Sans"; print("⚠⚠ 找不到中文字型：", _e)
plt.rcParams.update({"font.sans-serif": [_fn, "DejaVu Sans"], "axes.unicode_minus": False,
                     "figure.dpi": 150, "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#898781", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # 類別色 1~3
MUTED, INK2 = "#898781", "#52514e"
OUT = "紀錄/new_figures_costfrontier"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  ->", p)


def lab(ax, x, y, s, dx=4, dy=4, ha="left", color=INK2, size=7.5):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=size, color=color)


# ── 圖 1：22k 同成本比較（b6，精確 Σtiles 中位）─────────────────────────────
def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    # conic 關
    ax = axs[0]
    cap = [(2.989, 28.035, "cap 0.7M"), (3.929, 28.660, "cap 1.2M"), (4.976, 28.953, "cap 1.7M"), (5.089, 29.046, "基準 cap 2.6M")]
    bud = [(1.736, 26.578, "cb25", "o"), (2.194, 27.236, "cb25+成本取樣", "^"), (2.537, 27.661, "cb50", "o"), (3.719, 28.443, "cb50+成本取樣", "^")]
    vpc = [(2.742, 29.005, "v/c trim"), (3.745, 28.991, "v/c trim α=0.5")]
    x, y = zip(*[(a, b) for a, b, _ in cap])
    ax.plot(x, y, "-o", color=C1, lw=2, ms=6, label="只調 cap（顆數上限）", zorder=3)
    for a, b, s in cap:
        lab(ax, a, b, s, dx=(-6 if "基準" in s else 5), dy=(7 if "基準" in s else -11), ha=("right" if "基準" in s else "left"))
    for a, b, s, m in bud:
        ax.plot(a, b, m, color=C2, ms=8, mec="#fcfcfb", mew=1.5, zorder=4)
        lab(ax, a, b, s, dx=6, dy=-3)
    ax.plot([], [], "o", color=C2, ms=8, label="預算閘門（代理單位）")
    for a, b, s in vpc:
        ax.plot(a, b, "*", color=C3, ms=14, mec="#fcfcfb", mew=1.2, zorder=5)
        lab(ax, a, b, s, dx=(-8 if "0.5" in s else 6), dy=(-14 if "0.5" in s else 5), ha=("right" if "0.5" in s else "left"), color=INK2)
    ax.plot([], [], "*", color=C3, ms=12, label="v/c trim（價值÷成本剪枝）")
    ax.set_title("conic 關（舊外接盒）", fontsize=10.5)
    ax.set_xlabel("渲染成本：精確 Load 中位（百萬 tile-顆對／視角，對數軸）")
    ax.set_ylabel("val PSNR（dB，21,920 步）")
    ax.legend(fontsize=8, loc="lower right")
    # conic 開
    ax = axs[1]
    capc = [(1.761, 28.09, "cap 0.7M"), (2.273, 28.68, "cap 1.2M"), (2.980, 28.99, "cap 1.7M"), (3.047, 29.05, "基準 cap 2.6M")]
    budx = [(1.593, 27.64, "cb50x", "o"), (1.665, 27.81, "cb50x+成本取樣", "^"), (1.015, 26.53, "cb25x", "o"), (1.040, 26.66, "cb25x+成本取樣", "^")]
    x, y = zip(*[(a, b) for a, b, _ in capc])
    ax.plot(x, y, "-o", color=C1, lw=2, ms=6, label="只調 cap（顆數上限）", zorder=3)
    for a, b, s in capc:
        lab(ax, a, b, s, dx=(-6 if "基準" in s else 5), dy=(7 if "基準" in s else -11), ha=("right" if "基準" in s else "left"))
    for a, b, s, m in budx:
        ax.plot(a, b, m, color=C2, ms=8, mec="#fcfcfb", mew=1.5, zorder=4)
        lab(ax, a, b, s, dx=6, dy=-3 if "成本" not in s else 4)
    ax.plot([], [], "o", color=C2, ms=8, label="預算閘門（精確單位）")
    ax.set_title("conic 開（現行預設）", fontsize=10.5)
    ax.set_xlabel("渲染成本：精確 Load 中位（百萬，對數軸）")
    ax.legend(fontsize=8, loc="lower right")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks([1, 1.5, 2, 3, 4, 5]); ax.set_xticklabels(["1", "1.5", "2", "3", "4", "5"])
        ax.set_ylim(26.2, 29.4)
    fig.suptitle("圖 1　同成本比較（b6，22k）：左上方＝同成本下品質更好。預算閘門都在 cap 線下方；v/c trim 在線的左上方", fontsize=11)
    save(fig, "f1_frontier_22k.png")


# ── 圖 2：60k（v/c trim 與 init）──────────────────────────────────────────────
def fig2():
    data = {
        "b6": dict(base=(4.729, 30.45), vpc=(2.391, 30.19), conic=(2.773, 30.48), dup4=(2.616, 30.75)),
        "b13": dict(base=(6.123, 29.86), vpc=(2.955, 29.32), conic=(3.337, 30.00), dup4=(3.059, 30.32)),
    }
    slope = 2.31          # 22k conic 關 cap 曲線 cap07->cap12 的斜率（dB / ln Load）
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    for ax, (blk, d) in zip(axs, data.items()):
        (bx, by), (vx, vy) = d["base"], d["vpc"]
        xs = np.linspace(vx * 0.9, bx * 1.02, 50)
        ax.plot(xs, by + slope * np.log(xs / bx), "--", color=MUTED, lw=1.3, zorder=1)
        lab(ax, xs[3], by + slope * np.log(xs[3] / bx), "只調 cap 的推估線（22k 斜率）", dx=4, dy=-12, color=MUTED)
        ax.plot([bx, vx], [by, vy], "-", color=C3, lw=1.2, alpha=0.6, zorder=2)
        ax.plot(bx, by, "o", color=C1, ms=9, zorder=4); lab(ax, bx, by, f"speed3（conic 關）\n{by:.2f}", dx=-6, dy=6, ha="right")
        ax.plot(vx, vy, "*", color=C3, ms=15, mec="#fcfcfb", mew=1.2, zorder=5); lab(ax, vx, vy, f"+v/c trim\n{vy:.2f}（Load {100 * (vx / bx - 1):+.0f}%）", dx=6, dy=-18)
        (cx, cy), (dx_, dy_) = d["conic"], d["dup4"]
        ax.plot([cx, dx_], [cy, dy_], "-", color=C2, lw=1.2, alpha=0.6, zorder=2)
        ax.plot(cx, cy, "s", color=C2, ms=8, mec="#fcfcfb", mew=1.2, zorder=4); lab(ax, cx, cy, f"conic 開 預設 init {cy:.2f}", dx=6, dy=-10)
        ax.plot(dx_, dy_, "D", color=C2, ms=8, mec="#fcfcfb", mew=1.2, zorder=4); lab(ax, dx_, dy_, f"conic 開 dup4 {dy_:.2f}", dx=6, dy=4)
        ax.set_title(f"{blk}（60k，N 都是 2.34M）", fontsize=10.5)
        ax.set_xlabel("渲染成本：精確 Load 中位（百萬）")
        ax.set_ylabel("val PSNR（dB，60,000 步）")
        ax.set_ylim(min(vy, by + slope * np.log(vx * 0.9 / bx)) - 0.3, max(dy_, by) + 0.45)
    axs[1].plot([], [], "o", color=C1, ms=8, label="conic 關 基準（speed3）")
    axs[1].plot([], [], "*", color=C3, ms=12, label="conic 關 + v/c trim")
    axs[1].plot([], [], "s", color=C2, ms=8, label="conic 開：預設 init → dup4")
    axs[1].plot([], [], "--", color=MUTED, label="只調 cap 的推估（22k 斜率，非 60k 實測）")
    axs[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("圖 2　60k：v/c trim 省一半渲染成本只付 0.26~0.54 dB（調 cap 推估要付 ~1.6 dB）；dup4 同 N 品質更好且更便宜", fontsize=11)
    save(fig, "f2_60k_vpc_init.png")


# ── 圖 3：逐 tile 前 K 名（事後剪枝，b6 60k conic 開）────────────────────────
def fig3():
    tk = [(0.23, 17.925, 1), (0.32, 19.356, 2), (0.45, 20.963, 4), (0.65, 22.825, 8), (0.92, 24.945, 16), (1.37, 27.231, 32), (1.92, 29.165, 64)]
    op = [(0.51, 18.139), (0.75, 19.522), (0.99, 21.025), (1.18, 23.124), (1.35, 25.218), (1.60, 27.200), (1.93, 28.834)]
    mc = [(1.28, 19.195), (1.42, 20.218), (1.62, 21.544), (1.83, 23.219), (2.05, 25.189), (2.31, 27.243), (2.56, 28.970)]
    rd = [(0.24, 15.460), (0.36, 16.936), (0.52, 18.468), (0.73, 20.103), (1.05, 22.001), (1.48, 24.210), (1.94, 26.543)]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(*zip(*rd), "--", color=MUTED, lw=1.4, label="隨機（參考基準）")
    ax.plot(*zip(*mc), "-^", color=C3, lw=2, ms=6, label="單視角最大貢獻（現行 trim 判準）")
    ax.plot(*zip(*op), "-s", color=C2, lw=2, ms=6, label="opacity 最高")
    ax.plot([p[0] for p in tk], [p[1] for p in tk], "-o", color=C1, lw=2, ms=7, label="逐 tile 前 K 名", zorder=4)
    for x, y, k in tk:
        lab(ax, x, y, f"K={k}", dx=-6, dy=5, ha="right", color=C1 if False else INK2)
    ax.plot(2.94, 30.481, "*", color="#0b0b0b", ms=13, zorder=5); lab(ax, 2.94, 30.481, "不剪 30.48", dx=-6, dy=6, ha="right")
    lab(ax, rd[-1][0], rd[-1][1], "隨機", dx=5, dy=-4, color=MUTED)
    lab(ax, mc[-1][0], mc[-1][1], "單視角最大貢獻", dx=5, dy=-6)
    lab(ax, op[3][0], op[3][1], "opacity", dx=6, dy=-10)
    ax.set_xlabel("渲染成本：精確 Load 中位（百萬）")
    ax.set_ylabel("val PSNR（dB）")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("圖 3　事後剪枝（沒有重新訓練）：同成本下「逐 tile 前 K 名」最好\n每個 K 的四條線是同一個保留顆數 N；前 K 名的點落在最左邊＝同顆數下最便宜", fontsize=10.5)
    save(fig, "f3_tile_topk_posthoc.png")


# ── 圖 4：60k init 軌跡 ───────────────────────────────────────────────────────
def fig4():
    b6s = [10960, 21920, 32880, 43840, 54800, 60000]
    b13s = [13340, 26680, 40020, 53360, 60000]
    b6 = {"預設 SfM": [24.12, 26.66, 29.22, 30.08, 30.47, 30.48], "sfmfill": [24.09, 26.64, 29.21, 30.08, 30.51, 30.52],
          "dup4": [24.91, 26.89, 29.44, 30.33, 30.72, 30.75]}
    b13 = {"預設 SfM": [25.67, 27.61, 29.23, 29.90, 30.00], "sfmfill": [25.68, 27.60, 29.23, 29.90, 30.01],
           "dup4": [26.33, 27.87, 29.52, 30.19, 30.32]}
    nstep = {"預設 SfM": ([0, 1000, 5000, 10000, 14300, 20000, 60000], [0.579, 0.52, 0.84, 1.46, 2.34, 2.34, 2.34]),
             "dup4": ([0, 1000, 2300, 5000, 60000], [2.373, 2.13, 2.34, 2.34, 2.34])}
    col = {"預設 SfM": C1, "sfmfill": C2, "dup4": C3}
    mk = {"預設 SfM": "o", "sfmfill": "s", "dup4": "D"}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, st, d, t in ((axs[0], b6s, b6, "b6"), (axs[1], b13s, b13, "b13")):
        for k, v in d.items():
            ax.plot(st, v, "-" + mk[k], color=col[k], lw=2, ms=5, label=k)
            if k == "dup4":
                lab(ax, st[-1], v[-1], f"dup4 {v[-1]:.2f}", dx=-4, dy=6, ha="right")
        lab(ax, st[-1], d["預設 SfM"][-1], f"預設 {d['預設 SfM'][-1]:.2f} ≈ sfmfill {d['sfmfill'][-1]:.2f}", dx=-4, dy=-14, ha="right")
        ax.set_title(f"{t}：val PSNR（60k，N 終點都 2.34M）", fontsize=10); ax.set_xlabel("步數"); ax.set_ylabel("PSNR（dB）")
        ax.legend(fontsize=8, loc="lower right")
    ax = axs[2]
    for k, (st, v) in nstep.items():
        ax.plot(st, v, "-" + mk[k], color=col[k], lw=2, ms=5, label=k)
    ax.axvline(14300, color=C1, lw=0.8, ls=":"); lab(ax, 14300, 1.2, "預設在 14.3k 步長滿", dx=4, dy=0)
    ax.axvline(2300, color=C3, lw=0.8, ls=":"); lab(ax, 2300, 1.7, "dup4 在 2.3k 步長滿", dx=4, dy=0)
    ax.set_xlim(-500, 30000)
    ax.set_title("b6：顆數 N 隨步數（取樣點；b13 幾乎相同）", fontsize=10); ax.set_xlabel("步數"); ax.set_ylabel("N（百萬）")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("圖 4　init：dup4 早 1.2 萬步長滿，領先從 22k 起穩定在 +0.25~0.3、到 60k 不縮小（不只是時間平移）", fontsize=11)
    save(fig, "f4_init60k.png")


# ── 圖 5：官方模型的 opacity 剪枝曲線（held-out 741 幀）────────────────────
def fig5():
    kept = [100, 90, 75, 50, 25]
    n = [24.48, 22.03, 18.36, 12.24, 6.12]
    ps = [25.789, 25.789, 25.754, 24.404, 19.004]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(n, ps, "-o", color=C1, lw=2, ms=7)
    for a, b, k in zip(n, ps, kept):
        lab(ax, a, b, f"留 {k}%（{a:.1f}M）\n{b:.2f}", dx=6, dy=(8 if k == 100 else (-28 if k in (90, 75) else 4)))
    ax.set_ylim(18.5, 26.6)
    ax.set_xlabel("保留顆數 N（百萬）"); ax.set_ylabel("held-out PSNR（dB）")
    ax.set_title("圖 5　官方模型依 opacity 剪枝（官方 test 評分）：前 25%（≈610 萬顆）幾乎白付", fontsize=10.5)
    ax.invert_xaxis()
    save(fig, "f5_official_prune.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
