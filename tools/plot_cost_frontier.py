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
        lab(ax, a, b, s, dx=(-8 if s == "cb50x+成本取樣" else 6), dy=(-3 if "成本" not in s else 4), ha=("right" if s == "cb50x+成本取樣" else "left"))
    ax.plot([], [], "o", color=C2, ms=8, label="預算閘門（精確單位）")
    ax.set_title("conic 開（現行預設）", fontsize=10.5)
    ax.set_xlabel("渲染成本：精確 Load 中位（百萬，對數軸）")
    ax.legend(fontsize=8, loc="lower right")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks([1, 1.5, 2, 3, 4, 5]); ax.set_xticklabels(["1", "1.5", "2", "3", "4", "5"])
        ax.set_ylim(26.2, 29.4)
    fig.suptitle("圖 1　同成本比較（b6，22k）：每個點＝一個跑完的模型；藍線只是把不同 cap 的跑次連起來（不是訓練過程）\n"
                 "越左越便宜、越上品質越好。預算閘門的點都在藍線下方；v/c trim 在藍線的左上方", fontsize=10.5, y=0.99)
    fig.subplots_adjust(top=0.80)
    save(fig, "f1_frontier_22k.png")


# ── 圖 2：60k 的「前 → 後」對照（長條；每一列是一組對照，不是訓練過程）──────────
def fig2():
    slope = 2.31          # 22k conic 關 cap 曲線 cap07->cap12 的斜率（dB / ln Load）
    rows = [  # (標籤, 前 Load, 後 Load, 前 PSNR, 後 PSNR, 顏色, 是否推估)
        ("v/c trim　b6", 4.729, 2.391, 30.45, 30.19, C3, False),
        ("v/c trim　b13", 6.123, 2.955, 29.86, 29.32, C3, False),
        ("dup4 init　b6", 2.773, 2.616, 30.48, 30.75, C2, False),
        ("dup4 init　b13", 3.337, 3.059, 30.00, 30.32, C2, False),
    ]
    # 對照：只調 cap 省下與 v/c trim 相同比例的成本，依 22k 斜率推估的 PSNR 變化
    for blk, bx, vx in (("b6", 4.729, 2.391), ("b13", 6.123, 2.955)):
        rows.append((f"只調 cap 省同樣成本（推估）　{blk}", bx, vx, 0.0, slope * np.log(vx / bx), MUTED, True))
    labels = [r[0] for r in rows]
    dload = [100 * (r[2] / r[1] - 1) for r in rows]
    dpsnr = [r[4] - r[3] for r in rows]
    cols = [r[5] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, vals, unit, title in ((axs[0], dload, "%", "渲染成本變化（精確 Load 中位，%）\n負＝更便宜"),
                                  (axs[1], dpsnr, " dB", "品質變化（val PSNR，dB）\n正＝更好")):
        for yy, v, c, r in zip(y, vals, cols, rows):
            ax.barh(yy, v, color=c, height=0.62, hatch=("///" if r[6] else None), edgecolor="#fcfcfb", lw=0)
            ax.annotate(f"{v:+.1f}{unit}" if unit == "%" else f"{v:+.2f}{unit}", (v, yy),
                        xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=8, color=INK2)
        ax.axvline(0, color="#0b0b0b", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels(labels, fontsize=8.5)
    axs[0].set_xlim(-64, 12); axs[1].set_xlim(-2.15, 0.6)
    fig.subplots_adjust(top=0.74, wspace=0.08)
    fig.suptitle("圖 2　60k 的前 → 後對照（每一列＝同一塊、只改一項設定的兩個跑次，比較它們跑完時的終點）\n"
                 "v/c trim：成本 -49~-52%，品質只 -0.26~-0.54 dB；同樣省成本若只調 cap，推估要 -1.6~-1.7 dB（灰色斜線＝推估）",
                 fontsize=10.5, y=0.99)
    save(fig, "f2_60k_vpc_init.png")


# ── 圖 3：逐 tile 前 K 名（事後剪枝，b6 60k conic 開）────────────────────────
def fig3():
    # 2026-10-01 改版：加入現行 trim 判準 v 與 v/c（先前誤把「總貢獻最大」標成現行判準）
    tk = [(0.23, 17.925, 1), (0.32, 19.356, 2), (0.45, 20.963, 4), (0.65, 22.825, 8), (0.92, 24.945, 16), (1.37, 27.231, 32), (1.92, 29.165, 64)]
    vt = [(0.17, 16.391), (0.24, 18.231), (0.37, 20.525), (0.58, 22.740), (0.85, 25.034), (1.25, 27.297), (1.78, 29.165)]
    vc = [(0.05, 10.790), (0.08, 11.406), (0.13, 12.286), (0.19, 13.496), (0.31, 15.043), (0.49, 17.043), (0.76, 19.807)]
    op = [(0.51, 18.139), (0.75, 19.522), (0.99, 21.025), (1.18, 23.124), (1.35, 25.218), (1.60, 27.200), (1.93, 28.834)]
    mc = [(1.28, 19.195), (1.42, 20.218), (1.62, 21.544), (1.83, 23.219), (2.05, 25.189), (2.31, 27.243), (2.56, 28.970)]
    rd = [(0.24, 15.460), (0.36, 16.936), (0.52, 18.468), (0.73, 20.103), (1.05, 22.001), (1.48, 24.210), (1.94, 26.543)]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    # 參考基準：灰色、不佔類別色（線型區分＋直接標註）
    ax.plot(*zip(*rd), "--", color=MUTED, lw=1.3, label="參考：隨機")
    ax.plot(*zip(*op), ":", color=MUTED, lw=1.6, label="參考：opacity 最高")
    ax.plot(*zip(*mc), "-.", color=MUTED, lw=1.3, label="參考：總貢獻最大（偏好又大又亮的顆粒）")
    lab(ax, rd[-1][0], rd[-1][1], "隨機", dx=5, dy=-4, color=MUTED)
    lab(ax, op[-1][0], op[-1][1], "opacity", dx=5, dy=-10, color=MUTED)
    lab(ax, mc[-1][0], mc[-1][1], "總貢獻最大", dx=5, dy=-4, color=MUTED)
    # 主角：三種判準
    ax.plot(*zip(*vc), "-^", color=C3, lw=2, ms=6, label="v/c（剪完不重訓時最差：只留小而暗的顆粒）")
    ax.plot(*zip(*vt), "-s", color=C2, lw=2, ms=6, label="現行 trim 判準 v（每像素平均貢獻）", zorder=4)
    ax.plot([p[0] for p in tk], [p[1] for p in tk], "-o", color=C1, lw=2, ms=6, label="逐 tile 前 K 名", zorder=4)
    for x, y, k in tk[3:]:
        lab(ax, x, y, f"K={k}", dx=-6, dy=5, ha="right")
    lab(ax, vt[-1][0], vt[-1][1], "現行 v", dx=-10, dy=4, ha="right")
    lab(ax, vc[-1][0], vc[-1][1], "v/c", dx=6, dy=-4)
    ax.plot(2.94, 30.481, "*", color="#0b0b0b", ms=13, zorder=5); lab(ax, 2.94, 30.481, "不剪 30.48", dx=-6, dy=6, ha="right")
    ax.set_xlabel("渲染成本：精確 Load 中位（百萬）"); ax.set_ylabel("val PSNR（dB）")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("圖 3　同一個訓練好的模型（b6 60k），用不同規則剪掉不同比例、剪完「不重新訓練」\n"
                 "每條線＝一種剪法，線上的點＝保留比例不同。越左越便宜、越上越好：現行 v 與逐 tile 前 K 名幾乎重疊（v 略佳）",
                 fontsize=9.5)
    save(fig, "f3_tile_topk_posthoc.png")


# ── 圖 6：剪完再 fine-tune（b6 60k -> 接續到 65k）───────────────────────────────
def fig6():
    arms = ["不剪", "前32名", "opacity\n（同 N）", "前64名", "opacity\n（同 N） "]
    pre = [(2.936, 30.481), (1.372, 27.231), (1.599, 27.200), (1.920, 29.165), (1.931, 28.834)]
    post = [(2.926, 30.681), (1.546, 29.838), (1.771, 29.872), (2.051, 30.335), (2.066, 30.317)]
    x = np.arange(len(arms)); w = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    ax.bar(x - w / 2, [p[1] for p in pre], w, color=MUTED, label="剪完、未重訓")
    ax.bar(x + w / 2, [p[1] for p in post], w, color=C1, label="再 fine-tune 5k 步（60k→65k）")
    for i, (a, b) in enumerate(zip(pre, post)):
        lab(ax, i - w / 2, a[1], f"{a[1]:.2f}", dx=0, dy=2, ha="center", size=7)
        lab(ax, i + w / 2, b[1], f"{b[1]:.2f}", dx=0, dy=2, ha="center", size=7)
    ax.set_ylim(26.5, 31.2); ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_title("val PSNR（dB）", fontsize=10); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax = axs[1]
    ax.bar(x - w / 2, [p[0] for p in pre], w, color=MUTED, label="剪完、未重訓")
    ax.bar(x + w / 2, [p[0] for p in post], w, color=C1, label="fine-tune 後")
    for i, (a, b) in enumerate(zip(pre, post)):
        lab(ax, i + w / 2, b[0], f"{b[0]:.2f}M", dx=0, dy=2, ha="center", size=7)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_title("渲染成本：精確 Load 中位（百萬；越低越便宜）", fontsize=10); ax.grid(axis="x", visible=False)
    fig.suptitle("圖 6　剪完再 fine-tune 5k 步：前 K 名與同顆數的 opacity 剪法補回後品質幾乎一樣（差 ≤0.04 dB）；\n"
                 "前 32 名的 Load 比 opacity 低 13%，前 64 名只低 1%。對照：不剪多跑 5k 步也 +0.20", fontsize=10.5, y=1.02)
    save(fig, "f6_prune_finetune.png")


# ── 圖 7：官方配方 trim 真的執行時（官方 block 3）──────────────────────────────────
def fig7():
    names = ["官方原版\n（trim 從未執行）", "官方＋一行修正\n（trim 執行）", "我方程式碼跑官方配方\n（trim 執行）"]
    N = [5.23, 0.91, 0.93]; ps = [31.69, 30.87, 30.91]; wall = [3.31, 2.29, None]; vr = [7.30, 2.75, 2.59]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, vals, t, fmt in ((axs[0], N, "終點顆數 N（百萬）", "{:.2f}M"), (axs[1], ps, "val PSNR（dB）", "{:.2f}"),
                             (axs[2], vr, "訓練峰值 VRAM（實際配置，GB）", "{:.2f}")):
        cols = [MUTED, C1, C2]
        ax.bar(range(3), vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            lab(ax, i, v, fmt.format(v), dx=0, dy=2, ha="center", size=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7.5); ax.set_title(t, fontsize=10)
        ax.grid(axis="x", visible=False)
    axs[1].set_ylim(29.5, 32.2)
    fig.suptitle("圖 7　官方 block 3：讓 trim 真的執行，顆數少 5.7 倍、峰值 VRAM 少 62%，val 只低 0.8 dB（val⊂train）；\n"
                 "我方程式碼與官方＋修正的結果一致（30.91 vs 30.87、0.93M vs 0.91M）=> 我方能重現官方的 trim 行為", fontsize=10.5, y=1.05)
    save(fig, "f7_official_trim.png")


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
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
