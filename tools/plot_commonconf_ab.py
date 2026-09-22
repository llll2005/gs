#!/usr/bin/env python
"""通用 config 新舊版對比圖（2026-09-22）。輸出到 紀錄/new_figures_commonconf/

⚠⚠ 這張比較「是什麼」與「不是什麼」，先讀清楚再引用：

2026-09-22 的 config 收斂一共改了 7 項，但其中 **6 項是行為無變化**的
（lambda_normal / depth_loss_weight / opacity_reg / cap_max / densify_until /
absgrad+fast_noise+noise_gate）—— 因為所有任務腳本**本來就在傳那些值**，
config 只是被對齊過去，讓新腳本不會漏。
⇒ 對「用任務腳本啟動的跑次」而言，新舊 config 的差 **完全等於 `exact_conic_aabb` 的差**。

所以本圖＝ conic off（舊）vs conic on（新）的乾淨 A/B：兩塊 x 兩長度、**兩臂同 N、同一個 .so**。
⛔ 本圖**不是**「舊 config 整體 vs 新 config 整體」—— 那個比較**不存在乾淨資料**，
   因為新年代裡沒有任何一次跑次真的用過那 6 項的舊值（腳本全都覆寫掉了）。

數據出處：紀錄/資源與效能量測彙整.md §9.9b（品質／時間／VRAM／離線 Load）。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from matplotlib import font_manager as _fm
try:
    _ff = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK TC"],
                         capture_output=True, text=True).stdout.strip()
    _fm.fontManager.addfont(_ff); _fn = _fm.FontProperties(fname=_ff).get_name()
except Exception as _e:
    _fn = "DejaVu Sans"; print("⚠⚠ 找不到中文字型：", _e)
plt.rcParams["font.sans-serif"] = [_fn, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "紀錄", "new_figures_commonconf")
os.makedirs(OUT, exist_ok=True)
OLD, NEW = "#B0B0B0", "#55A868"

def save(fig, name):
    p = os.path.join(OUT, name); fig.tight_layout()
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  ->", p)

# ── 1. 品質：四組比較 x 四個指標的 Δ（新 − 舊）──────────────────────────────
grp = ["b6 22k", "b13 22k", "b6 60k", "b13 60k"]
d = {"ΔPSNR (dB)":   [0.000, 0.080, 0.060, 0.160],
     "ΔSSIM x1e3":   [1.0,   2.0,   1.0,   3.0],
     "−ΔLPIPS x1e3": [0.0,   1.0,   1.0,   3.0],      # 取負號 => 正值＝更好
     "Δ紋理比 x1e3":  [1.0,   2.0,   5.0,   5.0]}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (k, v) in zip(axes, d.items()):
    ax.bar(grp, v, color=[NEW if x >= 0 else "#C44E52" for x in v])
    ax.axhline(0, color="k", lw=0.8)
    if k.startswith("ΔPSNR"):
        # 同配方重複樣本的實測差（22k）＝ 這張圖唯一有噪音底的指標
        ax.axhspan(-0.05, 0.05, color="#CCB974", alpha=0.35, zorder=0)
        ax.text(0.02, 0.95, "黃帶＝同配方重複樣本\n實測差（b6 0.05／b13 0.03）",
                transform=ax.transAxes, va="top", fontsize=7)
    ax.set_title(k, fontsize=10); ax.tick_params(labelsize=8)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:+.3g}", ha="center",
                va="bottom" if x >= 0 else "top", fontsize=8)
fig.suptitle("品質：新（exact_conic_aabb=true）− 舊（false）　正值＝新版較好　"
             "16 個差沒有一個是負的", fontsize=11)
save(fig, "cc1_quality_delta.png")

# ── 2. 時間：隔離量測（step_breakdown，[solo]，兩臂同 N）────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (blk, n, fo, fn_, bo, bn) in zip(axes, [
        ("b6", "N=1,850,094", 26.35, 19.44, 73.88, 62.32),
        ("b13", "N=2,302,234", 40.47, 29.98, 101.25, 83.98)]):
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, [fo, bo], w, label="舊（conic off）", color=OLD)
    ax.bar(x + w/2, [fn_, bn], w, label="新（conic on）", color=NEW)
    for i, (a, b) in enumerate([(fo, fn_), (bo, bn)]):
        ax.text(i + w/2, b, f"{b:.1f}\n{100*(b/a-1):+.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i - w/2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["光柵化 forward", "forward+backward"], fontsize=9)
    ax.set_ylabel("ms"); ax.set_title(f"{blk}（{n}）", fontsize=10); ax.legend(fontsize=8)
fig.suptitle("時間：同一 process／CUDA event／單一相機／[solo]　⚠ 訓練牆鐘不可用（三槽平行）", fontsize=10)
save(fig, "cc2_time.png")

# ── 3. 離線 Load（精確 Σtiles，全部相機）＋ 峰值 VRAM ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
lab = ["b6 max", "b6 中位", "b13 max", "b13 中位"]
old = [7345817, 5089279, 10108848, 6476070]
new = [5149466, 3046590, 5669418, 3512081]
x = np.arange(4); w = 0.36
ax.bar(x - w/2, np.array(old)/1e6, w, label="舊", color=OLD)
ax.bar(x + w/2, np.array(new)/1e6, w, label="新", color=NEW)
for i, (a, b) in enumerate(zip(old, new)):
    ax.text(i + w/2, b/1e6, f"{100*(b/a-1):+.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("精確 Σtiles（百萬）"); ax.legend(fontsize=8)
ax.set_title("離線 Load：同工具、**全部**相機、兩臂同 N", fontsize=10)
ax = axes[1]
lab2 = ["b13 22k", "b6 60k", "b13 60k"]
o2, n2 = [4.10, 4.58, 4.68], [3.94, 4.35, 4.37]
x = np.arange(3)
ax.bar(x - w/2, o2, w, label="舊", color=OLD)
ax.bar(x + w/2, n2, w, label="新", color=NEW)
for i, (a, b) in enumerate(zip(o2, n2)):
    ax.text(i + w/2, b, f"{b:.2f}\n{100*(b/a-1):+.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lab2, fontsize=9)
ax.set_ylabel("峰值實佔 VRAM（GB）"); ax.set_ylim(0, 5.6); ax.legend(fontsize=8)
ax.set_title("峰值 VRAM（同 N）　⚠ b6 22k 未取到", fontsize=10)
save(fig, "cc3_load_vram.png")

# ── 4. 副作用：代理成本指標在新版下會脫鉤 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.8))
lab3 = ["b6", "b13"]
o3, n3 = [1.282, 1.070], [2.272, 2.234]
x = np.arange(2)
ax.bar(x - w/2, o3, w, label="舊（conic off）", color=OLD)
ax.bar(x + w/2, n3, w, label="新（conic on）", color="#C44E52")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.axhline(1.8, color="#C44E52", lw=0.8, ls=":")
ax.text(1.45, 1.83, "控制器警告門檻 1.8", fontsize=7, color="#C44E52")
for i, (a, b) in enumerate(zip(o3, n3)):
    ax.text(i - w/2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lab3)
ax.set_ylabel("代理 Σ(2r/16)² ÷ 精確 Σtiles")
ax.set_title("⚠ 副作用：新版讓代理成本指標脫鉤\n"
             "（radii 被刻意凍結成線性化半徑以保護尺寸語意）", fontsize=10)
ax.legend(fontsize=8)
save(fig, "cc4_proxy_decoupled.png")
print("完成。⚠ 本圖＝conic on/off 的 A/B，不是「舊 config 整體 vs 新 config 整體」—— 見檔頭。")
