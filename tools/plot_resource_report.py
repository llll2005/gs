#!/usr/bin/env python
"""資源與效能量測的圖（2026-09-17）。數據全部是量測值，出處見 紀錄/資源與效能量測彙整.md。
用法：python tools/plot_resource_report.py   => 輸出到 紀錄/figures/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ⚠ 2026-09-17：matplotlib 的字型快取沒收錄系統的 Noto Sans CJK => 只寫名稱會退回 DejaVu Sans、中文全變方框
#   => 用 fc-match 找字型檔，addfont 後取它的真實名稱
import subprocess
from matplotlib import font_manager as _fm
try:
    _ff = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK TC"], capture_output=True, text=True).stdout.strip()
    _fm.fontManager.addfont(_ff)
    _fn = _fm.FontProperties(fname=_ff).get_name()
except Exception as _e:
    _fn = "DejaVu Sans"
    print("⚠⚠ 找不到中文字型，圖上中文會變方框：", _e)
plt.rcParams["font.sans-serif"] = [_fn, "DejaVu Sans"]
print("字型：", _fn)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "紀錄", "figures")
os.makedirs(OUT, exist_ok=True)
C = {"backward": "#4C72B0", "forward": "#55A868", "trim": "#C44E52", "optimizer": "#8172B2",
     "loss": "#CCB974", "loopout": "#64B5CD", "other": "#B0B0B0"}
LAB = {"backward": "backward", "forward": "forward（光柵化）", "trim": "週期 trim（按 500 步攤平）",
       "optimizer": "optimizer.step", "loss": "loss（L1+SSIM）", "loopout": "迴圈外（dataloader／搬運）", "other": "其他"}

def save(fig, name):
    p = os.path.join(OUT, name); fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  ->", p)

# ── 圖 1：每步時間拆解（增生期，trim 按真實週期 500 步攤平）──
rows = [
    ("09-12 舊資料 b12\nN 1.79M（284 台相機）", dict(backward=181.58, forward=118.29, trim=101.83, optimizer=51.56, loss=15.79, loopout=18.53, other=0)),
    ("新年代 b6 低顆數\nN 0.63M（548 台相機）", dict(backward=99.48, forward=31.54, trim=30891.22/500, optimizer=14.37, loss=15.64, loopout=4.10, other=2.90)),
    ("新年代 b6 高顆數\nN 1.54M", dict(backward=172.05, forward=95.79, trim=75512.24/500, optimizer=45.51, loss=15.91, loopout=9.00, other=2.66)),
    ("新年代 b6 高顆數\n＋max_split_size_mb:128", dict(backward=176.60, forward=106.19, trim=76719.21/500, optimizer=57.91, loss=15.87, loopout=8.82, other=2.69)),
]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [3, 2]})
for ax, pct in zip(axes, (False, True)):
    for i, (name, d) in enumerate(rows[::-1]):
        tot = sum(d.values()); left = 0
        for k in C:
            v = d[k] / tot * 100 if pct else d[k]
            ax.barh(i, v, left=left, color=C[k], label=LAB[k] if i == 0 else None, edgecolor="white", linewidth=0.5)
            if (v > (6 if pct else 25)):
                ax.text(left + v / 2, i, f"{v:.0f}{'%' if pct else ''}", ha="center", va="center", fontsize=8, color="white")
            left += v
        if not pct:
            ax.text(left + 5, i, f"{tot:.0f} ms", va="center", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xlabel("佔每步時間 (%)" if pct else "每步時間 (ms，增生期)")
    if pct: ax.set_yticklabels([]); ax.set_xlim(0, 100)
axes[0].legend(loc="upper center", bbox_to_anchor=(0.8, -0.14), ncol=4, fontsize=8, frameon=False)
fig.suptitle("圖 1　訓練時間花在哪：運算（backward／forward／trim）為主，搬運只佔約 2%", fontsize=12)
save(fig, "fig1_time_breakdown.png")

# ── 圖 2：max_split_size_mb:128 在真實迴圈上的代價 ──
segs = ["backward", "forward", "optimizer", "loss", "loopout"]
off = [172.05, 95.79, 45.51, 15.91, 9.00]; on = [176.60, 106.19, 57.91, 15.87, 8.82]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [3, 2]})
x = range(len(segs)); w = 0.38
axes[0].bar([i - w/2 for i in x], off, w, label="不開", color="#4C72B0")
axes[0].bar([i + w/2 for i in x], on, w, label="開 max_split_size_mb:128", color="#DD8452")
for i, (a, b) in enumerate(zip(off, on)):
    axes[0].text(i + w/2, b + 2, f"{(b/a-1)*100:+.0f}%", ha="center", fontsize=8)
axes[0].set_xticks(list(x)); axes[0].set_xticklabels([LAB[s] for s in segs], fontsize=8, rotation=10)
axes[0].set_ylabel("ms/步"); axes[0].legend(fontsize=8)
axes[0].set_title("每段時間（真實每步 340.8 → 367.9 ms，+8.0%）", fontsize=10)
axes[1].bar([0 - w/2, 1 - w/2], [2818, 3860], w, color="#4C72B0", label="不開")
axes[1].bar([0 + w/2, 1 + w/2], [2823, 5526], w, color="#DD8452", label="開")
for xx, v in [(-w/2, 2818), (w/2, 2823), (1 - w/2, 3860), (1 + w/2, 5526)]:
    axes[1].text(xx, v + 60, f"{v:,}", ha="center", fontsize=8)
axes[1].axhline(6246, color="k", ls="--", lw=1); axes[1].text(1.45, 6300, "本機 6.1 GiB", fontsize=8, ha="right")
axes[1].axhline(5796, color="gray", ls=":", lw=1); axes[1].text(1.45, 5550, "lab 上限 5.66 GiB", fontsize=8, ha="right", color="gray")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["段內峰值配置", "段末保留（含配置器快取）"])
axes[1].set_ylabel("MiB"); axes[1].set_ylim(0, 7000); axes[1].legend(fontsize=8, loc="upper left")
axes[1].set_title("VRAM：實際配置不變，快取 +1.67 GB", fontsize=10)
fig.suptitle("圖 2　max_split_size_mb:128：慢 8%、不增加實際配置；只在鎖 VRAM 上限時用來防碎片 OOM（本機 b6，N 1.54M）", fontsize=11)
save(fig, "fig2_maxsplit_ab.png")

# ── 圖 3：訓練中各段的 VRAM ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (title, d, resv) in zip(axes, [
    ("低顆數 N 0.63M", {"迴圈外（常駐）": 631, "forward": 1081, "optimizer": 1051, "trim": 932, "loss": 1350, "backward": 1324}, 2262),
    ("高顆數 N 1.54M", {"迴圈外（常駐）": 1634, "forward": 2278, "optimizer": 2439, "trim": 2468, "loss": 2547, "backward": 2818}, 3860)]):
    ks = list(d); vs = [d[k] for k in ks]
    bars = ax.bar(ks, vs, color=["#64B5CD", "#55A868", "#8172B2", "#C44E52", "#CCB974", "#4C72B0"])
    for b, v in zip(bars, vs): ax.text(b.get_x() + b.get_width()/2, v + 50, f"{v:,}", ha="center", fontsize=8)
    ax.axhline(resv, color="#DD8452", ls="--", lw=1.2); ax.text(5.4, resv + 60, f"段末保留 {resv:,}（不開 max_split）", fontsize=8, ha="right", color="#DD8452")
    ax.axhline(6246, color="k", ls="--", lw=1); ax.text(5.4, 6300, "本機 6.1 GiB 牆", fontsize=8, ha="right")
    ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelsize=8, rotation=15)
axes[1].axhline(5526, color="#DD8452", ls=":", lw=1.2); axes[1].text(5.4, 5300, "段末保留 5,526（開 max_split）", fontsize=8, ha="right", color="#DD8452")
axes[0].set_ylabel("段內峰值配置 (MiB)"); axes[0].set_ylim(0, 7000)
fig.suptitle("圖 3　訓練中 VRAM：峰值在 backward；常駐 ≈ 928 B/顆（參數＋梯度＋Adam）；保留與配置之差是配置器快取", fontsize=11)
save(fig, "fig3_vram_segments.png")

# ── 圖 4：ckpt 裡存了什麼 ──
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 2]})
axes[0].bar(["ckpt（N 1.85M）"], [0.400], color="#4C72B0", label="參數 0.400 GiB（33%）")
axes[0].bar(["ckpt（N 1.85M）"], [0.799], bottom=[0.400], color="#C44E52", label="Adam 狀態 0.799 GiB（67%）")
axes[0].set_ylabel("GiB"); axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08)); axes[0].set_ylim(0, 1.4)
fields = [("SH 高階 shs_rest", 180), ("rotations", 16), ("means", 12), ("SH 常數 shs_dc", 12), ("scales（2DGS 2 軸）", 8), ("opacities", 4), ("Adam exp_avg", 232), ("Adam exp_avg_sq", 232)]
cols = ["#C44E52", "#55A868", "#55A868", "#DD8452", "#55A868", "#55A868", "#B0B0B0", "#8C8C8C"]
left = 0
for (n, v), c in zip(fields, cols):
    axes[1].barh(0, v, left=left, color=c, edgecolor="white")
    if v >= 12: axes[1].text(left + v/2, 0, f"{n}\n{v} B", ha="center", va="center", fontsize=7, color="white" if v > 30 else "black")
    left += v
axes[1].set_yticks([]); axes[1].set_xlabel("bytes／顆（ckpt 合計約 696 B／顆）")
axes[1].set_title("參數 232 B／顆：SH 高階佔 77.6%，幾何（位置／旋轉／縮放／不透明度）只有 40 B", fontsize=10)
fig.suptitle("圖 4　成果檔的組成：2/3 是優化器狀態；參數裡主要是顏色（SH3）", fontsize=11)
save(fig, "fig4_ckpt_storage.png")

# ── 圖 5：CPU RAM 影像快取 ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].bar(["現行（float32 影像＋深度圖）", "提案（uint8 RGB，不載深度）"], [17.28, 4.32], color="#4C72B0", label="影像")
axes[0].bar(["現行（float32 影像＋深度圖）", "提案（uint8 RGB，不載深度）"], [8.29, 0], bottom=[17.28, 4.32], color="#C44E52", label="深度圖（權重 0 仍載入）")
axes[0].text(0, 26.2, "25.6 MB", ha="center"); axes[0].text(1, 5.0, "4.3 MB（-83%）", ha="center")
axes[0].set_ylabel("MB／訓練視角"); axes[0].legend(fontsize=8); axes[0].set_ylim(0, 30)
axes[0].set_title("每個視角（1600x900 影像；深度 1920x1080）\n載入時另有 float64 暫存 34.6 MB", fontsize=10)
_lb = ["估算：現行 b6\n（548 台，含深度）", "估算：提案 b6", "估算：現行\nx lab 三槽", "估算：提案\nx lab 三槽", "本機實測 RSS\n原樣（未載深度）", "本機實測 RSS\nuint8＋不載深度"]
_v = [14.0, 2.37, 42.1, 7.1, 12.35, 4.71]
axes[1].bar(_lb, _v, color=["#C44E52", "#55A868", "#C44E52", "#55A868", "#DD8452", "#64B5CD"])
for i, v in enumerate(_v): axes[1].text(i, v + 0.8, f"{v:.1f} GB", ha="center", fontsize=8)
axes[1].tick_params(axis="x", labelsize=7)
axes[1].axhline(62, color="k", ls="--", lw=1); axes[1].text(5.4, 58.5, "lab 實體 RAM 62 GB", ha="right", fontsize=8)
axes[1].set_ylabel("GB"); axes[1].set_ylim(0, 70)
axes[1].set_title("整塊快取（未含 val 與載入暫存；lab 實見每跑次 17~20 GB）", fontsize=10)
fig.suptitle("圖 5　CPU RAM：lab 的快取把 float32 影像與「權重為 0 的深度圖」一起存（本機深度圖是舊檔、實際沒載入）；本機實測 uint8 即降 62%", fontsize=10)
save(fig, "fig5_ram_cache.png")

# ── 圖 6：一個 60k 跑次目錄（lab speed3 b6，7.8 GB）──
steps = ["499", "1,499", "14,999", "29,999", "41,999", "60,000"]
ck = [403003354, 419871514, 1809617434, 1809617434, 1628657434, 1628657434]
ply = [15633315, 16287687, 70200235, 70200235, 63180235, 63180235]
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(steps, [c/2**30 for c in ck], color=["#B0B0B0"]*5 + ["#4C72B0"], label="ckpt（參數＋Adam）")
ax.bar(steps, [p/2**30 for p in ply], bottom=[c/2**30 for c in ck], color="#C44E52", label="-xyz_rgb.ply（不可渲染）")
for i, (c, p) in enumerate(zip(ck, ply)): ax.text(i, (c+p)/2**30 + 0.03, f"{(c+p)/2**30:.2f}", ha="center", fontsize=8)
ax.set_xlabel("存檔步數"); ax.set_ylabel("GiB"); ax.legend(fontsize=8)
ax.set_title("圖 6　一個 60k 跑次的檔案：中間 ckpt 佔 checkpoints/ 的 78%；xyz_rgb.ply 6 份約 300 MB（14,999／29,999 時 N=2.6M，終點 2.34M）", fontsize=10)
save(fig, "fig6_run_dir.png")

# ── 圖 7：lab 三槽平行 vs 本機獨佔 的牆鐘時間 ──
lab = {"b6 21,920 步\nN 1.85M": [10107, 9627, 9952, 10313, 9285, 8932, 8702],
       "60k 步\nN 2.34M": [28055, 22863, 23132, 21503, 17873, 29945, 30765, 29575],
       "20k 步 init\n機制全關": [6971, 7041, 5143, 6491, 7281, 7591]}
loc = {"b6 21,920 步\nN 1.85M": [9943], "60k 步\nN 2.34M": [29403, 31614], "20k 步 init\n機制全關": []}
fig, ax = plt.subplots(figsize=(10, 4.4))
for i, k in enumerate(lab):
    ax.scatter([i - 0.12]*len(lab[k]), [v/3600 for v in lab[k]], color="#DD8452", s=28, label="lab 3090（三槽平行）" if i == 0 else None, zorder=3)
    if loc[k]: ax.scatter([i + 0.12]*len(loc[k]), [v/3600 for v in loc[k]], color="#4C72B0", s=40, marker="s", label="本機 4050（獨佔）" if i == 0 else None, zorder=3)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(list(lab)); ax.set_ylabel("牆鐘時間（小時）")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
ax.set_title("圖 7　單跑牆鐘：lab 三槽平行時與本機相近（22k）到略快（60k）；lab 的優勢是一次 3 個\n（本機 60k 為舊年代計時；計時不經過影像配對，仍有效）", fontsize=10)
save(fig, "fig7_wall_time.png")
# ── 圖 8：trim 判準 v vs v/c 的時間與成本（同塊、同 N、60k 終點模型）──
# 數據＝scripts/lab/task_60k_cost.sh（lab [solo]，tools/step_breakdown.py 與 cost_budget_calibrate.py）
# 五臂的 N 都是 2,340,000；離線 Load 用該塊全部相機（b6 548 台、b13 667 台）
T = {  # 臂: (b6 光柵化fwd ms, b6 fwd+bwd ms, b13 光柵化fwd, b13 fwd+bwd)
    "speed3（v，現行）": (74.13, 232.03, 102.11, 342.61),
    "+trimvpc（v/c）":   (61.32, 185.06,  80.74, 266.95),
}
L = {  # 臂: (b6 Load中位, b6 Load平均, b13 中位, b13 平均)
    "speed3（v，現行）": (7665730, 8001109, 8223874, 8236999),
    "+trimvpc（v/c）":   (1768517, 1874065, 2142386, 2192127),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2, 2, 1.5]})
w = 0.36
for ax, (bi, title) in zip(axes[:2], [(0, "b6（548 台相機）"), (2, "b13（667 台相機）")]):
    seg = ["光柵化 forward", "forward+backward"]
    base = [T["speed3（v，現行）"][bi], T["speed3（v，現行）"][bi + 1]]
    vpc  = [T["+trimvpc（v/c）"][bi],  T["+trimvpc（v/c）"][bi + 1]]
    x = range(len(seg))
    ax.bar([i - w/2 for i in x], base, w, color="#4C72B0", label="trim 判準 v（現行）")
    ax.bar([i + w/2 for i in x], vpc,  w, color="#55A868", label="trim 判準 v/c")
    for i, (a_, b_) in enumerate(zip(base, vpc)):
        ax.text(i - w/2, a_ + 6, f"{a_:.0f}", ha="center", fontsize=9)
        ax.text(i + w/2, b_ + 6, f"{b_:.0f}", ha="center", fontsize=9)
        ax.text(i, max(a_, b_) + 26, f"{(b_/a_-1)*100:+.1f}%", ha="center", fontsize=10, color="#C44E52")
    ax.set_xticks(list(x)); ax.set_xticklabels(seg, fontsize=9)
    ax.set_ylabel("ms（同一 process、CUDA event、lab [solo]）" if bi == 0 else "")
    ax.set_ylim(0, 400); ax.set_title(title, fontsize=11)
    if bi == 0: ax.legend(fontsize=9)
ax = axes[2]
lab2 = ["b6 中位", "b6 平均", "b13 中位", "b13 平均"]
bl = [L["speed3（v，現行）"][i] / 1e6 for i in (0, 1, 2, 3)]
vl = [L["+trimvpc（v/c）"][i] / 1e6 for i in (0, 1, 2, 3)]
x = range(4)
ax.bar([i - w/2 for i in x], bl, w, color="#4C72B0")
ax.bar([i + w/2 for i in x], vl, w, color="#55A868")
for i, (a_, b_) in enumerate(zip(bl, vl)):
    ax.text(i, max(a_, b_) + 0.3, f"{(b_/a_-1)*100:+.0f}%", ha="center", fontsize=9, color="#C44E52")
ax.set_xticks(list(x)); ax.set_xticklabels(lab2, fontsize=8, rotation=12)
ax.set_ylabel("離線 Load（百萬）"); ax.set_title("渲染成本 Load", fontsize=11)
fig.suptitle("圖 8　trim 判準 v -> v/c：同一塊、同 N=2,340,000（60k 終點模型）\n"
             "品質代價 PSNR b6 30.452->30.186（-0.27）／b13 29.865->29.323（-0.54）", fontsize=11)
save(fig, "fig8_trimvpc_time.png")

# ── 圖 9：各段隨顆數的縮放，以及全程（增生期 vs 收割期）的分配 ──
# 數據＝本機 _StepProfiler 同一支工具的兩個顆數點（scripts/task_stepcost2.sh）
LO = {"backward": 99.48, "forward": 31.54, "trim": 30891.22/500, "optimizer": 14.37, "loss": 15.64, "loopout": 4.10}
HI = {"backward": 172.05, "forward": 95.79, "trim": 75512.24/500, "optimizer": 45.51, "loss": 15.91, "loopout": 9.00}
NLO, NHI = 0.63, 1.54
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
ks = ["forward", "optimizer", "trim", "loopout", "backward", "loss"]
r = [HI[k] / LO[k] for k in ks]
cols = [C["forward"], C["optimizer"], C["trim"], C["loopout"], C["backward"], C["loss"]]
bars = ax.bar(range(len(ks)), r, color=cols)
for i, v in enumerate(r):
    ax.text(i, v + 0.06, f"x{v:.2f}", ha="center", fontsize=10)
ax.axhline(NHI/NLO, color="k", ls="--", lw=1.2)
ax.text(len(ks)-0.4, NHI/NLO + 0.07, f"顆數本身 x{NHI/NLO:.2f}（線性參考線）", ha="right", fontsize=9)
ax.set_xticks(range(len(ks))); ax.set_xticklabels([LAB[k] for k in ks], fontsize=8, rotation=12)
ax.set_ylabel(f"N {NLO}M -> {NHI}M 時的倍率"); ax.set_ylim(0, 3.6)
ax.set_title("各段隨顆數的縮放：forward／optimizer 超線性，backward 次線性，loss 不隨 N 變", fontsize=10)
ax = axes[1]
den = sum(HI.values()); har = den - HI["trim"]
for i, (nm, d) in enumerate([("增生期\n(step 1k~30k)", HI), ("收割期\n(step 30k~60k)\ntrim 停止", {**HI, "trim": 0.0})]):
    left = 0
    for k in ["backward", "forward", "trim", "optimizer", "loss", "loopout"]:
        v = d[k]
        if v <= 0: continue
        ax.barh(i, v, left=left, color=C[k], edgecolor="white", linewidth=0.5)
        if v > 25: ax.text(left + v/2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
        left += v
    ax.text(left + 6, i, f"{left:.0f} ms", va="center", fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["增生期\n(1k~30k)", "收割期\n(30k~60k)"], fontsize=9)
ax.set_xlabel("每步時間 (ms)"); ax.set_xlim(0, 560)
ax.set_title(f"全程分配（60k、N 1.54M）：trim 佔增生期 {HI['trim']/den*100:.0f}%，\n但收割期為 0 => 整趟約 {HI['trim']/(den+har)*100:.0f}%", fontsize=10)
fig.suptitle("圖 9　時間的兩個結構：隨顆數怎麼長、以及 trim 只在半場出現", fontsize=12)
save(fig, "fig9_scaling_phase.png")

print("完成")
