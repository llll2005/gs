"""★ 2026-11-19 使用者「整理當前訓練過程的時間分配圖表（含資料準備）、另外把訓練中每步各部分耗時畫一張」。

兩張圖（寫到 紀錄/）：
  step   每步各部分耗時：讀 task_mech_time.sh 的逐段計時 log（現行配方整段 17k、穩態從第 20 步起）
         => 紀錄/圖_每步耗時_1119.png
  pipe   整條流程時間分配：資料準備（prep_pipeline.sh 的 prep_timing.tsv）＋16 塊訓練（byblock 工具的 run_summary 表）＋合併＋評測
         => 紀錄/圖_時間分配_1119.png
用法：python tools/plot_time_alloc.py step <mech_time log>
      python tools/plot_time_alloc.py pipe <prep_timing.tsv> <byblock log> --merge_s 180 --test_s 810
"""
import argparse, os, re, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    FONT_FILE = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK TC"], capture_output=True, text=True).stdout.strip()
    _fm.fontManager.addfont(FONT_FILE); _fn = _fm.FontProperties(fname=FONT_FILE).get_name()
except Exception as _e:
    _fn = "DejaVu Sans"; print("⚠ 找不到中文字型：", _e)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.sans-serif": [_fn, "DejaVu Sans"], "axes.unicode_minus": False, "figure.dpi": 150,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "legend.frameon": False})
# 類別色（dataviz 參考色盤固定順序；validate_palette.js light 7 色全過，三色對比 < 3:1 => 每段都直接標數字）
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def parse_mech(path):
    t = open(path, encoding="utf-8").read()
    seg = {}
    for m in re.finditer(r"^(\d+) (.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)%", t, re.M):
        seg[int(m.group(1))] = (m.group(2).strip(), float(m.group(4)), int(m.group(5).replace(",", "")), float(m.group(6)))
    real = float(re.search(r"（真實步時間·含重疊）\s+([\d.]+)", t).group(1))
    hooks = float(re.search(r"on_train_batch_end hooks.*?：([\d.]+) ms／步", t).group(1))
    cls = {}
    for m in re.finditer(r"^(lean|full|full＋增生事件)\s+([\d,]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)% \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t, re.M):
        g = [float(x.replace(",", "")) for x in m.groups()[1:]]
        cls[m.group(1)] = dict(n=g[0], frac=g[1], real=g[2], tot=g[3], share=g[4], outer=g[5], fwd=g[6], loss=g[7], bwd=g[8],
                               dens=g[9], hooks=g[10], optim=g[11], trim=g[12])
    n = int(re.search(r"穩態取樣 ([\d,]+) 步", t).group(1).replace(",", ""))
    N = int(re.search(r"N = ([\d,]+)", t).group(1).replace(",", ""))
    return seg, real, hooks, cls, n, N


def plot_step(path, out):
    seg, real, hooks, cls, nsteps, N = parse_mech(path)
    w = {k: v[1] for k, v in seg.items()}
    # 7 類（固定順序＝固定顏色）：forward、loss、backward、密度控制、optimizer、迴圈外、其他
    other = sum(v for k, v in w.items() if k not in (3, 4, 6, 7, 9, 0))
    cats = [("光柵器 forward", w[3]), ("loss（L1＋SSIM＋正則）", w[4]), ("backward（整個反向：光柵器＋SSIM＋正則）", w[6]),
            ("密度控制（增生／搬家／合併）", w[7]), ("optimizer", w[9]),
            ("迴圈外（MCMC 雜訊等 hook＋資料載入）", w[0]), ("其他小段", other)]
    GAP = "#c9c8c2"   # 未歸類（真實步時間 − 各段之和；中性灰，不是類別色）
    cats.append(("未歸類（段與段之間的空檔）", max(real - sum(v for _, v in cats), 0.0)))
    col = {nm: CAT[i] for i, (nm, _) in enumerate(cats[:-1])}; col[cats[-1][0]] = GAP
    tot = sum(v for _, v in cats)

    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.7)
    # (a) 平均一步：由大到小
    ax = fig.add_subplot(gs[0])
    srt = sorted(cats, key=lambda x: -x[1])
    ys = range(len(srt))[::-1]
    for y, (nm, v) in zip(ys, srt):
        ax.barh(y, v, height=0.62, color=col[nm], edgecolor=SURF, linewidth=2)
        ax.text(v + 0.15, y, f"{v:.2f} ms（{100 * v / tot:.0f}%）", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(ys)); ax.set_yticklabels([nm for nm, _ in srt], fontsize=9.5, color=INK)
    ax.set_xlim(0, max(v for _, v in cats) * 1.32); ax.grid(axis="y", visible=False)
    ax.set_xlabel("每步平均耗時（ms）", fontsize=9, color=INK2)
    ax.set_title(f"(a) 平均一步 {real:.1f} ms 花在哪（{1000 / real:.1f} it/s）", loc="left", fontsize=11.5, color=INK)
    ax.text(0, -0.30, f"迴圈外 {w[0]:.2f} ms 之中，MCMC 雜訊、面內限制等每步 hook 佔 {hooks:.2f} ms，其餘是資料載入與 Lightning。"
                      f"「其他小段」＝scheduler、log、hook 前置、trim 等 {other:.2f} ms。",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")

    # (b) 依步型：一般步 vs 每 8 步的完整路徑步（＋增生事件那一步另註）
    ax = fig.add_subplot(gs[1])
    rows = [("一般步", cls["lean"]), ("完整路徑步（每 8 步：法向＋深度上限＋dist）", cls["full"])]
    keys = [("光柵器 forward", "fwd"), ("loss（L1＋SSIM＋正則）", "loss"), ("backward（整個反向：光柵器＋SSIM＋正則）", "bwd"),
            ("密度控制（增生／搬家／合併）", "dens"), ("optimizer", "optim"), ("迴圈外（MCMC 雜訊等 hook＋資料載入）", "outer"),
            ("其他小段", "small"), ("未歸類（段與段之間的空檔）", "gap")]
    for c in (cls["lean"], cls["full"]):
        c["small"] = c["hooks"] + c["trim"]
        c["gap"] = max(c["real"] - sum(c[k] for k in ("fwd", "loss", "bwd", "dens", "optim", "outer", "small")), 0.0)
    for y, (lab, c) in zip([1, 0], rows):
        x = 0.0
        for nm, k in keys:
            v = c[k]
            ax.barh(y, v, left=x, height=0.55, color=col[nm], edgecolor=SURF, linewidth=2)
            if v >= 1.6:
                ax.text(x + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color=INK)
            x += v
        ax.text(x + 0.4, y, f"{c['real']:.1f} ms／步｜佔步數 {c['frac']:.0f}%、佔總時間 {c['share']:.0f}%", va="center", fontsize=9, color=INK)
    ev = cls["full＋增生事件"]
    ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(0, cls["full"]["real"] * 1.75); ax.set_ylim(-0.6, 1.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("每步耗時（ms）；段內數字＝ms", fontsize=9, color=INK2)
    ax.set_title("(b) 依步型", loc="left", fontsize=11.5, color=INK)
    fig.text(0.01, -0.075, f"另有增生事件步（每 150 步一次、到 70% 步數為止，共 {int(ev['n'])} 次）：一步 {ev['real']:.0f} ms，其中密度控制 {ev['dens']:.0f} ms；"
                      f"只佔步數 {ev['frac']:.1f}%，但佔總時間 {ev['share']:.1f}%。",
            fontsize=8.5, color=INK2, va="top")
    hs = [plt.Rectangle((0, 0), 1, 1, color=col[nm]) for nm, _ in cats]
    fig.legend(hs, [nm for nm, _ in cats], loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"訓練中每步各部分耗時——現行配方（新底 9＋平均 1.5M），b4、{nsteps:,} 步、終點 {N / 1e6:.2f}M 顆、lab RTX 3090",
                 x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
    fig.text(0.01, 0.955, "逐段計時每段都同步一次 GPU，總時間比正常跑偏高約 5～8%，看比例。來源：logs/mech_time_tune_b4_s17000_1119_1146.log",
             fontsize=8.5, color=MUTED)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("=>", out)


def parse_byblock(path):
    t = open(path, encoding="utf-8").read()
    m = re.search(r"^ 合計\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t, re.M)
    return dict(startup=float(m.group(2)), train=float(m.group(3)), save=float(m.group(4)), proc=float(m.group(5)), task=float(m.group(6)))


def plot_pipe(prep_tsv, byblock, merge_s, test_s, out):
    prep = {}   # 同一步重跑過（PREP_ONLY）取最後一次、且只取成功的（rc=0）；依步驟編號排序
    for ln in open(prep_tsv, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 3 and p[1].isdigit() and p[2] == "0":
            k = int(p[0].split()[0]); prep[k] = (re.sub(r"^\d+\s*", "", p[0]), int(p[1]))
    prep = [prep[k] for k in sorted(prep)]
    bb = parse_byblock(byblock)
    stages = [(f"準備：{n}", s, 0) for n, s in prep]
    stages += [("訓練：啟動（16 塊合計）", bb["startup"], 1), ("訓練：純訓練（16 塊合計）", bb["train"], 1),
               ("訓練：存檔＋行程收尾", bb["save"] + (bb["task"] - bb["proc"]), 1), ("合併 16 塊", merge_s, 2)]
    tot = sum(s for _, s, _ in stages)
    grp = {0: ("資料準備（一次）", CAT[0]), 1: ("16 塊訓練", CAT[1]), 2: ("合併", CAT[2])}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 2.2 + 0.36 * len(stages)), gridspec_kw={"height_ratios": [0.9, len(stages) * 0.5]})
    # (a) 一條長條：三大段
    x = 0.0
    for g in (0, 1, 2):
        v = sum(s for _, s, gg in stages if gg == g)
        a1.barh(0, v / 3600, left=x, height=0.6, color=grp[g][1], edgecolor=SURF, linewidth=2)
        if v / tot >= 0.08:
            sub = f"\n純訓練 {bb['train'] / 3600:.2f} h＋啟動／存檔 {(v - bb['train']) / 3600:.2f} h" if g == 1 else ""
            a1.text(x + v / 7200, 0, f"{grp[g][0]}\n{v / 3600:.2f} h（{100 * v / tot:.0f}%）{sub}", ha="center", va="center", fontsize=9, color=INK)
        else:   # 太窄的段：標在條的右邊
            a1.text(x + v / 3600 + tot / 3600 * 0.01, 0, f"{grp[g][0]} {v / 60:.0f} 分", ha="left", va="center", fontsize=9, color=INK)
        x += v / 3600
    a1.set_yticks([]); a1.set_xlim(0, tot / 3600 * 1.12); a1.grid(axis="y", visible=False)
    a1.set_title(f"(a) 從原始影像到可用模型：合計 {tot / 3600:.2f} h（不含評測）", loc="left", fontsize=11.5, color=INK)
    a1.set_xlabel("小時", fontsize=9, color=INK2)
    # (b) 每一階段
    ys = range(len(stages))[::-1]
    for y, (n, s, g) in zip(ys, stages):
        a2.barh(y, s / 60, height=0.62, color=grp[g][1], edgecolor=SURF, linewidth=2)
        a2.text(s / 60 + 0.6, y, f"{s / 60:.1f} 分（{100 * s / tot:.1f}%）", va="center", fontsize=8.5, color=INK)
    a2.set_yticks(list(ys)); a2.set_yticklabels([n for n, _, _ in stages], fontsize=9, color=INK)
    a2.set_xlim(0, max(s for _, s, _ in stages) / 60 * 1.25); a2.grid(axis="y", visible=False)
    a2.set_xlabel("分鐘", fontsize=9, color=INK2)
    a2.set_title(f"(b) 各階段（評測不算在內：741 幀 test 約 {test_s / 60:.0f} 分）", loc="left", fontsize=11.5, color=INK)
    fig.suptitle("整條流程時間分配——現行配方（新底 9＋平均 1.5M）、MatrixCity aerial 4x4、lab RTX 3090 單卡",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("=>", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["step", "pipe"])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--merge_s", type=float, default=180)
    ap.add_argument("--test_s", type=float, default=810)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mode == "step":
        plot_step(a.inputs[0], a.out or os.path.join(ROOT, "紀錄", "圖_每步耗時_1119.png"))
    else:
        plot_pipe(a.inputs[0], a.inputs[1], a.merge_s, a.test_s, a.out or os.path.join(ROOT, "紀錄", "圖_時間分配_1119.png"))
