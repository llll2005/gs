#!/usr/bin/env python
"""proxy 驗證開獎：壓縮 5 倍的跑次，名次與全長版一致嗎？小效應分得開嗎？

⚠ 這裡**不看 proxy 的絕對分數**。proxy 的 LR 積分只有 1/5，分數本來就比較低，
   低多少不是重點。重點只有兩個：
     1. 名次（Spearman rho，以及逐對的正負號）
     2. 解析度（全長版差 0.222 dB 的那一對，proxy 分得開嗎）
"""
import glob
import itertools
import sys

from tensorboard.backend.event_processing import event_accumulator as ea

# (proxy 跑次, 全長跑次)
PAIRS = [
    ("px_noprior_b12", "noprior_b12"),
    ("px_sched30_b12", "sched30_b12"),
    ("px_sh3_nonormal_b12", "sh3_nonormal_b12"),
    ("px_dssim05_b12", "dssim05_b12"),
    ("px_dssim08_b12", "dssim08_b12"),
]

# 指標 -> (越高越好, 噪音底)   噪音底來源：紀錄/_ctx.md（n=2 單對，是差值的一次抽樣不是 sigma）
METRICS = {
    "val/psnr": (True, 0.0325),
    "val/ssim": (True, 0.00091),
    "val/lpips": (False, 0.00200),
    "val/texratio": (True, 0.00314),
}


def final(run, tag):
    evs = sorted(glob.glob(f"outputs/{run}/**/events.out.tfevents.*", recursive=True))
    if not evs:
        return None
    a = ea.EventAccumulator(evs[-1], size_guidance={ea.SCALARS: 0})
    a.Reload()
    if tag not in a.Tags()["scalars"]:
        return None
    return a.Scalars(tag)[-1].value


def spearman(a, b):
    n = len(a)
    if n < 3:
        return float("nan")

    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, i in enumerate(o):
            r[i] = pos + 1
        return r

    ra, rb = rank(a), rank(b)
    m = (n + 1) / 2
    num = sum((x - m) * (y - m) for x, y in zip(ra, rb))
    da = sum((x - m) ** 2 for x in ra) ** 0.5
    db = sum((y - m) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    missing = [p for p, _ in PAIRS if final(p, "val/psnr") is None]
    if missing:
        print(f"⚠ 缺這些 proxy 跑次，結果不完整: {', '.join(missing)}\n")

    verdicts = []
    for tag, (higher, floor) in METRICS.items():
        rows = []
        for px, full in PAIRS:
            vp, vf = final(px, tag), final(full, tag)
            if vp is None or vf is None:
                continue
            rows.append((px.replace("px_", "").replace("_b12", ""), vp, vf))
        if len(rows) < 3:
            print(f"--- {tag}: 資料不足 ---\n")
            continue

        name = tag.split("/")[1]
        sgn = 1 if higher else -1
        rho = spearman([sgn * r[1] for r in rows], [sgn * r[2] for r in rows])
        rows_by_full = sorted(rows, key=lambda r: -sgn * r[2])

        print(f"===== {name}   Spearman rho = {rho:.3f} =====")
        print(f"{'臂':>14} {'proxy':>9} {'全長':>9} {'名次(proxy/全長)':>18}")
        order_px = sorted(rows, key=lambda r: -sgn * r[1])
        for r in rows_by_full:
            i_px = order_px.index(r) + 1
            i_f = rows_by_full.index(r) + 1
            mark = "" if i_px == i_f else "  <- 名次不同"
            print(f"{r[0]:>14} {r[1]:>9.4f} {r[2]:>9.4f} {i_px:>10}/{i_f}{mark}")

        # 逐對：正負號對不對、proxy 分不分得開
        print(f"\n  {'配對':>28} {'全長差':>9} {'proxy差':>9}  判定")
        ok = bad = blind = 0
        for (a, b) in itertools.combinations(rows_by_full, 2):
            df = sgn * (a[2] - b[2])
            dp = sgn * (a[1] - b[1])
            if abs(dp) < floor:
                v, blind = "分不開（低於噪音底）", blind + 1
            elif dp * df > 0:
                v, ok = "✅ 方向對且分得開", ok + 1
            else:
                v, bad = "⛔ 方向相反", bad + 1
            print(f"  {a[0]+' vs '+b[0]:>28} {df:>+9.4f} {dp:>+9.4f}  {v}")
        print(f"\n  對 {ok} / 反 {bad} / 分不開 {blind}     "
              f"（噪音底 {floor}）\n")
        verdicts.append((name, rho, ok, bad, blind))

    if not verdicts:
        sys.exit(0)
    print("=" * 60)
    print(f"{'指標':>10} {'rho':>7} {'對':>4} {'反':>4} {'分不開':>7}")
    for n, r, o, b, bl in verdicts:
        print(f"{n:>10} {r:>7.3f} {o:>4} {b:>4} {bl:>7}")
    tot_bad = sum(v[3] for v in verdicts)
    tot_blind = sum(v[4] for v in verdicts)
    print()
    if tot_bad == 0 and tot_blind == 0:
        print("✅ 判定：proxy 成立。所有配對方向正確且分得開 => 之後篩選一律用 proxy（9.6h -> 1.9h）")
    elif tot_bad == 0:
        print(f"🟨 判定：方向全對，但有 {tot_blind} 對分不開 => proxy 只能篩大效應，"
              "小效應仍需全長。看上表哪個指標分得開（可能不是 PSNR）")
    else:
        print(f"⛔ 判定：有 {tot_bad} 對方向相反 => 時間壓縮改變了排名，不可用作篩子。"
              "檢查是哪一類機制翻轉（排程/先驗/loss 權重），那決定下一步。")
    print("\n⚠ 還要跑 `python tools/audit_all.py` 看最差10%與建築低頻，並看圖。")


if __name__ == "__main__":
    main()
