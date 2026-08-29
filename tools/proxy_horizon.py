#!/usr/bin/env python
"""提早停損的回測：中途的 val 指標能不能排出跟 60k 一樣的名次？

問題（2026-08-25）：單次跑次 9.6 小時，實驗做不完。若「step k 的排名 == 60k 的排名」，
就可以只跑到 k 步當篩子，只有勝出的配方才跑滿。

做法：每個跑次的 tensorboard 都有 ~11 個 epoch 的 val 指標。對每個 epoch e，算
跨跑次的 Spearman rho( metric@e , metric@final )。rho 夠高的最小 e 就是可用的地平線。

⚠ 只納入同世代（GT 修正後）且跑滿 max_steps 的跑次 —— 半途死掉的跑次會讓 rho 灌水。
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

from tensorboard.backend.event_processing import event_accumulator as ea

GT_FIX_TS = 1786529317  # 2026-08-12 10:08:37 UTC，GT 錯位修正的分界


def load_run(run_dir, tags):
    evs = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not evs:
        return None
    # 多個 version_* 時取最後一個（重跑會新增；早期 version 是失敗的嘗試）
    acc = ea.EventAccumulator(evs[-1], size_guidance={ea.SCALARS: 0})
    acc.Reload()
    avail = set(acc.Tags()["scalars"])
    out = {}
    for t in tags:
        if t not in avail:
            return None
        out[t] = [(s.step, s.value) for s in acc.Scalars(t)]
    # 開跑時刻用最早的 event 檔 mtime 不準（會被改寫）；用檔名裡的時戳
    m = re.search(r"tfevents\.(\d+)\.", os.path.basename(evs[-1]))
    out["_t0"] = int(m.group(1)) if m else 0
    out["_ev"] = evs[-1]
    return out


def spearman(a, b):
    n = len(a)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--metric", default="val/psnr")
    ap.add_argument("--min-t0", type=int, default=1786000000,
                    help="unix 時戳下限；GT 修正 = 2026-08-12 10:08:37")
    ap.add_argument("--block", default=None, help="只看名字含此字串的跑次，例如 _b12")
    ap.add_argument("--top", type=float, default=None,
                    help="只看最終值在 (best - top) 以內的跑次，測小效應的解析度")
    ap.add_argument("--higher-better", action="store_true", default=True)
    args = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.outputs, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if args.block and args.block not in name:
            continue
        r = load_run(d, [args.metric])
        if r is None:
            continue
        if r["_t0"] < args.min_t0:
            continue
        pts = r[args.metric]
        if len(pts) < 6:            # 太短的多半是半途死掉
            continue
        runs[name] = pts

    if len(runs) < 4:
        print(f"可用跑次太少 ({len(runs)})，無法回測", file=sys.stderr)
        for k, v in runs.items():
            print(f"  {k}: {len(v)} 點, 最後 step {v[-1][0]}")
        return

    # 只留跑到最常見的最終步數的跑次（過濾半途死掉）
    finals = defaultdict(list)
    for n, p in runs.items():
        finals[p[-1][0]].append(n)
    tgt = max(finals, key=lambda s: len(finals[s]))
    keep = {n: runs[n] for n in finals[tgt]}
    dropped = sorted(set(runs) - set(keep))

    if args.top is not None:
        best = max(p[-1][1] for p in keep.values())
        keep = {n: p for n, p in keep.items() if p[-1][1] >= best - args.top}

    names = sorted(keep, key=lambda n: -keep[n][-1][1])
    n_ep = min(len(keep[n]) for n in names)
    final = [keep[n][-1][1] for n in names]

    print(f"指標 {args.metric}   跑次 {len(names)}   最終步數 {tgt}")
    if dropped:
        print(f"排除（未跑到 {tgt}）: {', '.join(dropped)}")
    if args.top is not None:
        print(f"只看最終值 >= {max(final) - args.top:.3f} 的跑次")
    print()
    print(f"{'epoch':>5} {'step':>7} {'rho vs final':>13} {'名次對了幾個':>13} {'第一名是誰':>22}")
    for e in range(n_ep):
        cur = [keep[n][e][1] for n in names]
        step = keep[names[0]][e][0]
        rho = spearman(cur, final)
        # 「前 k 名集合」是否一致
        top3_cur = set(sorted(names, key=lambda n: -keep[n][e][1])[:3])
        top3_fin = set(names[:3])
        win = max(names, key=lambda n: keep[n][e][1])
        print(f"{e:>5} {step:>7} {rho:>13.3f} {len(top3_cur & top3_fin):>10}/3    {win:>22}")

    print()
    print("跑次（依最終值排序）:")
    for n in names:
        traj = "  ".join(f"{keep[n][e][1]:.2f}" for e in range(n_ep))
        print(f"  {n:>24}  {traj}")


if __name__ == "__main__":
    main()
