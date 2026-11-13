"""★ 2026-11-13 跑次時間總表（使用者：「最近量的各部分速度、跑次的結果資料有沒有整理到紀錄」）。
從 logs/quad_progress.log（internal/callbacks.py 與 runner 寫的 START／DONE／DIED）依序配對：
  runner TASK START -> gs START（跑次、塊）-> gs DONE／DIED（步數、N、最後 it/s、VRAM 峰值實佔）-> runner TASK DONE（整個任務秒數，含啟動與圖表資料包）
另外讀該跑次 tfevents 的 train/loss 首尾＝純訓練時間（與 eval_arms_lab.sh 的「時間」同一個量法）。
⚠ 只在單槽 runner（一次一個任務）期間配對可靠；多槽並行時 gs START／DONE 會交錯 => 用 --since 限定區間。
用法：python tools/run_ledger.py --since "11-12 08:30" [--until "11-13 23:59"] [--log logs/quad_progress.log] [--md]
"""
import argparse
import glob
import os
import re


def pure_minutes(run, block):
    fs = sorted(glob.glob(f"outputs/lab/{run}/blocks/block_{block}/lightning_logs/*/events*"))
    if not fs:
        return None
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
        a = ea.EventAccumulator(fs[-1], size_guidance={ea.SCALARS: 0})
        a.Reload()
        ev = a.Scalars("train/loss")
        return (ev[-1].wall_time - ev[0].wall_time) / 60 if len(ev) >= 2 else None
    except Exception:
        return None


def key(ts):
    m = re.match(r"(\d\d)-(\d\d) (\d\d):(\d\d)", ts)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/quad_progress.log")
    ap.add_argument("--since", required=True, help="MM-DD HH:MM（UTC，與 log 同）")
    ap.add_argument("--until", default="12-31 23:59")
    ap.add_argument("--md", action="store_true", help="輸出 markdown 表")
    a = ap.parse_args()
    lo, hi = key(a.since), key(a.until)
    rows, cur = [], None
    for line in open(a.log, encoding="utf-8", errors="replace"):
        ts = line[:11]
        if not (lo <= key(ts) <= hi):
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) < 3:
            continue
        who, msg = parts[1], parts[2]
        if who == "gs" and msg.startswith("START"):
            m = re.search(r"outputs/lab/([^/]+)/blocks/block_(\d+)", msg)
            if m:
                cur = dict(t0=ts, run=m.group(1), block=int(m.group(2)), status="?", N="", its="", vram="", task_s="")
        elif who == "gs" and cur and (msg.startswith("DONE") or msg.startswith("DIED")):
            cur["status"] = msg.split()[0]
            m = re.search(r"step=([\d,]+)/([\d,]+)", msg); cur["step"] = m.group(1) if m else ""
            m = re.search(r"N=([\d.]+M)", msg); cur["N"] = m.group(1) if m else ""
            m = re.search(r"([\d.]+)it/s", msg); cur["its"] = m.group(1) if m else ""
            m = re.search(r"峰值實佔([\d.]+)", msg); cur["vram"] = m.group(1) if m else ""
            cur["t1"] = ts
        elif who == "runner" and cur and msg.startswith("TASK") and ("DONE" in msg.split()[1] or "FAIL" in msg.split()[1]):
            m = re.search(r"\((\d+)s\)\s*$", msg.strip())
            if m and "t1" in cur:
                cur["task_s"] = int(m.group(1))
                rows.append(cur)
                cur = None
    for r in rows:
        r["pure"] = pure_minutes(r["run"], r["block"])
    if a.md:
        print("| 開始（UTC） | 跑次 | 塊 | 狀態 | 純訓練 分 ↓ | 整個任務 分 ↓ | 任務 − 純訓練 | 終點 N | 最後 it/s | VRAM 峰值實佔 GB |")
        print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pure = f"{r['pure']:.1f}" if r["pure"] else "—"
        task = f"{r['task_s'] / 60:.1f}" if r["task_s"] != "" else "—"
        over = f"{r['task_s'] / 60 - r['pure']:.1f}" if (r["pure"] and r["task_s"] != "") else "—"
        name = r["run"].replace("_best1maskdistrfed10b4", "")
        if a.md:
            print(f"| {r['t0']} | `{name}` | b{r['block']} | {r['status']} | {pure} | {task} | {over} | {r['N']} | {r['its']} | {r['vram']} |")
        else:
            print(f"{r['t0']}  {name:58} b{r['block']:<2} {r['status']:5} 純訓練 {pure:>5} 分  任務 {task:>5} 分  差 {over:>4}  N {r['N']:>6}  {r['its']:>6} it/s  VRAM {r['vram']}")


if __name__ == "__main__":
    main()
