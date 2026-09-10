#!/usr/bin/env python
"""佇列裡還有什麼、現在在跑什麼、剛跑完的結果如何 —— 一個指令看完。

為什麼需要：`scripts/queue.txt` 是「註解區塊 + 任務行」交錯，直接 `cat` 很難讀，
而且 runner 是用**緊鄰任務行上方的註解區塊**當標籤 —— 區塊一旦錯位（搬動任務時很容易），
台帳就會把 A 的標題掛在 B 的跑次上（2026-09-05 發生過兩次，一次顯示「(未命名)」、
一次把 O0 的臂標成「參數冗餘稽核」）。本工具把「標籤 ↔ 指令」的對應**明確印出來**，
錯位就一眼看得到。

用法: python tools/queue_status.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "scripts", "queue.txt")
LOG = os.path.join(ROOT, "logs", "runner.log")


def parse_queue():
    """回傳 [(標題, 指令, 註解行數)]；標題取任務行上方連續註解區塊的第一行 ★。"""
    if not os.path.exists(Q):
        return []
    lines = open(Q, errors="ignore").read().split("\n")
    out = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        j = i - 1
        block = []
        while j >= 0 and lines[j].startswith("#"):
            block.append(lines[j])
            j -= 1
        block.reverse()
        title = next((b.lstrip("# ").strip() for b in block if "★" in b), None)
        if title is None:
            title = block[0].lstrip("# ").strip() if block else "（無標題）"
        out.append((title, s, len(block)))
    return out


def running():
    try:
        ps = subprocess.run(["ps", "-eo", "etimes,cmd"], capture_output=True,
                            text=True, timeout=10).stdout
    except Exception:
        return []
    hits = []
    for ln in ps.splitlines():
        if "scripts/task_" in ln and "queue_status" not in ln and ln.strip().split()[1:2] == ["bash"]:
            et = int(ln.strip().split()[0])
            hits.append((et, "scripts/" + ln.split("scripts/")[1].split()[0]))
    return hits


def recent_ledger(n=6):
    if not os.path.exists(LOG):
        return []
    out = []
    for ln in open(LOG, errors="ignore"):
        if "| ✔ DONE" in ln or "| ✘ FAIL" in ln:
            out.append(ln.rstrip())
    return out[-n:]


def main():
    q = parse_queue()
    run = running()

    print(f"\n=== 現在在跑 ===")
    if run:
        for et, path in run:
            print(f"  {path}   已跑 {et//3600}h{et%3600//60:02d}m")
    else:
        print("  （沒有 task 腳本在執行）")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        print(f"  GPU: {gpu}")
    except Exception:
        pass

    print(f"\n=== 佇列（{len(q)} 個待執行；最上面的最先跑）===")
    if not q:
        print("  （空）")
    for k, (title, cmd, nblk) in enumerate(q, 1):
        mark = "▶ 執行中" if any(cmd.endswith(p) for _, p in run) else f"  {k}."
        print(f"{mark} {title[:74]}")
        print(f"      $ {cmd}" + ("" if nblk else "   ⚠ 這行**沒有註解區塊** => 台帳會顯示上一個區塊的標題"))

    print(f"\n=== 最近完成 ===")
    for ln in recent_ledger():
        print("  " + ln[:118])

    print("""
=== 你可以自己看的三個指令 ===
  python tools/queue_status.py     本畫面
  $EDITOR scripts/queue.txt        直接編輯（**跑中也可以改**，只有正在執行的那一行是固定的）
  tail -f logs/runner.log          即時輸出
  cat outputs/<name>/blocks/block_*/train_status.txt    跑動中跑次的進度與最近 val

  touch scripts/queue.stop         做完目前這個就停
""")


if __name__ == "__main__":
    sys.exit(main())
