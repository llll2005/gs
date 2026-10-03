#!/usr/bin/env python
"""佇列裡還有什麼、現在在跑什麼、剛跑完的結果如何 —— 一個指令看完。

為什麼需要：`scripts/queue.txt` 是「註解區塊 + 任務行」交錯，直接 `cat` 很難讀，
而且 runner 是用**緊鄰任務行上方的註解區塊**當標籤 —— 區塊一旦錯位（搬動任務時很容易），
台帳就會把 A 的標題掛在 B 的跑次上（2026-09-05 發生過兩次，一次顯示「(未命名)」、
一次把 O0 的臂標成「參數冗餘稽核」）。本工具把「標籤 ↔ 指令」的對應**明確印出來**，
錯位就一眼看得到。

用法: python3 tools/queue_status.py   （＝ bash scripts/runner.sh status；lab 從本機看：lab.py q）
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
        # ⚠ 2026-09-13：原本只比對 `scripts/task_`，而 **lab 的任務是 `scripts/lab/task_`**
        #   => 在 lab 上「現在在跑」永遠印「（沒有 task 腳本在執行）」，即使三個槽都滿的。
        #   典型的「顯示正常但沒作用」。改成比對 `task_` 並自己還原完整路徑。
        if "task_" not in ln or "queue_status" in ln:
            continue
        f = ln.strip().split()
        if f[1:2] != ["bash"]:
            continue
        # ⚠ 本機的 `scripts/_noedit.sh` 加固會把**整支腳本的原始碼**塞進 cmdline
        #   （`exec bash -c "$code" "$s" "$@"`）=> 掃 token 會撈到腳本**內文**裡的字串
        #   （實測撈到 `用法: task_cmp.sh <block_id> <arm>` 這段錯誤訊息）。
        #   真正的 $0 排在程式碼 blob **之後** => 取**最後一個**看起來像路徑的 token。
        cands = [t for t in f[1:] if "task_" in t and t.endswith(".sh") and "/" in t]
        if not cands:
            continue
        path = cands[-1]
        # 帶上後面**全部**的位置參數（`task_cmp.sh 6 cb50` 的「6 cb50」＝塊號與臂名）。
        # ⚠ 2026-09-13：曾經截到 2 個 => 3 個參數的任務（`task_step_timing.sh 21920 lab/cs_base
        #   lab/cs_trimvpc`）尾端比對不上，佇列行就不會被標成「執行中」。上限 8 只防異常長度。
        idx = len(f) - 1 - f[::-1].index(path)
        args = [t for t in f[idx + 1:] if not t.startswith("-")][:8]
        hits.append((int(f[0]), " ".join([path] + args)))
    # ⚠ 任務腳本裡的 `{ ... } | tee` 管線會 fork 出**命令列完全相同**的子 shell
    #   => 同一個任務會出現兩次。以「路徑+參數」去重，保留跑最久的那個。
    best = {}
    for et, key in hits:
        best[key] = max(et, best.get(key, -1))
    return [(et, key) for key, et in best.items()]


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TS = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d) \| ")


def _age(ts):
    """runner.log 的時間戳（mm-dd HH:MM:SS，沒有年份）到現在幾秒。"""
    import datetime as dt
    now = dt.datetime.now()
    t = dt.datetime.strptime(f"{now.year}-{ts}", "%Y-%m-%d %H:%M:%S")
    if t > now:                                    # 跨年
        t = t.replace(year=now.year - 1)
    return int((now - t).total_seconds())


def runner_running():
    """runner.log 裡最後一次「runner 啟動」之後、還沒有 DONE/FAIL 的 START。

    ⚠ 2026-10-03：平行模式**啟動前就把該行從佇列移除**（避免兩槽搶同一行）
      => 跑中的任務不在 queue.txt 裡，只看佇列會以為「沒在跑」。台帳才是準的。
    回傳 [(槽或 None, 標籤, 指令, 已跑秒數, slot log 路徑或 None)]"""
    if not os.path.exists(LOG):
        return []
    lines = open(LOG, errors="ignore").read().split("\n")
    k0 = max([i for i, ln in enumerate(lines) if "runner 啟動" in ln] or [0])
    run = {}                                       # key = 槽（序列模式用 None）
    for i in range(k0, len(lines)):
        ln = lines[i]
        m = TS.match(ln)
        if not m:
            continue
        body = ln[m.end():]
        ms = re.match(r"▶ START\s+(?:\[槽 (\d+)\] )?(.*)$", body)
        if ms:
            slot = ms.group(1)
            cmd, slog = "", None
            if i + 1 < len(lines):
                nxt = TS.sub("", lines[i + 1]).strip()
                mo = re.match(r"(.*?)\s+（輸出 -> (\S+)）$", nxt)
                cmd, slog = (mo.group(1), mo.group(2)) if mo else (nxt, None)
            run[slot] = (slot, ms.group(2).strip(), cmd, _age(m.group(1)), slog)
            continue
        md = re.match(r"[✔✘] (?:DONE|FAIL)\s+(?:\[槽 (\d+)\] )?", body)
        if md:
            run.pop(md.group(1), None)
    return sorted(run.values(), key=lambda r: (r[0] or ""))


def last_progress(path, nbytes=65536):
    """slot log 尾端最後一個像進度的片段（tqdm 的 n/N、step、val）。"""
    try:
        with open(os.path.join(ROOT, path), "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - nbytes))
            tail = f.read().decode("utf-8", "ignore")
    except Exception:
        return ""
    segs = [ANSI.sub("", x).strip() for x in re.split(r"[\r\n]", tail)]
    segs = [x for x in segs if x and "pkg_resources" not in x and "declare_namespace" not in x]
    for x in reversed(segs):
        if re.search(r"\d+/\d+ \[|step|val|PSNR|Epoch", x):
            return x[-120:]
    return segs[-1][-120:] if segs else ""


def runner_alive():
    try:
        ps = subprocess.run(["ps", "-eo", "pid,etimes,cmd"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for ln in ps.splitlines():
        if "scripts/runner.sh" in ln and "grep" not in ln and "status" not in ln and "--help" not in ln:
            f = ln.split()
            return f"pid {f[0]}，已跑 {int(f[1])//3600}h{int(f[1])%3600//60:02d}m"
    return ""


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
    rr = runner_running()
    alive = runner_alive()

    print(f"\n=== runner ===")
    print(f"  {'執行中（' + alive + '）' if alive else ('⚠ 沒有在跑 —— 佇列不會前進' if alive == '' else '（無法判斷）')}")
    print(f"\n=== 現在在跑 ===")
    if rr and alive == "":
        print("  ⚠ runner 已不在，下面是它最後一次啟動後沒收到 DONE/FAIL 的任務 —— 可能已死、或是孤兒行程仍在跑")
    if rr:
        for slot, lbl, cmd, age, slog in rr:
            print(f"  {'[槽 ' + slot + '] ' if slot else ''}{lbl[:90]}")
            print(f"      $ {cmd}   已跑 {age//3600}h{age%3600//60:02d}m")
            if slog:
                pg = last_progress(slog)
                if pg:
                    print(f"      … {pg}")
    elif run:
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
        # ⚠ 2026-09-13：不可只比腳本路徑 —— 同一支 task_cmp3.sh 在跑一臂時，佇列裡**所有**
        #   用 task_cmp3.sh 的行都會被標成「執行中」（實際發生：四個待跑的複製臂全被誤標）。
        #   也不可用子字串：`task_cmp3.sh 6 cb25` 是 `task_cmp3.sh 6 cb25cost` 的子字串。
        #   正解：佇列行的**尾端 token** 與「跑中的 路徑+參數」逐 token 完全相同。
        ct = cmd.split()
        mark = ("▶ 執行中" if any(len(pt := p.split()) <= len(ct) and ct[-len(pt):] == pt
                                   for _, p in run) else f"  {k}.")
        tag = " [solo 獨佔]" if cmd.startswith("[solo]") else (" [cpu]" if cmd.startswith("[cpu]") else "")
        print(f"{mark} {title[:74]}{tag}")
        print(f"      $ {cmd}" + ("" if nblk else "   ⚠ 這行**沒有註解區塊** => 台帳會顯示上一個區塊的標題"))

    print(f"\n=== 最近完成 ===")
    for ln in recent_ledger():
        print("  " + ln[:118])

    print("""
=== 指令 ===
  bash scripts/runner.sh status        本畫面（lab 上從本機看：JTOK=... python scripts/setup/lab.py q）
  bash scripts/runner.sh --help        佇列行的寫法（[cpu]／[solo]／標籤）
  $EDITOR scripts/queue.txt            直接編輯（跑中也可以改；平行模式下跑中的那行已不在檔裡）
  tail -f logs/runner.log              runner 的事件；各槽輸出在 logs/runner.slot<N>.log
  touch scripts/queue.stop             做完目前的就停
""")



if __name__ == "__main__":
    sys.exit(main())
