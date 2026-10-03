#!/usr/bin/env python
r"""lab 主機的遠端操作（只有 Jupyter、沒有 ssh）。在**本機**執行；token 只用環境變數 JTOK 帶，**不寫檔、不進 commit**。

用法: JTOK=... python scripts/setup/lab.py <子指令> ...      （`-h`／`help` 不需要 JTOK）

## 最常用（2026-10-03）
```
同步程式碼   JTOK=... bash scripts/setup/lab_pull.sh [--no-build]     本機 commit+push 後一律用這個（取代逐檔 put）
看佇列       lab.py q                    = lab 上的 python3 tools/queue_status.py（跑中的槽／待跑／最近完成）
排一個任務   lab.py run "cd gs && printf '%s\n' '# 標籤' '<佇列行>' >> scripts/queue.txt"
             （佇列行的寫法：bash scripts/runner.sh --help；臂／模式：bash scripts/lab/task_cmp.sh --help）
看某支 log   lab.py run "cd gs && tail -n 40 logs/<檔名>"
對比家族     lab.py run "cd gs && python tools/lab_cmp_report.py --blk 6"
```

## 子指令
```
run  <cmd>                 短指令（預設逾時 180 s，`T=600` 加長）；工作目錄＝ROOT（hdd/11213），repo 在 gs/
                           輸出已去掉終端機的回顯、提示字元與控制碼，只留指令本身的輸出；結尾印 [rc=N]
q                          看 lab 佇列（同上「看佇列」）
bg   <name> <cmd>          背景執行（setsid nohup），log 寫到 <ROOT>/labrun/<name>.log
log  <name> [行數]         讀 bg 的 log 尾部（contents API）
ps                         bg 工作還在不在（labrun/*.pid）
ls   <相對路徑>            列目錄（相對 ROOT；contents API）
put  <本機檔> <遠端相對路徑>  上傳單一小檔（<10MB；相對 ROOT，repo 裡的檔要自己加 gs/）
                           ⚠ 就地覆寫（inode 不變）=> 不可拿來改正在執行的腳本；程式碼一律走 lab_pull.sh
ckpts <run>                某個跑次有哪些 block / step / 大小（run 要帶 lab/，例：lab/cs60_conic）
get  <run> <block> <step> [本機目錄]   抓特定 ckpt（含同名 PLY）
pull [run|*]               把 lab 上的 ckpt 抓回 outputs/lab/（跳過 aborted 與已存在的；run 不帶 lab/）
getfile <repo相對路徑> [本機路徑]       抓任意單一檔案（相對 ROOT/gs）
```
⚠ 路徑基準：`run`／`bg`／`ls`／`put` 相對 ROOT（hdd/11213）；`get`／`getfile`／`ckpts` 相對 ROOT/gs。
  少加 `gs/` 拿到的是 **HTTP 500**（父目錄不存在），不是「找不到檔案」。
⚠ **web view 要 ckpt，不是 PLY** —— `-xyz_rgb.ply` 只有 x/y/z＋法線＋RGB，沒有 opacity/scale/rotation/SH。
⚠ `get`／`getfile` 走 `/files/` 端點（原始位元組，先寫 .part、比對大小才改名）；下行實測約 10 MiB/s。
⚠ lab 狀態由使用者自己看；Claude 不輪詢、不開監控（使用者 2026-09-14）。

## 為什麼短指令走 websocket、長工作走背景
`run` 走 Jupyter 終端機 websocket：對短指令沒問題，但下載／編譯／訓練這種幾十分鐘的工作綁在 websocket 上，
斷線就沒了 => 那些用 `bg`（或排進 runner 佇列）。終端機清理（DELETE）逾時不視為失敗（2026-09-12 實測，指令其實已成功）。
"""
import base64
import json
import os
import re
import shlex
import sys
import time
import uuid

BASE = os.environ.get("LABURL", "http://140.119.164.19:8888")
ROOT = os.environ.get("LABROOT", "hdd/11213")
REPO = os.environ.get("LABREPO", "gs")   # ROOT 底下的 repo 目錄
TOK = os.environ.get("JTOK", "")


def sess():
    import requests
    s = requests.Session()
    s.headers["Authorization"] = f"token {TOK}"
    return s


# 終端機的控制碼：CSI（顏色、bracketed paste 的 ESC[?2004h…）、OSC（視窗標題 ESC]0;…BEL）、其他兩字元序列
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Za-z0-9]|\x1b[=>78]")


def _clean(txt):
    """去掉控制碼；同一行裡被 \r 覆蓋的進度列只留最後一段。"""
    out = []
    for ln in _ANSI.sub("", txt.replace("\r\n", "\n")).split("\n"):
        if "\r" in ln:
            parts = [p for p in ln.split("\r") if p.strip()]
            ln = parts[-1] if parts else ""
        out.append(ln)
    return "\n".join(out)


def sh(cmd, timeout=180, stream=True):
    """在 lab 的終端機執行 cmd，回傳 (rc, 輸出)。

    ⚠ 2026-10-03：以前把終端機原樣印出 —— 每一行指令的回顯、`(base) root@…#` 提示字元、
      bracketed paste 與視窗標題的控制碼全混在輸出裡（lab_pull.sh 的回報因此很難讀）。
      現在：先關回顯與提示字元（並把終端機寬度設成 1000、關掉 pager —— 否則 git 經 less 輸出時，
      寬字元碰到第 80 欄會被補空白），再用「開始／結束」兩個標記夾住指令，只取中間那段，並去掉控制碼。
      標記由 `$S` 組出來，所以終端機回顯的那行（字面上是 `$S`）不會被誤認成標記。"""
    import websocket
    s = sess()
    name = s.post(f"{BASE}/api/terminals", timeout=60).json()["name"]
    tag = "__L_" + uuid.uuid4().hex[:8]
    beg = re.compile(re.escape(tag) + r"_B\r?\n")
    end = re.compile(re.escape(tag) + r"_E(\d+)")
    buf, start, shown, done = "", None, 0, False
    try:
        ws = websocket.create_connection(
            f"{BASE.replace('http://', 'ws://')}/terminals/websocket/{name}",
            header=[f"Authorization: token {TOK}"], timeout=60)
        time.sleep(1.0)
        ws.send(json.dumps(["stdin", "stty -echo cols 1000 2>/dev/null; export PS1= PS2= PROMPT_COMMAND= PAGER=cat GIT_PAGER=cat; "
                                     "bind 'set enable-bracketed-paste off' 2>/dev/null; "
                                     f"S={tag}\n"]))
        time.sleep(0.3)
        ws.send(json.dumps(["stdin", 'printf "%s_B\\n" "$S"\n']))
        ws.send(json.dumps(["stdin", f"cd {shlex.quote(ROOT)} 2>/dev/null; {cmd}\n"]))
        ws.send(json.dumps(["stdin", 'echo "${S}_E$?"\n']))
        t0 = time.time(); ws.settimeout(5)
        while time.time() - t0 < timeout:
            try:
                k, p = json.loads(ws.recv())[:2]
            except Exception:
                continue
            if k != "stdout":
                continue
            buf += p
            if start is None:
                m = beg.search(buf)
                if not m:
                    continue
                start = m.end()
            seg = buf[start:]
            me = end.search(seg)
            body = seg[:me.start()] if me else seg[:seg.rfind("\n") + 1]
            lines = _clean(body).split("\n")
            if not me:
                lines = lines[:-1]              # 最後一段還沒換行 => 先不印
            elif lines and lines[-1] == "":
                lines = lines[:-1]
            if stream:
                for ln in lines[shown:]:
                    print(ln, flush=True)
            shown = len(lines)
            if me:
                done = True
                break
        ws.close()
    finally:
        try:
            s.delete(f"{BASE}/api/terminals/{name}", timeout=60)
        except Exception:
            pass          # ⚠ 清理失敗不該讓已成功的指令變成例外（原本的坑）
    if start is None:
        return 1, f"⚠ 沒等到開始標記（終端機 {timeout}s 內沒有回應）"
    seg = buf[start:]
    me = end.search(seg)
    out = _clean(seg[:me.start()] if me else seg).rstrip("\n")
    if not done:
        msg = f"⚠ 逾時（{timeout}s）：沒等到結束標記 —— 指令可能還在 lab 上跑；要等久一點用 T=<秒>"
        if stream:
            print(msg, flush=True)
        return 124, out
    return int(me.group(1)), out


def contents(path):
    s = sess()
    r = s.get(f"{BASE}/api/contents/{ROOT}/{path}", timeout=180,
              params={"content": "1"})
    r.raise_for_status()
    return r.json()


def listdir(path):
    """列目錄條目。⚠ Jupyter contents API 有時回 `{"content": [...]}`、有時直接回 list
    —— 2026-09-13 踩過兩次（佇列讀取、ckpt 列表），而 `.get` 打在 list 上就是 AttributeError。
    所以一律經過這個正規化，不要直接用 contents() 的回傳值取 content。"""
    d = contents(path)
    if isinstance(d, list):
        return d
    c = d.get("content")
    return c if isinstance(c, list) else []


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if not TOK:
        raise SystemExit("⛔ 需要環境變數 JTOK（只能放在單一指令前面，例：JTOK=... python scripts/setup/lab.py q）")
    c = sys.argv[1]
    if c == "run":
        rc, _ = sh(" ".join(sys.argv[2:]), timeout=int(os.environ.get("T", "180")))
        print(f"[rc={rc}]")
        return 0 if rc == 0 else 1
    if c == "q":
        rc, _ = sh(f"cd {shlex.quote(REPO)} && python3 tools/queue_status.py", timeout=int(os.environ.get("T", "120")))
        return 0 if rc == 0 else 1
    if c == "bg":
        name, cmd = sys.argv[2], " ".join(sys.argv[3:])
        # 寫成腳本再跑：避免多層引號地獄；log 與 pid 都放 labrun/
        script = f"labrun/{name}.sh"
        body = "#!/usr/bin/env bash\nset -u\n" + cmd + "\n"
        s = sess()
        s.put(f"{BASE}/api/contents/{ROOT}/labrun", timeout=60,
              json={"type": "directory"})
        s.put(f"{BASE}/api/contents/{ROOT}/{script}", timeout=120,
              json={"type": "file", "format": "text", "content": body}).raise_for_status()
        # ⚠ 不要再 cd：sh() 已經 `cd ROOT`。2026-09-12 多 cd 一層導致 `&&` 短路，
        #   背景工作**根本沒啟動**，而 pid 檔還是被寫出來（$! 取到舊的）=> 看起來正常。
        # ⚠⚠ 2026-09-13：**不可以用 `$!` 判死活。** `setsid` 若不是行程組長會先 fork 再讓
        #   父行程立刻退出 ⇒ `$!` 抓到的是那個已經結束的包裝行程 ⇒ 這裡固定印
        #   「⛔ 行程沒起來」，而真正的工作**活得好好的**（2026-09-13 因此誤殺並重發一次）。
        #   正解：按**腳本名**找真正的行程，順便把真 pid 寫回 pid 檔（`ps` 子指令要用）。
        rc, out = sh(f"setsid nohup bash {script} > labrun/{name}.log 2>&1 < /dev/null & "
                     f"sleep 2; "
                     f"P=$(ps -eo pid,cmd | grep 'bash labrun/{name}.sh' | grep -v grep "
                     f"| awk '{{print $1}}' | head -1); "
                     f"if [ -n \"$P\" ]; then echo $P > labrun/{name}.pid; "
                     f"echo \"確認：行程活著 pid=$P\"; "
                     f"else echo '⛔ 行程沒起來'; fi", timeout=120)
        print(f"\n[已背景啟動 {name}  rc={rc}]  看進度：lab.py log {name}")
        return 0
    if c == "log":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        try:
            d = contents(f"labrun/{sys.argv[2]}.log")
        except Exception as e:
            print(f"讀不到 log：{e}")
            return 1
        txt = d.get("content") or ""
        if d.get("format") == "base64":
            txt = base64.b64decode(txt).decode("utf-8", "replace")
        lines = txt.replace("\r\n", "\n").split("\n")
        print("\n".join(lines[-n:]))
        return 0
    if c == "ps":
        rc, _ = sh("for p in labrun/*.pid; do [ -f \"$p\" ] || continue; "
                   "n=$(basename $p .pid); q=$(cat $p); "
                   "if kill -0 $q 2>/dev/null; then echo \"  執行中 $n (pid $q)\"; "
                   "else echo \"  已結束 $n\"; fi; done", timeout=120)
        return 0
    if c == "ls":
        for x in sorted(listdir(sys.argv[2] if len(sys.argv) > 2 else ""),
                        key=lambda y: (y["type"], y["name"])):
            print(f"  {x['type']:>9}  {x.get('size') or '':>12}  {x['name']}")
        return 0
    if c in ("ckpts", "get", "getfile", "pull"):
        import re as _re
        s_ = sess()

        def _fetch(rel, dest, expect=None):
            """走 /files/ 端點抓原始位元組（不經 base64），邊下載邊報進度。

            ⚠⚠ 2026-09-13：先寫 `<dest>.part`，**比對大小通過才改名**。
              原本直接寫最終檔名 => 傳輸被中斷（我 kill 了 pull）就把半個檔案留在
              最終檔名下，而且事後沒有任何東西比對大小 => 載入時才炸
              `PytorchStreamReader failed reading zip archive`。實際發生過 1 次
              （772.0 MiB vs 應有的 1,725.8 MiB，缺 55.3%）。這是本專案一再出現的
              「靜默壞掉」形狀，而這次是我自己造的。
            """
            url = f"{BASE}/files/{ROOT}/{REPO}/{rel}"
            t0, got = time.time(), 0
            os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
            part = dest + ".part"
            try:
                with s_.get(url, stream=True, timeout=1800) as rr:
                    if rr.status_code != 200:
                        print(f"  ⛔ HTTP {rr.status_code}  {rel}")
                        return False
                    if expect is None:
                        cl = rr.headers.get("Content-Length")
                        expect = int(cl) if cl and cl.isdigit() else None
                    with open(part, "wb") as f:
                        nxt = 1 << 26          # 每 64 MiB 才報一次，不要洗版
                        for chunk in rr.iter_content(1 << 22):
                            f.write(chunk); got += len(chunk)
                            if got >= nxt:
                                nxt += 1 << 26
                                el = time.time() - t0
                                pct = f" {100*got/expect:5.1f}%" if expect else ""
                                print(f"\r  {got/2**20:8.0f} MiB{pct}  "
                                      f"{got/2**20/max(el,1e-9):6.2f} MiB/s", end="", flush=True)
            except Exception as e:
                print(f"\r  ⛔ 傳輸中斷：{type(e).__name__} {str(e)[:80]}")
                if os.path.exists(part): os.remove(part)
                return False
            if expect is not None and got != expect:
                print(f"\r  ⛔ 大小不符：收到 {got:,} 應為 {expect:,}（缺 "
                      f"{100*(1-got/max(expect,1)):.1f}%）=> 丟棄，不寫成最終檔名")
                os.remove(part)
                return False
            os.replace(part, dest)
            el = time.time() - t0
            print(f"\r  ✔ {dest}  {got/2**20:.1f} MiB / {el:.1f}s "
                  f"({got/2**20/max(el,1e-9):.2f} MiB/s)")
            return True

        if c == "getfile":
            rel = sys.argv[2]
            dest = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(rel)
            return 0 if _fetch(rel, dest) else 1

        if c == "pull":
            # 把 lab 上**所有**（或指定 run 的）ckpt 抓回 outputs/lab/ 供本機分析。
            # 跳過 `.aborted_*` 殘留，也跳過本機已存在且大小相同的（可續傳、可重跑）。
            import re as _re2
            what = sys.argv[2] if len(sys.argv) > 2 else "*"
            rc, out = sh(f"cd {REPO} && stat -c '@CK %s %n' "
                         f"outputs/lab/{what}/blocks/*/checkpoints/*.ckpt 2>/dev/null || true",
                         timeout=300, stream=False)
            todo, skip_n, skip_b, ab_n = [], 0, 0, 0
            for ln in out.split("\n"):
                m = _re2.match(r"@CK (\d+) (\S+\.ckpt)\s*$", ln.strip())
                if not m:
                    continue
                sz, rel = int(m.group(1)), m.group(2)
                if ".aborted_" in rel:
                    ab_n += 1
                    continue
                dest = rel                      # 遠端路徑就是本機要放的相對路徑
                if os.path.exists(dest) and os.path.getsize(dest) == sz:
                    skip_n += 1; skip_b += sz; continue
                todo.append((rel, dest, sz))
            tot = sum(x[2] for x in todo)
            print(f"要抓 {len(todo)} 個 / {tot/2**30:.2f} GiB"
                  f"（已有 {skip_n} 個跳過 {skip_b/2**30:.2f} GiB；aborted 殘留 {ab_n} 個不抓）")
            ok = 0
            for i, (rel, dest, sz) in enumerate(todo, 1):
                print(f"[{i}/{len(todo)}] {dest}  {sz/2**20:.0f} MiB")
                if _fetch(rel, dest, expect=sz):
                    ok += 1
            print(f"\n完成 {ok}/{len(todo)}")
            return 0 if ok == len(todo) else 1

        run = sys.argv[2]
        base = f"outputs/{run}/blocks"
        # ⚠⚠ 這裡**不能用 contents API 列 checkpoints/** —— 2026-09-13 實測它對
        #   `.../block_6/checkpoints` 回傳 **HTTP 200 + 空 list**（父目錄卻看得到它是
        #   directory）=> 會靜默變成「沒有 ckpt」。改走終端機，用加標記的輸出再解析。
        #   下載仍走 /files/（那條是好的，實測 10.18 MiB/s）。
        rc, out = sh(f"cd {REPO} && stat -c '@CK %s %n' "
                     f"{base}/*/checkpoints/*.ckpt 2>/dev/null || true",
                     timeout=180, stream=False)
        found = {}
        for ln in out.split("\n"):
            m = _re.match(r"@CK (\d+) (\S+\.ckpt)\s*$", ln.strip())
            if not m:
                continue
            sz, path = int(m.group(1)), m.group(2)
            mb = _re.search(r"/(block_[^/]+)/checkpoints/(.*step=(\d+)\.ckpt)$", path)
            if not mb or ".aborted_" in mb.group(1):
                continue
            found[(mb.group(1), int(mb.group(3)))] = (mb.group(2), sz)
        if c == "ckpts":
            if not found:
                print(f"（{run} 沒有 ckpt；rc={rc}）")
                return 1
            print(f"{run}：")
            for (b, st), (nm, sz) in sorted(found.items()):
                print(f"  {b:>22}  step={st:<7} {sz/2**20:8.1f} MiB  {nm}")
            print("\n抓其中一個：lab.py get <run> <block編號> <step> [本機目錄]")
            print("⚠ web view 要用 **ckpt**；同名的 -xyz_rgb.ply 只有位置/法線/顏色，"
                  "沒有 opacity/scale/SH，進 viewer 不會有反應")
            return 0

        blk, step = sys.argv[3], int(sys.argv[4])
        outdir = sys.argv[5] if len(sys.argv) > 5 else f"outputs/{run}/blocks/block_{blk}/checkpoints"
        key = (f"block_{blk}", step)
        if key not in found:
            print(f"⛔ {run} 沒有 block_{blk} 的 step={step}。現有的：")
            for (b, st) in sorted(found):
                print(f"    {b}  step={st}")
            return 1
        nm, sz = found[key]
        print(f"抓 {run} / block_{blk} / step={step}   {sz/2**20:.1f} MiB -> {outdir}/")
        ok = _fetch(f"{base}/block_{blk}/checkpoints/{nm}", os.path.join(outdir, nm))
        # 同名的 PLY 也抓一份。⚠⚠ **它不能當 web view 用** —— 2026-09-13 使用者指出
        #   「.ply 進 web view 沒反應，一直以來都是給 ckpt」，查證屬實：那是
        #   `-xyz_rgb.ply`，只有 x/y/z + 法線 + RGB，**沒有 opacity/scale/rotation/SH**
        #   => 純點雲，不是 Gaussian splat 模型，只適合幾何檢視。web view 一律用 ckpt。
        ply = nm.replace(".ckpt", "-xyz_rgb.ply")
        _fetch(f"{base}/block_{blk}/checkpoints/{ply}", os.path.join(outdir, ply))
        return 0 if ok else 1

    if c == "put":
        lp, rp = sys.argv[2], sys.argv[3]
        raw = open(lp, "rb").read()
        s = sess()
        r = s.put(f"{BASE}/api/contents/{ROOT}/{rp}", timeout=600,
                  json={"type": "file", "format": "base64",
                        "content": base64.b64encode(raw).decode()})
        r.raise_for_status()
        print(f"  上傳 {len(raw):,} bytes -> {rp}")
        return 0
    raise SystemExit(f"⛔ 未知子指令 {c}（看全部：python scripts/setup/lab.py --help）")


if __name__ == "__main__":
    sys.exit(main())
