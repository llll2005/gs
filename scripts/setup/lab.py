#!/usr/bin/env python
"""lab 主機的遠端操作（只有 Jupyter、沒有 ssh）：短指令直跑、長工作背景化 + 輪詢 log。

## 為什麼不直接用 deploy.py 的 sh()

`deploy.py` 的 `sh()` 走 Jupyter 終端機 websocket，對**短**指令沒問題，但：
```
① finally 裡 `s.delete(terminal, timeout=30)` 逾時會把整個呼叫變成例外
   —— 指令其實已經成功了（2026-09-12 實測，check 就是這樣掛的）
② 下載 / 編譯 / 訓練這種幾十分鐘的工作綁在 websocket 上，斷線就沒了
```
本工具：短指令用 websocket（並把 DELETE 變成非致命），長工作用
`setsid nohup ... > log 2>&1 &` 丟到背景，再用 **contents API** 讀 log
（不吃 websocket，斷線也沒差）。

## 子指令

```
run  <cmd>                 短指令（預設 180s）
bg   <name> <cmd>          背景執行，log 寫到 <root>/labrun/<name>.log
log  <name> [行數]         讀 log 尾部（contents API）
ps                         看背景工作還在不在（用 labrun/*.pid）
ls   <相對路徑>            列目錄（contents API）
put  <本機檔> <遠端相對路徑>  上傳單一小檔（contents API，<10MB）
```
token 從環境變數 `JTOK` 讀，**不寫檔、不進 commit**。

用法: JTOK=... python scripts/setup/lab.py run "nvidia-smi -L"
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
TOK = os.environ.get("JTOK", "")


def sess():
    import requests
    s = requests.Session()
    s.headers["Authorization"] = f"token {TOK}"
    return s


def sh(cmd, timeout=180, stream=True):
    import websocket
    s = sess()
    name = s.post(f"{BASE}/api/terminals", timeout=60).json()["name"]
    tag = "__L_" + uuid.uuid4().hex[:8]
    want = re.compile(re.escape(tag) + r"_(\d+)")
    buf = ""
    try:
        ws = websocket.create_connection(
            f"{BASE.replace('http://', 'ws://')}/terminals/websocket/{name}",
            header=[f"Authorization: token {TOK}"], timeout=60)
        time.sleep(1.0)
        ws.send(json.dumps(["stdin", f"S={tag}\n"])); time.sleep(0.3)
        ws.send(json.dumps(["stdin", f"cd {shlex.quote(ROOT)} 2>/dev/null; {cmd}\n"]))
        ws.send(json.dumps(["stdin", 'echo "${S}_$?"\n']))
        t0 = time.time(); ws.settimeout(5)
        while time.time() - t0 < timeout:
            try:
                k, p = json.loads(ws.recv())[:2]
            except Exception:
                continue
            if k != "stdout":
                continue
            buf += p
            if stream:
                sys.stdout.write(p); sys.stdout.flush()
            if want.search(buf):
                break
        ws.close()
    finally:
        try:
            s.delete(f"{BASE}/api/terminals/{name}", timeout=60)
        except Exception:
            pass          # ⚠ 清理失敗不該讓已成功的指令變成例外（原本的坑）
    txt = buf.replace("\r\n", "\n")
    m = want.search(txt)
    return (int(m.group(1)) if m else 1), (txt[:m.start()] if m else txt)


def contents(path):
    s = sess()
    r = s.get(f"{BASE}/api/contents/{ROOT}/{path}", timeout=180,
              params={"content": "1"})
    r.raise_for_status()
    return r.json()


def main():
    if not TOK:
        raise SystemExit("需要環境變數 JTOK")
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    c = sys.argv[1]
    if c == "run":
        rc, _ = sh(" ".join(sys.argv[2:]), timeout=int(os.environ.get("T", "180")))
        print(f"\n[rc={rc}]")
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
        rc, out = sh(f"setsid nohup bash {script} > labrun/{name}.log 2>&1 < /dev/null & "
                     f"echo $! > labrun/{name}.pid; sleep 2; "
                     f"echo pid=$(cat labrun/{name}.pid); "
                     f"kill -0 $(cat labrun/{name}.pid) 2>/dev/null && echo '確認：行程活著' "
                     f"|| echo '⛔ 行程沒起來'", timeout=120)
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
        d = contents(sys.argv[2] if len(sys.argv) > 2 else "")
        for x in sorted(d.get("content", []), key=lambda y: (y["type"], y["name"])):
            print(f"  {x['type']:>9}  {x.get('size') or '':>12}  {x['name']}")
        return 0
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
    raise SystemExit(f"未知子指令 {c}")


if __name__ == "__main__":
    sys.exit(main())
