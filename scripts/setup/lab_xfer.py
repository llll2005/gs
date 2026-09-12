#!/usr/bin/env python
"""把一個目錄搬到 lab 主機（打包 -> 可續傳的切片上傳 -> 遠端重組 -> md5 驗證 -> 解開）。

## 為什麼要這樣搬

lab 主機（140.119.164.19:8888）**只開 Jupyter，port 22 不通、遠端也沒有 rsync**
=> 唯一能傳檔的通道是 Jupyter 的 contents API，而它**沒有 append**
   （每次 PUT 整個取代目標檔）=> 大檔只能切成獨立的 `.partNNNN` 再遠端 `cat` 回去。

實測頻寬（2026-09-12）：
```
我們上傳     單條 0.59 MB/s   8 條平行 1.58 MB/s   <- 瓶頸在我們的上行（約 14 Mbps）
lab 下載     71.4 MB/s                            <- 差 45 倍
```
=> **能讓 lab 自己拉的就不要用推的**。這支只用在「只有我們有」的檔案
   （我方的 COLMAP sparse/、partition/、depth_init/）。
⚠ rsync 幫不上：它快的是「只傳差異」與「可續傳」，不是頻寬；而且遠端沒有 rsync。
   續傳這支自己做（見下）。

## 三個必須驗的東西（少一個就會靜默壞掉）

```
1 片數    remote 片數 == ceil(size/CHUNK)        少一片 cat 出來仍是「看起來能解開」的壞 tar
2 md5     remote md5 == local md5                逐位元一致才算傳到
3 內容    解開後的目錄清單與大小                   確認 tar 的路徑層級對
```
⚠ 只做 `cat` 不驗 md5 是這個專案最常踩的形狀：**不會報錯**。

## 續傳

上傳前先問遠端「哪些 `.partNNNN` 已經在、大小對不對」，只補缺的。
=> 5 小時的上傳中斷後可以接著跑，不用重來。

用法:
  python scripts/setup/lab_xfer.py --src <本機目錄的父層>:<要打包的子目錄...> --name input
  例：--src data/matrix_city/aerial/train/block_all:input --name input
環境變數: JTOK（Jupyter token，**不要寫進檔案**）、JBASE（預設 lab 的位址）
"""
import argparse, base64, hashlib, json, os, subprocess, sys, time
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("JBASE", "http://140.119.164.19:8888")
CHUNK = 16 * 1024 * 1024
PAR = 8
REMOTE_ROOT = "hdd/11213"


def sess():
    s = requests.Session()
    s.headers["Authorization"] = f"token {os.environ['JTOK']}"
    return s


def remote_sh(cmd, timeout=600):
    """在遠端 Jupyter 終端機跑指令（那台沒有 ssh）。細節見 tools/ 的 labrun 說明。"""
    import uuid, re
    import websocket
    s = sess()
    name = s.post(f"{BASE}/api/terminals", timeout=30).json()["name"]
    tag = "__D_" + uuid.uuid4().hex[:8]
    want = re.compile(re.escape(tag) + r"_(\d+)")
    try:
        ws = websocket.create_connection(
            f"{BASE.replace('http://','ws://')}/terminals/websocket/{name}",
            header=[f"Authorization: token {os.environ['JTOK']}"], timeout=30)
        time.sleep(1.0)
        # ⚠ 終端機會回顯輸入 => 標記必須在遠端由兩段組合，否則會提前匹配到回顯
        ws.send(json.dumps(["stdin", f"S={tag}\n"])); time.sleep(0.3)
        ws.send(json.dumps(["stdin", cmd + "\n"]))
        ws.send(json.dumps(["stdin", 'echo "${S}_$?"\n']))
        buf, t0 = "", time.time(); ws.settimeout(5)
        while time.time() - t0 < timeout:
            try:
                k, p = json.loads(ws.recv())[:2]
            except Exception:
                continue
            if k == "stdout":
                buf += p
                if want.search(buf):
                    break
        ws.close()
        txt = buf.replace("\r\n", "\n")
        m = want.search(txt)
        out = txt[:m.start()] if m else txt
        out = "\n".join(l for l in out.split("\n")
                        if tag not in l and l.strip() != cmd.strip()
                        and not l.strip().startswith('echo "${S}'))
        return out.strip(), (m.group(1) if m else "?")
    finally:
        s.delete(f"{BASE}/api/terminals/{name}", timeout=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="<父目錄>:<子目錄1>[,<子目錄2>...]")
    ap.add_argument("--name", required=True, help="tar 檔名（不含 .tar）")
    ap.add_argument("--dest", default="data/matrix_city/aerial/train/block_all",
                    help="遠端解開的目標（相對 hdd/11213）")
    ap.add_argument("--stage", default="/tmp/labx/stage")
    a = ap.parse_args()
    parent, subs = a.src.split(":", 1)
    os.makedirs(a.stage, exist_ok=True)
    tar = os.path.join(a.stage, a.name + ".tar")

    # ── 1. 打包（已存在就沿用，方便續傳）──
    if not os.path.isfile(tar):
        print(f"打包 {parent} :: {subs} -> {tar}", flush=True)
        subprocess.check_call(["tar", "cf", tar, "-C", parent] + subs.split(","))
    size = os.path.getsize(tar)
    nparts = (size + CHUNK - 1) // CHUNK
    h = hashlib.md5()
    with open(tar, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    local_md5 = h.hexdigest()
    print(f"  {size:,} bytes  {nparts} 片  md5 {local_md5}", flush=True)

    # ── 2. 問遠端哪些片已經在且大小正確（續傳）──
    rdir = f"{REMOTE_ROOT}/xfer"
    remote_sh(f"mkdir -p /workspace/data/{rdir}")
    out, _ = remote_sh(f"cd /workspace/data/{rdir} && "
                       f"for f in {a.name}.tar.part*; do "
                       f"[ -f \"$f\" ] && echo \"$f $(stat -c%s \"$f\")\"; done")
    have = {}
    for ln in out.split("\n"):
        p = ln.strip().split()
        if len(p) == 2 and p[1].isdigit():
            have[p[0]] = int(p[1])
    todo = []
    for i in range(nparts):
        nm = f"{a.name}.tar.part{i:04d}"
        exp = min(CHUNK, size - i * CHUNK)
        if have.get(nm) != exp:
            todo.append(i)
    print(f"  遠端已有 {nparts - len(todo)}/{nparts} 片（大小正確），要補 {len(todo)} 片", flush=True)

    # ── 3. 平行上傳缺的片 ──
    done = [0]; t0 = time.time()
    def up(i):
        s = sess()
        with open(tar, "rb") as f:
            f.seek(i * CHUNK); b = f.read(CHUNK)
        for k in range(4):
            try:
                r = s.put(f"{BASE}/api/contents/{rdir}/{a.name}.tar.part{i:04d}",
                          json={"type": "file", "format": "base64",
                                "content": base64.b64encode(b).decode()}, timeout=1800)
                if r.status_code in (200, 201):
                    done[0] += 1
                    if done[0] % 16 == 0 or done[0] == len(todo):
                        el = time.time() - t0
                        print(f"    {done[0]}/{len(todo)}  "
                              f"{done[0]*CHUNK/1e6:.0f} MB  "
                              f"{done[0]*CHUNK/1e6/max(el,1e-9):.2f} MB/s", flush=True)
                    return True
            except Exception as e:
                print(f"    part{i:04d} 重試 {k+1}: {type(e).__name__}", flush=True)
            time.sleep(2 * (k + 1))
        print(f"    ❌ part{i:04d} 四次都失敗"); return False
    if todo:
        with ThreadPoolExecutor(PAR) as ex:
            if not all(ex.map(up, todo)):
                print("⛔ 有片上傳失敗 —— 直接再跑一次本指令即可續傳（已成功的片會被跳過）")
                return 1

    # ── 4. 遠端重組 + 驗片數 + 驗 md5 ──
    print("  遠端重組並驗證…", flush=True)
    out, rc = remote_sh(
        f"cd /workspace/data/{rdir} && N=$(ls {a.name}.tar.part* | wc -l) && "
        f"echo PARTS=$N && [ \"$N\" = \"{nparts}\" ] || exit 9 && "
        f"cat $(ls {a.name}.tar.part* | sort) > ../{a.name}.tar && "
        f"cd .. && stat -c%s {a.name}.tar && md5sum {a.name}.tar", timeout=1800)
    print("   ", out.replace("\n", "\n    "))
    if local_md5 not in out:
        print(f"⛔ md5 不符（本機 {local_md5}）—— **不要解開**，重跑本指令續傳")
        return 1
    print(f"  ✅ md5 一致 {local_md5}")

    # ── 5. 解開 + 清片段 ──
    out, rc = remote_sh(
        f"cd /workspace/data/{REMOTE_ROOT} && mkdir -p {a.dest} && "
        f"tar xf {a.name}.tar -C {a.dest} && rm -f {a.name}.tar xfer/{a.name}.tar.part* && "
        f"du -sh {a.dest}/*", timeout=1800)
    print("  解開後：\n    " + out.replace("\n", "\n    "))
    print(f"✅ {a.name} 完成（{size/1e9:.2f} GB / {time.time()-t0:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
