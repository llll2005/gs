#!/usr/bin/env python
"""把這個專案部署到**任何**一台機器：探測 -> 程式碼 -> 資料 -> 建環境 -> 驗證。

## 為什麼需要這一支（`lab_xfer.py` 不夠的地方）

`lab_xfer.py` 只做「打包 -> 切片上傳 -> 驗 md5 -> 解開」，而且把 `hdd/11213` 與
lab 的位址寫死了。實際部署還缺三件事，而每一件都曾經是**啟動後才發現**的問題：
```
① 遠端探測      磁碟/GPU/外網/工具鏈/CUDA 版本 —— lab 的系統 CUDA 是 12.9 而我方 torch 是
                cu118，`bootstrap.sh` 原本沒裝 env 內的 nvcc 11.8 => 編光柵器必失敗（已補）
② 傳輸後端      有 ssh 的機器該用 rsync（快幾十倍）；只有 Jupyter 的才走切片上傳
③ 啟動檢查      「裝好」不等於「裝對」=> 必須遠端跑 verify_env.py（它做的是**功能**驗證：
                渲染合成高斯 + backward，檢查 means2D.grad[:,2] 非零 => ABSGRAD 真的編進去了）
```

## 後端二選一

```
--ssh user@host --root /path       有 ssh 就走這條：rsync（增量、可續傳、快）
--jupyter http://host:8888 --root hdd/11213    只有 Jupyter：切片上傳（contents API 沒有 append）
                                   token 從環境變數 JTOK 讀，**不寫檔、不進 commit**
```

## 資料子集（`--data` 用逗號分隔）

```
sfm            sparse/ + partition/        1.8 GB   訓練必需（COLMAP 姿態）
depth_init     depth_init/                 2.2 GB   depth-init 起點（SfM-init 不需要）
depthmin       scales.json + 8 張 .npy      47 MB   **硬性必需**，見下
input          input/                     20.4 GB   訓練影像
test           ../../test/                 9.3 GB   只有官方 741 幀評測要
depths_full    estimated_depths/            31 GB   只有要開深度損失才需要
```
⚠⚠ `depthmin` 為什麼是硬性必需：`estimated_depth_colmap_dataparser.py` 有兩道硬檢查 ——
  缺 `estimated_depth_scales.json` 直接 `FileNotFoundError`；深度圖**全**缺則
  `assert loaded_depth_count > 0` 當場死。但缺**單張**只印 warning
  ⇒ 現行配方（`depth_loss_weight.init 0.0`）只要這 47 MB 就跑得起來，不用等 31 GB。

用法:
  python scripts/setup/deploy.py check   --jupyter http://140.119.164.19:8888 --root hdd/11213
  python scripts/setup/deploy.py code    --jupyter ... --root ...
  python scripts/setup/deploy.py data    --jupyter ... --root ... --data sfm,depthmin,input
  python scripts/setup/deploy.py all     --ssh user@host --root /scratch/gs --data sfm,depthmin,input
"""
import argparse, base64, hashlib, json, os, re, shlex, subprocess, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLOCK = "data/matrix_city/aerial/train/block_all"
DEST = BLOCK                      # 遠端相對 --root 的解開位置
CHUNK = 16 * 1024 * 1024
PAR = 8
GIT_URL = "git@github.com:llll2005/gs.git"

SUBSETS = {
    # 名稱: (打包的父目錄, [子路徑...], 遠端解開位置)
    "sfm":        (BLOCK, ["sparse", "partition"], DEST),
    "depth_init": (BLOCK, ["depth_init"], DEST),
    "input":      (BLOCK, ["input"], DEST),
    "depths_full":(BLOCK, ["estimated_depths"], DEST),
    "test":       ("data/matrix_city/aerial", ["test"], "data/matrix_city/aerial"),
}


_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[=>]")


def _clean(text, cmd, tag):
    """把互動式終端機的雜訊剝掉。

    ⚠ Jupyter 的終端機是**真的 pty**，所以輸出裡混著：
      ① ANSI 逃脫碼（顏色、括號貼上模式 `[?2004h/l`、OSC 標題 `]0;...`）
      ② shell 提示字元（`(base) root@host:/path#`）
      ③ **被回顯的輸入指令本身**
    不剝掉就會像 2026-09-12 第一版那樣，把 `echo "${S}_$?"` 和一長串提示字元印進報告裡。
    """
    out = []
    for ln in text.split("\n"):
        ln = _ANSI.sub("", ln)
        ln = re.sub(r"\(base\)\s*", "", ln)
        ln = re.sub(r"[\w.-]+@[\w.-]+:[^#$]*[#$]\s*", "", ln)   # 提示字元
        # ⚠ 指令輸出若沒有結尾換行，下一行被回顯的輸入會**接在同一行尾**
        #   （實測：`106072986 B/echo "${S}_$?"`）=> 不能只比對行首，要整行剝掉。
        ln = ln.split('echo "${S}')[0]
        if tag in ln:
            continue
        if cmd.strip() and cmd.strip() in ln:
            continue
        ln = ln.strip()
        if ln:
            out.append(ln)
    return "\n".join(out).strip()


# ══ 後端 ══════════════════════════════════════════════════════════════════════
class Ssh:
    kind = "ssh"

    def __init__(self, host, root):
        self.host, self.root = host, root

    def sh(self, cmd, timeout=1800, stream=False):
        full = ["ssh", "-o", "BatchMode=yes", self.host, f"cd {shlex.quote(self.root)} 2>/dev/null; {cmd}"]
        if stream:
            return subprocess.call(full), ""
        p = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()

    def put_tree(self, parent, subs, dest):
        """有 rsync 就用 rsync —— 增量、可續傳，比打包上傳快幾十倍。"""
        self.sh(f"mkdir -p {shlex.quote(dest)}")
        srcs = [os.path.join(ROOT, parent, s) + "/" for s in subs]
        for s, sub in zip(srcs, subs):
            rc = subprocess.call(["rsync", "-a", "--partial", "--info=progress2",
                                  s, f"{self.host}:{self.root}/{dest}/{sub}/"])
            if rc != 0:
                return False
        return True


class Jup:
    kind = "jupyter"

    def __init__(self, base, root):
        import requests, websocket           # noqa: F401
        self.base, self.root = base.rstrip("/"), root
        self.tok = os.environ.get("JTOK", "")
        if not self.tok:
            sys.exit("⛔ 需要環境變數 JTOK（Jupyter token）—— 不要把它寫進檔案或 commit")

    def _s(self):
        import requests
        s = requests.Session()
        s.headers["Authorization"] = f"token {self.tok}"
        return s

    def sh(self, cmd, timeout=1800, stream=False):
        """Jupyter 終端機（websocket）。⚠ 終端機**會回顯輸入** => 結束標記必須在遠端
        由兩段組合（`S=__D_x; echo "${S}_$?"`），否則會匹配到回顯而提前結束。"""
        import websocket
        s = self._s()
        name = s.post(f"{self.base}/api/terminals", timeout=30).json()["name"]
        tag = "__D_" + uuid.uuid4().hex[:8]
        want = re.compile(re.escape(tag) + r"_(\d+)")
        try:
            ws = websocket.create_connection(
                f"{self.base.replace('http://','ws://')}/terminals/websocket/{name}",
                header=[f"Authorization: token {self.tok}"], timeout=30)
            time.sleep(1.0)
            ws.send(json.dumps(["stdin", f"S={tag}\n"])); time.sleep(0.3)
            ws.send(json.dumps(["stdin", f"cd {shlex.quote(self.root)} 2>/dev/null; {cmd}\n"]))
            ws.send(json.dumps(["stdin", 'echo "${S}_$?"\n']))
            buf, t0 = "", time.time(); ws.settimeout(5)
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
            txt = buf.replace("\r\n", "\n")
            m = want.search(txt)
            out = txt[:m.start()] if m else txt
            return int(m.group(1)) if m else 1, _clean(out, cmd, tag)
        finally:
            # ⚠ 2026-09-12：這裡原本是裸的 `s.delete(..., timeout=30)`，逾時就把
            #   **已經成功的指令**變成例外（第一次連 lab 的 `check` 就這樣掛掉）。
            #   清理失敗不該影響結果 => 包起來並放寬逾時。
            try:
                s.delete(f"{self.base}/api/terminals/{name}", timeout=60)
            except Exception:
                pass

    def put_tree(self, parent, subs, dest, stage=None):
        """contents API **沒有 append** => 打包成 tar、切 16MB 片上傳、遠端 cat 回去。
        四道驗證缺一不可（少一片 cat 出來仍是「看起來能解開」的壞 tar）："""
        stage = stage or os.path.expanduser("~/labstage")
        os.makedirs(stage, exist_ok=True)
        name = "_".join(subs).replace("/", "_")
        tar = os.path.join(stage, name + ".tar")
        if not os.path.isfile(tar):
            print(f"  打包 {parent} :: {','.join(subs)}", flush=True)
            subprocess.check_call(["tar", "cf", tar, "-C", os.path.join(ROOT, parent)] + subs)
        size = os.path.getsize(tar)
        nparts = (size + CHUNK - 1) // CHUNK
        h = hashlib.md5()
        with open(tar, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        md5 = h.hexdigest()
        print(f"  {size:,} bytes  {nparts} 片  md5 {md5}", flush=True)

        rdir = f"{self.root}/xfer"
        self.sh(f"mkdir -p {shlex.quote(rdir)}")
        # ── 續傳：只補「不存在或大小不對」的片 ──
        _, out = self.sh(f"cd {shlex.quote(rdir)} && for f in {name}.tar.part*; do "
                         f"[ -f \"$f\" ] && echo \"$f $(stat -c%s \"$f\")\"; done")
        have = {}
        for ln in out.split("\n"):
            p = ln.strip().split()
            if len(p) == 2 and p[1].isdigit():
                have[p[0]] = int(p[1])
        todo = [i for i in range(nparts)
                if have.get(f"{name}.tar.part{i:04d}") != min(CHUNK, size - i * CHUNK)]
        print(f"  遠端已有 {nparts-len(todo)}/{nparts} 片（大小正確），補 {len(todo)} 片", flush=True)

        done = [0]; t0 = time.time()
        def up(i):
            s = self._s()
            with open(tar, "rb") as f:
                f.seek(i * CHUNK); b = f.read(CHUNK)
            for k in range(4):
                try:
                    r = s.put(f"{self.base}/api/contents/{rdir}/{name}.tar.part{i:04d}",
                              json={"type": "file", "format": "base64",
                                    "content": base64.b64encode(b).decode()}, timeout=1800)
                    if r.status_code in (200, 201):
                        done[0] += 1
                        if done[0] % 16 == 0 or done[0] == len(todo):
                            el = time.time() - t0
                            print(f"    {done[0]}/{len(todo)}  {done[0]*CHUNK/1e6:.0f} MB  "
                                  f"{done[0]*CHUNK/1e6/max(el,1e-9):.2f} MB/s", flush=True)
                        return True
                except Exception as e:
                    print(f"    part{i:04d} 重試 {k+1}: {type(e).__name__}", flush=True)
                time.sleep(2 * (k + 1))
            print(f"    ❌ part{i:04d} 四次都失敗"); return False
        if todo:
            with ThreadPoolExecutor(PAR) as ex:
                if not all(ex.map(up, todo)):
                    print("  ⛔ 有片失敗 —— 直接再跑同一行即可續傳"); return False
        # ── 驗片數 + 驗 md5 + 解開 ──
        rc, out = self.sh(
            f"cd {shlex.quote(rdir)} && N=$(ls {name}.tar.part* | wc -l) && echo PARTS=$N && "
            f"[ \"$N\" = \"{nparts}\" ] || exit 9 && cat $(ls {name}.tar.part* | sort) > ../{name}.tar && "
            f"cd .. && md5sum {name}.tar", timeout=3600)
        print("   ", out.replace("\n", "\n    "))
        if md5 not in out:
            print(f"  ⛔ md5 不符（本機 {md5}）—— **不要解開**，重跑本指令續傳"); return False
        print(f"  ✅ md5 一致")
        rc, out = self.sh(f"mkdir -p {shlex.quote(dest)} && tar xf {name}.tar -C {shlex.quote(dest)} && "
                          f"rm -f {name}.tar xfer/{name}.tar.part* && du -sh {shlex.quote(dest)}/*",
                          timeout=3600)
        print("  解開後：\n    " + out.replace("\n", "\n    "))
        return True


# ══ 子命令 ════════════════════════════════════════════════════════════════════
def cmd_check(be):
    print("\n=== 遠端探測 ===")
    probes = [
        ("身分/主機", "whoami; hostname"),
        ("GPU", "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"),
        ("磁碟（目標目錄）", "df -h . | tail -1"),
        ("外網", "timeout 10 curl -sI https://github.com | head -1"),
        ("下載速度", 'timeout 60 curl -sL -r 0-52428800 -o /dev/null -w "%{speed_download} B/s" '
                     '"https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl"'),
        ("工具", 'for c in git curl conda python nvcc gcc rsync; do command -v $c >/dev/null && printf "%s " $c; done'),
        ("系統 CUDA", "nvcc --version 2>/dev/null | tail -2 | head -1"),
        ("CPU/RAM", "nproc; free -g | sed -n 2p"),
    ]
    for label, c in probes:
        rc, out = be.sh(c, timeout=120)
        print(f"  {label:<16} {out.replace(chr(10), ' | ')[:150]}")
    print("""
⚠ 判讀：
  系統 CUDA 是 12.x 也沒關係 —— bootstrap.sh 會把 **11.8 工具鏈裝進 conda env**
  （`nvidia/label/cuda-11.8.0`）並強制 `CUDA_HOME=$CONDA_PREFIX`，不動系統。
  `rsync` 有 => 用 --ssh 後端會快幾十倍；只有 Jupyter 就只能切片上傳。""")
    return 0


def cmd_code(be):
    print("\n=== 程式碼 ===")
    rc, out = be.sh("timeout 15 curl -sI https://github.com | head -1", timeout=60)
    if "200" in out or "301" in out:
        print("  遠端有外網 => 讓它自己 clone（我們的上行是瓶頸，能拉就別推）")
        rc, out = be.sh(f"[ -d gs/.git ] && (cd gs && git pull --ff-only) || "
                        f"git clone --depth 1 {GIT_URL} gs", timeout=1800, stream=True)
        print(f"\n  rc={rc}")
        return 0 if rc == 0 else 1
    print("  遠端沒有外網 => 打包追蹤中的檔案上傳")
    subprocess.check_call("git ls-files -z | tar czf /tmp/code.tgz --null -T -", shell=True, cwd=ROOT)
    return 0 if be.put_tree("/tmp", ["code.tgz"], "gs") else 1


def cmd_data(be, names):
    for n in names:
        print(f"\n=== 資料：{n} ===")
        if n == "depthmin":
            # 硬性必需的最小集合：scales json + 少量 .npy（見模組 docstring）
            st = os.path.expanduser("~/labstage"); os.makedirs(st, exist_ok=True)
            tar = os.path.join(st, "depthmin.tar")
            if not os.path.isfile(tar):
                d = os.path.join(ROOT, BLOCK)
                npys = sorted(os.listdir(os.path.join(d, "estimated_depths")))[:8]
                subprocess.check_call(["tar", "cf", tar, "-C", d,
                                       "estimated_depth_scales.json"] +
                                      [f"estimated_depths/{x}" for x in npys])
            ok = be.put_tree(st, ["depthmin.tar"], DEST) if be.kind == "ssh" else \
                 be.put_tree(st, ["depthmin.tar"], DEST, stage=st)
            # tar-of-tar：ssh 後端是 rsync 一個檔，要再解一次
            if be.kind == "ssh":
                be.sh(f"cd {DEST} && tar xf depthmin.tar && rm -f depthmin.tar")
            if not ok:
                return 1
            continue
        if n not in SUBSETS:
            print(f"  ⛔ 不認得的子集 {n}（可用：{', '.join(list(SUBSETS)+['depthmin'])}）"); return 1
        parent, subs, dest = SUBSETS[n]
        if not be.put_tree(parent, subs, dest):
            return 1
    return 0


def cmd_bootstrap(be):
    print("\n=== 建環境（遠端跑 bootstrap.sh）===")
    rc, _ = be.sh("cd gs && bash scripts/setup/bootstrap.sh", timeout=7200, stream=True)
    print(f"\n  rc={rc}")
    return 0 if rc == 0 else 1


def cmd_verify(be):
    print("\n=== 啟動檢查（裝好 != 裝對）===")
    rc, out = be.sh("cd gs && conda run -n gspl python scripts/setup/verify_env.py", timeout=1800)
    print("  " + out.replace("\n", "\n  ")[:4000])
    print(f"  rc={rc}   （0 = 全過）")
    return 0 if rc == 0 else 1


def cmd_status(be, names):
    """傳輸進度 —— **直接問遠端**，不依賴本機 log。

    ⚠ 為什麼不看 log：背景上傳是用 `conda run` 起的，而它**會緩衝 stdout**
      （本專案記過這個坑：`[dynamic-K]` 的列印在跑動中完全看不到）=> log 可能是空的。
      問遠端「有幾片、多大」才是可信的進度。
    """
    print("\n=== 傳輸進度（直接問遠端）===")
    stage = os.path.expanduser("~/labstage")
    rc, out = be.sh(f"cd {shlex.quote(be.root)}/xfer 2>/dev/null && "
                    f"for f in *.tar.part0000; do b=${{f%.tar.part0000}}; "
                    f"n=$(ls $b.tar.part* 2>/dev/null | wc -l); "
                    f"echo \"$b $n\"; done", timeout=300)
    remote = {}
    for ln in out.split("\n"):
        q = ln.strip().split()
        if len(q) == 2 and q[1].isdigit():
            remote[q[0]] = int(q[1])
    if not remote:
        print("  xfer/ 目前沒有進行中的切片")
    for base, got in sorted(remote.items()):
        tar = os.path.join(stage, base + ".tar")
        if os.path.isfile(tar):
            sz = os.path.getsize(tar)
            exp = (sz + CHUNK - 1) // CHUNK
            print(f"  {base:<16} {got:>5}/{exp:<5} 片  "
                  f"{got*CHUNK/1e9:>6.2f}/{sz/1e9:.2f} GB  {100*got/exp:>5.1f}%")
        else:
            print(f"  {base:<16} {got:>5} 片（本機的 tar 已刪或在別處，算不出總數）")
    print("\n=== 遠端已就位的資料 ===")
    rc, out = be.sh(f"du -sh {shlex.quote(be.root)}/{DEST}/* 2>/dev/null; "
                    f"du -sh {shlex.quote(be.root)}/data/matrix_city/aerial/test 2>/dev/null", timeout=300)
    print("  " + (out.replace("\n", "\n  ") if out else "（還沒有）"))
    return 0


def cmd_pull(be, remote_path, out_path):
    """從遠端**抓**檔案回來（Jupyter 的 `/files/` 端點吃 range => 可續傳）。

    ⚠ 只有 Jupyter 後端需要這個；ssh 後端直接用 `rsync`/`scp` 更快。
    ⚠ 路徑是**相對 --root**，例如 `outputs/lab_xxx/blocks/block_12/results.txt`。
    """
    if be.kind != "jupyter":
        print(f"  ssh 後端請直接用： rsync -a {be.host}:{be.root}/{remote_path} {out_path}")
        return 0
    import requests
    url = f"{be.base}/files/{be.root.lstrip('/')}/{remote_path}"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    have = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
    h = {"Authorization": f"token {be.tok}"}
    if have:
        h["Range"] = f"bytes={have}-"
        print(f"  已有 {have:,} bytes => 續傳")
    t0 = time.time()
    with requests.get(url, headers=h, stream=True, timeout=1800) as r:
        if r.status_code not in (200, 206):
            print(f"  ⛔ HTTP {r.status_code}  {url}")
            return 1
        mode = "ab" if r.status_code == 206 else "wb"
        got = have if r.status_code == 206 else 0
        with open(out_path, mode) as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch); got += len(ch)
                if got % (64 << 20) < (1 << 20):
                    print(f"    {got/1e6:.0f} MB  {got/1e6/max(time.time()-t0,1e-9):.2f} MB/s",
                          flush=True)
    print(f"  ✅ {out_path}  {os.path.getsize(out_path):,} bytes  "
          f"{time.time()-t0:.0f}s")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "code", "data", "bootstrap", "verify",
                                "all", "status", "pull"])
    ap.add_argument("--remote", default="", help="pull：遠端路徑（相對 --root）")
    ap.add_argument("--out", default="", help="pull：本機輸出路徑")
    ap.add_argument("--jupyter"); ap.add_argument("--ssh")
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default="sfm,depthmin,input")
    a = ap.parse_args()
    if bool(a.jupyter) == bool(a.ssh):
        sys.exit("⛔ 要指定 --jupyter <url> 或 --ssh <user@host> 其中之一")
    be = Jup(a.jupyter, a.root) if a.jupyter else Ssh(a.ssh, a.root)
    names = [x for x in a.data.split(",") if x]
    if a.cmd == "all":
        for f in (lambda: cmd_check(be), lambda: cmd_code(be),
                  lambda: cmd_data(be, names), lambda: cmd_bootstrap(be),
                  lambda: cmd_verify(be)):
            if f() != 0:
                print("⛔ 中止（上一步失敗）"); return 1
        print("\n✅ 部署完成")
        return 0
    if a.cmd == "pull":
        if not a.remote:
            sys.exit("⛔ pull 需要 --remote <遠端路徑>（相對 --root）")
        return cmd_pull(be, a.remote, a.out or os.path.basename(a.remote))
    return {"check": lambda: cmd_check(be), "code": lambda: cmd_code(be),
            "data": lambda: cmd_data(be, names), "bootstrap": lambda: cmd_bootstrap(be),
            "verify": lambda: cmd_verify(be), "status": lambda: cmd_status(be, names)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
