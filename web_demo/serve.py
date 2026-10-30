"""
Usage:  python web_demo/serve.py [port=8080]

Endpoints:
  GET  /files.json    -> list of .ply files in this directory
  GET  /ckpts.json    -> local checkpoints under outputs/**/{checkpoints,deploy}/ (newest first) — no upload needed
  POST /convert_local -> body {"path": "outputs/.../x.ckpt"}, converts IN PLACE on the server
  GET  /current.json  -> {"file": "block_9.ply", "ts": 1234567890}
  GET  /bounds.json?file=x.ply -> 穩健範圍 {"ok": true, "lo": [...], "hi": [...], "n": N}（viewer 自動置中／縮放用；快取在 x.ply.bounds.json）
  POST /select        -> body {"file": "xxx.ply"}, updates current and ts
  POST /convert       -> body: raw .ckpt bytes, header X-Filename: name.ckpt
                         returns {"ok": true, "file": "name.ply"} or {"ok": false, "error": "..."}
  GET  /*             -> static file serving
"""
import json
import os
import sys
import time
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT      = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))

_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


def ply_bounds(path, n_sample=2_000_000):
    """10-30 使用者：viewer 預設把模型放在畫面中間、大小合適。
    min/max 會被離群點撐爆（SfM 有 z 到 -461 的點、海面浮塊、單塊模型 7 成顆粒在盒外）=> 用分位數：
    只取不透明顆粒（sigmoid(opacity) > 0.5），xy 取 5~95 百分位、z 取 2~99（高樓頂不要被切掉）。
    （外圈塊的塊內顆粒一路延伸到場景邊：full44 塊內 b3 的 x 最大 23.8、z 最小 -21.3；合併模型 x -72.8、z -82.8）
    只讀 x/y/z/opacity 欄（memmap、最多抽 200 萬顆）；結果快取在 <ply>.bounds.json（檔案大小／mtime 變了就重算）。"""
    import numpy as np
    st = os.stat(path)
    cache = path + ".bounds.json"
    if os.path.exists(cache):
        try:
            c = json.load(open(cache))
            if c.get("v") == 2 and c.get("size") == st.st_size and c.get("mtime") == int(st.st_mtime):
                return c
        except Exception:
            pass
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("不是 PLY")
        fmt, n, props, in_vertex = None, 0, [], False
        while True:
            ln = f.readline()
            if not ln:
                raise ValueError("PLY 標頭沒有 end_header")
            w = ln.decode("ascii", "replace").split()
            if not w:
                continue
            if w[0] == "format":
                fmt = w[1]
            elif w[0] == "element":
                in_vertex = w[1] == "vertex"
                if in_vertex:
                    n = int(w[2])
            elif w[0] == "property" and in_vertex:
                if w[1] == "list":
                    raise ValueError("vertex 有 list 屬性，不支援")
                props.append((w[2], "<" + _PLY_TYPES[w[1]]))
            elif w[0] == "end_header":
                off = f.tell()
                break
    if fmt != "binary_little_endian":
        raise ValueError(f"只支援 binary_little_endian（這個是 {fmt}）")
    arr = np.memmap(path, dtype=np.dtype(props), mode="r", offset=off, shape=(n,))
    idx = np.linspace(0, n - 1, min(n, n_sample)).astype(np.int64)
    sub = arr[idx]
    X = np.stack([sub["x"], sub["y"], sub["z"]], 1).astype(np.float64)
    ok = np.all(np.isfinite(X), 1)
    if "opacity" in [p[0] for p in props]:
        o = 1 / (1 + np.exp(-np.clip(sub["opacity"].astype(np.float64), -60, 60)))   # 3DGS PLY 存的是 logit
        if (ok & (o > 0.5)).sum() >= 1000:
            ok &= o > 0.5
    X = X[ok]
    if len(X) == 0:
        raise ValueError("沒有有限座標的顆粒")
    lo = [float(np.percentile(X[:, 0], 5)), float(np.percentile(X[:, 1], 5)), float(np.percentile(X[:, 2], 2))]
    hi = [float(np.percentile(X[:, 0], 95)), float(np.percentile(X[:, 1], 95)), float(np.percentile(X[:, 2], 99))]
    out = {"ok": True, "v": 2, "lo": lo, "hi": hi, "n": int(n), "size": st.st_size, "mtime": int(st.st_mtime)}
    try:
        json.dump(out, open(cache, "w"))
    except OSError:
        pass
    return out


_state      = {"file": None, "ts": 0}
_lock       = threading.Lock()
_converting = threading.Lock()   # only one conversion at a time


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/ckpts.json":
            # 檔案本來就在同一台機器上 => 拖放/上傳整段都是白做的。
            # 這條讓瀏覽器只送一個路徑，伺服器自己讀自己轉。
            # （60k 的 ckpt 是 1.6 GB，走上傳光讀進瀏覽器記憶體就要很久。）
            out = []
            root = os.path.join(PROJECT_ROOT, "outputs")
            for dirpath, _dirs, files in os.walk(root):
                # 10-29：半精度部署檔（tools/half_ckpt.py）放在 deploy/，不放 checkpoints/ => 兩個都掃
                if os.path.basename(dirpath) not in ("checkpoints", "deploy"):
                    continue
                for fn in files:
                    if not fn.endswith(".ckpt"):
                        continue
                    full = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    out.append({"path": os.path.relpath(full, PROJECT_ROOT),
                                "mb": round(st.st_size / 1e6, 1),
                                "mtime": int(st.st_mtime)})
            out.sort(key=lambda d: -d["mtime"])          # 最新的排最前面
            self._json(out)
        elif self.path == "/files.json":
            plys = sorted(f for f in os.listdir(SERVE_DIR) if f.endswith(".ply"))
            self._json(plys)
        elif self.path == "/current.json":
            with _lock:
                self._json(_state.copy())
        elif self.path.startswith("/bounds.json"):
            from urllib.parse import urlparse, parse_qs
            name = os.path.basename(parse_qs(urlparse(self.path).query).get("file", [""])[0])
            try:
                self._json(ply_bounds(os.path.join(SERVE_DIR, name)))
            except Exception as e:
                self._json({"ok": False, "error": str(e)})
        else:
            super().do_GET()

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path == "/select":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            with _lock:
                _state["file"] = body.get("file")
                _state["ts"]   = int(time.time() * 1000)
            self._json({"ok": True})

        elif self.path == "/upload":
            filename = self.headers.get("X-Filename", "upload.ply")
            ply_path = os.path.join(SERVE_DIR, filename)
            if os.path.exists(ply_path):
                print(f"[cache] {filename}")
                self._json({"ok": True, "file": filename, "cached": True})
                return
            length = int(self.headers.get("Content-Length", 0))
            print(f"[upload-ply] {filename} ({length/1e6:.1f} MB)")
            with open(ply_path, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            print(f"[done] {filename}")
            self._json({"ok": True, "file": filename, "cached": False})

        elif self.path == "/convert_local":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length) or b"{}")
            rel    = body.get("path", "")
            # ── 路徑檢查：只接受 PROJECT_ROOT/outputs 底下的 .ckpt ──
            full = os.path.normpath(os.path.join(PROJECT_ROOT, rel))
            outs = os.path.join(PROJECT_ROOT, "outputs") + os.sep
            if not full.startswith(outs) or not full.endswith(".ckpt") or not os.path.isfile(full):
                self._json({"ok": False, "error": f"不接受的路徑：{rel}"})
                return
            # 命名帶上跑次與 block，否則每個跑次的 epoch=212-step=60000.ckpt 會互相蓋掉
            parts = os.path.relpath(full, os.path.join(PROJECT_ROOT, "outputs")).split(os.sep)
            run   = parts[0] if parts else "run"
            if run == "lab" and len(parts) > 1:      # 10-14：outputs/lab/<跑次>/… 只取 "lab" 會讓不同跑次的同一塊撞名、讀到別人的快取
                run = f"lab_{parts[1]}"
            blk   = next((x for x in parts if x.startswith("block_")), "")
            stem  = os.path.splitext(os.path.basename(full))[0]
            tag   = f"step{stem.split('step=')[-1]}" if "step=" in stem else stem   # merged_fp16 => 不加 step 前綴
            from utils.export_splat_ply import FORMAT_TAG      # 10-14：格式修正後換檔名，不讀舊的錯誤快取
            ply_name = f"{run}{('_' + blk) if blk else ''}_{tag}_{FORMAT_TAG}.ply"
            ply_path = os.path.join(SERVE_DIR, ply_name)

            if os.path.exists(ply_path):
                print(f"[cache] {ply_name}")
                self._json({"ok": True, "file": ply_name, "cached": True})
                return
            if not _converting.acquire(blocking=False):
                self._json({"ok": False, "error": "已經有一個轉檔在跑"})
                return
            try:
                print(f"[convert-local] {rel} -> {ply_name}")
                from utils.export_splat_ply import load_gaussians, save_3dgs_ply
                g = load_gaussians(full)
                save_3dgs_ply(ply_path, g)
                print(f"[done] {ply_name} ({os.path.getsize(ply_path)/1e6:.1f} MB)")
                self._json({"ok": True, "file": ply_name, "cached": False})
            except Exception as e:
                import traceback
                traceback.print_exc()
                if os.path.exists(ply_path):
                    os.unlink(ply_path)
                self._json({"ok": False, "error": str(e)})
            finally:
                _converting.release()

        elif self.path == "/convert":
            filename = self.headers.get("X-Filename", "upload.ckpt")
            from utils.export_splat_ply import FORMAT_TAG
            ply_name = os.path.splitext(filename)[0] + f"_{FORMAT_TAG}.ply"
            ply_path = os.path.join(SERVE_DIR, ply_name)

            # cache hit — already converted
            if os.path.exists(ply_path):
                print(f"[cache] {ply_name}")
                self._json({"ok": True, "file": ply_name, "cached": True})
                return

            if not _converting.acquire(blocking=False):
                self._json({"ok": False, "error": "Another conversion is already running"})
                return

            tmp_path = os.path.join(SERVE_DIR, "_converting.ckpt")
            try:
                # stream upload to disk
                length = int(self.headers.get("Content-Length", 0))
                print(f"[upload] {filename} ({length/1e6:.1f} MB)")
                with open(tmp_path, "wb") as f:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(65536, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)

                print(f"[convert] {filename} → {ply_name}")
                from utils.export_splat_ply import load_gaussians, save_3dgs_ply
                g = load_gaussians(tmp_path)
                save_3dgs_ply(ply_path, g)
                print(f"[done] {ply_name} ({os.path.getsize(ply_path)/1e6:.1f} MB)")
                self._json({"ok": True, "file": ply_name, "cached": False})

            except Exception as e:
                import traceback
                traceback.print_exc()
                if os.path.exists(ply_path):
                    os.unlink(ply_path)
                self._json({"ok": False, "error": str(e)})
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                _converting.release()
        else:
            self.send_error(404)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


print(f"Serving http://localhost:{PORT}  (dir: {SERVE_DIR})")
ThreadingHTTPServer(("", PORT), Handler).serve_forever()
