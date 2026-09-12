"""
Usage:  python web_demo/serve.py [port=8080]

Endpoints:
  GET  /files.json    -> list of .ply files in this directory
  GET  /ckpts.json    -> local checkpoints under outputs/ (newest first) — no upload needed
  POST /convert_local -> body {"path": "outputs/.../x.ckpt"}, converts IN PLACE on the server
  GET  /current.json  -> {"file": "block_9.ply", "ts": 1234567890}
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
                if os.path.basename(dirpath) != "checkpoints":
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
            blk   = next((x for x in parts if x.startswith("block_")), "")
            stem  = os.path.splitext(os.path.basename(full))[0]
            step  = stem.split("step=")[-1] if "step=" in stem else stem
            ply_name = f"{run}{('_' + blk) if blk else ''}_step{step}.ply"
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
            ply_name = os.path.splitext(filename)[0] + ".ply"
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
