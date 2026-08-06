"""
Usage:  python web_demo/serve.py [port=8080]

Endpoints:
  GET  /files.json    -> list of .ply files in this directory
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
        if self.path == "/files.json":
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
