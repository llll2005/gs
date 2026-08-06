"""
Convert a .ckpt to .ply (cached) and push it to the Spark web viewer.

Usage:
  python utils/ckpt_preview.py path/to/epoch=X-step=Y.ckpt [--port 8080]

Yazi keymap example (add to ~/.config/yazi/keymap.toml):
  [[manager.prepend_keymap]]
  on   = ["p"]
  run  = 'shell "cd /home/LnoArch/Projects/專題/CityGaussian && python utils/ckpt_preview.py \"$0\"" --confirm'
  desc = "Preview checkpoint in browser"
"""
import argparse
import os
import re
import sys
import json
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR      = os.path.join(PROJECT_ROOT, "web_demo")
sys.path.insert(0, PROJECT_ROOT)


def ply_name_for(ckpt_path: str) -> str:
    """Derive a stable, human-readable PLY filename from a checkpoint path."""
    ckpt_path = os.path.abspath(ckpt_path)

    # outputs/<exp>/blocks/block_N/checkpoints/epoch=X-step=Y.ckpt
    m = re.search(r"outputs/([^/]+)/blocks/(block_\d+)/checkpoints/.*step=(\d+)", ckpt_path)
    if m:
        exp, block, step = m.group(1), m.group(2), m.group(3)
        # shorten long experiment names to last 2 segments
        exp_short = "_".join(exp.split("_")[-4:]) if len(exp) > 40 else exp
        return f"{exp_short}__{block}__step{step}.ply"

    # fallback: just strip extension
    return os.path.basename(ckpt_path).replace(".ckpt", ".ply")


def export_if_needed(ckpt_path: str, ply_path: str):
    if os.path.exists(ply_path):
        size_mb = os.path.getsize(ply_path) / 1e6
        print(f"[cache] {os.path.basename(ply_path)} ({size_mb:.1f} MB)")
        return

    print(f"[convert] {os.path.basename(ckpt_path)} → {os.path.basename(ply_path)}")
    from utils.export_splat_ply import load_gaussians, save_3dgs_ply
    g = load_gaussians(ckpt_path)
    save_3dgs_ply(ply_path, g)


def push_to_browser(ply_name: str, port: int):
    url  = f"http://localhost:{port}/select"
    body = json.dumps({"file": ply_name}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=2)
        print(f"[browser] switched to {ply_name}")
    except Exception as e:
        print(f"[warn] could not reach server on port {port}: {e}")
        print(f"  Start it with:  python web_demo/serve.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_path", help=".ckpt checkpoint path")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    ckpt_path = args.ckpt_path
    if not ckpt_path.endswith(".ckpt"):
        sys.exit(f"Not a .ckpt file: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        sys.exit(f"File not found: {ckpt_path}")

    ply_name = ply_name_for(ckpt_path)
    ply_path = os.path.join(WEB_DIR, ply_name)

    export_if_needed(ckpt_path, ply_path)
    push_to_browser(ply_name, args.port)


if __name__ == "__main__":
    main()
