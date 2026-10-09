"""把初始化 PLY 隨機抽成指定顆數（2026-10-09，「init 就給到上限、不再增生」用：dup5 約 2.96M -> cap 2.6M）。
用法：python tools/subsample_ply.py <in.ply> <out.ply> --n 2600000 [--seed 0]
"""
import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
v = PlyData.read(a.src)["vertex"].data
if len(v) <= a.n:
    raise SystemExit(f"⛔ {a.src} 只有 {len(v):,} 點 <= {a.n:,}")
idx = np.sort(np.random.default_rng(a.seed).choice(len(v), a.n, replace=False))
os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
PlyData([PlyElement.describe(v[idx], "vertex")]).write(a.dst)
print(f"✅ {a.src} {len(v):,} -> {a.dst} {a.n:,}")
