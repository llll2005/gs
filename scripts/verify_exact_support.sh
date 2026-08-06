#!/bin/bash
# Verify EXACT_SUPPORT is a pure memory win: renders must come out byte-identical, tile
# intersections must drop. Run this AFTER the DAR arm finishes -- reinstalling the rasterizer
# swaps the .so out from under any running job.
#
# Baseline note: auxiliary.h has `#define TIGHTBBOX 0`, so the shipped build bounds every
# primitive at a flat 3 sigma. EXACT_SUPPORT caps the radius where alpha provably falls under
# the 1/255 the blend loop already discards (both passes), and drops o <= 1/255 outright.
# With EXACT_SUPPORT_GROW=0 the radius is additionally capped at 3, so nothing that currently
# contributes can be lost -> renders must match bit for bit.
#
# Estimated on b12@30k: 8.71x fewer intersections, N_max x2.05 (1.70M -> 3.48M at K=1).
# If the render differs at all, EXACT_SUPPORT is wrong -- do not proceed to the caps rebuild.
set -eu
cd /home/LnoArch/Projects/專題/CityGaussian
CKPT=$(ls outputs/mcmc_sb_ns_b12_cap1m_K1/blocks/block_12/checkpoints/*step=29999*.ckpt | head -1)
OUT=/tmp/claude-1000/exact_support

mkdir -p $OUT
echo "== 1. 用現行(已安裝)二進位算基準 =="
conda run --no-capture-output -n gspl python tools/render_fixed_ckpt.py --ckpt "$CKPT" --out $OUT/before.npz --views 8

echo "== 2. 重編 rasterizer (EXACT_SUPPORT=1, GROW=0) =="
conda run --no-capture-output -n gspl pip install --no-build-isolation -e submodules/diff-surfel-rasterization-trim-pp

echo "== 3. 同一 ckpt 同一視角重算 =="
conda run --no-capture-output -n gspl python tools/render_fixed_ckpt.py --ckpt "$CKPT" --out $OUT/after.npz --views 8

echo "== 4. 位元級比對 + tile 相交數 =="
conda run --no-capture-output -n gspl python - <<'PY'
import numpy as np
a=np.load('/tmp/claude-1000/exact_support/before.npz'); b=np.load('/tmp/claude-1000/exact_support/after.npz')
img_ok=all(np.array_equal(a[k],b[k]) for k in a.files if k.startswith('img'))
print(f"影像位元級相同: {'✅ PASS' if img_ok else '❌ FAIL — EXACT_SUPPORT 有誤,勿繼續'}")
if not img_ok:
    for k in a.files:
        if k.startswith('img') and not np.array_equal(a[k],b[k]):
            d=np.abs(a[k].astype(np.float64)-b[k].astype(np.float64)); print(f"  {k}: maxdiff={d.max():.3e} mean={d.mean():.3e}")
for k in a.files:
    if k.startswith('isect'): print(f"  {k}: {int(a[k])} -> {int(b[k])}  ({int(a[k])/max(int(b[k]),1):.2f}x 少)  [估計 8.71x]")
PY
echo ""
echo "PASS 之後：(1) 重生 block_caps.csv（tau 已變）(2) 用新 cap 重跑 b12@2M+"
