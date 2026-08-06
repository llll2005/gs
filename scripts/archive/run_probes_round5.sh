#!/bin/bash
# Probe round 5: (1) beta render+metrics on its finished 30k save (training done in
# round 4, eval OOM'd with training state resident — standalone fits), (2) build
# probe_b7_small (70k init points; triangle/convex are per-primitive heavy and OOM at
# 217k init on 6GB — documented per-family accommodation), (3) triangle & convex on it.
cd /home/LnoArch/Projects/專題 || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SCENE=/home/LnoArch/Projects/專題/CityGaussian/data/probe_b7
SMALL=/home/LnoArch/Projects/專題/CityGaussian/data/probe_b7_small
LOGD=/home/LnoArch/Projects/專題/CityGaussian/logs
PROG=$LOGD/quad_progress.log

echo "[probe5] beta eval start $(date +%F_%T)" >> $PROG
(cd beta-splatting && PYTHONPATH=/home/LnoArch/Projects/專題/beta-splatting/_pylibs \
  python render.py -m output/probe_b7 --skip_train > $LOGD/probe_beta_render.log 2>&1 \
  && PYTHONPATH=/home/LnoArch/Projects/專題/beta-splatting/_pylibs \
  python metrics.py -m output/probe_b7 > $LOGD/probe_beta_metrics.log 2>&1)
echo "[probe5] beta eval done rc=$? $(date +%F_%T)" >> $PROG

python - <<'EOF'
import os, sys, random, shutil
sys.path.insert(0, "CityGaussian")
from internal.utils import colmap as cu
src = "CityGaussian/data/probe_b7"; dst = "CityGaussian/data/probe_b7_small"
os.makedirs(f"{dst}/sparse/0", exist_ok=True)
if not os.path.exists(f"{dst}/images"):
    os.symlink(os.path.abspath(f"{src}/images"), f"{dst}/images")
shutil.copy(f"{src}/val_names.txt", f"{dst}/val_names.txt")
for f in ["cameras.bin", "images.bin"]:
    shutil.copy(f"{src}/sparse/0/{f}", f"{dst}/sparse/0/{f}")
pts = cu.read_points3D_binary(f"{src}/sparse/0/points3D.bin")
random.seed(20260711)
keep = set(random.sample(list(pts.keys()), 70000))
cu.write_points3D_binary({k: v for k, v in pts.items() if k in keep}, f"{dst}/sparse/0/points3D.bin")
p = f"{dst}/sparse/0/points3D.ply"
os.path.exists(p) and os.remove(p)
print("probe_b7_small ready: 70000 points")
EOF

run_one() {
  local name=$1 dir=$2 extra_args=$3
  echo "[probe5] $name start $(date +%F_%T)" >> $PROG
  (cd $dir && python train.py -s $SMALL --eval -r 1 $extra_args \
      -m output/probe_b7 > $LOGD/probe_${name}_train.log 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    (cd $dir && python render.py -m output/probe_b7 --skip_train > $LOGD/probe_${name}_render.log 2>&1 \
      && python metrics.py -m output/probe_b7 > $LOGD/probe_${name}_metrics.log 2>&1)
    rc=$?
  fi
  echo "[probe5] $name done rc=${rc} $(date +%F_%T)" >> $PROG
}

rm -rf triangle-splatting/output/probe_b7 convex-splatting/output/probe_b7
run_one triangle triangle-splatting "--outdoor --max_shapes 1000000"
run_one convex   convex-splatting   "--outdoor --light"
echo "[probe5] ALL DONE $(date +%F_%T)" >> $PROG
