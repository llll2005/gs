#!/bin/bash
# Post-harvest-arms experiment batch (Gemini round-6 program):
#  1) dust-ablation render     — decisive zombie check (predictions: Gemini <0.1dB)
#  2) arm-B working-set audit  — did cap 2.0M grow WS95 (A=7,154) or just the dust pile?
#  3) condensation surgery     — prune 42k ckpt to top-20% by render mass, finetune 5k
#     steps with ONLY color/SB learnable (geometry/opacity/scale locked, noise off,
#     trim off, densify off). Recovers ~21.99 => harvest jump = appearance convergence;
#     collapses => condensation's continuous mass transfer is the engine.
# Waits until the harvest-alpha arms have all finished.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log

while ! grep -q "sb_harvest_noreg done" $PROG || pgrep -f "main.py fit" > /dev/null; do sleep 600; done
sleep 60

ACKPT=$(find outputs/mcmc_sb_60k_aggr17_b12_cap1m/blocks/block_12/checkpoints -name "*step=41999.ckpt" | head -1)

echo "[surgery] batch start $(date +%F_%T)" >> $PROG

# 1) dust ablation (uses the 60k ckpt)
CK60=$(find outputs/mcmc_sb_60k_aggr17_b12_cap1m/blocks/block_12/checkpoints -name "*step=60000.ckpt" | head -1)
conda run -n gspl python -u tools/render_dust_ablation.py "$CK60" --block 12 --sb --n_views 6 > logs/dust_ablation.log 2>&1
grep -E "removed|PSNR|agreement" logs/dust_ablation.log | tail -3 >> $PROG

# 2) arm-B audits (CPU) — final + mid ckpts if present
for CK in $(find outputs/mcmc_sb_60k_aggr17_b12_capfx/blocks/block_12/checkpoints -name "*.ckpt" 2>/dev/null | sort); do
  conda run -n gspl python -u tools/audit_working_set.py "$CK" >> logs/armB_audit.log 2>&1
done
grep -E "AUDIT|working set" logs/armB_audit.log | tail -8 >> $PROG

# 3) condensation surgery
conda run -n gspl python -u tools/make_pruned_ckpt.py "$ACKPT" /tmp/claude-1000/surgery_pruned_42k.ckpt --keep_frac 0.2 > logs/surgery_prune.log 2>&1
echo "[surgery] finetune start $(date +%F_%T)" >> $PROG
rm -rf outputs/sb_surgery_coloronly
conda run -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from /tmp/claude-1000/surgery_pruned_42k.ckpt \
  --data.parser.block_id 12 \
  --trainer.max_steps 5000 \
  --model.density.init_args.densify_from_iter 999999 \
  --model.density.init_args.noise_lr 0.0 \
  --model.renderer.init_args.diable_trimming true \
  --model.gaussian.init_args.optimization.means_lr_init 0.0 \
  --model.gaussian.init_args.optimization.opacities_lr 0.0 \
  --model.gaussian.init_args.optimization.scales_lr 0.0 \
  --model.gaussian.init_args.optimization.rotations_lr 0.0 \
  -n sb_surgery_coloronly > logs/sb_surgery_coloronly.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/sb_surgery_coloronly/blocks/block_12/results.txt 2>/dev/null | head -1)
echo "[surgery] colessonly done rc=$rc $psnr (target ~21.99, baseline42k ~19.4) $(date +%F_%T)" >> $PROG
