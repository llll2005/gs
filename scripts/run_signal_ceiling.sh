#!/bin/bash
# Does any candidate densification signal have concentration a sampler could exploit?
# Inference only, ~10 min, three existing checkpoints. Decides whether DBP/DGD is worth reviving.
#
# 2026-07-29 rejected error-guided densification on ceiling(top5%)/mean = 1.18-1.27, but that
# measured PHOTOMETRIC error over unsplit views on b12 -- and b12 is roughly half flat water, whose
# low uniform error drags any concentration statistic toward 1.0. DGD chases the SSIM gradient
# instead, precisely because L1 is insensitive to blur. Both objections are testable here.
#
# Three checkpoints so the answer is not an artifact of one model's failure mode:
#   b12_cap2m_reg000_exact  the best b12 model (1.80M)
#   oreg_0p007_b12          same block, strong opacity_reg (different geometry regime)
#   mcmc_60k_sh3_aggr17_b12 SH3 colour instead of SB (different colour capacity)
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
OUT=紀錄/signal_ceiling.txt
: > $OUT
for spec in \
  "b12_cap2m_reg000_exact:outputs/b12_cap2m_reg000_exact/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt" \
  "oreg_0p007_b12:outputs/oreg_0p007_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt" \
  "mcmc_60k_sh3_aggr17_b12:outputs/mcmc_60k_sh3_aggr17_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt" ; do
  name=${spec%%:*}; ck=${spec#*:}
  [ -f "$ck" ] || { echo "跳過 $name（缺 ckpt）" | tee -a $OUT; continue; }
  echo "" | tee -a $OUT; echo "########## $name ##########" | tee -a $OUT
  conda run --no-capture-output -n gspl python tools/measure_signal_ceiling.py \
    --ckpt "$ck" --block 12 2>&1 | grep -vE "Warning|warn|pkg_resources" | tee -a $OUT
done
echo "[彙整] $OUT"
