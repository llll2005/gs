#!/bin/bash
# Re-score the 2026-07-19..28 b12 checkpoints with a content-split metric whose resolution is
# checked in the same run. Pure inference on checkpoints that already exist -- no training.
#
# Those runs were filed as "no effect" on the strength of val LPIPS differing by <0.022. That
# reading does not hold: over the same window the runs that OOM'd at 30k scored LPIPS 0.729 while
# the ones that finished 60k scored 0.737, so the metric ranked a crippled model above a healthy
# one. The experiments are UNTESTED, not disproven, and the checkpoints are all still on disk.
#
# The four dead runs go in as --known_bad. If a metric cannot separate them from the healthy runs
# by more than the spread among the healthy runs themselves, it cannot rank the healthy runs
# either, and the tool says so instead of letting us believe it.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
conda run --no-capture-output -n gspl python tools/rescore_by_content.py --block 12 \
  --runs mcmc_sb_ns_b12_cap1m_K1 sb_harvest_noreg sb_harvest_freeze sb_harvest_cprime \
         mcmc_sb_b12_cap1m_harvestdust mcmc_60k_sh3_aggr17_b12 mcmc_sb_60k_aggr17_b12_cap1m \
         oreg_0p0_b12 oreg_0p002_b12 oreg_0p007_b12 b12_cap2m_reg000_exact \
  --known_bad mcmc_sb_b12_cap1m_condense mcmc_sb_b12_cap1m_vpc mcmc_sb_ns_b12_cap2m_K2 \
  2>&1 | grep -vE "Warning|warn|pkg_resources" | tee 紀錄/rescore_plateau.txt
