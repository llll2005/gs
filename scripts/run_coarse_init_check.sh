#!/bin/bash
# Does coarse-init survive our SB colour model? 200 steps, just to read the kernel banner.
#
# The coarse model is sh0. Our mainline is Spherical Beta (F=25). If loading a coarse checkpoint
# makes the run fall back to sh0 -- memory records that overwrite_config kept sh0 after loading an
# sh0 coarse once before -- then any coarse-init arm silently compares a different colour model
# and is worthless as a control. Cheaper to find out in 200 steps than in four hours.
#
# The coarse itself is usable: nearest-neighbour distance from current SfM points to
# coarse_mc_aerial_1.2x is P50 0.055 against a scene 610 units across, i.e. the same frame, no
# Sim3 drift. (memory old_data_coarse_invalid names only aerial_train_block_all_3x, not this one.)
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
CK="outputs/coarse_mc_aerial_1.2x/checkpoints/epoch=6-step=30000.ckpt"
rm -rf outputs/coarse_init_check
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from "$CK" --data.parser.block_id 12 \
  --model.density.init_args.cap_max 1000000 \
  --trainer.max_steps 200 \
  -n coarse_init_check > logs/coarse_init_check.log 2>&1
echo "──── kernel 橫幅（要看到 color=SH0+SB2 floats/point=25，不是 sh0）────"
grep -o '\[kernel\].*' logs/coarse_init_check.log | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -2
grep -oE 'active_sh_degree[^ ]*|sh_degree[^ ]*' logs/coarse_init_check.log | tail -3
