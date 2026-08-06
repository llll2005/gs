#!/bin/bash
cd "$(dirname "$0")"
python main.py fit \
  --config configs/citygsv2_mc_aerial_sh2_trim24.yaml \
  --data.parser.block_id 0 \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_0.ply \
  -n=depth_init_block0_full
