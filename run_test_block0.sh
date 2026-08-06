#!/bin/bash
cd "$(dirname "$0")"
# The trained checkpoint used sh_degree=0 (bug now fixed for future runs).
# Override sh_degree to 0 here so the model shape matches the checkpoint.
python main.py test \
  --config outputs/depth_init_block0_full/blocks/block_0/lightning_logs/version_0/config.yaml \
  --model.gaussian.init_args.sh_degree 0 \
  --model.save_val_output true \
  --save_val
