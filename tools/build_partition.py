"""Materialise the dataparser's partition cache for a given block_dim, without training.

utils/depth_init_blocks.py reads <dataset>/partition/partitions-dim_<BX>_<BY>_visibility_<t>/ but
does not create it -- the partition is built by the dataparser when a training run loads the data.
So a fresh grid needs this step first; running depth-init straight away fails with
"Partition directory not found" (2026-07-29).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "probes/p5_cost_aware"))

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--block", type=int, default=0)
args = ap.parse_args()

from train_p5 import build_sets

train_set, _ = build_sets(args.config, args.data, args.block, "/tmp/claude-1000/_bp")
print(f"[partition] built; block {args.block} has {len(train_set.cameras)} cameras")
pdir = os.path.join(args.data, "partition")
for d in sorted(os.listdir(pdir)):
    print(f"  {d}")
