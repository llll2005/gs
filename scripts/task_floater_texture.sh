#!/bin/bash
# Are floaters where the gradient is starved? Premise test for the whole "gradient starvation"
# account, CPU only (--views reads GT images; no model rendering).
#
# 2026-08-09: water ghosting and near-camera floaters were unified under one mechanism -- a
# primitive close to the camera projects a huge screen footprint, and over flat content (water,
# sky) it can explain the mean colour cheaply and then receives almost no gradient telling it to
# move. That account predicts floaters sit preferentially over LOW-texture regions.
# It also survived a falsification the init account did not: uniform_60k (no shape at all) still
# produces floaters, 0.463% vs depth-init's 1.802% -- so init amplifies ~4x but does not cause.
#
# If floaters turn out to be spread evenly across texture levels, the mechanism is wrong and the
# cross-view geometric consistency idea built on it should not be implemented.
set -u
cd "$(dirname "$0")/.." || exit 1
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/measure_floater_texture.py \
  --runs cap4m_b12 uniform_60k_b12 oreg_0p002_b12 blurbudget_b12 --views 12
