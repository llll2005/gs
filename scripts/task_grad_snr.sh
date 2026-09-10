#!/usr/bin/env bash
# Adam 看見的是 m/sqrt(v)，不是 |g|。§11.80 的「幅度少 4.6 倍」在 Adam 之下站不站得住？
# Adam 對梯度幅度尺度不變 => 均勻縮小不改變步長。真正會崩的是訊噪比（抵消壓 m 不壓 v）。
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/grad_snr.py agd2_b12 sched30_b12 --blk 12
