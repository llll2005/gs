#!/usr/bin/env bash
# 足跡當增生訊號 + 它與 |g| 的重疊。回答「現行最佳配方 absgrad 的增益到底來自哪個訊號」。
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/footprint_signal.py --run agd2_b12 --with-grad
