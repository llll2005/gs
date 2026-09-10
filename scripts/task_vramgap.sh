#!/bin/bash
# ★★★★★ 追查 VRAM 的 2.4 GB 缺口（~5 分鐘）—— 使用者要求「必須嚴察」
# step_breakdown 只量裸 renderer 呼叫（forward 1.214 / fwd+bwd 1.901），
# 不含 optimizer step / density controller / **trim pass（渲染全部 284 台相機）**。
# 若 trim 佔最大 => 峰值由**每 500 步一次的間歇事件**決定，而 cap_max 是為它被壓低的
# => 「trim 分批」可能直接換到更高的 cap。
set -u
cd "$(dirname "$0")/.." || exit 1
conda run -n gspl --no-capture-output python -u tools/vram_gap.py --run agd2_b12 --step 60000
