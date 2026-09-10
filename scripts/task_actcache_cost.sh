#!/usr/bin/env bash
# 代換 4 的前置（~2 分）：重複的 get_scales()/get_opacities() 到底值多少 ms/step
# 靜態估算只有 **0.06%**（exp/sigmoid over 2.34M，冗餘 2 次），遠低於我原本隨口說的 0.5%。
# 加快取的風險是「陳舊快取 = 靜默錯誤」（本專案已在不報錯的 bug 上摔過 7 次）
# => 先量到再決定要不要換這個風險。
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u - <<'PY'
import sys, glob, numpy as np, torch
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
dev = torch.device("cuda")
ck = sorted(glob.glob("outputs/agd2_b12/**/*step=60000.ckpt", recursive=True))[0]
model, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
    ck, device=dev, eval_mode=False, pre_activate=False)
def t(fn, n=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        a,b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return float(np.median(ts))
ms_s = t(lambda: model.get_scales()); ms_o = t(lambda: model.get_opacities())
print(f"  N = {model.n_gaussians:,}")
print(f"  get_scales()    {ms_s:.4f} ms   get_opacities() {ms_o:.4f} ms")
red = 2*(ms_s+ms_o)          # 每步約 3 次，其中 2 次冗餘
print(f"  每步冗餘（各 2 次）= **{red:.3f} ms** = **{100*red/602:.3f}%** of 602 ms")
print(f"  判準：>0.5% 才值得為它加快取（陳舊快取=靜默錯誤的風險）；<0.2% => 不做")
PY
