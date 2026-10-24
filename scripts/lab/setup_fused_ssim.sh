#!/bin/bash
# 2026-10-24：在 lab 的 gspl env 裝 fused_ssim（Taming 3DGS 作者的融合 SSIM CUDA kernel，github.com/rahul-goel/fused-ssim）
#   用途：metric.fused_ssim=true（現行 internal/utils/ssim.py 是 5 次 11x11 depthwise conv2d）。
#   commit 與本機 env 裝的同一版（a7c48d6）；只裝這個套件（--no-deps），不動光柵器與 torch。
#   lab 編譯要強制 CUDA_HOME 指到 env（容器預設是系統 nvcc 12.9，torch 是 cu118；同 setup/lab_pull.sh）。
#   裝完自我檢查：與現行 ssim、float64 真值比 SSIM 值與梯度，並計時。
set -e
cd "$(dirname "$0")/../.."
#   ⚠ 容器的 PYTHONPATH 指向系統 python3.12 的 torch => 一律走 conda run（同 _common.sh），不直接呼叫 env 的 python
G=/root/miniconda3/envs/gspl
if ! conda run -n gspl python -c "import fused_ssim" >/dev/null 2>&1; then
  conda run -n gspl --no-capture-output env CUDA_HOME=$G CUDA_PATH=$G TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps \
    "git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38" 2>&1 | tail -25
fi
conda run -n gspl --no-capture-output python - <<'EOF'
import sys, time, torch
sys.path.insert(0, ".")
from fused_ssim import fused_ssim
from internal.utils.ssim import ssim
torch.manual_seed(0)
gt = torch.rand(3, 900, 1600, device="cuda")
gt = torch.nn.functional.avg_pool2d(gt[None], 5, 1, 2)[0]          # 帶點空間相關，接近真實影像
pr0 = (gt + 0.05 * torch.randn_like(gt)).clamp(0, 1)
r = {}
for name, f, dt in (("ssim", ssim, torch.float32), ("fused", lambda a, b: fused_ssim(a[None], b[None]), torch.float32),
                    ("ssim64", ssim, torch.float64)):
    pr = pr0.clone().to(dt).requires_grad_(True); v = f(pr, gt.to(dt)); v.backward()
    r[name] = (v.detach().double(), pr.grad.detach().double())
for k in ("ssim", "fused"):
    v, g = r[k]; v2, g2 = r["ssim64"]
    print(f"[fused-ssim 檢查] {k}: SSIM {float(v):.8f}，對 float64 差 {abs(float(v - v2)):.2e}，梯度相對 L2 {float((g - g2).norm() / g2.norm()):.2e}")
assert abs(float(r["fused"][0] - r["ssim64"][0])) < 1e-5, "fused_ssim 值與真值差太多"
for name, f in (("現行 ssim", ssim), ("fused_ssim", lambda a, b: fused_ssim(a[None], b[None]))):
    pr = pr0.clone().requires_grad_(True)
    for _ in range(5): f(pr, gt).backward()
    torch.cuda.synchronize(); t = time.perf_counter(); n = 50
    for _ in range(n):
        pr.grad = None; f(pr, gt).backward()
    torch.cuda.synchronize(); print(f"[fused-ssim 檢查] {name}: forward+backward {1000 * (time.perf_counter() - t) / n:.2f} ms（卡上可能有別的行程）")
print("✅ fused_ssim 可用")
EOF
