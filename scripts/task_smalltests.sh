#!/bin/bash
# ★★★★ 小測試系列（約 20 分鐘）—— 使用者指示：先把機制問題測完再決定要不要投大實驗
#
# 依據：2026-08-26/27 四個 9.6h 的介入全輸，而所有真正的產出都來自 5~10 分鐘的診斷
#      （天花板 1.22 vs 16.68、rho(v,c)=0.127、疊影與配方無關、噪音底、零權重 loss −9%）。
#      **診斷能拒絕整個家族，介入一次只能測一個點。**
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "################ 1. fused_ssim：數值/梯度一致性 + 速度 ################"
# `vanilla_metrics.py:22` 的 `fused_ssim` 旗標**預設 False**，但套件已安裝。
# fallback 每次呼叫都 create_window（重建 11x11 高斯窗）再做 5 次非可分離 conv2d，
# 3 通道 x 1.44M 像素，且 RGB SSIM 每步都在主 loss 裡（lambda_dssim=0.2）。
# ⚠ 不是位元等價 => 先看梯度差多大，再決定敢不敢換。
conda run -n gspl python - <<'PY'
import torch, time, sys
sys.path.insert(0,'.')
from internal.utils.ssim import ssim
from fused_ssim import fused_ssim
torch.manual_seed(0); d='cuda'; H,W=900,1600
a=torch.rand(3,H,W,device=d,requires_grad=True); b=torch.rand(3,H,W,device=d)
v1=float(ssim(a,b)); v2=float(fused_ssim(a.unsqueeze(0),b.unsqueeze(0)))
print('數值  內建 %.6f  fused %.6f  絕對差 %.2e'%(v1,v2,abs(v1-v2)))
g1=torch.autograd.grad(ssim(a,b),a)[0]; g2=torch.autograd.grad(fused_ssim(a.unsqueeze(0),b.unsqueeze(0)),a)[0]
rel=float((g1-g2).abs().max()/g1.abs().max().clamp_min(1e-30))
print('梯度  最大絕對差 %.2e  相對 %.2e  corr %.6f'%(float((g1-g2).abs().max()),rel,
      float(torch.corrcoef(torch.stack([g1.flatten(),g2.flatten()]))[0,1])))
def bench(f,n=30):
    for _ in range(3): torch.autograd.grad(f(),a)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): torch.autograd.grad(f(),a)
    torch.cuda.synchronize(); return (time.time()-t)/n*1000
t1=bench(lambda: ssim(a,b)); t2=bench(lambda: fused_ssim(a.unsqueeze(0),b.unsqueeze(0)))
print('速度(fwd+bwd)  內建 %.2f ms  fused %.2f ms  => %.2fx，每步省 %.2f ms'%(t1,t2,t1/t2,t1-t2))
print('   （對照：整步 ~490 ms @2.34M；metric 閘門已省 ~39 ms）')
PY

echo
echo "################ 2. tau_current：K-strip 那條線值不值得復活 ################"
conda run -n gspl python tools/measure_tau.py --run sched30_b12 \
  --block-list data/matrix_city/aerial/train/block_all/partition/partitions-dim_5_5_visibility_0.08/002_002.txt \
  || echo "!! tau 量測失敗"

echo
echo "################ 完成。判讀見各段輸出 ################"
