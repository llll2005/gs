#!/bin/bash
# ★★★ acs 的安全檢查（純 CPU，數秒）：反覆減半有沒有把尺度打到零？
#
# shrink 在**每個 densify 事件**（150 步）都做，30k 步內約 193 次。
# 若「AC 前 5%」的成員固定不變，尾端會是 0.5^193 => 被消滅。
# step 1499（約 3 次事件）實測 p0.1 只降到 0.51x（同一批應為 0.125x）=> 成員有在換。
# 本檢查在 14999（約 93 次事件）再驗一次；尾端若崩就該中止跑次。
set -u
cd "$(dirname "$0")/.." || exit 1
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import torch, glob, numpy as np, sys, os
sys.path.insert(0, os.getcwd())
print(f"{'跑次@步數':>20} {'N':>10} {'scale中位':>12} {'p1':>11} {'p0.1':>11} {'<1e-4':>8}")
bad = False
for r in ("agd2_b12", "acs2_b12"):
    p = glob.glob(f"outputs/{r}/**/*step=14999.ckpt", recursive=True)
    if not p:
        print(f"{r+'@14999':>20}  （尚無 ckpt）"); continue
    sc = torch.exp(torch.load(p[0], map_location="cpu")["state_dict"]["gaussian_model.gaussians.scales"].float())
    m = sc.max(dim=1).values.numpy()
    frac = float((m < 1e-4).mean())
    print(f"{r+'@14999':>20} {len(m):>10,} {np.median(m):>12.3e} {np.percentile(m,1):>11.3e} "
          f"{np.percentile(m,0.1):>11.3e} {frac:>7.2%}")
    if r == "acs2_b12" and frac > 0.01:
        bad = True
print("\n判準：acs 的 <1e-4 比例 > 1% 或 p0.1 比 agd2 低兩個數量級 => 尺度正在塌，應中止")
print("⚠ 尺度塌陷會表現成「顆數正常但畫面變空」——分數看不出成因")
sys.exit(1 if bad else 0)
PY
