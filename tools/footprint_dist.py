#!/usr/bin/env python
"""兩個跑次在同一步的**投影半徑分布**比較（純 CPU，訓練中可跑）。

用途：驗證「改變 densify 瞄準」有沒有真的改變族群的足跡分布。
2026-09-05 在 `fpd_b12`（cost_aware_densify -0.5）vs `agd2_b12` 的 14999 上量到
**比值 p50 0.9923 / p90 1.0109 / p99 1.0139 / >300px 0.9981** —— 幾乎完全相同，
而同時取樣分布的 top10% 已被換掉一半（重疊 50.83%、ESS 483k->109k）。
⇒ 「改變往哪裡增生」不改變足跡分布 ⇒ 成本分布 c_i 像是**平衡態性質**。

用法: python tools/footprint_dist.py agd2_b12 fpd_b12 --step 29999
"""
import sys, glob; sys.path.insert(0, "/home/LnoArch/Projects/專題/CityGaussian")
import torch, numpy as np
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("runs", nargs=2)
_ap.add_argument("--step", type=int, default=14999)
_ap.add_argument("--ncam", type=int, default=24)
_A = _ap.parse_args(); RUNS = _A.runs


def radii(run, step=None, ncam=None):
    step = step or _A.step; ncam = ncam or _A.ncam
    p = glob.glob(f"outputs/{run}/**/*step={step}.ckpt", recursive=True)[0]
    c = torch.load(p, map_location="cpu"); sd = c["state_dict"]
    mu = sd["gaussian_model.gaussians.means"].numpy().astype(np.float64)
    s  = np.exp(sd["gaussian_model.gaussians.scales"].numpy().astype(np.float64)).max(1)
    dmh = c["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"], output_path="/tmp", global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    mx = np.zeros(mu.shape[0]); seen = np.zeros(mu.shape[0], np.int32)
    for i in range(min(ncam, len(cams))):
        cam = cams[i]
        R = np.asarray(cam.R.cpu() if torch.is_tensor(cam.R) else cam.R, np.float64)
        T = np.asarray(cam.T.cpu() if torch.is_tensor(cam.T) else cam.T, np.float64)
        fx = float(cam.fx); W, H = int(cam.width), int(cam.height)
        pc = mu @ R.T + T; z = pc[:,2]; zc = np.clip(z, .2, None)
        r = 3.0*fx*s/zc
        u = fx*pc[:,0]/zc + W/2; v = fx*pc[:,1]/zc + H/2
        vis = (z>.2)&(u>-r)&(u<W+r)&(v>-r)&(v<H+r)
        mx[vis] = np.maximum(mx[vis], r[vis]); seen[vis]+=1
    return mx[seen>0], s
print(f"{'run':>10} {'顆數(可見)':>11} {'半徑p50':>9} {'p90':>9} {'p99':>9} {'max':>10} "
      f"{'>300px':>8} {'>12px':>8}")
out={}
for r in RUNS:
    m, s = radii(r); out[r]=(m,s)
    print(f"{r:>10} {len(m):>11,} {np.percentile(m,50):>9.2f} {np.percentile(m,90):>9.2f} "
          f"{np.percentile(m,99):>9.2f} {m.max():>10.1f} "
          f"{100*(m>300).mean():>7.3f}% {100*(m>12).mean():>7.2f}%")
a,_=out[RUNS[0]]; b,_=out[RUNS[1]]
print(f"\n  比值(fpd/agd2)  p50 {np.percentile(b,50)/np.percentile(a,50):.4f}  "
      f"p90 {np.percentile(b,90)/np.percentile(a,90):.4f}  "
      f"p99 {np.percentile(b,99)/np.percentile(a,99):.4f}  "
      f">300px {(b>300).mean()/max((a>300).mean(),1e-12):.4f}")
