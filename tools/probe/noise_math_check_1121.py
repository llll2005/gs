#!/usr/bin/env python
"""★ 11-21 noise_math 整合檢查（不訓練）：真實 density 設定＋b2 模型，現行（fast_noise＋閘門）vs noise_math 各跑 R 次，
比「哪些顆粒被移動」與位移平方和的期望（同分布 => 比值 ≈ 1，誤差 ~ sqrt(2/(R·有效自由度))）、以及法向分量（應為 0）。"""
import os, sys, types
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from internal.utils.gaussian_model_loader import GaussianModelLoader
CK = sys.argv[1] if len(sys.argv) > 1 else "outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt"
R = 300
dev = torch.device("cuda")
model, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(CK, device=dev, eval_mode=False, pre_activate=False)
ck = torch.load(CK, map_location="cpu", weights_only=False)
ctl = ck["hyper_parameters"]["density"].instantiate()
pl = types.SimpleNamespace(gaussian_optimizers=[types.SimpleNamespace(param_groups=[{"name": "means", "lr": 1.6e-5}])], is_final_step=False)
m0 = model.means.detach().clone()
q = torch.nn.functional.normalize(model.gaussians["rotations"].detach(), dim=-1); w, a, b, c = q.unbind(-1)
nrm = torch.stack([2 * (a * c + w * b), 2 * (b * c - w * a), 1 - 2 * (a * a + b * b)], -1)     # R 的第 3 欄（面元法向）
res = {}
for nm in (False, True):
    ctl.config.noise_math = nm
    acc = torch.zeros(m0.shape[0], device=dev, dtype=torch.float64); moved = torch.zeros(m0.shape[0], dtype=torch.bool, device=dev); nmax = 0.0
    for _ in range(R):
        with torch.no_grad():
            model.means.copy_(m0)
        ctl._add_xyz_noise({}, None, model, 5000, pl)
        d = (model.means.detach() - m0)
        acc += (d.double() ** 2).sum(-1); moved |= (d != 0).any(-1)
        nmax = max(nmax, float(((d * nrm).sum(-1).abs() / d.norm(dim=-1).clamp_min(1e-30)).max()))
    res[nm] = (acc / R, moved, nmax)
    print(f"noise_math={nm}：被移動 {int(moved.sum()):,} 顆｜Σ E|Δ|² {float(acc.sum() / R):.4e}｜法向分量／位移 max {nmax:.1e}")
a0, mv0, _ = res[False]; a1, mv1, _ = res[True]
print(f"被移動的集合相同：{bool(torch.equal(mv0, mv1))}｜Σ E|Δ|² 比值 {float(a1.sum() / a0.sum()):.4f}｜"
      f"逐顆 E|Δ|² 比值（前 10% 大位移的顆粒）中位 {float((a1 / a0.clamp_min(1e-40))[a0 > a0.quantile(0.9)].median()):.3f}")
with torch.no_grad():
    model.means.copy_(m0)

# ── 診斷：法向分量的分布、位移平方和被誰主導、原寫法的 R 慣例
from internal.utils.gaussian_projection import build_rotation_matrix
Rp = build_rotation_matrix(model.gaussians["rotations"].detach())
q2 = torch.nn.functional.normalize(model.gaussians["rotations"].detach(), dim=-1); w, a, b, c = q2.unbind(-1)
t1c = torch.stack([1 - 2 * (b * b + c * c), 2 * (a * b + w * c), 2 * (a * c - w * b)], -1)
print(f"build_rotation_matrix 第 1 欄 vs CUDA 慣例第 1 欄：max 差 {float((Rp[:, :, 0] - t1c).abs().max()):.1e}｜第 1 列 vs 第 1 欄 max 差 {float((Rp[:, 0, :] - t1c).abs().max()):.1e}")
for nm in (False, True):
    ctl.config.noise_math = nm
    rat = []
    for _ in range(20):
        with torch.no_grad():
            model.means.copy_(m0)
        ctl._add_xyz_noise({}, None, model, 5000, pl)
        d = (model.means.detach() - m0); mv = (d != 0).any(-1)
        rat.append(((d[mv] * nrm[mv]).sum(-1).abs() / d[mv].norm(dim=-1)))
    rat = torch.cat(rat)
    print(f"noise_math={nm}：法向分量／位移 中位 {float(rat.median()):.1e}、p99 {float(rat.quantile(0.99)):.1e}、> 0.5 的比例 {float((rat > 0.5).float().mean()):.2%}")
a0, mv0, _ = res[False]; a1, mv1, _ = res[True]
top = torch.argsort(a0, descending=True)[:10]
s_ = torch.exp(model.gaussians["scales"].detach())
print("位移平方最大的 10 顆：E|Δ|²（原／新）、s1、s2、o")
for i in top.tolist():
    print(f"   {float(a0[i]):.3e} / {float(a1[i]):.3e}  s=({float(s_[i, 0]):.4f}, {float(s_[i, 1]):.4f})  o={float(torch.sigmoid(model.gaussians['opacities'][i])):.5f}")
print(f"前 10 顆佔原寫法總和 {float(a0[top].sum() / a0.sum()):.1%}")
with torch.no_grad():
    model.means.copy_(m0)
