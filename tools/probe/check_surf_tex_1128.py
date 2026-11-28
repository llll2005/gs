"""11-28 面內亮度漸變（surf_tex）的正確性檢查（本機 GPU、不訓練）。
  (A) 沒有 tex vs tex 全 0：render 逐位元相同？其他參數的梯度差 vs 同一設定跑兩次的雜訊底。
  (B) tex 隨機非 0：方向導數 有限差分 vs autograd —— tex 本身、位置（顏色斜率推交點那一項）；對照組＝不加 tex 時位置的同一檢查（量測本身的準度）。
兩條路徑：lean（訓練一般步，renderCUDA_opt）與完整（幾何步／評估，renderCUDA GEOM）。
用法：python tools/probe/check_surf_tex_1128.py [--ckpt outputs/lab/view_block_1_base9c15/checkpoints/view_block_1_base9c15.ckpt]
"""
import argparse
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "."); sys.path.insert(0, "tools")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_1_base9c15/checkpoints/view_block_1_base9c15.ckpt")
ap.add_argument("--block", type=int, default=1)
ap.add_argument("--m39", type=int, default=1, help="1＝開反單應交點等 m39 的光柵器路徑（現行配方）")
ap.add_argument("--keep", type=float, default=1.0, help="只留這個比例的顆粒（本機 6 GB 卡完整路徑會 OOM）")
ap.add_argument("--paths", default="lean,full")
a = ap.parse_args()

from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot

dev = torch.device("cuda")
torch.manual_seed(0)
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=False, pre_activate=False)
for f in ("absgrad_gate", "record_reduce", "tile_cull", "trim_running_vc", "trim_running_warp"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
for f in ("lean_train", "bwd_sat_skip", "bwd_reduce", "bwd_pair_bit", "bwd_pair_bit_warp", "skip_box"):
    if hasattr(renderer, f):
        setattr(renderer, f, True)
if a.m39:
    for f in ("inv_homog", "exact_strip_cull"):
        setattr(renderer, f, True)
if a.keep < 1.0:
    with torch.no_grad():
        km = torch.rand(model.n_gaussians, device=dev) < a.keep
        for k in list(model.gaussians.keys()):
            model.gaussians[k] = nn.Parameter(model.gaussians[k].data[km].clone())
N = model.n_gaussians
names0 = tuple(model._names)
print(f"[模型] {a.ckpt} N={N:,}｜renderer lean={getattr(renderer, 'lean_train', None)} inv_homog={getattr(renderer, 'inv_homog', None)}", flush=True)

names_t, cams = eot.load_test_cameras("data/matrix_city/aerial/test/block_all_test_official2", 1.2)
lo, hi, _ = eot.block_bounds("data/matrix_city/aerial/train/block_all", a.block, [4, 4])
cen = np.array([(-cams.R[i].numpy().T @ cams.T[i].numpy()) for i in range(len(names_t))])
cand = [i for i in range(len(names_t)) if np.all(cen[i] >= lo) and np.all(cen[i] <= hi)]
bg = torch.zeros(3, device=dev)
best, ci = -1, cand[0]
with torch.no_grad():
    for i in cand[::4]:                          # 挑看到最多顆粒的那一幀
        nv = int((renderer(cams[i].to_device(dev), model, bg_color=bg)["radii"] > 0).sum())
        if nv > best:
            best, ci = nv, i
cam = cams[ci].to_device(dev)
print(f"[相機] test 幀 {names_t[ci]}，畫面內 {best:,} 顆", flush=True)
W = None


def set_tex(t):
    if t is None:
        model._names = names0
        if "tex" in model.gaussians:
            del model.gaussians["tex"]
    else:
        model._names = names0 + ("tex",)
        model.gaussians["tex"] = t


def loss_and_grads(lean, keys):
    global W
    for p in model.gaussians.values():
        if isinstance(p, torch.Tensor) and p.grad is not None:
            p.grad = None
    out = renderer(cam, model, bg_color=bg, **({"_lean": True} if lean else {}))
    img = out["render"]
    if W is None:
        W = torch.randn_like(img)
    L = (img * W).sum()
    L.backward()
    return float(L), img.detach().clone(), {k: model.gaussians[k].grad.detach().clone() for k in keys if model.gaussians[k].grad is not None}


def loss_only(lean):
    with torch.no_grad():
        out = renderer(cam, model, bg_color=bg, **({"_lean": True} if lean else {}))
        return float((out["render"] * W).sum().double())


keys = ("means", "scales", "rotations", "opacities", "shs_dc")
for lean in [p == "lean" for p in a.paths.split(",")]:
    path = "lean" if lean else "完整"
    print(f"\n===== 路徑：{path} =====", flush=True)
    set_tex(None)
    L0, I0, G0 = loss_and_grads(lean, keys)
    L0b, I0b, G0b = loss_and_grads(lean, keys)
    set_tex(nn.Parameter(torch.zeros((N, 2), device=dev)))
    L1, I1, G1 = loss_and_grads(lean, keys + ("tex",))
    print(f"(A) render：沒有 tex vs tex 全 0 最大差 {float((I0 - I1).abs().max()):.3e}（同設定兩次 {float((I0 - I0b).abs().max()):.3e}）")
    for k in keys:
        if k in G0 and k in G1:
            nz = float((G0[k] - G0b[k]).norm() / G0[k].norm().clamp_min(1e-30)); dz = float((G0[k] - G1[k]).norm() / G0[k].norm().clamp_min(1e-30))
            print(f"    {k:<10} 梯度相對差 {dz:.2e}（雜訊底 {nz:.2e}）")
    print(f"    tex 梯度（tex＝0 時）範數 {float(G1['tex'].norm()):.3e}")

    # (B) 方向導數
    tex0 = (0.15 * torch.randn((N, 2), device=dev))
    set_tex(nn.Parameter(tex0.clone()))
    L, _, G = loss_and_grads(lean, ("tex", "means"))
    v = torch.randn_like(tex0)
    for eps in (1e-2, 3e-3):
        model.gaussians["tex"].data = tex0 + eps * v; Lp = loss_only(lean)
        model.gaussians["tex"].data = tex0 - eps * v; Lm = loss_only(lean)
        num = (Lp - Lm) / (2 * eps); ana = float((G["tex"] * v).sum())
        print(f"(B) tex 方向導數 eps {eps:g}：有限差分 {num:.6e}｜autograd {ana:.6e}｜相對差 {abs(num - ana) / max(abs(ana), 1e-30):.2e}")
    model.gaussians["tex"].data = tex0
    # 位置：tex 那一項單獨檢查——同一個位置擾動下 [L(tex) − L(0)] 的有限差分 vs autograd 的 [∂L/∂m(tex) − ∂L/∂m(0)]
    #   alpha 截斷／排序這些不連續在兩邊相同（幾何一樣），相減後大半抵掉，剩下的才是顏色斜率推交點的那一項
    gm = G["means"].norm(dim=-1)
    idx = torch.topk(gm, 2000).indices
    vm = torch.zeros_like(model.gaussians["means"]); vm[idx] = torch.randn((2000, 3), device=dev)
    m0 = model.gaussians["means"].data.clone()
    model.gaussians["tex"].data = tex0; _, _, Gt = loss_and_grads(lean, ("means",))
    model.gaussians["tex"].data = torch.zeros_like(tex0); _, _, Gz = loss_and_grads(lean, ("means",))
    ana = float(((Gt["means"] - Gz["means"]) * vm).sum())
    for eps in (3e-4, 1e-4, 3e-5):
        vals = []
        for t in (tex0, torch.zeros_like(tex0)):
            model.gaussians["tex"].data = t
            model.gaussians["means"].data = m0 + eps * vm; Lp = loss_only(lean)
            model.gaussians["means"].data = m0 - eps * vm; Lm = loss_only(lean)
            model.gaussians["means"].data = m0
            vals.append((Lp - Lm) / (2 * eps))
        num = vals[0] - vals[1]
        print(f"(B) 位置方向導數裡 tex 那一項 eps {eps:g}：有限差分 {num:.6e}｜autograd {ana:.6e}｜相對差 {abs(num - ana) / max(abs(ana), 1e-30):.2e}"
              f"（整體量級 {abs(vals[0]):.3e}）")
    model.gaussians["tex"].data = tex0
    set_tex(None)
print("\nDONE")
