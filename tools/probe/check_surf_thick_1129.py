"""11-29 固定厚度（surf_thick）的正確性檢查（本機 GPU、不訓練）。
  (A) 單顆合成面元：光柵器的 alpha vs Python 數值解（沿射線密集取樣 3D Mahalanobis 的最小值），含斜看與幾乎側對；
      只比 ray-splat 分支的像素（rho' < rho2d），厚度 0 時也比一次（對照：原本 2DGS 的 |s|^2）。
  (B) 真模型：厚度比例 0（逐顆 property thick 全 0）vs 不開 => render 逐位元相同？
  (C) 真模型開厚度：逐顆有限差分 vs autograd（位置、log 尺度、旋轉；挑梯度最大的 K 顆、一次動一顆）；
      對照組＝不開厚度的同一檢查（量測本身的準度）。lean（訓練一般步）與完整路徑各一次。
用法：python tools/probe/check_surf_thick_1129.py [--ckpt ...] [--ratio 0.3] [--keep 0.3]
"""
import argparse
import math
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "."); sys.path.insert(0, "tools")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_1_texref1129/checkpoints/view_block_1_texref1129.ckpt")
ap.add_argument("--block", type=int, default=1)
ap.add_argument("--ratio", type=float, default=0.3)
ap.add_argument("--keep", type=float, default=0.3, help="只留這個比例的顆粒（本機 6 GB 卡）")
ap.add_argument("--K", type=int, default=24)
ap.add_argument("--paths", default="lean,full")
ap.add_argument("--skip", default="", help="跳過的段落（A,B,C）")
a = ap.parse_args()

from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot

dev = torch.device("cuda")
torch.manual_seed(0)
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=False, pre_activate=False)
for f in ("absgrad_gate", "record_reduce", "tile_cull", "trim_running_vc", "trim_running_warp"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
for f in ("lean_train", "bwd_sat_skip", "bwd_reduce", "bwd_pair_bit", "bwd_pair_bit_warp", "skip_box", "inv_homog", "exact_strip_cull"):
    if hasattr(renderer, f):
        setattr(renderer, f, True)
names0 = tuple(model._names)
bg = torch.zeros(3, device=dev)
names_t, cams = eot.load_test_cameras("data/matrix_city/aerial/test/block_all_test_official2", 1.2)
lo, hi, _ = eot.block_bounds("data/matrix_city/aerial/train/block_all", a.block, [4, 4])
cen = np.array([(-cams.R[i].numpy().T @ cams.T[i].numpy()) for i in range(len(names_t))])
cand = [i for i in range(len(names_t)) if np.all(cen[i] >= lo) and np.all(cen[i] <= hi)]


def set_thick(r):
    """r=None：不開；r=float：renderer 比例；r=Tensor：逐顆 property"""
    model._names = names0
    if "thick" in model.gaussians:
        del model.gaussians["thick"]
    renderer.surf_thick_ratio = 0.0
    if isinstance(r, torch.Tensor):
        model._names = names0 + ("thick",)
        model.gaussians["thick"] = nn.Parameter(r, requires_grad=False)
    elif r is not None:
        renderer.surf_thick_ratio = float(r)


# ───────── (A) 單顆合成面元 ─────────
def quat_from_R(Rm):
    m = Rm
    w = math.sqrt(max(1e-12, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return np.array([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])


if "A" not in a.skip:
    print("===== (A) 單顆合成面元：光柵器 alpha vs 數值解 =====", flush=True)
    cam = cams[cand[0]].to_device(dev)
    Rc = cam.R.double().cpu().numpy(); Tc = cam.T.double().cpu().numpy()
    C = -Rc.T @ Tc
    fwd = Rc[2]                                                   # 相機前方（世界）
    keep_props = {k: v.data.clone() for k, v in model.gaussians.items()}
    rng = np.random.default_rng(1)
    W, H = int(cam.width), int(cam.height)
    fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
    for trial, (tilt, ratio) in enumerate([(30, 0.0), (30, 0.3), (60, 0.3), (80, 0.3), (88, 1.0), (89.5, 0.3)]):
        mu = C + fwd * 0.8                                        # 相機前方 0.8 單位
        # 法向與視線夾角 = tilt
        a_ = rng.normal(size=3); a_ -= a_.dot(fwd) * fwd; a_ /= np.linalg.norm(a_)
        n = math.cos(math.radians(tilt)) * (-fwd) + math.sin(math.radians(tilt)) * a_
        tu = np.cross(n, rng.normal(size=3)); tu /= np.linalg.norm(tu); tv = np.cross(n, tu)
        Rm = np.stack([tu, tv, n], 1)
        q = quat_from_R(Rm)
        su, sv = 0.02, 0.012
        with torch.no_grad():
            for k, v in model.gaussians.items():
                model.gaussians[k] = nn.Parameter(v.data[:1].clone(), requires_grad=v.requires_grad)
            model.gaussians["means"].data[:] = torch.tensor(mu, dtype=torch.float32, device=dev)
            model.gaussians["rotations"].data[:] = torch.tensor(q, dtype=torch.float32, device=dev)
            sc = model.gaussians["scales"].data
            sc[:] = torch.log(torch.tensor([su, sv] + [su] * (sc.shape[1] - 2), dtype=torch.float32, device=dev))
            model.gaussians["opacities"].data[:] = 10.0               # sigmoid ≈ 1 => alpha = min(0.99, G)
        set_thick(ratio if ratio > 0 else None)
        with torch.no_grad():
            out = renderer(cam, model, bg_color=bg)
        al = out["rend_alpha"][0].double().cpu().numpy()
        # 數值解：每個像素的射線，沿射線取 3D Mahalanobis 最小值
        sw = ratio * min(su, sv)
        ys, xs = np.nonzero(al > 1e-4)
        if len(xs) == 0:
            print(f"  tilt {tilt}° 比例 {ratio}：畫面上沒有像素"); continue
        sel = rng.choice(len(xs), size=min(3000, len(xs)), replace=False)
        xs, ys = xs[sel], ys[sel]
        dc = np.stack([(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, np.ones_like(xs, dtype=float)], 1)   # 相機座標
        dw = dc @ Rc                                                  # 世界方向（R^T d）
        o = C - mu
        P = np.stack([o @ tu / su, o @ tv / sv, o @ n / (sw if sw > 0 else 1e-30)])            # 相機在面元正規化座標
        D = np.stack([dw @ tu / su, dw @ tv / sv, dw @ n / (sw if sw > 0 else 1e-30)], 1)      # 方向
        if sw > 0:
            t = -(D @ P) / (D * D).sum(1)
            X = P[None] + t[:, None] * D
            rho = (X * X).sum(1)
        else:                                                         # 2DGS：交平面
            t = -P[2] / D[:, 2]
            X = P[None, :2] + t[:, None] * D[:, :2]
            rho = (X * X).sum(1)
        # 低通：螢幕上到中心的距離
        pc = Rc @ mu + Tc
        px, py = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
        rho2d = 2.0 * ((xs + 0.5 - px) ** 2 + (ys + 0.5 - py) ** 2)    # FilterInvSquare = 2
        m = rho < rho2d - 1e-3
        ana = np.minimum(0.99, np.exp(-0.5 * rho))
        ok = m & (ana > 1.5 / 255)
        err = np.abs(al[ys, xs] - ana)[ok]
        print(f"  法向與視線夾角 {tilt:>5}° 厚度比例 {ratio}：ray-splat 像素 {int(ok.sum()):>5}｜|alpha − 數值解| 中位 {np.median(err) if err.size else float('nan'):.2e}"
              f" 最大 {err.max() if err.size else float('nan'):.2e}｜畫面 alpha>0.01 像素 {int((al > 0.01).sum())}", flush=True)
    # (A2) 同一顆合成面元的梯度：單顆、opacity 0.5（不碰 0.99 上限）=> 唯一的不連續是 1/255 截斷，有限差分可靠
    print("===== (A2) 單顆合成面元：有限差分 vs autograd（lean／完整；位置、log 尺度、旋轉各 4 個方向）=====", flush=True)
    Wa = None
    for tilt, ratio in [(45, 0.0), (45, 0.3), (75, 0.3), (85, 1.0)]:
        mu = C + fwd * 0.8
        a_ = rng.normal(size=3); a_ -= a_.dot(fwd) * fwd; a_ /= np.linalg.norm(a_)
        n = math.cos(math.radians(tilt)) * (-fwd) + math.sin(math.radians(tilt)) * a_
        tu = np.cross(n, rng.normal(size=3)); tu /= np.linalg.norm(tu); tv = np.cross(n, tu)
        q = quat_from_R(np.stack([tu, tv, n], 1))
        with torch.no_grad():
            model.gaussians["means"].data[:] = torch.tensor(mu, dtype=torch.float32, device=dev)
            model.gaussians["rotations"].data[:] = torch.tensor(q, dtype=torch.float32, device=dev)
            sc = model.gaussians["scales"].data
            sc[:] = torch.log(torch.tensor([0.02, 0.012] + [0.02] * (sc.shape[1] - 2), dtype=torch.float32, device=dev))
            model.gaussians["opacities"].data[:] = 0.0
        set_thick(ratio if ratio > 0 else None)
        with torch.no_grad():   # 權重只放在 alpha > 0.05 的像素（遠離 1/255 截斷的邊緣 => 有限差分不會吃到截斷跳動）
            o0 = renderer(cam, model, bg_color=bg)
            Wa = torch.randn_like(o0["render"]) * (o0["rend_alpha"] > 0.05).float()
        for lean in (True, False):
            for p_ in model.gaussians.values():
                p_.requires_grad_(True); p_.grad = None
            img = renderer(cam, model, bg_color=bg, **({"_lean": True} if lean else {}))["render"]
            (img * Wa).sum().backward()
            out = []
            for key, eps in (("means", 1e-4), ("scales", 1e-3), ("rotations", 1e-3)):   # 位置 1e-4（座標約 5 單位，1e-5 只有 20 個 float ULP）
                g = model.gaussians[key].grad[0].detach().clone()
                rel = []
                for _ in range(4):
                    v = torch.randn(g.shape[0], device=dev); v /= v.norm()
                    p0 = model.gaussians[key].data[0].clone()
                    vals = []
                    for sg in (1, -1):
                        model.gaussians[key].data[0] = p0 + sg * eps * v
                        with torch.no_grad():
                            vals.append(float((renderer(cam, model, bg_color=bg, **({"_lean": True} if lean else {}))["render"] * Wa).double().sum()))
                    model.gaussians[key].data[0] = p0
                    num = (vals[0] - vals[1]) / (2 * eps); an = float((g * v).sum())
                    rel.append(abs(num - an) / max(abs(an), abs(num), 1e-12))
                out.append(f"{key} {np.median(rel):.1e}")
            print(f"  夾角 {tilt}° 厚度比例 {ratio} {'lean' if lean else '完整'}：相對差中位 " + "｜".join(out), flush=True)
    with torch.no_grad():
        for k, v in keep_props.items():
            model.gaussians[k] = nn.Parameter(v.clone(), requires_grad=True)
    set_thick(None)

if a.keep < 1.0:
    with torch.no_grad():
        km = torch.rand(model.n_gaussians, device=dev) < a.keep
        for k in list(model.gaussians.keys()):
            model.gaussians[k] = nn.Parameter(model.gaussians[k].data[km].clone())
N = model.n_gaussians
best, ci = -1, cand[0]
with torch.no_grad():
    for i in cand[::4]:
        nv = int((renderer(cams[i].to_device(dev), model, bg_color=bg)["radii"] > 0).sum())
        if nv > best:
            best, ci = nv, i
cam = cams[ci].to_device(dev)
print(f"\n[模型] N={N:,}｜test 幀 {names_t[ci]}，畫面內 {best:,} 顆", flush=True)
Wt = None


def loss_grads(lean, keys):
    global Wt
    for p in model.gaussians.values():
        if isinstance(p, torch.Tensor) and p.grad is not None:
            p.grad = None
    img = renderer(cam, model, bg_color=bg, **({"_lean": True} if lean else {}))["render"]
    if Wt is None:
        Wt = torch.randn_like(img)
    L = (img * Wt).sum()
    L.backward()
    return float(L), img.detach().clone(), {k: model.gaussians[k].grad.detach().clone() for k in keys if model.gaussians[k].grad is not None}


def loss_only(lean):
    with torch.no_grad():
        return float((renderer(cam, model, bg_color=bg, **({"_lean": True} if lean else {}))["render"] * Wt).double().sum())


for lean in [p == "lean" for p in a.paths.split(",")]:
    path = "lean" if lean else "完整"
    print(f"\n===== 路徑：{path} =====", flush=True)
    if "B" not in a.skip:
        set_thick(None); _, I0, _ = loss_grads(lean, ("means",)); _, I0b, _ = loss_grads(lean, ("means",))
        set_thick(torch.zeros((N, 1), device=dev)); _, I1, _ = loss_grads(lean, ("means",))
        set_thick(a.ratio); _, I2, _ = loss_grads(lean, ("means",))
        print(f"(B) render：不開 vs 逐顆厚度 0 最大差 {float((I0 - I1).abs().max()):.3e}（同設定兩次 {float((I0 - I0b).abs().max()):.3e}）"
              f"｜開比例 {a.ratio}：最大差 {float((I0 - I2).abs().max()):.3e}、改動 >1/255 的像素 {100 * float(((I0 - I2).abs().amax(0) > 1 / 255).float().mean()):.2f}%")
    if "C" in a.skip:
        continue
    for label, r in (("對照（不開厚度）", None), (f"固定厚度 {a.ratio}", a.ratio)):
        set_thick(r)
        _, _, G = loss_grads(lean, ("means", "scales", "rotations"))
        res = {}
        for key, eps in (("means", 1e-5), ("scales", 1e-3), ("rotations", 1e-3)):
            g = G[key]
            idx = torch.topk(g.norm(dim=-1), a.K).indices
            rel = []
            for i in idx.tolist():
                v = torch.randn(g.shape[1], device=dev); v /= v.norm()
                p0 = model.gaussians[key].data[i].clone()
                model.gaussians[key].data[i] = p0 + eps * v; Lp = loss_only(lean)
                model.gaussians[key].data[i] = p0 - eps * v; Lm = loss_only(lean)
                model.gaussians[key].data[i] = p0
                num = (Lp - Lm) / (2 * eps); an = float((g[i] * v).sum())
                rel.append(abs(num - an) / max(abs(an), abs(num), 1e-12))
            rel = np.array(rel)
            res[key] = (np.median(rel), np.mean(rel < 0.05))
        print(f"(C) {label:<14} 逐顆有限差分 vs autograd（梯度最大 {a.K} 顆）："
              + "｜".join(f"{k} 相對差中位 {m:.2e}、<5% 的 {100 * f:.0f}%" for k, (m, f) in res.items()), flush=True)
    set_thick(None)
print("\nDONE")
