#!/usr/bin/env python
"""★ 11-21 晚 RAST_F32_CONST（auxiliary.h：double 常數改 float）的數值檢查：同一個 process 同時載入兩份光柵器，
對 lean／dist lean／完整路徑三種步，比較所有輸出通道與各參數梯度。另外同一份 .so 跑兩次當雜訊底（backward 用 atomicAdd）。

  python tools/probe/f32_const_check_1121.py --so_a <預設>/diff_trim_surfel_rasterization --so_b <float 常數>/diff_trim_surfel_rasterization
"""
import argparse, importlib.util, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt")
ap.add_argument("--so_a", required=True)
ap.add_argument("--so_b", required=True)
ap.add_argument("--ncam", type=int, default=3)
a = ap.parse_args()

import torch
import internal.renderers.sep_depth_trim_2dgs_renderer as RM
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.ssim import ssim


def load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


SO = {"a": load_pkg("rast_chk_a", a.so_a), "b": load_pkg("rast_chk_b", a.so_b)}
dev = torch.device("cuda")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=False, pre_activate=False)
renderer._lean_announced = True
renderer.bwd_sat_skip = renderer.bwd_reduce = renderer.bwd_pair_bit = renderer.record_reduce = True
renderer.absgrad_gate = True; renderer._absgrad_announced = True
renderer.inv_homog = True; renderer.exact_strip_cull = True; renderer.skip_box = True
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
pp = ck["datamodule_hyper_parameters"]["parser"]
if getattr(pp, "points_from", None) == "ply":
    pp.points_from = "random"
cams = pp.instantiate(path=ck["datamodule_hyper_parameters"]["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                      global_rank=0).get_outputs().train_set.cameras
del ck
idx = np.linspace(0, len(cams) - 1, a.ncam).round().astype(int).tolist()
cams = [cams[i].to_device(dev) for i in idx]
params = dict(model.gaussians.items())
for p in params.values():
    p.requires_grad_(True)
bg = torch.zeros(3, device=dev)
tgts = [torch.rand((3, int(c.height), int(c.width)), device=dev, generator=torch.Generator(device=dev).manual_seed(1000 + i)) for i, c in enumerate(cams)]


def run(which, kind, c, t):
    R = SO[which]
    RM.GaussianRasterizationSettings = R.GaussianRasterizationSettings; RM.GaussianRasterizer = R.GaussianRasterizer
    for p in params.values():
        p.grad = None
    renderer.lean_train = kind != "full"
    TF = {"_absgrad": True}
    if kind == "dist":
        TF["_lean_dist"] = True
    out = renderer.training_forward(0, None, c, model, bg, **TF)
    loss = 0.8 * (out["render"] - t).abs().mean() + 0.2 * (1 - ssim(out["render"], t))
    outs = {"render": out["render"].detach().cpu()}
    keys = {"lean": (), "dist": ("rend_dist",), "full": ("rend_dist", "surf_depth", "rend_normal", "surf_normal")}[kind]
    for k in keys:
        if k in out and torch.is_tensor(out[k]) and out[k].requires_grad:
            outs[k] = out[k].detach().cpu()
            loss = loss + 0.1 * out[k].float().abs().mean()
    loss.backward()
    g = {n: p.grad.detach().double().cpu() for n, p in params.items() if p.grad is not None}
    return outs, g


def cmp(A, B):
    (oa, ga), (ob, gb) = A, B
    r = {}
    for k in oa:
        d = (oa[k].double() - ob[k].double()).abs()
        r["out_" + k] = (float(d.max()), float(d.max() / (oa[k].double().abs().max() + 1e-30)))
    for n in ga:
        x, y = ga[n], gb[n]
        r["grad_" + n] = (float((x - y).norm() / (x.norm() + 1e-30)), float(((x - y).abs().max()) / (x.abs().max() + 1e-30)))
    return r


for kind in ("lean", "dist", "full"):
    print(f"══ {kind}（每格：輸出＝最大絕對差／相對最大；梯度＝相對範數差／相對最大差）")
    rows = {}
    for ci, (c, t) in enumerate(zip(cams, tgts)):
        A1 = run("a", kind, c, t); A2 = run("a", kind, c, t); B1 = run("b", kind, c, t)
        for k, v in cmp(A1, A2).items():
            rows.setdefault(k, [[], []])[0].append(v)
        for k, v in cmp(A1, B1).items():
            rows.setdefault(k, [[], []])[1].append(v)
    for k, (noise, diff) in rows.items():
        nm = (max(x[0] for x in noise), max(x[1] for x in noise))
        dm = (max(x[0] for x in diff), max(x[1] for x in diff))
        print(f"  {k:<22} 同一份兩次 {nm[0]:.2e}／{nm[1]:.2e}   預設 vs float 常數 {dm[0]:.2e}／{dm[1]:.2e}")
