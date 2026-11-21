#!/usr/bin/env python
"""★ 11-21 檢查「geomState.aih 改成需要才配置」（紀錄/實驗分析/39 §4.7：全場景 test 渲染峰值 5.69 -> 6.23 GiB 的根因）。

同一組相機、三種設定（旗標全關／反單應＋幾何配對遮罩＋skip_box／只有 skip_box），每種量：
  訓練步（lean forward＋backward）的影像與各參數梯度、峰值配置；評估渲染（no_grad、非 lean）的影像與峰值配置。
存成 npz（只存每個陣列的 sha256＋形狀，不存陣列本身：全梯度一份約 3 GB）；用舊 .so 跑一次、新 .so（PYTHONPATH）跑一次，再 --compare 兩個 npz：影像與梯度要逐位元相同，峰值應下降（旗標關）／不變（旗標開）。

  python tools/probe/aih_alloc_check_1121.py --out a.npz                       # 目前環境的 .so
  PYTHONPATH=<新 build> python tools/probe/aih_alloc_check_1121.py --out b.npz
  python tools/probe/aih_alloc_check_1121.py --compare a.npz b.npz
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt")
ap.add_argument("--ncam", type=int, default=3)
ap.add_argument("--out")
ap.add_argument("--compare", nargs=2)
a = ap.parse_args()

if a.compare:
    A, B = (np.load(f) for f in a.compare)
    keys = sorted(k for k in A.files if not k.startswith(("peak", "gstat")))
    bad = [k for k in keys if k not in B.files or str(A[k]) != str(B[k])]
    # 梯度：backward 用 atomicAdd => 同一個 .so 兩次跑也不逐位元相同；比「範數、絕對和、4 個固定隨機投影」的相對差
    rel = {}
    for k in sorted(k for k in A.files if k.startswith("gstat")):
        x, y = A[k], B[k]; rel[k] = float(np.max(np.abs(x - y) / (np.abs(x) + 1e-30)))
    if rel:
        w = max(rel, key=rel.get)
        print(f"梯度摘要 {len(rel)} 組：相對差 中位 {np.median(list(rel.values())):.2e}、最大 {rel[w]:.2e}（{w}）")
    print(f"影像＋梯度 {len(keys)} 個陣列：逐位元相同 {len(keys) - len(bad)}，不同 {len(bad)}", bad[:8])
    for k in sorted(k for k in A.files if k.startswith("peak")):
        print(f"{k:<28} {A[k] / 2**20:9.1f} MiB -> {B[k] / 2**20:9.1f} MiB（{(B[k] - A[k]) / 2**20:+8.1f}）")
    sys.exit(1 if bad else 0)

import hashlib
import torch
import diff_trim_surfel_rasterization as R


def dig(x):
    x = np.ascontiguousarray(x.detach().float().cpu().numpy())
    return np.array(f"{x.shape}:{hashlib.sha256(x.tobytes()).hexdigest()}")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.ssim import ssim
print("光柵器：", R.__file__)
dev = torch.device("cuda")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=False, pre_activate=False)
renderer._lean_announced = True
renderer.bwd_sat_skip = renderer.bwd_reduce = renderer.bwd_pair_bit = renderer.record_reduce = True
renderer.absgrad_gate = True; renderer._absgrad_announced = True
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
print(f"N = {params['means'].shape[0]:,}｜相機 {len(cams)} 台 {int(cams[0].width)}x{int(cams[0].height)}")

SETS = {"off": dict(inv_homog=False, exact_strip_cull=False, skip_box=False),
        "sbox": dict(inv_homog=False, exact_strip_cull=False, skip_box=True),
        "m39r": dict(inv_homog=True, exact_strip_cull=True, skip_box=True)}
res = {}
for tag, kv in SETS.items():
    for k, v in kv.items():
        setattr(renderer, k, v)
    for ci, (c, t) in enumerate(zip(cams, tgts)):
        # 訓練步
        renderer.lean_train = True
        for p in params.values():
            p.grad = None
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        out = renderer.training_forward(0, None, c, model, bg, _absgrad=True)
        img = out["render"]
        loss = 0.8 * (img - t).abs().mean() + 0.2 * (1 - ssim(img, t))
        loss.backward()
        torch.cuda.synchronize()
        res[f"peak_train_{tag}_{ci}"] = np.int64(torch.cuda.max_memory_allocated() - base)
        res[f"img_train_{tag}_{ci}"] = dig(img)
        for n, p in params.items():
            if p.grad is not None:
                res[f"grad_{n}_{tag}_{ci}"] = dig(p.grad)
                g = p.grad.detach().double().flatten().cpu()
                ix = torch.arange(g.numel(), dtype=torch.float64)
                res[f"gstat_{n}_{tag}_{ci}"] = np.array([float(g.norm()), float(g.abs().sum())] + [float((g * torch.cos(ix * w)).sum()) for w in (0.1234, 0.5678, 1.2345, 2.3456)])
                del g, ix
        del out, img, loss
        # 評估渲染
        renderer.lean_train = False
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        with torch.no_grad():
            o = renderer(c, model, bg)
        torch.cuda.synchronize()
        res[f"peak_eval_{tag}_{ci}"] = np.int64(torch.cuda.max_memory_allocated() - base)
        res[f"img_eval_{tag}_{ci}"] = dig(o["render"])
        del o
    print(tag, "訓練步峰值增量 %.1f MiB｜評估 %.1f MiB" % (res[f"peak_train_{tag}_0"] / 2**20, res[f"peak_eval_{tag}_0"] / 2**20))
np.savez(a.out, **res)
print("=>", a.out)
