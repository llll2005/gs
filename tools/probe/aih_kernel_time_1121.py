#!/usr/bin/env python
"""★ 11-21 反單應交點（純數學化簡 #1）的 kernel 層級計時：同一個 process、同一組相機，交替開關 renderer.inv_homog，
用 torch.profiler 把每個 CUDA kernel 的 GPU 時間按名稱加總（每步平均）。不訓練（不更新參數）。

要用新編的光柵器：PYTHONPATH=<build>/lib.linux-x86_64-cpython-39 python tools/probe/aih_kernel_time_1121.py --ckpt <ckpt>
"""
import argparse, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt")
ap.add_argument("--ncam", type=int, default=6)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--dist", type=int, default=0, help="1＝dist 版 lean（目前反單應不作用，當對照）")
ap.add_argument("--toggle", default="aih", choices=["aih", "xcull", "both", "tcull", "nobox", "homog", "so"], help="aih＝切換反單應；xcull＝切換幾何配對遮罩（反單應照 --aih_fixed）；tcull＝切換 tile 精確剔除（反單應＋遮罩固定開）；nobox＝遮罩開時切換 skip_box")
ap.add_argument("--aih_fixed", type=int, default=1, help="toggle=xcull 時反單應固定開（1）或關（0）")
ap.add_argument("--so_a", default="", help="toggle=so：設定 0 用的光柵器套件目錄（…/diff_trim_surfel_rasterization）")
ap.add_argument("--so_b", default="", help="toggle=so：設定 1 用的光柵器套件目錄；同一個 process 交替兩份 .so（跨 process 時脈飄移太大）")
ap.add_argument("--lean", type=int, default=1, help="0＝完整路徑（MODE 0 forward＋GEOM backward；量每 8 步那種步）")
a = ap.parse_args()

import diff_trim_surfel_rasterization as R
from torch.profiler import profile, ProfilerActivity
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.ssim import ssim
print("光柵器：", R.__file__, "｜有 inv_homog：", "inv_homog" in R.GaussianRasterizationSettings._fields)
dev = torch.device("cuda")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=False, pre_activate=False)
renderer.lean_train = bool(a.lean); renderer._lean_announced = True
renderer.bwd_sat_skip = renderer.bwd_reduce = renderer.bwd_pair_bit = renderer.record_reduce = True
renderer.absgrad_gate = True; renderer._absgrad_announced = True
TF = {"_absgrad": True}
if a.dist:
    TF["_lean_dist"] = True
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


def step(cam, tgt):
    for p in params.values():
        p.grad = None
    out = renderer.training_forward(0, None, cam, model, bg, **TF)
    img = out["render"]
    loss = 0.8 * (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
    if a.dist:
        loss = loss + 100.0 * out["rend_dist"].mean()
    loss.backward()


def short(name):
    n = re.sub(r"\(.*", "", name)
    n = re.sub(r"^void ", "", n)
    return n[:70]


# ★ 這台筆電的 CUPTI 對 GPU 回報 INVALID_DEVICE（拿不到 kernel 層級時間）=> 改用 CUDA event 量 forward／backward，
#   同一 process、逐步交替兩種設定、多輪，報每台相機的成對差（同相機、相鄰時間）=> 共用開銷（preprocess／排序／SSIM）在差值裡抵掉。
def _set(setting):
    if a.toggle == "aih":
        renderer.inv_homog = bool(setting); renderer.exact_strip_cull = False
    elif a.toggle == "xcull":
        renderer.inv_homog = bool(a.aih_fixed); renderer.exact_strip_cull = bool(setting)
    elif a.toggle == "tcull":   # ★ 11-21 晚：tile 精確剔除（10-03 的 tile_cull）在現行配方＋m39 上
        renderer.inv_homog = True; renderer.exact_strip_cull = True; renderer.skip_box = True; renderer.tile_cull = bool(setting)
    elif a.toggle == "so":   # ★ 11-21 晚：同一個 process 交替兩份光柵器（例：預設 vs RAST_F32_CONST=1）；渲染設定＝m39 的 lean
        renderer.inv_homog = True; renderer.exact_strip_cull = True; renderer.skip_box = True
        _R = _SO[setting]
        _RMOD.GaussianRasterizationSettings = _R.GaussianRasterizationSettings; _RMOD.GaussianRasterizer = _R.GaussianRasterizer
    elif a.toggle == "homog":   # ★ 11-21 晚 #9：反單應＋遮罩開，切換齊次性少歸約
        renderer.inv_homog = True; renderer.exact_strip_cull = True; renderer.skip_box = True; renderer.aih_homog = bool(setting)
    elif a.toggle == "nobox":   # ★ 11-21 晚：遮罩開時 skip_box 還有沒有用（1＝拿掉 skip_box）
        renderer.inv_homog = True; renderer.exact_strip_cull = True; renderer.skip_box = not bool(setting)
    else:   # both：全關 vs 反單應＋幾何配對遮罩
        renderer.inv_homog = bool(setting); renderer.exact_strip_cull = bool(setting)


_SO, _RMOD = {}, None
if a.toggle == "so":
    import importlib.util
    import internal.renderers.sep_depth_trim_2dgs_renderer as _RMOD

    def _load_pkg(name, path):
        spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    _SO = {0: _load_pkg("rast_so_a", a.so_a), 1: _load_pkg("rast_so_b", a.so_b)}
    print("toggle so：0 =", _SO[0]._C.__file__, "｜1 =", _SO[1]._C.__file__)


def timed(cam, tgt):
    for p in params.values():
        p.grad = None
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    e0.record()
    out = renderer.training_forward(0, None, cam, model, bg, **TF)
    img = out["render"]
    loss = 0.8 * (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
    if a.dist:
        loss = loss + 100.0 * out["rend_dist"].mean()
    e1.record()
    loss.backward()
    e2.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1), e1.elapsed_time(e2)

for setting in (0, 1, 0, 1):                # 暖機
    _set(setting)
    for c, t in zip(cams, tgts):
        timed(c, t)
F = {0: [], 1: []}; B = {0: [], 1: []}
for r in range(a.rounds):
    for ci, (c, t) in enumerate(zip(cams, tgts)):
        for setting in ((0, 1) if (r + ci) % 2 == 0 else (1, 0)):
            _set(setting)
            f, b = timed(c, t)
            F[setting].append(f); B[setting].append(b)
F0, F1, B0, B1 = (np.array(x) for x in (F[0], F[1], B[0], B[1]))
def rep(tag, x0, x1):
    d = (x1 - x0) / x0 * 100
    bs = [np.median(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
    print(f"{tag:<14} 關 中位 {np.median(x0):7.3f} ms｜開 {np.median(x1):7.3f} ms｜成對差 中位 {np.median(d):+.2f}%（95% 區間 {np.percentile(bs, 2.5):+.2f}～{np.percentile(bs, 97.5):+.2f}%）")
print(f"{len(F0)} 對（{a.rounds} 輪 x {len(cams)} 台相機）")
rep("forward+loss", F0, F1)
rep("backward", B0, B1)
rep("合計", F0 + B0, F1 + B1)
