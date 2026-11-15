"""11-16 Fable A-1／A-3 計時版：同一個 .so 內用 CITYGS_SKIPF 切換（1＝A-1 forward、2＝A-1 backward、4＝A-3 逐 warp 配對位元），
走訓練用的 lean 路徑（renderer.forward(_lean=True)），量每視角 forward／backward 的 GPU 時間，並和不開的版本比渲染與梯度。
PYTHONPATH 指到 scratchpad/rast_skip。只渲染＋backward，不訓練、不更新參數。
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--nview", type=int, default=24)
ap.add_argument("--rep", type=int, default=3); ap.add_argument("--phase", default="grow,harvest")
a = ap.parse_args()
import diff_trim_surfel_rasterization as RZ
print("光柵器", RZ.__file__, flush=True)
from internal.utils.gaussian_model_loader import GaussianModelLoader
dev = torch.device("cuda")
model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=False, pre_activate=False)
dmh = ck["datamodule_hyper_parameters"]; _pp = dmh["parser"]
if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
    _pp.points_from = "random"
ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0).get_outputs().train_set
del ck
idx = np.linspace(0, len(ts.cameras) - 1, a.nview).round().astype(int).tolist()
cams = [ts.cameras[i].to_device(dev) for i in idx]
params = [p for p in model.parameters() if p.requires_grad]
print(f"N={model.n_gaussians:,}｜可求梯度參數 {len(params)} 組｜視角 {len(cams)}", flush=True)
bg = torch.zeros(3, device=dev)
g = torch.Generator(device=dev).manual_seed(0)
W = None
ARMS = [("不開", 0, False), ("不開（重跑，量雜訊）", 0, False), ("A-1 fwd", 1, False), ("A-1 bwd", 2, False), ("A-1 fwd+bwd", 3, False),
        ("配對 bit（tile 層，現有）", 0, True), ("A-3（逐 warp 位元）", 4, True), ("A-1 fwd＋A-3", 5, True)]


def run(cam, absg):
    out = renderer(cam, model, bg_color=bg, _lean=True, _absgrad=absg)
    return out


def one_pass(absg, record):
    global W
    tf, tb, imgs, grads = [], [], [], []
    for k, cam in enumerate(cams):
        for p in params:
            p.grad = None
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        out = run(cam, absg)
        e1.record()
        img = out["render"]
        if W is None or len(W) <= k:
            W = (W or []) + [torch.rand(img.shape, device=dev, generator=g) - 0.5]
        loss = (img * W[k]).sum()
        loss.backward()
        e2.record()
        torch.cuda.synchronize()
        tf.append(e0.elapsed_time(e1)); tb.append(e1.elapsed_time(e2))
        if record and k < 2:
            imgs.append(img.detach().cpu())
            grads.append([p.grad.detach().cpu() if p.grad is not None else None for p in params])
    return np.array(tf), np.array(tb), imgs, grads


for phase in a.phase.split(","):
    grow = phase == "grow"
    renderer._acc_stop = not grow          # 增生期：訓練中累加 V/C（WACC）開；收割期關
    absg = grow                            # absgrad 只在增生期累加（absgrad_gate）
    print(f"\n══ {'增生期（V/C 累加＋absgrad 開）' if grow else '收割期（V/C 累加、absgrad 關）'} ══", flush=True)
    res = {}
    ref = None
    for name, f, pb in ARMS:
        os.environ["CITYGS_SKIPF"] = str(f); renderer.bwd_pair_bit = pb
        one_pass(absg, False)                                   # 暖機
        TF, TB = [], []
        for r in range(a.rep):
            tf, tb, imgs, grads = one_pass(absg, r == 0)
            TF.append(tf); TB.append(tb)
            if r == 0:
                cur = (imgs, grads)
        TF, TB = np.median(np.stack(TF), 0), np.median(np.stack(TB), 0)
        if ref is None:
            ref = cur; dimg = dgr = 0.0; dg_rel = 0.0
        else:
            dimg = max(float((x - y).abs().max()) for x, y in zip(cur[0], ref[0]))
            num = den = 0.0
            for G1, G0 in zip(cur[1], ref[1]):
                for x, y in zip(G1, G0):
                    if x is None or y is None:
                        continue
                    num = max(num, float((x - y).abs().max())); den = max(den, float(y.abs().max()))
            dgr = num; dg_rel = num / max(den, 1e-30)
        res[name] = (TF.sum(), TB.sum())
        b = res["不開"]
        print(f"  {name:22} forward {TF.sum():8.1f} ms（{100 * (TF.sum() / b[0] - 1):+5.1f}%）｜backward {TB.sum():8.1f} ms（{100 * (TB.sum() / b[1] - 1):+5.1f}%）"
              f"｜合計 {100 * ((TF.sum() + TB.sum()) / (b[0] + b[1]) - 1):+5.1f}%｜渲染最大差 {dimg:.2e}｜梯度最大差（相對）{dg_rel:.2e}", flush=True)
os.environ["CITYGS_SKIPF"] = "0"
