"""11-08 使用者「減少 tile 就能減少時間」「都試試；VRS 的判準多作幾個假設」——訓練前先量（不訓練）：
① 每像素工作量（稽核 kernel 的 n_proc＝飽和前走過的清單長度；n_blend＝真的混合的）：平均、分布、飽和門檻 1e-4 vs 1e-3 的差。
② 跳過 tile 的理論省量：逐 tile 工作量（整塊一起走 => 256 × tile 內最大 n_proc）落在哪；random／error／texture 三種 π（保留平均 0.6、下限 0.25）
   各自期望略過多少工作量（error 用這個模型對 GT 的逐 tile L1、texture 用 GT 梯度強度）。
③ 實測：同一 ckpt、同一組訓練相機、GT 當目標（L1＋SSIM），lean forward＋backward 的時間：預設／門檻 1e-3／三種跳過。交錯兩輪抵消溫度漂移。
用法：python tools/probe/work_bench_1108.py --ckpt <ckpt> [--nview 24] [--ntime 10]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nview", type=int, default=24)
    ap.add_argument("--ntime", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--keep", type=float, default=0.6)
    ap.add_argument("--min_keep", type=float, default=0.25)
    a = ap.parse_args()
    from PIL import Image
    import diff_trim_surfel_rasterization as R
    from internal.metrics import vrs as V
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    assert "t_sat_1e3" in R.GaussianRasterizationSettings._fields, f"光柵器沒有 t_sat_1e3（沒重編）：{R.__file__}"
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"
    ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0).get_outputs().train_set
    del ck
    idx = np.linspace(0, len(ts.cameras) - 1, a.nview).round().astype(int).tolist()
    bg = torch.zeros(3, device=dev)

    def load_gt(i, cam):
        im = Image.open(ts.image_paths[i]).convert("RGB").resize((int(cam.width), int(cam.height)), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im)).to(dev).permute(2, 0, 1).float() / 255.0

    # ── ①② 稽核（不可微）
    print(f"N={model.n_gaussians:,}｜視角 {len(idx)}｜{a.ckpt}", flush=True)
    tot = {"proc": 0.0, "proc3": 0.0, "blend": 0.0, "blend3": 0.0, "tile": 0.0, "tile3": 0.0, "pix": 0}
    hist = []
    sav = {"random": [0.0, 0.0], "error": [0.0, 0.0], "texture": [0.0, 0.0]}     # [Σ(1−π)·W_tile, ΣW_tile]
    lowshare = {"error": 0.0, "texture": 0.0}
    timing_views = []
    with torch.no_grad():
        for k, i in enumerate(idx):
            cam = ts.cameras[i].to_device(dev)
            gt = load_gt(i, cam)
            renderer.t_sat_1e3 = False
            _r, _u, _rad, others, _t = renderer(cam, model, bg_color=bg, _audit=True)
            n_proc, n_blend = others[0].float(), others[2].float()
            renderer.t_sat_1e3 = True
            _r, _u, _rad, others3, _t = renderer(cam, model, bg_color=bg, _audit=True)
            renderer.t_sat_1e3 = False
            n_proc3, n_blend3 = others3[0].float(), others3[2].float()
            H, W = n_proc.shape
            tmax = torch.nn.functional.max_pool2d(n_proc[None, None], T, T, ceil_mode=True)[0, 0] * (T * T)
            tmax3 = torch.nn.functional.max_pool2d(n_proc3[None, None], T, T, ceil_mode=True)[0, 0] * (T * T)
            tot["proc"] += float(n_proc.sum()); tot["proc3"] += float(n_proc3.sum())
            tot["blend"] += float(n_blend.sum()); tot["blend3"] += float(n_blend3.sum())
            tot["tile"] += float(tmax.sum()); tot["tile3"] += float(tmax3.sum()); tot["pix"] += H * W
            hist.append(n_proc.flatten()[::97].cpu())
            out = renderer(cam, model, bg_color=bg)
            img = (out["render"] if isinstance(out, dict) else out[0]).clamp(0, 1)
            err = V.tile_mean((img - gt).abs().mean(0), T)
            tex = V.texture_score(gt, T)
            for nm, sc in (("random", None), ("error", err), ("texture", tex)):
                pi = torch.full_like(tmax, a.keep) if sc is None else V.keep_probs(sc, a.keep, a.min_keep)
                sav[nm][0] += float(((1 - pi) * tmax).sum()); sav[nm][1] += float(tmax.sum())
                if sc is not None:
                    lo = sc <= torch.quantile(sc.flatten(), 0.4)
                    lowshare[nm] += float(tmax[lo].sum())
            if k < a.ntime:
                timing_views.append((cam, gt, err, tex))
    hp = torch.cat(hist).numpy()
    print("① 每像素工作量（飽和前走過的清單長度 n_proc；真的混合的 n_blend）")
    print(f"   n_proc 平均 {tot['proc'] / tot['pix']:.1f}（q50 {np.median(hp):.0f}／q90 {np.quantile(hp, 0.9):.0f}／q99 {np.quantile(hp, 0.99):.0f}）｜"
          f"n_blend 平均 {tot['blend'] / tot['pix']:.1f}（混合佔 {tot['blend'] / max(tot['proc'], 1):.1%}）")
    print(f"   飽和門檻 1e-3：n_proc {tot['proc3'] / max(tot['proc'], 1):.3f} 倍、n_blend {tot['blend3'] / max(tot['blend'], 1):.3f} 倍、"
          f"整塊一起走的工作量 {tot['tile3'] / max(tot['tile'], 1):.3f} 倍")
    print(f"② 跳過 tile 的理論省量（保留平均 {a.keep}、下限 {a.min_keep}；工作量＝256 × tile 內最大 n_proc）")
    for nm, (s, w) in sav.items():
        extra = f"｜分數最低 40% 的 tile 佔工作量 {lowshare[nm] / max(w, 1):.1%}" if nm in lowshare else ""
        print(f"   {nm:8} 期望略過工作量 {s / max(w, 1):.1%}{extra}")

    # ── ③ 實測時間
    renderer.lean_train = True
    renderer._lean_announced = renderer._lean_dist_announced = True
    renderer.bwd_sat_skip = renderer.bwd_reduce = renderer.bwd_zero_grad_skip = True
    renderer.bwd_pair_bit = renderer.skip_unscored_pixels = False
    renderer.trim_running_vc = True; renderer.trim_running_warp = True
    renderer._acc_announced = True; renderer._acc_stop = False
    renderer._acc_add = lambda *x, **k: None
    renderer.trim_hidden = False
    renderer.sh_adam_in_backward = False
    renderer.absgrad_gate = True; renderer._absgrad_announced = True
    renderer._pix_skip_announced = True
    for p in dict(model.gaussians.items()).values():
        p.requires_grad_(True)
    params = dict(model.gaussians.items())
    g = torch.Generator(device=dev).manual_seed(0)
    masks = {}
    for j, (cam, gt, err, tex) in enumerate(timing_views):
        H, W = gt.shape[-2:]
        for nm, sc in (("random", None), ("error", err), ("texture", tex)):
            pi = torch.full_like(err, a.keep) if sc is None else V.keep_probs(sc, a.keep, a.min_keep)
            kt = torch.rand(pi.shape, device=dev, generator=g) < pi
            masks[(j, nm)] = ((~V.tile_to_pix(kt, H, W, T)).to(torch.uint8).contiguous().view(-1), V.tile_to_pix(kt, H, W, T)[None])

    def bench(mode):
        renderer.t_sat_1e3 = (mode == "tsat")
        renderer._vrs_pix = mode in ("random", "error", "texture")
        tt = []
        for j, (cam, gt, _e, _x) in enumerate(timing_views):
            kw = {}
            kp = None
            if renderer._vrs_pix:
                pix, kp = masks[(j, mode)]
                kw["_pix_skip"] = pix
            ts_ = []
            for r_ in range(a.repeat + 2):
                for p in params.values():
                    p.grad = None
                e0, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                out = renderer.training_forward(0, None, cam, model, bg, _absgrad=True, _geo=False, _lean_dist=False, **kw)
                r = out["render"]
                g2 = gt if kp is None else torch.where(kp, gt, r.detach())
                rr = r if kp is None else torch.where(kp, r, r.detach())
                loss = 0.8 * (rr - g2).abs().mean() + 0.2 * (1 - ssim(rr, g2))
                loss.backward()
                e2.record()
                torch.cuda.synchronize()
                if r_ >= 2:
                    ts_.append(e0.elapsed_time(e2))
            tt.append(np.median(ts_))
        renderer.t_sat_1e3 = False; renderer._vrs_pix = False
        return float(np.mean(tt))

    res = {}
    for rnd in range(2):
        for mode in ("base", "tsat", "random", "error", "texture"):
            res.setdefault(mode, []).append(bench(mode))
    b = np.mean(res["base"])
    print(f"③ 實測 lean forward＋backward（{len(timing_views)} 台 x {a.repeat} 次 x 2 輪交錯）")
    for mode, v in res.items():
        print(f"   {mode:8} {np.mean(v):7.2f} ms（{np.mean(v) / b - 1:+.1%}）｜兩輪 {v[0]:.2f}／{v[1]:.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
