"""11-09 飽和門檻 1e-5 臂（地下 1.48 -> 1.28、其他面向平）的時間被桌面使用污染（外部 GPU 2.24 GB、每 3k 段從第一段就慢 7～15%）
=> 拆開「1e-5 本身多算」與「外部干擾」：同一 ckpt、同一組訓練相機、GT 當目標（L1＋SSIM），lean forward＋backward，
門檻 1e-4（預設）／1e-3／1e-5 交錯三輪抵消溫度漂移；另用稽核 kernel 量每像素走過的清單長度（n_proc）與真的混合的（n_blend）。
只量時間、不訓練；ckpt 用 30k 終點 => 量到的是收割期的單步成本（增生期 opacity 較低、差距可能更大）。
用法：python tools/probe/tsat_bench_1109.py --ckpt <ckpt> [--nview 24] [--ntime 12]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODES = ("1e-4", "1e-3", "1e-5")


def set_mode(renderer, mode):
    renderer.t_sat_1e3 = mode == "1e-3"
    renderer.t_sat_1e5 = mode == "1e-5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nview", type=int, default=24)
    ap.add_argument("--ntime", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    from PIL import Image
    import diff_trim_surfel_rasterization as R
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    assert "t_sat_1e5" in R.GaussianRasterizationSettings._fields, f"光柵器沒有 t_sat_1e5（沒重編）：{R.__file__}"
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

    # ── ① 稽核（不可微）
    print(f"N={model.n_gaussians:,}｜視角 {len(idx)}｜{a.ckpt}", flush=True)
    tot = {m: [0.0, 0.0] for m in MODES}     # [Σn_proc, Σn_blend]
    pix = 0
    views = []
    with torch.no_grad():
        for k, i in enumerate(idx):
            cam = ts.cameras[i].to_device(dev)
            for m in MODES:
                set_mode(renderer, m)
                _r, _u, _rad, others, _t = renderer(cam, model, bg_color=bg, _audit=True)
                tot[m][0] += float(others[0].float().sum()); tot[m][1] += float(others[2].float().sum())
            set_mode(renderer, "1e-4")
            pix += int(cam.width) * int(cam.height)
            if k < a.ntime:
                views.append((cam, load_gt(i, cam)))
    print("① 每像素工作量（n_proc＝飽和前走過的清單長度；n_blend＝真的混合的）")
    for m in MODES:
        print(f"   門檻 {m}  n_proc 平均 {tot[m][0] / pix:6.1f}（{tot[m][0] / tot['1e-4'][0]:.3f} 倍）｜n_blend 平均 {tot[m][1] / pix:6.1f}（{tot[m][1] / tot['1e-4'][1]:.3f} 倍）")

    # ── ② 實測時間（同 work_bench_1108 的 lean 設定）
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
    renderer._vrs_pix = False
    params = dict(model.gaussians.items())
    for p in params.values():
        p.requires_grad_(True)

    def bench(mode):
        set_mode(renderer, mode)
        tt = []
        for cam, gt in views:
            ts_ = []
            for r_ in range(a.repeat + 2):
                for p in params.values():
                    p.grad = None
                e0, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                out = renderer.training_forward(0, None, cam, model, bg, _absgrad=True, _geo=False, _lean_dist=False)
                r = out["render"]
                loss = 0.8 * (r - gt).abs().mean() + 0.2 * (1 - ssim(r, gt))
                loss.backward()
                e2.record()
                torch.cuda.synchronize()
                if r_ >= 2:
                    ts_.append(e0.elapsed_time(e2))
            tt.append(np.median(ts_))
        set_mode(renderer, "1e-4")
        return float(np.mean(tt))

    res = {}
    for rnd in range(3):
        for mode in (MODES if rnd % 2 == 0 else MODES[::-1]):
            res.setdefault(mode, []).append(bench(mode))
    b = np.mean(res["1e-4"])
    print(f"② 實測 lean forward＋backward（{len(views)} 台 x {a.repeat} 次 x 3 輪交錯）")
    for mode in MODES:
        v = res[mode]
        print(f"   門檻 {mode}  {np.mean(v):7.2f} ms（{np.mean(v) / b - 1:+.1%}）｜三輪 " + "／".join(f"{x:.2f}" for x in v))
    print("DONE")


if __name__ == "__main__":
    main()
