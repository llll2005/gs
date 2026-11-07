"""11-07 alpha 上限 0.999 那一跑（PF2DA）每步慢 +23%（各階段一致、從 step 200 起就慢，那時幾乎沒有 α > 0.99 的顆粒）
=> 分開兩個嫌疑：①重編後的光柵器連預設模式都變慢（熱迴圈裡讀 __device__ 全域變數 g_alpha_max）②0.999 本身。
同一個 ckpt、同一組相機、隨機目標（L1＋SSIM），量訓練版 forward（lean）＋backward 的時間；四組交錯跑（a99、a999、a99、a999）抵消溫度漂移。
用 PYTHONPATH 指到別處編好的光柵器比較不同版本（不碰共用環境）：
  PYTHONPATH=<編好的目錄> python tools/probe/alpha_speed_bench_1107.py --ckpt <ckpt> --tag <名稱>
--vc 1：增生期（forward 同時累加 V/C、backward 累加 |g|）；0：收割期。舊版光柵器沒有 alpha_999 欄位 => 只跑 a99。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ncam", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--vc", type=int, default=1)
    a = ap.parse_args()

    import diff_trim_surfel_rasterization as R
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    dev = torch.device("cuda")
    has999 = "alpha_999" in getattr(R.GaussianRasterizationSettings, "_fields", ())
    print(f"[{a.tag}] 光柵器：{os.path.dirname(R.__file__)}｜有 alpha_999：{has999}", flush=True)
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    renderer.lean_train = True
    renderer._lean_announced = renderer._lean_dist_announced = True
    renderer.bwd_sat_skip = renderer.bwd_reduce = renderer.bwd_zero_grad_skip = True
    renderer.bwd_pair_bit = renderer.skip_unscored_pixels = False
    renderer.trim_running_vc = bool(a.vc); renderer.trim_running_warp = True
    renderer._acc_announced = True; renderer._acc_stop = False
    renderer._acc_add = lambda *x, **k: None
    renderer.trim_hidden = False
    renderer.sh_adam_in_backward = False
    renderer.absgrad_gate = True; renderer._absgrad_announced = True
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"
    ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                                   global_rank=0).get_outputs().train_set
    del ck
    cams = ts.cameras
    idx = np.linspace(0, len(cams) - 1, a.ncam).round().astype(int).tolist()
    for p in dict(model.gaussians.items()).values():
        p.requires_grad_(True)
    params = dict(model.gaussians.items())
    bg = torch.zeros(3, device=dev)
    data = []
    for c in idx:
        cam = cams[c].to_device(dev)
        g = torch.Generator(device=dev).manual_seed(1000 + c)
        data.append((cam, torch.rand((3, int(cam.height), int(cam.width)), device=dev, generator=g)))
    print(f"[{a.tag}] N={model.n_gaussians:,}｜相機 {len(data)} 台 x {a.repeat} 次｜vc={a.vc}", flush=True)

    def bench(al999):
        renderer.alpha_max_999 = al999
        tf, tb = {}, {}
        for i, (cam, tgt) in enumerate(data):
            for r_ in range(a.repeat + 2):
                for p in params.values():
                    p.grad = None
                e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                e0.record()
                out = renderer.training_forward(0, None, cam, model, bg, _absgrad=bool(a.vc), _geo=False, _lean_dist=False)
                r = out["render"]
                loss = 0.8 * (r - tgt).abs().mean() + 0.2 * (1 - ssim(r, tgt))
                e1.record()
                loss.backward()
                e2.record()
                torch.cuda.synchronize()
                if r_ >= 2:
                    tf.setdefault(i, []).append(e0.elapsed_time(e1)); tb.setdefault(i, []).append(e1.elapsed_time(e2))
        f = float(np.mean([np.median(v) for v in tf.values()])); b = float(np.mean([np.median(v) for v in tb.values()]))
        return f, b

    order = [False, True, False, True] if has999 else [False, False]
    res = {}
    for al in order:
        f, b = bench(al)
        res.setdefault(al, []).append((f, b))
        print(f"[{a.tag}] alpha {'0.999' if al else '0.99 '}｜forward {f:6.2f} ms｜backward {b:6.2f} ms｜合計 {f + b:6.2f} ms", flush=True)
    for al, v in res.items():
        print(f"[{a.tag}] 平均 alpha {'0.999' if al else '0.99 '}：合計 {np.mean([f + b for f, b in v]):6.2f} ms（forward {np.mean([f for f, _ in v]):.2f}／backward {np.mean([b for _, b in v]):.2f}）", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
