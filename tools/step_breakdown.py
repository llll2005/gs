#!/usr/bin/env python
"""穩態每步成本的逐段拆解（載 ckpt 直接微基準，約 2 分鐘）。

## 為什麼要重做一個工具

先前兩次量測都把我帶偏，三個坑各踩過：
1. **跨跑次比 wall time** —— 台帳實測噪音約 **5%**（`agd` vs `agd2` 差 5.2%，
   而強度 `w` 完全不改變計算量）⇒ 小於 ~10% 的改善量不出來。
2. **cProfile / advanced profiler** —— CPU 時間會把 CUDA 非同步工作記到下一個
   同步點（`.to()` 每次 0.17 秒搬相機參數，實際是在等前一次光柵化）。
3. **短視窗 profile** —— 40 步的窗被**只跑一次**的起始 trim 佔掉 80%，
   量到的完全不是穩態。

本工具的做法：**同一個 process 內、CUDA event 計時、每段重複多次取中位數、
不含任何一次性事件**。所有段落共用同一份 ckpt 與同一台相機 ⇒ 段落之間可直接相加比較。

## 用法
    python tools/step_breakdown.py --run agd2_b12 [--step 60000] [--repeat 20]
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cuda_time(fn, repeat, warmup=3):
    """CUDA event 計時：回傳中位數毫秒。warmup 讓 cuDNN/記憶體池穩定下來。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="agd2_b12")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")

    cks = sorted(glob.glob(f"outputs/{args.run}/**/*.ckpt", recursive=True),
                 key=lambda p: int(p.split("step=")[-1].split(".")[0]))
    if not cks:
        raise SystemExit(f"找不到 {args.run} 的 ckpt")
    if args.step is not None:
        cks = [p for p in cks if int(p.split("step=")[-1].split(".")[0]) == args.step] or cks[-1:]
    ck_path = cks[-1]
    print(f"ckpt: {os.path.basename(ck_path)}")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    dev = torch.device("cuda")
    # ⚠ pre_activate=False：預設 True 會把 activation 換成 identity
    # （`vanilla_gaussian.py:379` self.opacity_activation = self._return_as_is），
    # 那會讓 `get_scales()` 回傳未激活的原始值 ⇒ noise 段量到的不是訓練時的東西。
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck_path, device=dev, eval_mode=False, pre_activate=False)
    n = model.n_gaussians
    print(f"N = {n:,}")

    # 相機用訓練集第一台（與訓練時同解析度、同 down_sample）
    ckpt = torch.load(ck_path, map_location="cpu")
    from internal.dataset import DataModule
    dm = DataModule(**ckpt["datamodule_hyper_parameters"])
    dm.setup("fit")
    cam = dm.dataparser_outputs.train_set.cameras[0].to_device(dev)
    print(f"影像 {int(cam.width)}x{int(cam.height)}")

    bg = torch.zeros((3,), device=dev)

    results = []

    def render():
        return renderer(cam, model, bg_color=bg)

    out = render()
    img = out["render"]
    gt_img = torch.rand_like(img)

    results.append(("光柵化 forward", cuda_time(lambda: render(), args.repeat)))

    # L1
    results.append(("L1 loss", cuda_time(lambda: torch.abs(img - gt_img).mean(), args.repeat)))

    # SSIM
    try:
        from internal.utils.ssim import ssim as _ssim
        results.append(("SSIM", cuda_time(lambda: _ssim(img, gt_img), args.repeat)))
    except Exception as ex:                                   # noqa: BLE001
        print(f"  （SSIM 略過：{ex}）")

    # forward+backward
    def fb():
        o = renderer(cam, model, bg_color=bg)
        loss = torch.abs(o["render"] - gt_img).mean()
        loss.backward()
        model.gaussians["means"].grad = None
    results.append(("forward+backward(L1)", cuda_time(fb, max(4, args.repeat // 3))))

    # MCMC noise：cov_3d 建構 + bmm
    from internal.density_controllers.mcmc_density_controller import compute_cov_3d
    if compute_cov_3d is not None:
        def noise_term():
            sc = model.get_scales()
            if sc.shape[-1] < 3:
                sc = torch.cat([sc, torch.zeros((sc.shape[0], 3 - sc.shape[-1]),
                                                dtype=sc.dtype, device=sc.device)], -1)
            cov = compute_cov_3d(scales=sc, scale_modifier=1., quaternions=model.get_rotations())
            nz = torch.randn_like(model.means)
            return torch.bmm(cov, nz.unsqueeze(-1)).squeeze(-1)
        results.append(("MCMC noise(cov3d+bmm)", cuda_time(noise_term, args.repeat)))

    tot = sum(v for _, v in results)
    print(f"\n{'段落':>24} {'ms':>9} {'佔已量':>8}")
    for k, v in results:
        print(f"{k:>24} {v:>9.2f} {v / tot:>7.1%}")
    print(f"{'（已量總和）':>24} {tot:>9.2f}")
    print("""
⚠ 「已量總和」不是每步總時間：optimizer step、資料搬運、Lightning 開銷未計入。
   要對照每步真實時間，用 台帳總秒數/60000（agd2_b12 = 602 ms/step）。
   兩者的差就是「還沒量到的部分」—— 那才是下一個該找的地方。""")


if __name__ == "__main__":
    main()
