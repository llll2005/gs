#!/usr/bin/env python
"""`cap_max` 到底能設多高？—— 直接把粒子複製到目標 N，跑**真實訓練步**看會不會 OOM。

## 為什麼要重量一次

`tools/vram_gap.py`（2026-09-04）量到：
```
allocated 峰值合計 ≈ 2.42 GB      reserved = 4.87 GB      => 缺口 2.45 GB 全是配置器碎片
階段峰值：forward+backward 1.901 > trim pass 1.239 ~ forward 1.222 > optimizer 1.012
```
**兩個發現都推翻既有假設**：
1. **trim pass 不是峰值主因**（它 `no_grad` 且逐台釋放）—— 我原本押它。
2. **台帳的 `VRAM=5.57/6.1G` 是 `torch.cuda.memory_reserved()`**（`callbacks.py:_vram`）
   ⇒ **`cap_max` 長年是照著一個含約 40% 碎片的數字調的。**

⚠ 但「碎片」不等於「可用」：PyTorch 配置失敗時會先釋放快取區塊再重試，
   所以真正的上限是 **allocated + 無法釋放的碎片**，那個數字只能實測。
   ⇒ 本工具不推估，直接把 N 推上去跑真實步驟，讓它自己撞牆。

## 量法

對每個目標 N：把既有粒子**整批複製**到該數量（複製對 VRAM 是等價的：
參數/梯度/Adam 狀態都按 N 線性，binning 緩衝也按覆蓋的 tile 數增長），
然後跑 `--steps` 個**完整**訓練步（forward + backward + optimizer.step + MCMC 噪音）
再加**一次 trim pass**，記錄 peak allocated / peak reserved，OOM 就記下來停。

⚠ **本工具是上界估計，比真實訓練樂觀**：沒有 Lightning、dataloader、
  以及長時間跑次累積的碎片。⇒ 採用時要留餘裕（建議取通過的最大 N 的 ~85%）。

用法: python tools/vram_scale.py --run agd2_b12 --targets 2.34 2.8 3.2 3.6 4.0
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GB = 1024 ** 3


def replicate(model, target_n):
    """把高斯整批複製/截斷到 target_n。means 加極小抖動避免完全重合（不影響 VRAM，
    但讓 binning 的行為更接近真實族群）。"""
    n0 = model.n_gaussians
    idx = torch.arange(target_n, device=model.get_xyz.device) % n0
    with torch.no_grad():
        for k, v in model.gaussians.items():
            nv = v.data[idx].clone()
            if k == "means":
                nv += (torch.rand_like(nv) - 0.5) * 1e-4
            model.gaussians[k] = torch.nn.Parameter(nv, requires_grad=v.requires_grad)
    return target_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="agd2_b12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[2.34, 2.8, 3.2, 3.6, 4.0],
                    help="目標顆數（百萬）")
    ap.add_argument("--steps", type=int, default=6, help="每個 N 跑幾個完整訓練步")
    ap.add_argument("--trim-cams", type=int, default=284)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    ck = sorted(glob.glob(f"outputs/{args.run}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {args.run} 的 step={args.step} ckpt")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    train_cams = dp.get_outputs().train_set.cameras
    tot = torch.cuda.get_device_properties(0).total_memory / GB
    print(f"{args.run} @ {args.step}   顯卡總量 {tot:.2f} GB   每個 N 跑 {args.steps} 步 + 1 次 trim\n")
    print(f"{'目標 N':>10} {'peak alloc':>12} {'peak resv':>11} {'碎片':>9} {'佔卡':>8}  結果")

    ok_max = None
    for t in args.targets:
        target_n = int(t * 1e6)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model, renderer, _ = GaussianModelLoader.\
                initialize_model_and_renderer_from_checkpoint_file(
                    ck[0], device=dev, eval_mode=False, pre_activate=False)
            replicate(model, target_n)
            params = list(model.gaussians.values())
            for p in params:
                p.requires_grad_(True)
            opt = torch.optim.Adam([{"params": params, "lr": 1e-6}])
            bg = torch.zeros((3,), device=dev)

            for s in range(args.steps):
                cam = train_cams[s % len(train_cams)].to_device(dev)
                o = renderer(cam, model, bg_color=bg)
                img = o["render"]
                gt = torch.rand_like(img)
                loss = 0.8 * torch.abs(img - gt).mean() + 0.2 * (1 - ssim_fn(img, gt))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():        # MCMC 噪音：每步都做，是常駐成本的一部分
                    model.gaussians["means"].data += torch.randn_like(
                        model.gaussians["means"]) * 1e-8

            from internal.renderers.sep_depth_trim_2dgs_renderer import contribution_accumulator
            push, gather = contribution_accumulator(getattr(renderer, "K", 5))
            with torch.no_grad():
                for i in range(min(args.trim_cams, len(train_cams))):
                    push(renderer(train_cams[i].to_device(dev), model,
                                  bg_color=bg, record_transmittance=True))
                gather()

            pa = torch.cuda.max_memory_allocated() / GB
            pr = torch.cuda.max_memory_reserved() / GB
            ok_max = target_n
            print(f"{target_n/1e6:>9.2f}M {pa:>12.3f} {pr:>11.3f} {pr-pa:>9.3f} "
                  f"{100*pr/tot:>7.1f}%  ✅ 通過")
        except torch.cuda.OutOfMemoryError:
            pr = torch.cuda.max_memory_reserved() / GB
            print(f"{target_n/1e6:>9.2f}M {'-':>12} {pr:>11.3f} {'-':>9} "
                  f"{100*pr/tot:>7.1f}%  ⛔ OOM")
            break
        finally:
            for v in ("model", "renderer", "opt", "push", "gather"):
                if v in dir():
                    pass
            torch.cuda.empty_cache()

    print(f"""
判讀：
  通過的最大 N = {ok_max/1e6 if ok_max else 0:.2f}M（現行 cap 2.6M，實測交付 0.90 x cap）
  ⇒ 若它明顯 > 2.6M，`cap_max` 一直被**碎片**而不是被真實容量壓著
     依 §11.59 的顆數槓桿（**+0.541 dB/加倍**），2.34M -> 3.5M = 0.58 個加倍 ≈ **+0.31 dB**
  ⇒ 若它 ~= 2.6M，容量是真的滿了，碎片其實會被配置器回收 ⇒ 這條收線

⚠ 本工具比真實訓練**樂觀**（無 Lightning/dataloader、無長跑累積的碎片）
  ⇒ 採用時取通過值的 ~85%，且第一次跑要盯 ledger 的 `(實佔 x.xx)` 欄位
    （2026-09-04 已把 allocated 加進台帳，之後每次跑次都看得到真實餘裕）。""")


if __name__ == "__main__":
    main()
