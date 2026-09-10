#!/usr/bin/env python
"""dynamic_strips 逼出 K=6~8 是真的需要，還是預測器太保守？（純 CPU）

## 起因（2026-09-11）

`speed3_sfminit_b12` 與 `speed3_b12` **同樣 2.34M 顆**，但前者 VRAM 峰值實佔只有 4.08 GB
而慢 24%（1.85 vs 2.44 it/s）。差別不在 SfM-init，在於它的 config **開了 `dynamic_strips`**：
`dynk_K.log` 600 個採樣點裡 **K>1 佔 77.7%、K>=6 佔 69.3%**。

⚠ 而 `_strip_forward_backward` 的 docstring 自承
「Image-space SSIM is computed per strip (**boundary-window approximation**)」
⇒ 77.7% 的訓練步跑在被改過的 loss 上 ⇒ **這次跑不能用來判定 SfM-init**。

## 量什麼

```
① 預測器要多少 K      對兩個 60k 模型、同一批相機跑 estimate_render_load + predict_num_strips
② 預算 vs 實測         predict 的 budget 對照「同 N 用 K=1 實際跑到多少峰值而沒 OOM」
```
判準：
```
depth-init 在同參數下也被判 K>=6，而它 K=1 實跑 4.87 GB 沒爆 => **預測器過保守**
只有 SfM-init 被判高 K                                      => 它的幾何真的比較貴
```
用法: python tools/strip_k_audit.py
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.strip_cameras import (estimate_render_load, predict_num_strips,  # noqa: E402
                                          A_RENDER, B_RENDER, BYTES_PER_ISECT)

GB = 2 ** 30
# (名稱, vram_target_gb, v_os_gb, safety, max_strips)
PARAMSETS = [("sfminit 用的 (5.2/0.8/0.85)", 5.2, 0.8, 0.85, 8),
             ("speed3 用的 (5.4/0.8/0.60)", 5.4, 0.8, 0.60, 8)]
# 同 N=2.34M、K=1 實測沒 OOM 的峰值（train_status.txt 的「峰值實佔」）
ACTUAL_K1 = {"speed3_b12": 4.87}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["speed3_b12", "speed3_sfminit_b12"])
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--max-cam", type=int, default=24)
    a = ap.parse_args()

    print(f"常數：A_RENDER={A_RENDER} B/pt  B_RENDER={B_RENDER} B/pt  "
          f"BYTES_PER_ISECT={BYTES_PER_ISECT}\n")
    for run in a.runs:
        ck = sorted(glob.glob(f"outputs/{run}/**/*step={a.step}.ckpt", recursive=True))
        if not ck:
            print(f"{run}: 找不到 ckpt"); continue
        c = torch.load(ck[0], map_location="cpu")
        sd = c["state_dict"]
        pre = "gaussian_model.gaussians."
        props = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        N = props["means"].shape[0]
        fpp = sum(int(np.prod(v.shape[1:])) if v.dim() > 1 else 1 for v in props.values())
        means = props["means"].float()
        scales = torch.exp(props["scales"].float())

        dmh = c["datamodule_hyper_parameters"]
        dp = dmh["parser"].instantiate(path=dmh["path"],
                                       output_path=os.path.dirname(os.path.dirname(ck[0])),
                                       global_rank=0)
        cams = dp.get_outputs().train_set.cameras
        loads = [estimate_render_load(means, scales, cams[i])
                 for i in range(min(a.max_cam, len(cams)))]
        loads = np.array([l if np.isfinite(l) else np.inf for l in loads])
        fin = loads[np.isfinite(loads)]

        model_b = N * fpp * 4 * 4
        fixed = model_b + A_RENDER * N
        print(f"=== {run} @ {a.step} ===")
        print(f"  N={N:,}  floats/point={fpp}  "
              f"model_state={model_b/GB:.3f} GiB  fixed(含 A_RENDER)={fixed/GB:.3f} GiB")
        print(f"  load（相交數）中位 {np.median(fin)/1e6:.1f}M  "
              f"p90 {np.percentile(fin,90)/1e6:.1f}M  "
              f"MONSTER(+inf) {int((~np.isfinite(loads)).sum())}/{len(loads)} 台")
        for lab, tgt, vos, saf, kmax in PARAMSETS:
            budget = saf * (tgt * GB - vos * GB)
            ks = [predict_num_strips(float(l), N, fpp, tgt, vos, saf, kmax) for l in loads]
            room = budget - fixed
            print(f"    {lab:>26}  budget={budget/GB:.3f} GiB  room={room/GB:+.3f} GiB"
                  f"  => K 中位 **{int(np.median(ks))}**  範圍 {min(ks)}~{max(ks)}")
        if run in ACTUAL_K1:
            print(f"  ★ 同 N 在 K=1 **實測**峰值 {ACTUAL_K1[run]:.2f} GB（沒 OOM，卡 6.1 GB）")
        print()
    print("""判讀：
  兩個模型在同參數下都被判高 K，而 depth-init 用 K=1 實跑 4.87 GB 沒爆
  => **預測器的 budget 訂得比硬體實際能吃的低約 1 GB** => K 是白付的
  只有 SfM-init 被判高 K => 它的足跡真的比較大，那 K 就不是白付的""")


if __name__ == "__main__":
    main()
