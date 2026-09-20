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
    ap.add_argument("--exact", action="store_true",
                    help="★ 2026-09-21：同時渲染同一組相機，取光柵器回傳的**精確** tile 數，"
                         "報估計/精確的比值。需要 2026-09-21 之後重編的光柵器（有 `tiles` 輸出）。")
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

        # ★★ 2026-09-21：估計 vs 精確。`estimate_render_load` 的檔頭自承是「保守上界」，
        #   而它同時踩了四個高估來源：用 s_max 撐**正方形**、假設**正面朝向**（無前縮）、
        #   `(2r/TILE)^2` **忽略 tile 量化**、**不做畫面裁切**。
        #   在今天之前沒辦法量它到底高估多少 —— 光柵器的逐顆 tile 數從未暴露。
        #   為什麼要量：K>1 是**有損**的（SSIM 系統性偏高、absgrad 退化成 no-op），
        #   而本工具已經量到「兩個 60k 模型都被判 K=6，但同 N 用 K=1 實測 4.87 GB 沒 OOM」
        #   => 預測器的保守度**直接害死過一個 60k 跑次**。高估倍率就是重新標定的依據。
        exact_tot = est_tot = None
        if a.exact:
            try:
                from internal.utils.gaussian_model_loader import GaussianModelLoader
                dev = torch.device("cuda")
                model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
                    ck[0], device=dev, eval_mode=True, pre_activate=False)
                bg = torch.zeros((3,), device=dev)
                ex, es = [], []
                with torch.no_grad():
                    for i in range(min(a.max_cam, len(cams))):
                        o = renderer(cams[i].to_device(dev), model, bg_color=bg)
                        if "tiles" not in o:
                            raise RuntimeError("光柵器沒回傳 `tiles` —— 需要 2026-09-21 之後重編")
                        ex.append(float(o["tiles"].double().sum()))
                        es.append(float(loads[i]) if np.isfinite(loads[i]) else float("nan"))
                ex = np.array(ex); es = np.array(es)
                m = np.isfinite(es) & (ex > 0)
                exact_tot, est_tot = ex[m], es[m]
            except Exception as e:
                print(f"  ⚠⚠ --exact 失敗，只報估計值：{e}")

        model_b = N * fpp * 4 * 4
        fixed = model_b + A_RENDER * N
        print(f"=== {run} @ {a.step} ===")
        print(f"  N={N:,}  floats/point={fpp}  "
              f"model_state={model_b/GB:.3f} GiB  fixed(含 A_RENDER)={fixed/GB:.3f} GiB")
        print(f"  load（相交數）中位 {np.median(fin)/1e6:.1f}M  "
              f"p90 {np.percentile(fin,90)/1e6:.1f}M  "
              f"MONSTER(+inf) {int((~np.isfinite(loads)).sum())}/{len(loads)} 台")
        if exact_tot is not None and len(exact_tot) > 0:
            r = est_tot / exact_tot
            print(f"  ★ 精確 tile 數（光柵器）中位 {np.median(exact_tot)/1e6:.1f}M  "
                  f"p90 {np.percentile(exact_tot,90)/1e6:.1f}M")
            print(f"  ★★ 估計/精確 = 中位 **{np.median(r):.2f}x**  "
                  f"範圍 {r.min():.2f}~{r.max():.2f}  "
                  f"（<1 代表**低估**＝預測器不安全）")
            print(f"     高估 {100*(np.median(r)-1):.0f}%。⚠ **但這不等於修了就能降 K** —— 見下方兩行 K 的對照。")
        for lab, tgt, vos, saf, kmax in PARAMSETS:
            budget = saf * (tgt * GB - vos * GB)
            ks = [predict_num_strips(float(l), N, fpp, tgt, vos, saf, kmax) for l in loads]
            room = budget - fixed
            line = (f"    {lab:>26}  budget={budget/GB:.3f} GiB  room={room/GB:+.3f} GiB"
                    f"  => K 中位 **{int(np.median(ks))}**  範圍 {min(ks)}~{max(ks)}")
            if exact_tot is not None and len(exact_tot) > 0:
                # ★ 用**精確** load 重算 K：回答「把估計修準，K 會不會降」。
                #   不能用推理代替 —— 峰值模型是 [B·N + γ·load]/K，而 B·N 那一項與 load 無關。
                kx = [predict_num_strips(float(l), N, fpp, tgt, vos, saf, kmax) for l in exact_tot]
                line += f"   ｜用精確 load 重算 => K 中位 **{int(np.median(kx))}** 範圍 {min(kx)}~{max(kx)}"
            print(line)
        if run in ACTUAL_K1:
            print(f"  ★ 同 N 在 K=1 **實測**峰值 {ACTUAL_K1[run]:.2f} GB（沒 OOM，卡 6.1 GB）")
        print()
    print("""判讀：
  兩個模型在同參數下都被判高 K，而 depth-init 用 K=1 實跑 4.87 GB 沒爆
  => **預測器的 budget 訂得比硬體實際能吃的低約 1 GB** => K 是白付的
  只有 SfM-init 被判高 K => 它的足跡真的比較大，那 K 就不是白付的""")


if __name__ == "__main__":
    main()
