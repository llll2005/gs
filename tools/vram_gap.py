#!/usr/bin/env python
"""VRAM 的 2.4 GB 缺口是誰佔的？（在**真實訓練步驟**的各階段量峰值）

## 缺口從哪來

`tools/step_breakdown.py` 量到（N=2.34M）：
```
常駐（參數 0.506 + 梯度 0.506 + Adam 動量 1.011）= 2.022 GB
渲染 forward 峰值增量                            = 1.214 GB
forward+backward 峰值增量                        = 1.901 GB
                                    小計約 3.9 GB
訓練期實測 reserved                              = 5.6~5.7 GB
                                    ⇒ **約 2.4 GB 未解釋**
```
⚠ 但 `step_breakdown` 量的是**裸 renderer 呼叫**，不含：
  optimizer step 的暫存／density controller 的緩衝／**trim pass（渲染全部 284 台相機）**。

## 已排除的候選（讀碼，零成本）

- **影像快取**：`image_on_cpu: true`（實際生效值）⇒ 影像留在 **CPU**，不佔 VRAM。
  （若為 false 會是 284 x 1600x900x3 x4B = **4.9 GB**，所以這個旗標很關鍵。）
- **相機**：`camera_on_cpu: false` ⇒ 在 GPU，但每台只有 R/T/內參，可忽略。

## 本工具量什麼

在**同一個 process** 內逐階段 `reset_peak_memory_stats()` 再量 `max_memory_allocated`：
```
A 常駐                    載入後（參數＋Adam 狀態建立後）
B forward                 單相機渲染
C forward+backward
D optimizer.step()        Adam 的暫存
E trim pass               **渲染全部 284 台相機並累積 contribution**  <- 最大嫌疑
F 加總 vs reserved        差額 = 配置器碎片
```
判讀：E 若吃掉一大塊 ⇒ **trim 是 VRAM 峰值的主因**，而它每 500 步才跑一次
⇒ 峰值由一個**間歇性事件**決定，而 `cap_max` 是為了那個峰值而被壓低的
⇒ 「錯開 trim 與高峰」或「trim 分批」可能直接換到更高的 cap。
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GB = 1024 ** 3


def peak(fn, label, out):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    p = torch.cuda.max_memory_allocated()
    out.append((label, (p - base) / GB))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="agd2_b12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--trim-cams", type=int, default=284,
                    help="trim pass 用幾台相機（284 = 與訓練時相同）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    ck = sorted(glob.glob(f"outputs/{args.run}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {args.run} 的 step={args.step} ckpt")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    model, renderer, _ = GaussianModelLoader.\
        initialize_model_and_renderer_from_checkpoint_file(ck[0], device=dev,
                                                           eval_mode=False, pre_activate=False)
    n = model.n_gaussians
    print(f"{args.run} @ step={args.step}   N = {n:,}")

    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    print(f"  image_on_cpu = {dmh.get('image_on_cpu')}   camera_on_cpu = {dmh.get('camera_on_cpu')}")
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                  output_path=os.path.dirname(os.path.dirname(ck[0])),
                                  global_rank=0)
    out_dp = dp.get_outputs()
    train_cams = out_dp.train_set.cameras
    cam = train_cams[0].to_device(dev)
    W, H = int(cam.width), int(cam.height)
    gt = torch.rand((3, H, W), device=dev)
    bg = torch.zeros((3,), device=dev)

    params = [p for p in model.gaussians.values()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": params, "lr": 1e-4}])

    res = []
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated() / GB
    print(f"\n{'階段':>26} {'峰值增量 GB':>13}")
    print(f"{'A 常駐（載入後 allocated）':>26} {resident:>13.3f}")

    peak(lambda: renderer(cam, model, bg_color=bg), "B forward", res)

    def fb():
        o = renderer(cam, model, bg_color=bg)
        loss = 0.8 * torch.abs(o["render"] - gt).mean() + 0.2 * (1 - ssim_fn(o["render"], gt))
        loss.backward()
    peak(fb, "C forward+backward", res)

    # Adam 狀態要先建立起來，否則第一次 step 的配置會混進來
    opt.step(); opt.zero_grad(set_to_none=True)
    fb()
    peak(lambda: opt.step(), "D optimizer.step()", res)
    opt.zero_grad(set_to_none=True)

    # ---- E：trim pass —— 渲染全部相機並累積 contribution ----
    from internal.renderers.sep_depth_trim_2dgs_renderer import contribution_accumulator
    K = getattr(renderer, "K", 5)

    def trim_pass():
        push, gather = contribution_accumulator(K)
        with torch.no_grad():
            for i in range(min(args.trim_cams, len(train_cams))):
                push(renderer(train_cams[i].to_device(dev), model,
                              bg_color=bg, record_transmittance=True))
            gather()
    peak(trim_pass, f"E trim pass（{args.trim_cams} 台）", res)

    for k, v in res:
        print(f"{k:>26} {v:>13.3f}")
    tot = resident + max(v for _, v in res)
    print(f"\n  常駐 + 最大單階段峰值 = {tot:.3f} GB")
    print(f"  目前 reserved         = {torch.cuda.memory_reserved() / GB:.3f} GB")
    print(f"  台帳訓練期 reserved    = 5.6~5.7 GB")
    print(f"""
判讀：
  E 佔最大 => **trim pass 決定 VRAM 峰值**，而它每 500 步才跑一次
             ⇒ 峰值由**間歇事件**決定，`cap_max` 是為了它被壓低的
             ⇒ 「trim 分批渲染」或「錯開峰值」可能直接換到更高的 cap
  各階段都小 => 缺口在配置器碎片（reserved 遠大於 allocated）⇒ 調 PYTORCH_CUDA_ALLOC_CONF
⚠ 本工具用**單一 process、乾淨起點**量，訓練時還有 dataloader/Lightning 的常駐，
  所以絕對值會低於台帳；要看的是**各階段的相對大小**。""")


if __name__ == "__main__":
    main()
