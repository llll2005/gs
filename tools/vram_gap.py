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
                    help="trim pass 用幾台相機（舊資料 284；新資料 b6=548 / b12=653 / b13=667）")
    ap.add_argument("--ckpt", default=None,
                    help="直接指定 ckpt 路徑。★ --run/--step 是用 glob 找的，"
                         "同一個 run 下多個 block 都有同一步數時會挑到錯的那個"
                         "（2026-09-13 踩到：lab/speed3 的 b6 與 b13 都會有 step=29999）")
    ap.add_argument("--frag", type=int, default=0, metavar="輪數",
                    help="碎片模式：跑 N 輪 trim pass 而**不清快取**，每輪報 "
                         "allocated/reserved/num_alloc_retries。0 = 不跑（預設走原本的逐階段量測）")
    args = ap.parse_args()

    # ★ 套用與訓練相同的 VRAM 上限。**不套就量不到碎片** ——
    #   lab 是 24 GiB，沒有上限就永遠有餘裕、retries 恆為 0，整個量測沒有意義。
    #   直接複用 main.py 那條路徑，不重寫（同一個環境變數 CITYGS_VRAM_CAP_GB）。
    from internal.entrypoints.gspl import _apply_vram_cap
    _apply_vram_cap()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    if args.ckpt:
        ck = sorted(glob.glob(args.ckpt))
        if not ck:
            raise SystemExit(f"--ckpt 指定的路徑找不到檔案：{args.ckpt}")
        if len(ck) > 1:
            raise SystemExit(f"--ckpt 展開成 {len(ck)} 個檔案，請指定到唯一一個：\n  "
                             + "\n  ".join(ck))
    else:
        ck = sorted(glob.glob(f"outputs/{args.run}/**/*step={args.step}.ckpt", recursive=True))
        if not ck:
            raise SystemExit(f"找不到 {args.run} 的 step={args.step} ckpt")
        if len(ck) > 1:
            print(f"⚠⚠ glob 找到 {len(ck)} 個 ckpt，用第一個 —— 這**可能是錯的那一個**，"
                  f"請改用 --ckpt 指定：")
            for c in ck:
                print(f"     {c}")
    print(f"  ckpt = {ck[0]}")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    model, renderer, _ = GaussianModelLoader.\
        initialize_model_and_renderer_from_checkpoint_file(ck[0], device=dev,
                                                           eval_mode=False, pre_activate=False)
    n = model.n_gaussians
    # ⚠ 用 --ckpt 時 args.run/args.step 還是預設值 => 原本這行會印出**錯的來源**
    #   （實際 b6@29999 卻標成 agd2_b12@60000）。標籤錯位在本專案已經害過多次，
    #   所以一律以真正載入的 ckpt 路徑為準。
    print(f"來源 {ck[0]}   N = {n:,}")

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

    # ── 碎片模式 ──
    # ⚠ 為什麼要獨立一段：上面的 `peak()` 每段前都會 `empty_cache()` +
    #   `reset_peak_memory_stats()` —— 那正好把「配置器手上留了多少、留成什麼形狀」
    #   清掉，也就是把要量的東西清掉了。docstring 裡的「F 加總 vs reserved」
    #   因此一直量不到真東西。這裡改成**連續跑、不清快取**，看它怎麼累積。
    # 要答的問題（2026-09-13）：同一個 N、同一個上限下，
    #   **相機數**是不是碎片的來源？trim pass 每輪走全部相機，而每台的 binning
    #   緩衝大小隨視角變動 ⇒ 相機愈多，進出配置器的「不同尺寸大塊」愈多。
    #   對照訊號：b6（548 台）gap 0.11G vs b13（667 台）gap 0.74G，而實佔幾乎相同。
    if args.frag > 0:
        ncam = min(args.trim_cams, len(train_cams))
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        print(f"\n★ 碎片模式：{ncam} 台相機 x {args.frag} 輪（不清快取）"
              f"   上限={os.environ.get('CITYGS_VRAM_CAP_GB') or '未設'}")
        print(f"{'輪':>4} {'allocated':>11} {'reserved':>11} {'gap':>9} "
              f"{'retries':>8} {'ooms':>6}")
        for r in range(args.frag):
            trim_pass()
            torch.cuda.synchronize()
            stt = torch.cuda.memory_stats()
            a = torch.cuda.memory_allocated() / GB
            rs = torch.cuda.memory_reserved() / GB
            print(f"{r + 1:>4} {a:>10.3f}G {rs:>10.3f}G {rs - a:>8.3f}G "
                  f"{stt.get('num_alloc_retries', 0):>8} {stt.get('num_ooms', 0):>6}")
        print("""
判讀（碎片模式）：
  retries 隨相機數上升        => **相機數確實在製造碎片**（配置器要先釋放快取才配得到）
  retries 恆為 0 而 gap 也平  => 相機數不是來源，回頭查 N 與間歇峰值
  gap 逐輪單調上升            => 累積型碎片（愈跑愈糟），與「跑到 step 14,000 才死」相符
⚠ 沒設 CITYGS_VRAM_CAP_GB 就不要解讀：24 GiB 上永遠有餘裕，retries 必為 0。""")

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
