#!/usr/bin/env python
"""逼近 VRAM 上限時變慢，是**配置器在搬**還是單純計算變多？（教授 2026-09-11 提的實驗）

## 為什麼這個問題還沒答案

「VRAM 佔用在哪」`tools/vram_gap.py` 已經答完（2026-09-04，N=2.34M）：
```
allocated 峰值合計 ≈ 2.42 GB    reserved = 4.87 GB   => 缺口 2.45 GB **全是配置器碎片**
階段峰值：forward+backward 1.901 > trim pass 1.239 ~ forward 1.222 > optimizer 1.012
⇒ 台帳的 `VRAM=5.57/6.1G` 是 memory_reserved()，含約 40% 碎片
```
**沒量過的是「代價」**：餘裕變小之後，每一步是不是變慢、慢在哪。

## 教授的提法與它對應的量

「故意讓他 OOM 然後設定 flag，看他是因為搬移才慢還是啥」
⇒ PyTorch 的快取配置器在 `cudaMalloc` 失敗時，會**釋放全部快取區塊再重試**
  （`release_cached_blocks()` -> 一連串 `cudaFree` + 裝置同步）。那就是「搬移」的成本，
  而且它有**直接的計數器**：`torch.cuda.memory_stats()["num_alloc_retries"]`。
```
retries == 0 而步時間平  => 沒有搬移成本，慢就只是計算變多
retries > 0 且與時間同步跳 => **就是配置器在抖**，且可用旗標處理
```

## 設計：把「計算量」與「記憶體壓力」分開

```
臂 A 純計算   餘裕充足，N 從小掃到大        => t_compute(N)，預期 retries 全 0
臂 B 純壓力   **N 固定**，用 ballast 張量把可用 VRAM 一格一格吃掉，直到 OOM
              => 計算量逐字相同，只有餘裕在變 => 時間的變化**只能**來自配置器
臂 C 旗標     在 B 最緊的可行點，比較 alloc conf 設定（由任務腳本用 env 分三個 process 跑）
```
⚠ `expandable_segments` 在 **torch 2.0.1 不支援**（2.1 才有）⇒ 可調的只有
  `max_split_size_mb` / `garbage_collection_threshold` / `roundup_power2_divisions`。
  現行任務腳本已經固定 `max_split_size_mb:128` ⇒ **那是一個已經被釘住的變數**，
  臂 C 的對照組之一必須是「完全不設」。

⚠ 本工具沿用 `tools/vram_scale.py` 的複製法（整批複製粒子到目標 N）：
  對 VRAM 等價，且保留 opacity/scale 分布（決定 binning 工作量）。
  它是**上界估計，比真實訓練樂觀**（沒有 Lightning／dataloader／長跑累積的碎片）。

用法:
  python tools/vram_pressure.py --arm a
  python tools/vram_pressure.py --arm b --n 2.34
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tools/vram_pressure.py --arm b --n 2.34
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GB = 1024 ** 3


def replicate(model, target_n):
    """整批複製/截斷到 target_n（與 tools/vram_scale.py 同一套，保持可比）。"""
    n0 = model.n_gaussians
    idx = torch.arange(target_n, device=model.get_xyz.device) % n0
    with torch.no_grad():
        for k, v in model.gaussians.items():
            nv = v.data[idx].clone()
            if k == "means":
                nv += (torch.rand_like(nv) - 0.5) * 1e-4
            model.gaussians[k] = torch.nn.Parameter(nv, requires_grad=v.requires_grad)
    return target_n


def build(ck, dev, target_n):
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, _ = GaussianModelLoader.\
        initialize_model_and_renderer_from_checkpoint_file(
            ck, device=dev, eval_mode=False, pre_activate=False)
    replicate(model, target_n)
    params = list(model.gaussians.values())
    for p in params:
        p.requires_grad_(True)
    return model, renderer, torch.optim.Adam([{"params": params, "lr": 1e-6}])


def run_steps(model, renderer, opt, cams, dev, steps, warmup=3):
    """跑 steps 個完整訓練步，用 CUDA event 逐步計時，回傳 (每步 ms 的中位, retries 增量)。

    ⚠ 用 CUDA event 而不是 wall clock：CPU 時間會把非同步工作記到下一個同步點
    （`tools/step_breakdown.py` 踩過，`.to()` 看起來 0.17 秒其實是在等前一次光柵化）。
    """
    from internal.utils.ssim import ssim as ssim_fn
    bg = torch.zeros((3,), device=dev)
    ts = []
    r0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    for s in range(steps + warmup):
        cam = cams[s % len(cams)].to_device(dev)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        o = renderer(cam, model, bg_color=bg)
        img = o["render"]
        gt = torch.rand_like(img)
        loss = 0.8 * torch.abs(img - gt).mean() + 0.2 * (1 - ssim_fn(img, gt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.gaussians["means"].data += torch.randn_like(model.gaussians["means"]) * 1e-8
        b.record()
        torch.cuda.synchronize()
        if s >= warmup:
            ts.append(a.elapsed_time(b))
    r1 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    return float(np.median(ts)), r1 - r0


def stats_line():
    d = torch.cuda.memory_stats()
    return (torch.cuda.max_memory_allocated() / GB,
            torch.cuda.max_memory_reserved() / GB,
            d.get("num_alloc_retries", 0), d.get("num_ooms", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b"], default="b")
    ap.add_argument("--run", default="speed3_b12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--n", type=float, default=2.34, help="臂 B 固定的顆數（百萬）")
    ap.add_argument("--ns", type=float, nargs="+",
                    default=[0.6, 1.2, 1.8, 2.34, 2.8], help="臂 A 掃的顆數（百萬）")
    ap.add_argument("--ballast", type=float, nargs="+",
                    default=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5], help="臂 B 吃掉的 GB")
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    ck = sorted(glob.glob(f"outputs/{a.run}/**/*step={a.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {a.run} 的 step={a.step} ckpt")
    ck = ck[0]
    ckpt = torch.load(ck, map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck)),
                                   global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    del ckpt
    tot = torch.cuda.get_device_properties(0).total_memory / GB
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "（未設）")
    print(f"\n{a.run} @ {a.step}   顯卡 {tot:.2f} GB   torch {torch.__version__}")
    print(f"PYTORCH_CUDA_ALLOC_CONF = {conf}")
    print(f"⚠ expandable_segments 在 torch 2.0.1 不支援（2.1 才有）\n")

    if a.arm == "a":
        print("=== 臂 A：餘裕充足，只有計算量在變（建立 t_compute(N) 基準線）===")
        print(f"{'N':>9} {'每步中位(ms)':>13} {'vs 最小 N':>10} {'peak alloc':>11} "
              f"{'peak resv':>10} {'碎片':>8} {'retries':>8}")
        base = None
        for t in a.ns:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                model, renderer, opt = build(ck, dev, int(t * 1e6))
                ms, dr = run_steps(model, renderer, opt, cams, dev, a.steps)
                pa, pr, rt, om = stats_line()
                base = base or ms
                print(f"{t:>8.2f}M {ms:>13.1f} {ms/base:>9.2f}x {pa:>11.3f} "
                      f"{pr:>10.3f} {pr-pa:>8.3f} {dr:>8}")
            except torch.cuda.OutOfMemoryError:
                print(f"{t:>8.2f}M {'⛔ OOM':>13}")
                break
            finally:
                # ⚠ `del locals()[v]` 在 CPython **無效**（locals() 是快照）=> 模型不會被釋放，
                #   下一輪會假性 OOM。必須明確重新指派。
                model = renderer = opt = None
                torch.cuda.empty_cache()
        print("""
判讀：這條線是**純計算**的成本（retries 應該全 0）。臂 B 相對於它的超出量，才是壓力的代價。""")
        return 0

    # ── 臂 B ──────────────────────────────────────────────────────────────────
    n = int(a.n * 1e6)
    print(f"=== 臂 B：N 固定 {a.n:.2f}M（**計算量逐字相同**），只有可用餘裕在變 ===")
    print(f"{'ballast':>9} {'每步中位(ms)':>13} {'vs 無壓力':>10} {'retries':>8} "
          f"{'num_ooms':>9} {'peak alloc':>11} {'peak resv':>10} {'碎片':>8} {'剩餘':>8}")
    base = None
    for bal in a.ballast:
        torch.cuda.empty_cache()
        ball = None    # list[Tensor]
        try:
            model, renderer, opt = build(ck, dev, n)
            if bal > 0:
                # 先建模型再吃 ballast：順序與真實訓練一致（模型常駐、暫存後到）。
                # ⚠ 分成 128 MB 小塊，不要單塊 1.5 GB —— 單塊可能自己就配置失敗，
                #   那會把「牆」量在 ballast 上而不是訓練步上，等於量錯對象。
                ball, need = [], int(bal * GB)
                chunk = 128 * 1024 * 1024
                while need > 0:
                    take = min(chunk, need)
                    ball.append(torch.empty(take, dtype=torch.uint8, device=dev))
                    need -= take
            torch.cuda.reset_peak_memory_stats()
            ms, dr = run_steps(model, renderer, opt, cams, dev, a.steps)
            pa, pr, rt, om = stats_line()
            base = base or ms
            print(f"{bal:>8.2f}G {ms:>13.1f} {ms/base:>9.3f}x {dr:>8} {om:>9} "
                  f"{pa-bal:>11.3f} {pr-bal:>10.3f} {pr-pa:>8.3f} {tot-pr:>8.3f}")
        except torch.cuda.OutOfMemoryError:
            d = torch.cuda.memory_stats()
            where = "配置 ballast 時" if (ball is None or
                                          sum(t.numel() for t in ball) < int(bal * GB)) \
                else "**訓練步**"
            print(f"{bal:>8.2f}G {'⛔ OOM':>13} {'':>10} "
                  f"{d.get('num_alloc_retries',0):>8} {d.get('num_ooms',0):>9}"
                  f"   <- 牆在這裡（OOM 發生在 {where}）")
            break
        finally:
            ball = model = renderer = opt = None      # 同上：不可用 del locals()[...]
            torch.cuda.empty_cache()
    print("""
判讀（計算量完全相同，所以時間的變化只能來自配置器）：
  retries 全 0 且時間平到 OOM 為止 => **沒有搬移成本**，只有一道硬牆
                                      => 效率問題全在計算，該往 §3 的計算面找
  retries > 0 且時間同步跳          => **就是配置器在抖**（cudaMalloc 失敗 -> 釋放全部快取
                                      -> 重試，過程會同步裝置）=> 臂 C 的旗標值得試
  retries 為 0 但時間仍升           => 既不是搬移也不是計算 => 查 kernel 佔用/時脈（nvidia-smi -q -d PERFORMANCE）
⚠ 本工具沒有 Lightning／dataloader／長跑累積的碎片 ⇒ 是**樂觀**估計；真實訓練只會更早撞牆。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
