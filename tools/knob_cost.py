#!/usr/bin/env python
"""兩個「冗餘」候選各值多少 ms/step？（CUDA event 直接量該函式，不受跨跑次 5% 噪音影響）

## 候選來源（2026-09-05 的靜態稽核）

`agd2_b12`（現行最佳）單次跑次的機制觸發統計，直接從 runner.log 的區段數出來：
```
[screen-size prune] 193 次（每個 densify 事件）  ⇒ 在做實事，總回收 189,433 顆 = cap 的 7.29%
[densify-blind]     193 次                      ⇒ **只印字，不影響訓練**
其餘全部              0 次（emergency/vpc/blur-split/harvest/ac-shrink/cost-budget/transparent）
```
而 `after_backward` 裡**唯一沒有守衛、每步都跑**的是 `_accumulate_error_score`
（全幅 `avg_pool2d` + N x 3 的投影矩陣乘），它的三個真正消費者
（`err_unlock_frac` / `err_guided_densify` / `transparent_corrector`）現行配方全是 0
⇒ 它只為了那 193 行 log 而每步執行。已加旗標 `densify_blind_report`（預設 False）。

第二個候選：`_add_xyz_noise` 實測 **59.4 ms/step**（總 602 ms 的 9.9%）。
gate `op_sigmoid(1-o)` 對高 opacity 粒子本來就 ~0：
```
eps      可跳過（14999/29999/60000 三個 ckpt）
1e-3      47.6% / 62.8% / **69.0%**
```
被跳過者每步位移 <= `scale^2*noise_lr*lr*eps` = 4.15e-8
⇒ 60,000 步隨機遊走累積 = 自身尺寸的 **0.15%**。已加旗標 `noise_gate_eps`（預設 0）。

## ⚠ 為什麼用 CUDA event 而不是跑完整跑次比 wall time

跨跑次 wall time 噪音約 **5%**（§11.29）⇒ 想量 1~10% 的改善，噪音會蓋過訊號。
本工具在**同一個 process** 內交錯量兩個版本、取中位數，把那個噪音消掉。

用法: python tools/knob_cost.py --run agd2_b12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeModule:
    """`_add_xyz_noise` 只用到 `gaussian_optimizers`（取 means 的 lr）與 `is_final_step`。"""
    def __init__(self, opt):
        self.gaussian_optimizers = [opt]
        self.is_final_step = False


def timeit(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="agd2_b12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--repeat", type=int, default=30)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    ck = sorted(glob.glob(f"outputs/{args.run}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {args.run} 的 step={args.step} ckpt")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.density_controllers.mcmc_2dgs_density_controller import (
        MCMC2DGSDensityController as Ctrl)
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck[0], device=dev, eval_mode=False, pre_activate=False)
    N = model.n_gaussians
    print(f"{args.run} @ {args.step}   N = {N:,}\n")

    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    out = dp.get_outputs()
    cam = out.train_set.cameras[0].to_device(dev)

    means = model.gaussians["means"]
    opt = torch.optim.Adam([{"params": [means], "lr": 1.767e-6, "name": "means"}])
    pl = _FakeModule(opt)

    cfg = Ctrl(cap_max=2600000)
    inst = cfg.instantiate() if hasattr(cfg, "instantiate") else None
    if inst is None:
        raise SystemExit("controller 不支援 instantiate()，請改用 config 直接建構")
    inst.config = cfg
    if hasattr(inst, "setup"):
        try:
            inst.setup("fit", type("M", (), {"gaussian_model": model, "device": dev})())
        except Exception:
            pass

    bg = torch.zeros((3,), device=dev)
    with torch.no_grad():
        o = renderer(cam, model, bg_color=bg)
    gt = torch.rand_like(o["render"])
    batch = (cam, (None, gt), None)

    print(f"{'項目':>34} {'ms/step':>10} {'佔 602ms':>10}")

    # ── 1. MCMC 噪音：全量 vs 稀疏 ──
    cfg.noise_gate_eps = 0.0
    t_full = timeit(lambda: inst._add_xyz_noise(o, batch, model, 40000, pl), args.repeat)
    cfg.noise_gate_eps = args.eps
    t_sp = timeit(lambda: inst._add_xyz_noise(o, batch, model, 40000, pl), args.repeat)
    g = inst.op_sigmoid(1.0 - model.get_opacities()).squeeze(-1)
    frac = float((g > args.eps).float().mean())
    print(f"{'_add_xyz_noise 全量':>34} {t_full:>10.2f} {100*t_full/602:>9.2f}%")
    print(f"{f'_add_xyz_noise 稀疏 eps={args.eps:g}':>34} {t_sp:>10.2f} {100*t_sp/602:>9.2f}%"
          f"   （實際計算 {100*frac:.1f}% 的粒子）")
    print(f"{'  => 省下':>34} {t_full - t_sp:>10.2f} {100*(t_full-t_sp)/602:>9.2f}%")

    # ── 2. 每步誤差累積（唯一無守衛者）──
    cfg.noise_gate_eps = 0.0
    t_err = timeit(lambda: inst._accumulate_error_score(o, batch, model), args.repeat)
    print(f"\n{'_accumulate_error_score':>34} {t_err:>10.2f} {100*t_err/602:>9.2f}%"
          f"   （消費者全為 0，只餵 193 行 log）")

    # ── 3. fast_noise（已實作但從沒開過）──
    cfg.fast_noise = True
    t_fast = timeit(lambda: inst._add_xyz_noise(o, batch, model, 40000, pl), args.repeat)
    cfg.fast_noise = False
    print(f"{'_add_xyz_noise + fast_noise':>34} {t_fast:>10.2f} {100*t_fast/602:>9.2f}%"
          f"   （代數改寫，vs 全量 {t_full:.2f}）")

    tot = (t_full - t_sp) + t_err
    print(f"""
  ★ 兩項合計可省 **{tot:.2f} ms/step = {100*tot/602:.2f}%**
    （9.6 小時的跑次 => 約省 {9.6*tot/602*60:.0f} 分鐘）

判讀與後續：
  省 >= 3%  => 值得，接著跑端到端驗分數（非位元等價：RNG 串流改變）
  省 <  1%  => 不值得冒品質風險，只留 `densify_blind_report`（那個是純 log，零風險）
⚠ `_accumulate_error_score` 的關閉是**訓練行為不變**（消費者全為 0），只少一行 log
  => 那部分可以直接採用，不需要驗分數。
⚠ 稀疏噪音改變 RNG 串流 => **必須**端到端驗四指標。""")


if __name__ == "__main__":
    main()
