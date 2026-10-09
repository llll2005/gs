"""trim 的 record 改成「只用部分相機」會剪錯多少？（2026-10-09 新年代重量；舊數字在年代分界前、只能當機制參考）

trim 判準 v_i = max_{訓練視角} 每像素平均 T·α（contribution_accumulator reduce="max"），每次剪 v 最低的 10%。
只用部分相機時，某些顆的最佳視角沒被抽到 => v 被低估 => 掉進底部被錯剪。本工具在同一個 ckpt 上一次掃完全部相機，
同時累積幾種取樣方案的 v，比底部 10% 遮罩與「全部相機」的重疊：
  half_even／half_odd   每 2 台取 1（兩次 trim 輪流用就是「輪流取樣」；同一 ckpt 上兩者取 max = 全部，staleness 量不到）
  quarter               每 4 台取 1（舊年代 TRIM_SUBSAMPLE=4）
  rand_half             隨機一半
另外報：被錯剪的那批在「全部相機」下的 v 是門檻的幾倍（接近 1 => 只是邊界上換人，影響小；遠大於 1 => 剪掉真的有用的）。

用法：python tools/record_subsample_overlap.py --ckpt <ckpt> [--ratio 0.1]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    train = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                                      global_rank=0).get_outputs().train_set
    del ck
    N, C = model.n_gaussians, len(train)
    rng = np.random.default_rng(a.seed)
    rh = set(rng.choice(C, C // 2, replace=False).tolist())
    schemes = {"full": lambda i: True, "half_even": lambda i: i % 2 == 0, "half_odd": lambda i: i % 2 == 1,
               "quarter": lambda i: i % 4 == 0, "rand_half": lambda i: i in rh}
    v = {k: torch.zeros(N, device=dev) for k in schemes}
    bg = torch.zeros(3, device=dev)
    with torch.no_grad():
        for i in range(C):
            t = renderer(train.cameras[i].to_device(dev), model, bg_color=bg, record_transmittance=True)
            if isinstance(t, tuple):
                t = t[0]
            t = t.reshape(-1)
            for k, f in schemes.items():
                if f(i):
                    v[k] = torch.maximum(v[k], t)
    k_cut = max(1, int(N * a.ratio))
    full = v["full"]
    thr = torch.kthvalue(full, k_cut).values
    m_full = full <= thr
    print(f"ckpt {a.ckpt}\n  N={N:,}  相機 {C}  剪 {100 * a.ratio:.0f}%（{k_cut:,} 顆）  全部相機的門檻 v={float(thr):.3e}"
          f"  v==0 {float((full == 0).float().mean()) * 100:.2f}%")
    print(f"  {'方案':<10} {'相機':>5} {'遮罩重疊':>9} {'錯剪顆數':>10} {'錯剪者 v_full/門檻 中位':>22} {'p90':>8} {'>2倍的':>8}")
    for k in schemes:
        if k == "full":
            continue
        vk = v[k]
        m = vk <= torch.kthvalue(vk, k_cut).values
        wrong = m & ~m_full
        ov = float((m & m_full).sum()) / max(float(m_full.sum()), 1.0)
        if int(wrong.sum()) > 0:
            r = (full[wrong] / thr.clamp_min(1e-12)).float()
            q = torch.quantile(r[torch.randperm(r.numel(), device=dev)[:1_000_000]], torch.tensor([.5, .9], device=dev)).tolist()
            gt2 = float((r > 2).float().mean()) * 100
        else:
            q, gt2 = [float("nan")] * 2, 0.0
        ncam = sum(1 for i in range(C) if schemes[k](i))
        print(f"  {k:<10} {ncam:>5} {100 * ov:8.2f}% {int(wrong.sum()):>10,} {q[0]:>22.2f} {q[1]:>8.2f} {gt2:>7.1f}%")


if __name__ == "__main__":
    main()
