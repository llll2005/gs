#!/usr/bin/env python
"""自適應 cap ①：用共同價格 lambda 把顆數預算分給 16 塊（`紀錄/實驗分析/14` §3.3，使用者 10-17「可以排」）。

## 問題與解法（共享問題，KKT）
```
max sum_b Q_b(N_b)   s.t.  sum_b N_b <= B
=> 各塊「最後一顆的邊際價值」相等：dQ_b/dN_b = lambda
```
邊際價值用每顆的 V/C 近似（＝ best1 的 trim 判準：Σ_視角 Σ_像素 T·α ÷ Σ_視角 覆蓋像素；無單位、跨塊可比）。
⇒ 對全部 16 塊的顆粒用**同一個門檻 lambda**：V/C >= lambda 的留下；lambda 選成總數 = 預算。
每塊的 cap_b = 留下顆數 / 0.9（cap 只交付 90%：2.6M -> 2.34M，10-17 仍成立）。

⚠ V/C 是「平均價值」不是「邊際價值」——這是近似，判決實驗（同總量 vs 統一 cap）就是在驗它。
⚠ C=0（任何訓練視角都沒混合到）的顆粒價值記為 -1：永遠不入選。

用法：
  python tools/price_alloc.py --run outputs/lab/g44s20k_best1run --frac 0.5 --out logs/price_alloc_xxx.json
輸出 JSON：{"frac":..,"lambda":..,"caps":{"0":cap,...},"uniform_cap":..,"blocks":{b:{N,seen,keep}}}
（uniform_cap = 平均 cap_b => 兩臂 sum cap 相同）
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def last_ckpt(d):
    cks = glob.glob(os.path.join(d, "checkpoints", "*.ckpt"))
    if not cks:
        return None
    return max(cks, key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))


def block_scores(ck, max_cam, dev):
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck, device=dev, eval_mode=False, pre_activate=False)
    N = model.n_gaussians
    ckpt = torch.load(ck, map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    del ckpt
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"      # 這裡只要相機；init PLY 不在本機時不讀它
    dp = _pp.instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(ck)), global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    n = len(cams) if max_cam <= 0 else min(max_cam, len(cams))
    stride = max(1, len(cams) // n)
    bg = torch.zeros((3,), device=dev)
    V = torch.zeros(N, device=dev)
    C = torch.zeros(N, device=dev)
    used = 0
    with torch.no_grad():
        for i in range(0, len(cams), stride):
            cam = cams[i].to_device(dev)
            mean, covered = renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True)
            cf = covered.float()
            V += mean * cf          # mean = Σ T·α ÷ 覆蓋像素 => 還原成總和（同 renderer 週期 trim 的 `blur`）
            C += cf
            used += 1
    score = torch.where(C > 0, V / torch.clamp_min(C, 1.0), torch.full_like(V, -1.0))
    out = score.cpu().numpy().astype(np.float32)
    del model, renderer, V, C, score
    torch.cuda.empty_cache()
    return out, used, len(cams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/lab/<跑次>（底下 blocks/block_0..15）")
    ap.add_argument("--frac", type=float, default=0.5, help="總預算 = frac x 探針 16 塊的總顆數")
    ap.add_argument("--max-cam", type=int, default=0, help="每塊用幾台訓練相機（0 = 全部）")
    ap.add_argument("--cap-min", type=int, default=300_000)
    ap.add_argument("--cap-max", type=int, default=4_500_000, help="每塊 cap 上限（VRAM：峰值增量 GB ≈ 0.355+0.592N(M)+0.057Σtiles(M)）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")

    blocks = {}
    for b in range(16):
        d = os.path.join(args.run, "blocks", f"block_{b}")
        ck = last_ckpt(d)
        if ck is None:
            raise SystemExit(f"⛔ {d} 沒有 ckpt（探針 16 塊要全部跑完）")
        cache = os.path.join(d, "price_vc.npy")
        if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(ck):
            s = np.load(cache); used = tot = -1
        else:
            s, used, tot = block_scores(ck, args.max_cam, dev)
            np.save(cache, s)
        blocks[b] = s
        seen = int((s >= 0).sum())
        print(f"block {b:>2}：N {len(s):>9,}  看得到 {seen:>9,}（{100 * seen / len(s):5.1f}%）  相機 {used}/{tot}  "
              f"V/C 中位 {float(np.median(s[s >= 0])) if seen else 0:.3e}  ← {os.path.basename(ck)}", flush=True)

    Ntot = sum(len(s) for s in blocks.values())
    T = int(args.frac * Ntot)
    pool = np.concatenate([s[s >= 0] for s in blocks.values()])
    if T >= len(pool):
        lam = 0.0
    else:
        lam = float(np.partition(pool, len(pool) - T)[len(pool) - T])
    caps, info = {}, {}
    for b, s in blocks.items():
        keep = int((s >= lam).sum()) if lam > 0 else int((s >= 0).sum())
        caps[b] = int(min(max(round(keep / 0.9), args.cap_min), args.cap_max))
        info[b] = {"N": int(len(s)), "seen": int((s >= 0).sum()), "keep": keep}
    ucap = int(round(sum(caps.values()) / 16))
    print(f"\n預算 {args.frac:.2f} x {Ntot:,} = {T:,} 顆；共同價格 lambda（V/C 門檻）= {lam:.4e}")
    print(f"{'塊':>4} {'探針 N':>10} {'留下':>10} {'cap_b':>10} {'相對統一':>8}")
    for b in range(16):
        print(f"{b:>4} {info[b]['N']:>10,} {info[b]['keep']:>10,} {caps[b]:>10,} {caps[b] / ucap:>7.2f}x")
    print(f"統一 cap（同 Σcap）= {ucap:,}；Σcap = {sum(caps.values()):,}")
    json.dump({"run": args.run, "frac": args.frac, "lambda": lam, "caps": {str(b): c for b, c in caps.items()},
               "uniform_cap": ucap, "blocks": {str(b): v for b, v in info.items()}},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"=> {args.out}")


if __name__ == "__main__":
    main()
