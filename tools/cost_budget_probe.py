#!/usr/bin/env python
"""成本預算 `Σc_i <= B` 的可行性探針（純 CPU，不碰 GPU）。

要回答的問題（§11.49，2026-08-29）：
  `cost_aware_densify` 六項全平，而三個臂的 N **完全相同**（2.34M = 0.9xcap）
  => 族群大小由 `cap_max`（顆數）決定，成本訊號沒有預算可以交易。
  命題要的是 `max Q s.t. max_view Load(view) <= B`，而那從未被實作。

  **但在寫程式之前必須先排除退化**：如果成本和顆數是鎖死的
  （每顆的 c_i 都差不多、或固定 N 下 Load 沒有自由度），
  那 `Σc_i <= B` 就只是 `N <= cap` 換個寫法，整條線不值得做。

量什麼（全部來自 ckpt + SfM 相機，零 GPU、零訓練）：
  c_i(view) = 該顆在該視角覆蓋的 16px tile 數 = (2r/TILE)^2，r 來自投影
              （與 `internal/utils/strip_cameras.projected_radius` 同一套公式，
                也與 `tools/measure_tau.py` 的 tau 定義一致）
  Load(view) = Σ_i c_i(view)   <- 光柵器 binning 的實際條目數，VRAM 的主項
  B          = max_view Load   <- 最壞視角決定 VRAM 峰值

三個判準：
  A. **固定 N 下 Load 有沒有自由度**：三個臂 N 相同，Load 差多少？
     差 <5% => 成本被顆數鎖死 => 退化，收線。
  B. **c_i 的離散度**：CV 與 top10% 佔 Load 的比例。
     top10% 佔比接近 10% => 每顆一樣貴 => 沒有「便宜的那批」可挑 => 退化。
  C. **槓桿上界**：同一個 B 下，若全部改用第 10 百分位那麼便宜的顆，能放幾顆？
     ⚠ 這是**上界**，而且刻意忽略 §3.1.1 的覆蓋硬下限
       （只用便宜顆會蓋不滿場景）=> 真實增益一定低於它。
       它的用途是**否證**：若上界本身就 <1.5x，這條線不值得做。
     對照：顆數槓桿實測 **+0.541 dB/加倍**（記憶 cheap_lever_scaling）。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.colmap import read_images_binary, read_cameras_binary  # noqa: E402

TILE = 16


def load_run(run):
    """`run` 或 `run@step`。後者用於量**同一次跑次的軌跡**：
    B 與 N 若在整條軌跡上等比例上升，代表「限制成本」等價於「限制顆數」
    ⇒ `Sigma c_i <= B` 動態上退化，不值得寫 controller（§11.50 只比了三個跑次的終點）。"""
    want_step = None
    if "@" in run:
        run, w = run.split("@", 1)
        want_step = int(w)
    cks = sorted(glob.glob(f"outputs/{run}/**/*.ckpt", recursive=True),
                 key=lambda p: int(p.split("step=")[-1].split(".")[0]))
    if not cks:
        raise SystemExit(f"找不到 {run} 的 ckpt")
    if want_step is not None:
        cks = [p for p in cks if int(p.split("step=")[-1].split(".")[0]) == want_step] or cks[-1:]
    ck = torch.load(cks[-1], map_location="cpu")
    g = ck["state_dict"]
    means = g["gaussian_model.gaussians.means"].float()
    scales = torch.exp(g["gaussian_model.gaussians.scales"].float())[:, :2]
    step = int(cks[-1].split("step=")[-1].split(".")[0])
    return means, scales, step, os.path.basename(cks[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block-list", required=True)
    ap.add_argument("--down-sample", type=float, default=1.2)
    ap.add_argument("--n-cams", type=int, default=24,
                    help="抽樣視角數（CPU 上全部 284 台太慢）。Load 的最大值對抽樣敏感 => 固定種子並全臂共用同一批")
    args = ap.parse_args()

    sp = os.path.join(args.data, "sparse/0")
    images = read_images_binary(os.path.join(sp, "images.bin"))
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    cam0 = list(cams.values())[0]
    fx = float(cam0.params[0]) / args.down_sample
    W = int(round(cam0.width / args.down_sample))
    H = int(round(cam0.height / args.down_sample))
    want = {l.strip() for l in open(args.block_list) if l.strip()}
    sel = [im for im in images.values() if im.name in want]
    rng = np.random.default_rng(0)
    if len(sel) > args.n_cams:                      # ★ 全臂共用同一批視角，否則 Load 不可比
        sel = [sel[i] for i in sorted(rng.choice(len(sel), args.n_cams, replace=False))]
    n_tiles_frame = (W / TILE) * (H / TILE)
    print(f"影像 {W}x{H}  fx {fx:.1f}  視角 {len(sel)}（種子 0，全臂共用）  "
          f"每幀 tile 數 {n_tiles_frame:,.0f}")

    rows = []
    for run in args.runs:
        means, scales, step, ckname = load_run(run)
        s_max = scales.max(dim=1).values
        n = means.shape[0]
        loads, ci_sum, ci_cnt = [], torch.zeros(n), torch.zeros(n)
        for im in sel:
            R = torch.tensor(im.qvec2rotmat(), dtype=torch.float32)
            T = torch.tensor(im.tvec, dtype=torch.float32)
            pc = means @ R.T + T
            z = pc[:, 2]
            vis = z > 0.2
            if not bool(vis.any()):
                continue
            idx = vis.nonzero(as_tuple=True)[0]
            pcv, zc = pc[idx], z[idx]
            u = fx * pcv[:, 0] / zc + W / 2
            v = fx * pcv[:, 1] / zc + H / 2
            stretch = torch.sqrt(1 + (pcv[:, 0] / zc) ** 2 + (pcv[:, 1] / zc) ** 2)
            r = 3.0 * fx * s_max[idx] / zc * stretch
            inf = (u > -r) & (u < W + r) & (v > -r) & (v < H + r)
            if not bool(inf.any()):
                continue
            k = idx[inf]
            tiles = ((2 * r[inf] / TILE) ** 2).clamp(min=1.0, max=n_tiles_frame)
            loads.append(float(tiles.sum()))
            ci_sum[k] += tiles
            ci_cnt[k] += 1
        loads = np.array(loads)
        seen = ci_cnt > 0
        ci = (ci_sum[seen] / ci_cnt[seen]).numpy()          # 逐顆平均成本（只在看得到它的視角上）
        B = loads.max()
        order = np.sort(ci)[::-1]
        top10_share = order[:max(1, len(order) // 10)].sum() / order.sum()
        p10, p50, p99 = np.percentile(ci, [10, 50, 99])
        # 槓桿上界：同一個 B，全用 p10 那麼便宜的顆
        n_vis_worst = None
        lever = B / (p10 * len(ci) / len(ci))               # = B / p10 顆數上限
        rows.append((run, n, step, B, loads.mean(), ci.mean(), ci.std() / ci.mean(),
                     p10, p50, p99, top10_share, lever / n, ckname))

    print(f"\n{'臂':>14} {'N':>10} {'B=maxLoad':>12} {'meanLoad':>12} "
          f"{'c̄':>7} {'CV':>6} {'c_p10':>7} {'c_p50':>7} {'c_p99':>8} {'top10%佔':>9} {'槓桿上界':>9}")
    for (run, n, step, B, ml, cm, cv, p10, p50, p99, t10, lev, ck) in rows:
        print(f"{run:>14} {n:>10,} {B:>12,.0f} {ml:>12,.0f} "
              f"{cm:>7.2f} {cv:>6.2f} {p10:>7.2f} {p50:>7.2f} {p99:>8.1f} "
              f"{t10:>8.1%} {lev:>8.2f}x")

    if len(rows) > 1:
        Bs = np.array([r[3] for r in rows])
        Ns = np.array([r[1] for r in rows])
        print(f"\n判準 A（固定 N 下 Load 的自由度）：")
        print(f"  N 全距 {Ns.max()/Ns.min()-1:+.2%}   B 全距 {Bs.max()/Bs.min()-1:+.2%}")
        if Bs.max() / Bs.min() - 1 < 0.05:
            print("  ⇒ **B 幾乎不動 => 成本被顆數鎖死 => `Σc_i<=B` 退化成 `N<=cap`，收線**")
        else:
            print("  ⇒ B 在固定 N 下有實質變動 => 成本是獨立的自由度，約束值得換")
    print("""
判準 B：top10% 佔 Load 的比例。接近 10% => 每顆一樣貴、沒有便宜的那批可挑。
判準 C：槓桿上界 = 同一個 B 下改用 p10 成本的顆能放幾倍顆數。
  ⚠ 這是**上界**，刻意忽略覆蓋硬下限（§3.1.1：只用便宜顆蓋不滿場景）
    => 真實增益必定低於它。用途是**否證**：上界 <1.5x 就不值得做。
  對照：顆數槓桿實測 +0.541 dB/加倍。""")


if __name__ == "__main__":
    main()
