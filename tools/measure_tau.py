#!/usr/bin/env python
"""量訓練後的 tau（最壞視角的逐點 tile 覆蓋），重算各 K 的 N_max。

為什麼要量（研究總覽 §11.31，2026-08-27）：7/24 報告的 N_max 表
`K=1 -> 1.70M / K=2 -> 2.50M / K=4 -> 3.26M / K=8 -> 3.86M` 是用 **init 時的 tau_MAX=73** 算的。
`freeze_b12_K2` 實測在 1.8M 就 OOM（預期 2.5M），死因是**訓練中高斯攤開、tau 漲到約 120**
=> static 公式偏樂觀。K-strip 那條線值不值得復活，取決於**當前配方的成品模型 tau 是多少**。

⚠ 這是 GT 無關的**記憶體量測**（VRAM/成本），不受 GT 錯位期影響。

N_max = (V_target - V_os) / (M*F*4 + gamma*tau/K)
  V_target = 6GB * 0.9 = 5.4G（碎片化留 10%）   V_os = 0.718G（實測）
  M = 4（params+grad+2*Adam）  F = 每點 floats（SH3 = 58）  gamma = 23 B/相交（實測）
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


class _Cam:
    """strip_cameras.projected_radius 需要的最小介面：R, T, fx"""
    def __init__(self, R, T, fx):
        self.R, self.T, self.fx = R, T, fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sched30_b12")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block-list", required=True, help="partition 的 XXX_YYY.txt")
    ap.add_argument("--down-sample", type=float, default=1.2, help="與訓練 config 一致")
    ap.add_argument("--V-target", type=float, default=6.0 * 0.9)
    ap.add_argument("--V-os", type=float, default=0.718)
    ap.add_argument("--gamma", type=float, default=23.0, help="bytes/相交（實測）")
    ap.add_argument("--F", type=int, default=58, help="每點 floats（SH3=58, SB=25）")
    ap.add_argument("--ply", default=None, help="改用 PLY（量 init 幾何），與 --run 二選一")
    args = ap.parse_args()

    if args.ply:
        from plyfile import PlyData
        d = PlyData.read(args.ply)["vertex"]
        means = torch.tensor(np.stack([d["x"], d["y"], d["z"]], 1), dtype=torch.float32)
        sc = [n for n in d.data.dtype.names if n.startswith("scale_")][:2]
        scales = torch.exp(torch.tensor(np.stack([d[n] for n in sc], 1), dtype=torch.float32))
        print(f"PLY {os.path.basename(args.ply)}   N = {means.shape[0]:,}")
    else:
        cks = sorted(glob.glob(f"outputs/{args.run}/**/*.ckpt", recursive=True),
                     key=lambda p: int(p.split("step=")[-1].split(".")[0]))
        if not cks:
            raise SystemExit(f"找不到 {args.run} 的 ckpt")
        ck = torch.load(cks[-1], map_location="cpu")
        g = ck["state_dict"]
        means = g["gaussian_model.gaussians.means"].float()
        scales = torch.exp(g["gaussian_model.gaussians.scales"].float())[:, :2]
        print(f"ckpt {os.path.basename(cks[-1])}   N = {means.shape[0]:,}")

    sp = os.path.join(args.data, "sparse/0")
    images = read_images_binary(os.path.join(sp, "images.bin"))
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    cam0 = list(cams.values())[0]
    fx = float(cam0.params[0]) / args.down_sample
    W = int(round(cam0.width / args.down_sample))
    H = int(round(cam0.height / args.down_sample))
    want = {l.strip() for l in open(args.block_list) if l.strip()}
    sel = [im for im in images.values() if im.name in want]
    print(f"塊內相機 {len(sel)}   影像 {W}x{H}   fx {fx:.1f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    means, scales = means.to(dev), scales.to(dev)
    s_max = scales.max(dim=1).values
    n_tiles_frame = (W / TILE) * (H / TILE)

    taus = []
    for im in sel:
        R = torch.tensor(im.qvec2rotmat(), dtype=torch.float32, device=dev)
        T = torch.tensor(im.tvec, dtype=torch.float32, device=dev)
        pc = means @ R.T + T
        z = pc[:, 2]
        # ⚠ 2026-08-28 修正：第一版對**全部** 2.34M 顆點取平均（含相機後方與視錐外，
        # `clamp_min(0.2)` 讓後方的點半徑爆掉）=> tau_MAX 灌到 3407（init 值是 73），
        # 推出 N_max=0.06M 而我方實際在跑 2.34M ⇒ 顯然是量測錯。
        # 光柵器只數**真的進 binning** 的點 ⇒ 這裡也只算視錐內的。
        vis = z > 0.2
        if int(vis.sum()) == 0:
            continue
        pcv, zc = pc[vis], z[vis]
        u = fx * pcv[:, 0] / zc + W / 2
        v = fx * pcv[:, 1] / zc + H / 2
        stretch = torch.sqrt(1 + (pcv[:, 0] / zc) ** 2 + (pcv[:, 1] / zc) ** 2)
        r = 3.0 * fx * s_max[vis] / zc * stretch            # 螢幕半徑（px）
        inframe = (u > -r) & (u < W + r) & (v > -r) & (v < H + r)
        if int(inframe.sum()) == 0:
            continue
        tiles = ((2 * r[inframe] / TILE) ** 2).clamp(max=n_tiles_frame)
        taus.append(float(tiles.mean()))
    taus = np.array(taus)
    tau_mean, tau_max = float(taus.mean()), float(taus.max())
    print(f"\ntau_mean = {tau_mean:.1f}   tau_MAX = {tau_max:.1f}   "
          f"（7/24 用的 init 值：mean 30.5 / MAX 73）")

    V = args.V_target - args.V_os
    print(f"\nN_max = ({args.V_target} - {args.V_os})G / (4*{args.F}*4 + {args.gamma}*tau/K)")
    print(f"{'K':>3} {'N_max@tau_MAX':>15} {'N_max@tau_mean':>16}   7/24 的表（init tau）")
    ref = {1: 1.70, 2: 2.50, 4: 3.26, 8: 3.86}
    for K in (1, 2, 4, 8):
        def nmax(tau):
            per_pt = 4 * args.F * 4 + args.gamma * tau / K      # bytes/point
            return V * 1e9 / per_pt / 1e6
        print(f"{K:>3} {nmax(tau_max):>14.2f}M {nmax(tau_mean):>15.2f}M   {ref[K]:>10.2f}M")
    print(f"""
判讀：若 tau_MAX 已遠高於 73（訓練中高斯攤開），整張 7/24 的表要下修，
      K 買到的餘裕比當初估的少 => K-strip 那條線的期望值下降。
⚠ 這是**上界估計**（不做 frame-clip、用 facing-worst-case 半徑），偏保守。
⚠ radix sort 暫存（約 2x 相交陣列）沒完全計入 gamma —— 實際更緊。""")


if __name__ == "__main__":
    main()
