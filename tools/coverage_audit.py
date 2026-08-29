#!/usr/bin/env python
"""模型裡有多少質量是「塊內相機根本沒看到」的？

起因（2026-08-28，使用者用 web viewer 截圖）：單塊 b12 的渲染裡，一部分俯視區清楚、
一部分整片灰，而**清楚/糊掉的位置跨配方固定**。

★ 關鍵推論：**我方的 val/test 視角全都是塊內相機**，它們看的是 block 中心的內容。
   viewer 看到的那些糊掉區在**block 邊緣**，**從來不會出現在任何 test 圖裡**
   => 指標說 26.4 dB，viewer 說一坨灰，兩者都沒錯，因為它們量的不是同一塊區域。

本工具量：每顆粒子被幾台**塊內**相機的視錐涵蓋。被 0~少數相機看到的粒子是
**未受約束的**（沒有光度監督），它們在單塊渲染裡是垃圾，而合併後會疊加到鄰塊上。
⇒ 這直接影響合併策略（block 之間要重疊多少、要不要在合併前剔除未受約束的質量）。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.colmap import read_images_binary, read_cameras_binary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sched30fast_b12")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block-list", required=True)
    ap.add_argument("--down-sample", type=float, default=1.2)
    args = ap.parse_args()

    cks = sorted(glob.glob(f"outputs/{args.run}/**/*.ckpt", recursive=True),
                 key=lambda p: int(p.split("step=")[-1].split(".")[0]))
    ck = torch.load(cks[-1], map_location="cpu")
    means = ck["state_dict"]["gaussian_model.gaussians.means"].float()
    op = torch.sigmoid(ck["state_dict"]["gaussian_model.gaussians.opacities"].float()).squeeze(-1)
    print(f"{args.run}   N = {means.shape[0]:,}")

    sp = os.path.join(args.data, "sparse/0")
    images = read_images_binary(os.path.join(sp, "images.bin"))
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    c0 = list(cams.values())[0]
    fx = float(c0.params[0]) / args.down_sample
    W, H = int(round(c0.width / args.down_sample)), int(round(c0.height / args.down_sample))
    want = {l.strip() for l in open(args.block_list) if l.strip()}
    sel = [im for im in images.values() if im.name in want]
    print(f"塊內相機 {len(sel)}   影像 {W}x{H}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    means = means.to(dev)
    seen = torch.zeros(means.shape[0], dtype=torch.int16, device=dev)
    for im in sel:
        R = torch.tensor(im.qvec2rotmat(), dtype=torch.float32, device=dev)
        T = torch.tensor(im.tvec, dtype=torch.float32, device=dev)
        pc = means @ R.T + T
        z = pc[:, 2]
        u = fx * pc[:, 0] / z.clamp_min(1e-6) + W / 2
        v = fx * pc[:, 1] / z.clamp_min(1e-6) + H / 2
        seen += ((z > 0.2) & (u > 0) & (u < W) & (v > 0) & (v < H)).to(torch.int16)
    seen = seen.cpu().numpy()
    op = op.numpy()

    print(f"\n{'被幾台塊內相機看到':>20} {'粒子佔比':>10} {'不透明度中位':>12} {'質量佔比(sum o)':>15}")
    tot_o = op.sum()
    for lo, hi, lab in [(0, 0, "0（完全沒看到）"), (1, 2, "1-2"), (3, 5, "3-5"),
                        (6, 15, "6-15"), (16, 10 ** 9, ">=16")]:
        m = (seen >= lo) & (seen <= hi)
        if m.sum() == 0:
            continue
        print(f"{lab:>20} {100*m.mean():>9.2f}% {np.median(op[m]):>12.4f} {100*op[m].sum()/tot_o:>14.2f}%")
    weak = seen <= 2
    print(f"""
判讀：`<=2 台相機` 的粒子幾乎沒有光度監督 => 單塊渲染裡是垃圾，合併後會疊到鄰塊上。
      本模型有 **{100*weak.mean():.1f}%** 的粒子、**{100*op[weak].sum()/tot_o:.1f}%** 的不透明度質量落在這一類。
⚠ 這些**從來不會出現在 val/test 圖裡**（那些視角都是塊內相機、看的是 block 中心）
  => 指標看不到它們，但 viewer 看得到，合併時也會受影響。""")


if __name__ == "__main__":
    main()
