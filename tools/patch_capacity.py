#!/usr/bin/env python
"""表徵容量的下界測試：**幾個 2D 高斯基底能不能擬合一個失敗 tile 的 GT？**

外部諮詢第二輪（Gemini 量測一）提的最便宜檢驗。目的是**把「表徵表達不了」這條路一次砍掉**：

  若少量 2D 高斯就能把失敗 tile 的 GT 擬合到 corr > 0.9
  ⇒ **高斯基底的表達力絕對足夠**，問題 100% 在多視角優化/梯度傳播，不在表徵。

⚠ **這個測試的邊界（必須誠實標註）**：
  它擬合的是**影像平面上的 2D 高斯**，不是「從 3D surfel 投影下來的 2DGS」。
  ⇒ 它給的是**表徵力的上界**（3D 投影還要滿足多視角一致性，只會更難）。
  ⇒ 因此「能擬合」**不能**證明 3D 情形也能；但「**連這都擬合不了**」就能一票否決表徵路線。
  ⇒ 定位：**單向否證工具**。

⚠ 同時做**對照組**（成功 tile 走完全相同流程）。若成功/失敗兩組都輕鬆擬合，
   差異就完全不在「這個 patch 難不難用高斯表示」。

用法: python tools/patch_capacity.py agd2_b12 sched30_b12 --blk 12 --n-gauss 20
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def fit_patch(gt, n_gauss, steps, dev, seed=0):
    """用 n_gauss 個各向異性 2D 高斯（位置/尺度/旋轉/顏色/不透明度全可學）擬合 gt。
    前向是簡單的加權和（不做深度排序 alpha 合成）—— 這正是「表徵力上界」的意思。"""
    H, W = gt.shape
    g = torch.tensor(gt, dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    mu = torch.rand(n_gauss, 2, device=dev) * torch.tensor([W, H], device=dev, dtype=torch.float32)
    logs = torch.log(torch.full((n_gauss, 2), W / (2 * n_gauss ** 0.5), device=dev))
    th = torch.rand(n_gauss, device=dev) * 3.1416
    col = torch.full((n_gauss,), float(g.mean()), device=dev)
    amp = torch.full((n_gauss,), 0.5, device=dev)
    bg = torch.tensor(float(g.mean()), device=dev)
    ps = [p.requires_grad_(True) for p in (mu, logs, th, col, amp, bg)]
    opt = torch.optim.Adam([{"params": [mu], "lr": 0.5},
                            {"params": [logs, th], "lr": 0.05},
                            {"params": [col, amp, bg], "lr": 0.05}])
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    for _ in range(steps):
        s = torch.exp(logs).clamp(0.3, W)
        c, si = torch.cos(th), torch.sin(th)
        dx = xx[None] - mu[:, 0, None, None]
        dy = yy[None] - mu[:, 1, None, None]
        a = (c[:, None, None] * dx + si[:, None, None] * dy) / s[:, 0, None, None]
        b = (-si[:, None, None] * dx + c[:, None, None] * dy) / s[:, 1, None, None]
        w = amp[:, None, None] * torch.exp(-0.5 * (a * a + b * b))
        img = bg + (w * col[:, None, None]).sum(0)
        loss = ((img - g) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        r = img.detach().cpu().numpy().ravel()
        t = gt.ravel()
        d = r.std() * t.std()
        corr = float(((r - r.mean()) * (t - t.mean())).mean() / d) if d > 1e-9 else 0.0
        return corr, float(img.std()), float(g.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--n-gauss", type=int, default=20)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-patch", type=int, default=12, help="每組取幾個 patch")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    fs = sorted(f for f in os.listdir(dirs[args.runs[0]]) if f.endswith(".png"))
    T = args.tile
    pats = {"失敗": [], "成功": []}
    for f in fs:
        if all(len(v) >= args.n_patch for v in pats.values()):
            break
        outs = {r: per_image(os.path.join(dirs[r], f), T, args.contrast_q) for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, *_x, nx, ny = outs[args.runs[0]]
        ms = np.stack([outs[r][1] <= args.r_min for r in args.runs])
        allb, neverb = ms.all(0), ~ms.any(0)
        im = Image.open(os.path.join(dirs[args.runs[0]], f))
        w = im.width // 2
        G = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
        for m, lab in ((allb, "失敗"), (neverb, "成功")):
            for t in keep[m][:3]:
                if len(pats[lab]) >= args.n_patch:
                    break
                ty, tx = int(t // nx), int(t % nx)
                pats[lab].append(G[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T].copy())

    print(f"每組 {args.n_patch} 個 48x48 patch；用 **{args.n_gauss} 個 2D 高斯**擬合 {args.steps} 步"
          f"（裝置 {dev}）\n")
    print(f"{'':>6} {'擬合後 corr 中位':>16} {'最小':>8} {'最大':>8} {'GT 紋理中位':>12}")
    for lab in ("失敗", "成功"):
        rs = [fit_patch(p, args.n_gauss, args.steps, dev) for p in pats[lab]]
        c = np.array([x[0] for x in rs]); gstd = np.array([x[2] for x in rs])
        print(f"{lab:>6} {np.median(c):>16.4f} {c.min():>8.4f} {c.max():>8.4f} {np.median(gstd):>12.4f}")
    print(f"""
判讀：
  失敗組 corr > 0.9  => **高斯基底的表達力足夠** ⇒ 表徵路線一票否決，
                       問題在多視角優化/梯度傳播（Gemini 的「空間梯度積分抵消」等）
  失敗組 corr 也低   => patch 本身就難用 {args.n_gauss} 個高斯表示 ⇒ 表徵/容量路線仍活
⚠ 本測試擬合的是**影像平面的 2D 高斯**（無多視角一致性約束、無深度排序 alpha 合成）
  ⇒ 給的是**上界**；「能擬合」不證明 3D 也能，但「連這都不行」可一票否決。""")


if __name__ == "__main__":
    main()
