"""What IS the low-frequency error? It carries 45% of the residual energy and has never been looked at.

error_spectrum.py found relative error of only 0.05 in the >=32px band, but that band holds so much
signal energy that it contributes ~45% of total MSE -- more than the 1-2px detail bands (21%) that
every count-scaling argument is about. Halving it would be worth ~1.1 dB.

Note the spectrum was computed on the BUILDING group (views sorted by GT gradient), so this is
large-scale error in city views, not the water views.

Two questions decide what kind of problem it is:
  signed error systematically non-zero   -> a photometric/appearance offset (exposure, ambient,
                                            missing view-dependent term). Cheap to fix.
  zero-mean but large                    -> structural: large regions rendered as the wrong thing.
                                            That is geometry, and expensive.
and, cutting the other way, whether the error concentrates on dark content (shadowed courtyards,
which a diffuse-only model with SB lobes may not reach) or bright content (roofs, sky-facing).
"""
import argparse, glob
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["blurbudget_b12"])
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--levels", type=int, default=5)
a = ap.parse_args()

GRAD = lambda t: float((t[:, 1:, :] - t[:, :-1, :]).abs().mean() + (t[:, :, 1:] - t[:, :, :-1]).abs().mean())

def lowpass(x, n):
    cur = x
    for _ in range(n):
        cur = F.avg_pool2d(cur.unsqueeze(0), 2).squeeze(0)
    return cur

for run in a.runs:
    d = sorted(glob.glob(f"outputs/{run}/blocks/block_12/test/*/"))
    files = sorted(glob.glob(d[-1] + "*.png"))
    pairs = []
    for f in files:
        im = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.).permute(2, 0, 1)
        w = im.shape[-1] // 2
        pairs.append((GRAD(im[..., :w]), im[..., :w], im[..., w:], f.split("/")[-1]))
    pairs.sort(key=lambda x: -x[0])
    pairs = pairs[: a.views]

    G, E, T = [], [], []
    for _, gt, rd, _ in pairs:
        g = lowpass(gt, a.levels); r = lowpass(rd, a.levels)
        tx = lowpass((gt[:, 1:] - gt[:, :-1]).abs().mean(0, keepdim=True).clamp(0, 1), a.levels)
        G.append(g.reshape(3, -1)); E.append((r - g).reshape(3, -1)); T.append(tx.reshape(-1))
    g = torch.cat(G, 1); e = torch.cat(E, 1); t = torch.cat(T)
    bright = g.mean(0)

    print(f"\n=== {run}  低頻(>= {2**a.levels}px)誤差歸因，{len(pairs)} 張建築視角 ===")
    print(f"  有號誤差均值 {float(e.mean()):+.5f}   |誤差| 均值 {float(e.abs().mean()):.5f}"
          f"   ⇒ 系統性偏移佔 {100*abs(float(e.mean()))/float(e.abs().mean()):.1f}%")
    print(f"  逐通道有號偏移  R {float(e[0].mean()):+.5f}  G {float(e[1].mean()):+.5f}  B {float(e[2].mean()):+.5f}")
    q = torch.quantile(bright, torch.tensor([0.25, 0.5, 0.75]))
    print(f"\n  {'GT 亮度分組':<14}{'有號誤差':>11}{'|誤差|':>10}{'GT 亮度':>10}{'像素佔比':>10}")
    for lab, sel in [("最暗 25%", bright <= q[0]), ("次暗", (bright > q[0]) & (bright <= q[1])),
                     ("次亮", (bright > q[1]) & (bright <= q[2])), ("最亮 25%", bright > q[2])]:
        print(f"  {lab:<14}{float(e[:, sel].mean()):>+11.5f}{float(e[:, sel].abs().mean()):>10.5f}"
              f"{float(bright[sel].mean()):>10.4f}{100*float(sel.float().mean()):>9.1f}%")
    qt = torch.quantile(t, torch.tensor([0.33, 0.67]))
    print(f"\n  {'GT 紋理分組':<14}{'有號誤差':>11}{'|誤差|':>10}")
    for lab, sel in [("平坦 33%", t <= qt[0]), ("中", (t > qt[0]) & (t <= qt[1])), ("有紋理 33%", t > qt[1])]:
        print(f"  {lab:<14}{float(e[:, sel].mean()):>+11.5f}{float(e[:, sel].abs().mean()):>10.5f}")
print("\n  系統性偏移佔比高 ⇒ 光度/外觀問題，便宜可修")
print("  接近 0（零均值）⇒ 結構性：大片區域渲染成錯的東西，屬幾何問題")
