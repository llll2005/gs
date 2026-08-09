"""Is the residual a kernel-resolution limit, or a misallocation? Decide it spectrally.

The count-scaling argument (+0.229 dB per doubling, so 26 dB needs 100x primitives) is extrapolated
from a law measured on an algorithm that HAS redundancy -- pruning 50% of primitives cost only
0.67 dB of building PSNR and left texture ratio flat. So the law may describe our allocator rather
than the representation.

The discriminating question: where does the error live in frequency?

  error concentrated in the top band      -> the kernels cannot resolve it. Count (or a finer
                                             kernel) is the only fix, and the extrapolation holds.
  error spread into low and mid bands     -> primitives are in the wrong place or wrongly valued.
                                             A better allocator fixes it at the SAME count, and
                                             the extrapolation is measuring our algorithm, not the
                                             representation's limit.

Natural images are sparse: if edges occupy ~10% of pixels, an average density of 0.72 primitives
per pixel is 7.2 per edge pixel, which is ample. A perfect allocator would not need 1/px.

Bands come from a Gaussian pyramid: band k = level k minus the upsampled level k+1, so band 0 is
the finest. Reported as error energy relative to GT energy IN THE SAME BAND, which is the only
comparison that is not dominated by the fact that natural images have far more energy at low
frequency.
"""
import argparse, glob
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["blurbudget_b12", "cap4m_b12"])
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--levels", type=int, default=5)
a = ap.parse_args()

def pyr(x, n):
    """Laplacian-style band decomposition of [C,H,W]."""
    cur, bands = x, []
    for _ in range(n):
        lo = F.avg_pool2d(cur.unsqueeze(0), 2).squeeze(0)
        up = F.interpolate(lo.unsqueeze(0), size=cur.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        bands.append(cur - up)
        cur = lo
    bands.append(cur)
    return bands

GRAD = lambda t: float((t[:, 1:, :] - t[:, :-1, :]).abs().mean() + (t[:, :, 1:] - t[:, :, :-1]).abs().mean())

for run in a.runs:
    d = sorted(glob.glob(f"outputs/{run}/blocks/block_12/test/*/"))
    if not d:
        print(f"{run}: 無 test 圖"); continue
    files = sorted(glob.glob(d[-1] + "*.png"))
    pairs = []
    for f in files:
        im = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.).permute(2, 0, 1)
        w = im.shape[-1] // 2
        pairs.append((GRAD(im[..., :w]), im[..., :w], im[..., w:]))
    pairs.sort(key=lambda x: -x[0])
    pairs = pairs[: a.views]

    eg = np.zeros(a.levels + 1); sg = np.zeros(a.levels + 1)
    for _, gt, rd in pairs:
        bg, be = pyr(gt, a.levels), pyr(gt - rd, a.levels)
        for k in range(a.levels + 1):
            sg[k] += float((bg[k] ** 2).mean()); eg[k] += float((be[k] ** 2).mean())
    print(f"\n=== {run}（建築組 {len(pairs)} 張）===")
    print(f"{'頻帶':>6}{'尺度(px)':>10}{'GT 能量':>12}{'誤差能量':>12}{'相對誤差':>10}{'解釋掉':>9}")
    for k in range(a.levels + 1):
        rel = eg[k] / max(sg[k], 1e-12)
        lab = f"1/{2**k}" if k < a.levels else f"<=1/{2**a.levels}"
        print(f"{k:>6}{lab:>10}{sg[k]/len(pairs):>12.2e}{eg[k]/len(pairs):>12.2e}"
              f"{rel:>10.3f}{100*(1-rel):>8.1f}%")
print("\n  相對誤差只在頻帶 0 高 ⇒ 核解析度極限，顆數是唯一解")
print("  中低頻帶（2~4）相對誤差也高 ⇒ 粒子放錯地方，同顆數下有改進空間")
