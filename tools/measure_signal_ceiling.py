"""How much could a perfect densification sampler win, per candidate signal?

Densification is an allocation problem: given a budget of new primitives, put them where they buy
the most quality. The sampler can only help if the deficit it chases is CONCENTRATED. If the thing
you are trying to fix is spread evenly over the frame, then any sampler -- perfect or random --
reaches the same average, and no amount of machinery on top changes that.

  ceiling = mean(deficit over the worst 5% of pixels) / mean(deficit over all pixels)

ceiling ~ 1.0  ->  deficit is uniform; the signal carries nothing a sampler can exploit
ceiling >> 1   ->  a sampler that finds those pixels has that much headroom

This re-opens a question that was closed on 2026-07-29, when the same statistic came out at
1.18-1.27 and error-guided densification was rejected on it. Two reasons that measurement does not
settle the matter:

  1. It measured PHOTOMETRIC error. DGD (CityGaussianV2 Eq.2) uses the SSIM gradient, and the
     paper's whole argument is that L1 is insensitive to blur -- flattening detail barely moves the
     mean-brightness error while destroying structure. A uniform photometric deficit is perfectly
     compatible with a concentrated structural one.
  2. It was measured on b12, which is roughly half flat water. Water has low, uniform error and
     drags any concentration statistic toward 1.0 -- the same dilution that made b12's val PSNR
     meaningless (water views score 32-40 dB with a near-uniform render).

So: three deficits, measured separately on water-heavy and building-heavy views.

  photometric   |I - I_hat|                     what the 07-29 run measured
  structural    1 - SSIM_map(I, I_hat)          what DGD actually chases
  texture       relu(|grad I| - |grad I_hat|)   missing high frequency, the thing we can SEE

Views come from the block's PARTITION LIST -- an AABB over its cameras is a superset that includes
views the block never trained on, which makes any model look broken.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader


def ssim_map(a, b, win=11, sigma=1.5):
    """Per-pixel SSIM, same Gaussian window as internal/utils/ssim.py, kept as a map."""
    g = torch.arange(win, dtype=torch.float32, device=a.device) - win // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2)); g = (g / g.sum())
    k = (g[:, None] @ g[None, :]).expand(3, 1, win, win)
    f = lambda x: F.conv2d(x.unsqueeze(0), k, padding=win // 2, groups=3).squeeze(0)
    mu_a, mu_b = f(a), f(b)
    saa, sbb, sab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + C1) * (2 * sab + C2))
            / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))).mean(0)


def grad_mag(x):
    dy = F.pad((x[:, 1:, :] - x[:, :-1, :]).abs(), (0, 0, 0, 1))
    dx = F.pad((x[:, :, 1:] - x[:, :, :-1]).abs(), (0, 1, 0, 0))
    return (dy + dx).mean(0)


def ceiling(m, frac=0.05):
    v = m.flatten()
    k = max(1, int(v.numel() * frac))
    return float(v.topk(k).values.mean() / v.mean().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--views", type=int, default=6, help="每組取幾張")
    ap.add_argument("--frac", type=float, default=0.05)
    a = ap.parse_args()

    dev = "cuda"
    names, cams = load_test_cameras(a.data, 1.2)
    files = sorted(f for f in os.listdir(f"{a.data}/images_1.2") if f.lower().endswith(".png"))
    W, H = int(cams.width[0]), int(cams.height[0])
    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        a.data, "partition",
        f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    idx = [i for i, n in enumerate(names) if n in want and 0 <= int(n[:-4]) - 1 < len(files)]

    def gt(i):
        p = Image.open(f"{a.data}/images_1.2/{files[int(names[i][:-4]) - 1]}").convert("RGB")
        return torch.from_numpy(np.array(p.resize((W, H), Image.LANCZOS), np.uint8)) \
                    .float().permute(2, 0, 1).to(dev) / 255.

    probe = idx[::max(1, len(idx) // 40)]
    tex = sorted((float(grad_mag(gt(i)).mean()), i) for i in probe)
    groups = {"水面": [i for _, i in tex[:a.views]], "建築": [i for _, i in tex[-a.views:]]}
    print(f"[視角] 分區 {len(want)} 張；水面組 GT 梯度 {tex[0][0]:.4f}~{tex[a.views-1][0]:.4f}"
          f"，建築組 {tex[-a.views][0]:.4f}~{tex[-1][0]:.4f}")

    model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, dev, eval_mode=True)
    bg = torch.zeros(3, device=dev)
    print(f"[模型] N={model.get_xyz.shape[0]:,}\n")
    print(f"{'組別':<8}{'訊號':<12}{'天花板(top5%/mean)':>20}{'平均缺口':>12}")
    out = {}
    for g, ii in groups.items():
        acc = {"光度": [], "結構": [], "紋理": []}
        lvl = {"光度": [], "結構": [], "紋理": []}
        with torch.no_grad():
            for i in ii:
                G = gt(i)
                R = rend(cams[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
                d = {"光度": (R - G).abs().mean(0),
                     "結構": (1 - ssim_map(R, G)).clamp_min(0),
                     "紋理": (grad_mag(G) - grad_mag(R)).clamp_min(0)}
                for k, m in d.items():
                    acc[k].append(ceiling(m, a.frac)); lvl[k].append(float(m.mean()))
        for k in acc:
            c, l = np.mean(acc[k]), np.mean(lvl[k])
            out[(g, k)] = c
            print(f"{g:<8}{k:<12}{c:>20.2f}{l:>12.4f}")
        print()

    print("判讀：天花板 ~1.0 = 缺口均勻，換取樣訊號拿不到東西；越大代表完美取樣器的空間越大")
    print(f"  2026-07-29 的舊量測（光度、未分組）＝ 1.18~1.27，據此否決了誤差導向 densify")
    b = out.get(("建築", "結構")); p = out.get(("建築", "光度"))
    if b and p:
        print(f"  建築區：結構 {b:.2f} vs 光度 {p:.2f}  → 結構訊號帶有光度看不到的資訊："
              f"{'是，值得換' if b > p * 1.3 else '否，兩者差不多'}")


if __name__ == "__main__":
    main()
