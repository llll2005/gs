"""Would a gradient-matching loss term help, or would it just add noise? CPU-only upper bound.

Texture ratio (rendered gradient energy / GT gradient energy) sits at 0.455 on buildings: the model
is systematically too smooth, which is what L1+SSIM asks for -- blur is the MSE-optimal answer under
uncertainty, so the optimiser has no reason to grow detail. The proposed fix is an explicit
    L_tex = | GRAD(render) - GRAD(gt) |
which costs no VRAM and targets the bias rather than the capacity.

Before implementing it, bound what it can buy. Sharpening the EXISTING render is the most
favourable thing such a loss could do to a finished model: it raises gradient energy without
moving any primitive. So sweep unsharp-mask strength and watch texture ratio against LPIPS.

    LPIPS improves as texratio -> 1   the missing high frequency is recoverable by amplification,
                                      so a texture loss has real headroom
    LPIPS worsens immediately         the high frequency is absent or misplaced, not merely weak;
                                      the loss would manufacture noise. Do not implement.

This is an UPPER bound in one direction only: a real loss also moves primitives during training,
which sharpening cannot do. A negative result here is therefore strong evidence against, while a
positive result only says "worth trying".
"""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

GRAD = lambda t: float((t[:, 1:, :] - t[:, :-1, :]).abs().mean() + (t[:, :, 1:] - t[:, :, :-1]).abs().mean())

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="blurbudget_b12")
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--amounts", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.5, 4.0])
a = ap.parse_args()

d = sorted(glob.glob(f"outputs/{a.run}/blocks/block_12/test/*/"))[-1]
files = sorted(glob.glob(d + "*.png"))
lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex")

pairs = []
for f in files:
    im = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.).permute(2, 0, 1)
    w = im.shape[-1] // 2
    gt, rd = im[..., :w], im[..., w:]
    pairs.append((GRAD(gt), gt, rd))
pairs.sort(key=lambda x: -x[0])                 # buildings first: most gradient energy
pairs = pairs[: a.views]
print(f"[{a.run}] 建築組 {len(pairs)} 張（GT 梯度 {pairs[-1][0]:.4f}~{pairs[0][0]:.4f}）\n")

def unsharp(x, amount, r=1):
    k = 2 * r + 1
    blur = torch.nn.functional.avg_pool2d(x.unsqueeze(0), k, 1, r).squeeze(0)
    return (x + amount * (x - blur)).clamp(0, 1)

print(f"{'銳化量':>8}{'紋理比':>9}{'LPIPS':>9}{'ΔLPIPS':>9}{'PSNR':>8}")
base = None
for amt in a.amounts:
    R, L, P = [], [], []
    for g_gt, gt, rd in pairs:
        s = unsharp(rd, amt)
        R.append(GRAD(s) / max(g_gt, 1e-9))
        L.append(float(lp(s.unsqueeze(0), gt.unsqueeze(0))))
        P.append(float(-10 * torch.log10(((s - gt) ** 2).mean().clamp_min(1e-12))))
    l = np.mean(L)
    base = l if base is None else base
    print(f"{amt:>8.1f}{np.mean(R):>9.3f}{l:>9.4f}{l - base:>+9.4f}{np.mean(P):>8.2f}")
print("\n  ΔLPIPS 隨銳化下降 ⇒ 高頻可用放大救回，紋理損失有空間")
print("  ΔLPIPS 一開始就上升 ⇒ 缺的高頻不在那裡（或位置錯），損失只會製造雜訊，別實作")
