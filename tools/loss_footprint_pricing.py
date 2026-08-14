"""Does SSIM price errors by screen footprint while L1 does not? CPU, no training.

dssim05_b12 (lambda_dssim 0.2 -> 0.5) cut the floater fraction 1.193% -> 0.958% while containing
nothing that targets floaters. The proposed mechanism: SSIM compares WINDOWS, so one primitive with
a large screen footprint destroys the structure of everything it covers, whereas L1 sees only a
faint tint spread thin. If true, SSIM is an implicit penalty proportional to c_i -- the screen
footprint this project tried and failed to price explicitly inside the density controller
(the degeneracy theorem: v_i and c_i both scale with footprint, so their ratio carries no signal).

The test needs no model. Take a real render, inject a FIXED TOTAL amount of error, and vary only
how it is spread: one big blob versus many small ones of the same combined area and amplitude.
    L1 is linear in the error, so it must be flat across spreads. Any slope belongs to SSIM.
    penalty ratio rising with blob radius  ->  SSIM prices by footprint; the implicit c_i is real
    flat                                   ->  the floater drop in dssim05 was a coincidence

⚠ This measures the LOSS's response to a fixed perturbation, not what the optimiser does with it.
  A confirmed slope makes the mechanism plausible, not proven; dssim08's floater fraction is still
  the empirical test.
"""
import argparse, glob
import numpy as np
import torch

from internal.utils.ssim import ssim as ssim_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dssim05_b12")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--radii", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--amp", type=float, default=0.25, help="每個斑塊的顏色偏移")
    ap.add_argument("--area_frac", type=float, default=0.01, help="被擾動的畫面比例，各半徑相同")
    a = ap.parse_args()

    from PIL import Image
    d = sorted(glob.glob(f"outputs/{a.run}/blocks/block_12/test/*/"))[-1]
    files = sorted(glob.glob(d + "*.png"))[: a.views]
    rng = np.random.default_rng(0)

    print(f"[{a.run}] {len(files)} 張   每個半徑擾動 {100*a.area_frac:.1f}% 的畫面，振幅 {a.amp}")
    print(f"\n{'半徑(px)':>9}{'斑塊數':>8}{'ΔL1':>12}{'Δ(1-SSIM)':>13}{'SSIM/L1':>10}")
    base = None
    for r in a.radii:
        dl1, dss = [], []
        for f in files:
            im = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.).permute(2, 0, 1)
            w = im.shape[-1] // 2
            gt, rd = im[:, :, :w], im[:, :, w:]
            H, W = rd.shape[-2], rd.shape[-1]
            n = max(1, int(a.area_frac * H * W / (np.pi * r * r)))
            pert = rd.clone()
            yy, xx = np.ogrid[:2 * r + 1, :2 * r + 1]
            disk = ((yy - r) ** 2 + (xx - r) ** 2) <= r * r
            m = torch.from_numpy(disk.astype(np.float32))
            for _ in range(n):
                cy = int(rng.integers(r, H - r)); cx = int(rng.integers(r, W - r))
                pert[:, cy - r:cy + r + 1, cx - r:cx + r + 1] += a.amp * m
            pert = pert.clamp(0, 1)
            dl1.append(float((pert - gt).abs().mean() - (rd - gt).abs().mean()))
            s0 = float(ssim_fn(rd.unsqueeze(0), gt.unsqueeze(0)))
            s1 = float(ssim_fn(pert.unsqueeze(0), gt.unsqueeze(0)))
            dss.append(s0 - s1)
        L, S = np.mean(dl1), np.mean(dss)
        ratio = S / max(L, 1e-9)
        base = ratio if base is None else base
        print(f"{r:>9}{n:>8}{L:>12.5f}{S:>13.5f}{ratio:>10.2f}")
    print("\n  ΔL1 幾乎不隨半徑變（誤差總量固定）；比值上升 = SSIM 按足跡定價 ⇒ 隱式 c_i 成立")
    print("  比值持平 = dssim05 的 floater 下降與此機制無關")


if __name__ == "__main__":
    main()
