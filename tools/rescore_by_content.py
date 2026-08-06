"""Re-score existing checkpoints with a metric that is checked for resolution first.

The 2026-07-19..28 b12 runs all landed within LPIPS 0.022 of each other, which looked like "none of
these mechanisms did anything". It was not. Splitting the same runs by whether they finished:

    OOM'd at 30-34k (n=4):  PSNR 19.39   LPIPS 0.729
    completed 60k   (n=9):  PSNR 21.97   LPIPS 0.737

A crippled half-trained model scored BETTER than a healthy one. A metric that cannot separate
"died at 30k" from "trained to 60k" has no resolution in that regime, so nothing measured there --
in either direction -- means anything. Those experiments are UNTESTED, not disproven.

So this tool does two things at once:

1. Splits views by CONTENT. b12 is roughly half flat water, where the render is a near-uniform
   colour blob that still scores 32-40 dB; averaging that with the building views is what hides
   everything. GT gradient energy separates the two cleanly and needs no labels.

2. Carries the known-bad runs (the 30k OOM checkpoints) through as a CONTROL, and reports the
   separation each metric achieves between known-good and known-bad. A metric that fails to
   separate them is reported as unusable rather than quietly believed.

Texture ratio (render gradient / GT gradient) is included because it is the quantity that
corresponds to what "blurry" looks like: on 2026-08-01 it gave 0.63-0.72 for b7's building views
against 0.23-0.64 for b12's, matching visual inspection, while PSNR put them within a few dB.

Views come from the block's PARTITION LIST, never from an AABB over its cameras -- the partition is
assigned by visibility and the AABB is a superset, so AABB selection feeds a block model cameras it
never trained on and makes any model look shattered.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.ssim import ssim as ssim_fn

GRAD = lambda t: float((t[:, 1:, :] - t[:, :-1, :]).abs().mean()
                       + (t[:, :, 1:] - t[:, :, :-1]).abs().mean())
PSNR = lambda a, b: float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="outputs/<name> 目錄名")
    ap.add_argument("--known_bad", nargs="*", default=[], help="已知壞掉的 run（尺的對照組）")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--views", type=int, default=8, help="每組（水面/建築）取幾張")
    ap.add_argument("--at_step", type=int, default=None,
                    help="compare at the checkpoint nearest this step instead of the last one; "
                         "use it to separate primitive count from schedule length")
    ap.add_argument("--out", default="紀錄/rescore_by_content.csv")
    a = ap.parse_args()

    dev = "cuda"
    names, cams = load_test_cameras(a.data, 1.2)
    files = sorted(f for f in os.listdir(f"{a.data}/images_1.2") if f.lower().endswith(".png"))
    W, H = int(cams.width[0]), int(cams.height[0])

    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    plist = os.path.join(a.data, "partition",
                         f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
                         f"{bx:03d}_{by:03d}.txt")
    want = {l.strip() for l in open(plist) if l.strip()}
    idx = [i for i, n in enumerate(names) if n in want and 0 <= int(n[:-4]) - 1 < len(files)]
    print(f"[視角] block {a.block} 分區清單 {len(want)} 張，可用 {len(idx)}")

    def gt(i):
        p = Image.open(f"{a.data}/images_1.2/{files[int(names[i][:-4]) - 1]}").convert("RGB")
        return torch.from_numpy(np.array(p.resize((W, H), Image.LANCZOS), np.uint8)) \
                    .float().permute(2, 0, 1).to(dev) / 255.

    probe = idx[::max(1, len(idx) // 40)]
    tex = sorted((GRAD(gt(i)), i) for i in probe)
    groups = {"水面": [i for _, i in tex[:a.views]], "建築": [i for _, i in tex[-a.views:]]}
    print(f"[分組] 水面組 GT 梯度 {tex[0][0]:.4f}~{tex[a.views-1][0]:.4f}   "
          f"建築組 {tex[-a.views][0]:.4f}~{tex[-1][0]:.4f}")
    gts = {g: {i: gt(i) for i in ii} for g, ii in groups.items()}

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)

    rows = []
    for run in a.runs + a.known_bad:
        d = f"outputs/{run}"
        ck = sorted((int(f[f.index("step=") + 5:f.index(".ckpt")]), os.path.join(r, f))
                    for r, _, fs in os.walk(d) for f in fs if f.endswith(".ckpt") and "step=" in f)
        if not ck:
            print(f"  {run}: 無 ckpt，跳過")
            continue
        # --at_step lets a dense 60k run be compared against a sparse 24k one at MATCHED steps.
        # Without it the count difference is confounded with the schedule difference: v1 reaches
        # 77,616 points in 24k steps while oreg_0p002 reaches 900,000 in 60k, and the texture-ratio
        # gap (0.170 vs 0.392) could be either cause. Picks the nearest available checkpoint.
        if a.at_step is not None:
            step, path = min(ck, key=lambda sp: abs(sp[0] - a.at_step))
        else:
            step, path = ck[-1]
        model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            path, dev, eval_mode=True)
        bg = torch.zeros(3, device=dev)
        rec = {"run": run, "step": step, "n": model.get_xyz.shape[0],
               "bad": run in a.known_bad}
        for g, ii in groups.items():
            P, S, L, R = [], [], [], []
            with torch.no_grad():
                for i in ii:
                    o = rend(cams[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
                    G = gts[g][i]
                    P.append(PSNR(o, G)); S.append(float(ssim_fn(o, G)))
                    L.append(float(lp(o.unsqueeze(0), G.unsqueeze(0))))
                    R.append(GRAD(o) / max(GRAD(G), 1e-9))
            rec |= {f"{g}_psnr": np.mean(P), f"{g}_ssim": np.mean(S),
                    f"{g}_lpips": np.mean(L), f"{g}_texratio": np.mean(R)}
        rows.append(rec)
        print(f"  {run:<34} step={step:>6} N={rec['n']:>9,}  "
              f"建築 PSNR={rec['建築_psnr']:.2f} LPIPS={rec['建築_lpips']:.3f} "
              f"紋理比={rec['建築_texratio']:.3f}{'   ← 已知壞' if rec['bad'] else ''}")
        del model, rend
        torch.cuda.empty_cache()

    good = [r for r in rows if not r["bad"]]
    bad = [r for r in rows if r["bad"]]
    print("\n=== 尺的鑑別力（已知好 vs 已知壞的分離度）===")
    print(f"{'指標':<20}{'已知好':>10}{'已知壞':>10}{'差距':>10}{'組內標準差':>12}{'判定':>8}")
    if bad:
        for key in ("建築_psnr", "建築_ssim", "建築_lpips", "建築_texratio",
                    "水面_psnr", "水面_texratio"):
            g, b = np.mean([r[key] for r in good]), np.mean([r[key] for r in bad])
            sd = np.std([r[key] for r in good])
            # separation must exceed the spread among good runs, or it cannot rank them either
            ok = "可用" if abs(g - b) > max(sd, 1e-9) else "無鑑別力"
            print(f"{key:<20}{g:>10.4f}{b:>10.4f}{g-b:>+10.4f}{sd:>12.4f}{ok:>8}")
    else:
        print("  （未指定 --known_bad，無法校驗尺）")

    if rows:
        import csv
        keys = list(rows[0].keys())
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
        print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
