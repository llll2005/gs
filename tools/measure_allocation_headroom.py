"""Is there headroom in WHERE the primitives are, or is the budget already in the right places?

The proposal on the table is to steer the marginal budget towards texture-complex regions. Before
building that, the premise has to hold: error must be high where primitive density is LOW. If
density already tracks texture, MCMC is allocating correctly and the remaining error is
representational, not allocational -- and the four "reallocate a fixed budget" lines that already
came back net-zero are the precedent.

Note the prior: an earlier error-guided densification attempt measured a ceiling of only 1.2x.

Method, per view, on a coarse grid of cells:
    density_c   primitives whose projected centre falls in cell c, per pixel
    texture_c   GT gradient energy in cell c          (the training image, not held-out GT --
                                                       these are our inputs, the photometric loss
                                                       already uses them)
    error_c     mean squared render error in cell c

Then three correlations, all across cells pooled over views:
    corr(texture, density)  does the allocator already follow content?
    corr(texture, error)    is complex content where the error lives?
    corr(density, error)    THE ONE THAT DECIDES: strongly negative => low density causes error
                            => steering helps. Near zero or positive => the primitives are already
                            there and more of them will not fix it.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader


def cell_stats(t, cell):
    """Mean of [C,H,W] or [H,W] over cell x cell blocks -> [h,w]."""
    if t.dim() == 3:
        t = t.mean(0)
    h, w = t.shape
    t = t[: h // cell * cell, : w // cell * cell]
    return t.reshape(h // cell, cell, -1, cell).mean(dim=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="oreg_0p002_b12")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cell", type=int, default=32, help="grid cell size in pixels")
    a = ap.parse_args()

    dev = "cuda"
    ck = glob.glob(f"outputs/{a.run}/blocks/block_{a.block}/checkpoints/*.ckpt")
    ck.sort(key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
    model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck[-1], dev, eval_mode=True)
    xyz = model.get_xyz.detach()
    print(f"[模型] {a.run}  {xyz.shape[0]:,} 顆")

    names, cams = load_test_cameras(a.data, 1.2)
    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        a.data, "partition", f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    idx = [i for i, n in enumerate(names) if n in want]

    # rank by GT gradient, keep the building half -- flat water has no texture to allocate towards
    # Same loader as tools/prune_curve.py: COLMAP names are 4-digit and 1-based, the files on disk
    # are 6-digit, so index by position with the -1 offset rather than by name.
    from PIL import Image
    files = sorted(f for f in os.listdir(f"{a.data}/images_1.2") if f.lower().endswith(".png"))
    W0, H0 = int(cams.width[0]), int(cams.height[0])
    idx = [i for i in idx if 0 <= int(names[i][:-4]) - 1 < len(files)]

    def gt_of(i):
        p = Image.open(f"{a.data}/images_1.2/{files[int(names[i][:-4]) - 1]}").convert("RGB")
        return torch.from_numpy(np.array(p.resize((W0, H0), Image.LANCZOS), np.uint8)) \
                    .float().permute(2, 0, 1).to(dev) / 255.

    grads = []
    for i in idx[:: max(1, len(idx) // 40)]:
        g = gt_of(i)
        grads.append((float((g[:, 1:] - g[:, :-1]).abs().mean() +
                            (g[:, :, 1:] - g[:, :, :-1]).abs().mean()), i))
    grads.sort(reverse=True)
    picked = [i for _, i in grads[: a.views]]
    print(f"[視角] 建築組 {len(picked)} 張（GT 梯度 {grads[a.views-1][0]:.4f}~{grads[0][0]:.4f}）")

    D, T, E = [], [], []
    with torch.no_grad():
        for i in picked:
            cam = cams[i].to_device(dev)
            gt = gt_of(i)
            out = rend(cam, model, bg_color=torch.zeros(3, device=dev))
            rd = out["render"].clamp(0, 1)
            H, W = rd.shape[-2], rd.shape[-1]

            # project primitive centres; count per pixel via a scatter into a [H,W] map
            p = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ cam.full_projection
            ok = p[:, 3] > 1e-6
            uv = p[ok, :2] / p[ok, 3:4]
            u = ((uv[:, 0] * 0.5 + 0.5) * W).long()
            v = ((uv[:, 1] * 0.5 + 0.5) * H).long()
            m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            dens = torch.zeros(H * W, device=dev)
            dens.scatter_add_(0, (v[m] * W + u[m]), torch.ones(int(m.sum()), device=dev))

            err = ((rd - gt) ** 2)
            tex = (gt[:, 1:] - gt[:, :-1]).abs().mean(0)
            tex = torch.nn.functional.pad(tex[None, None], (0, 0, 0, 1))[0, 0]

            D.append(cell_stats(dens.view(H, W), a.cell).flatten())
            T.append(cell_stats(tex, a.cell).flatten())
            E.append(cell_stats(err, a.cell).flatten())
            del out; torch.cuda.empty_cache()

    d = torch.cat(D).cpu().numpy(); t = torch.cat(T).cpu().numpy(); e = torch.cat(E).cpu().numpy()
    c = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    print(f"\n[格子] {len(d):,} 個 {a.cell}x{a.cell} 儲存格")
    print(f"  corr(紋理, 密度) = {c(t, d):+.3f}   配置有沒有跟著內容走")
    print(f"  corr(紋理, 誤差) = {c(t, e):+.3f}   誤差是不是集中在複雜內容")
    print(f"  corr(密度, 誤差) = {c(d, e):+.3f}   ★ 決定性：強負 ⇒ 密度不足導致誤差，定向投放有空間")
    print(f"                                        接近 0 或正 ⇒ 粒子已經在那裡，加更多沒用")

    # split cells by texture tercile and report density/error, so the correlation is not the only view
    q = np.quantile(t, [1 / 3, 2 / 3])
    for lab, sel in [("低紋理", t <= q[0]), ("中", (t > q[0]) & (t <= q[1])), ("高紋理", t > q[1])]:
        print(f"  {lab:<6} 密度 {d[sel].mean():7.3f} 顆/px   誤差 {e[sel].mean():.5f}"
              f"   每顆分到的誤差 {e[sel].mean()/max(d[sel].mean(),1e-9):.5f}")


if __name__ == "__main__":
    main()
