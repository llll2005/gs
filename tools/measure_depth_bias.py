"""Is the rendered surface at the right depth?

Our per-block models put the rendered surface at ~0.75x the true depth -- a blanket at roughly
rooftop height that never resolves the streets below. The bias is fully formed by step 499 and
barely moves over 60k steps, and it survives every other explanation we tested. This tool is how
that was measured, so that any candidate fix can be judged the same way.

Reference geometry is `sparse/0`: 3.6M points triangulated against the dataset's own poses. It is
trustworthy here -- projecting it into a test camera lands the points exactly on rooftops, facades
and kerbs -- and it is the only metric geometry available locally (MatrixCity's real depth was never
downloaded; `block_all/depth/` is an 8-bit visualisation with 256 levels, not usable).

Two mistakes this encodes so they are not repeated:

  1. **Use only the points COLMAP recorded as visible in that image** (`point3D_ids`). Projecting the
     whole cloud includes points occluded by buildings, whose true depth is legitimately larger than
     the visible surface, and that alone manufactures a one-sided error of tens of percent.
  2. **Filter the cloud to p1-p99 first.** Without it, stray points at z up to 34 in a 19-unit scene
     dominate the ratio and shifted a measured bias from -34% to -28%.

`surf_depth` is camera-space z, not ray distance: the rasterizer writes `depths[idx] = p_view.z`
(cuda_rasterizer/forward.cu:304) and the median-depth path carries that same quantity through to
MIDDEPTH_OFFSET. So the ratio below is directly interpretable, with no convention correction.

Read `slope` as the headline: the least-squares ratio of rendered to true depth over all sampled
points. 1.0 is correct; 0.75 means the surface floats a quarter of the way toward the camera.
`corr` guards against reading a slope off noise -- it should stay near 0.8+ if the render tracks
real depth at all.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.colmap import read_images_binary, read_points3D_binary
from internal.utils.gaussian_model_loader import GaussianModelLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--block", type=int, default=None, help="限定某塊的分區清單；省略=全場景取樣")
    ap.add_argument("--block_dim", type=int, nargs=2, default=[3, 3])
    ap.add_argument("--content_bounds", action="store_true",
                    help="同時把參考點限制在該塊的空間範圍內。逐塊微調的模型會保留整城的粗略粒子"
                         "（實測 blk5：塊內 4.0× 成長、塊外 0.83× 只被略剪），航拍視錐又很廣，"
                         "不加這道過濾等於拿未優化的區域去評分。跨模型比較時務必兩邊都加。")
    ap.add_argument("--min_alpha", type=float, default=0.5,
                    help="只採納累積 alpha 達此值的像素；稀疏模型上沒有這道過濾，讀數是噪音")
    a = ap.parse_args()

    P3 = read_points3D_binary(f"{a.data}/sparse/0/points3D.bin")
    xyz_all = np.array([p.xyz for p in P3.values()])
    lo, hi = np.percentile(xyz_all, 1, 0), np.percentile(xyz_all, 99, 0)   # mistake 2
    pid = {k: v.xyz for k, v in P3.items()}
    imgs = read_images_binary(f"{a.data}/sparse/0/images.bin")
    by_name = {im.name: im for im in imgs.values()}

    names, cams = load_test_cameras(a.data, 1.2)
    if a.block is None:
        pool = list(range(len(names)))
    else:
        by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
        want = {l.strip() for l in open(os.path.join(
            a.data, "partition",
            f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
            f"{bx:03d}_{by:03d}.txt")) if l.strip()}
        pool = [i for i, n in enumerate(names) if n in want]
        if a.content_bounds:
            from eval_official_test import block_bounds
            _lo, _hi, _ = block_bounds(a.data, a.block, a.block_dim)
            lo = np.maximum(lo, _lo - 0.5)
            hi = np.minimum(hi, _hi + 0.5)
            print(f"[範圍] 參考點限制在 blk{a.block} 的 x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}]")
    probe = [pool[j] for j in np.linspace(0, len(pool) - 1, min(a.views, len(pool))).astype(int)]

    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, "cuda", eval_mode=True)
    bg = torch.zeros(3, device="cuda")
    print(f"[模型] {a.ckpt}\n        N={model.get_xyz.shape[0]:,}")

    zr_all, zt_all = [], []
    covered = total = 0
    for i in probe:
        im = by_name.get(names[i])
        if im is None:
            continue
        ids = im.point3D_ids
        ids = ids[ids >= 0]
        X = np.array([pid[j] for j in ids if j in pid])           # mistake 1
        if len(X) == 0:
            continue
        X = X[np.all((X > lo) & (X < hi), 1)]
        if len(X) < 200:
            continue
        cam = cams[i].to_device("cuda")
        with torch.no_grad():
            out = renderer(cam, model, bg_color=bg)
        dep = out["surf_depth"].squeeze()
        # A pixel with no surface in front of it still carries a `surf_depth` value, and on a sparse
        # model most pixels are like that. Reading those as geometry is what made the 500k-primitive
        # official coarse report slope 0.356 / corr +0.585 on one sample and 0.240 / corr -0.193 on
        # another -- a negative correlation means the numbers were noise, not depth. Require real
        # coverage, i.e. the accumulated alpha at that pixel.
        alpha = out["rend_alpha"].squeeze()
        Xc = (cam.R @ torch.from_numpy(X).float().cuda().T).T + cam.T
        z = Xc[:, 2]
        m = z > 0.05
        u = cam.fx * Xc[m, 0] / z[m] + cam.cx
        v = cam.fy * Xc[m, 1] / z[m] + cam.cy
        H, W = dep.shape
        k = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        vv, uu = v[k].long(), u[k].long()
        zr = dep[vv, uu]
        zt = z[m][k]
        ok = (zr > 1e-4) & (alpha[vv, uu] >= a.min_alpha)
        covered += int(ok.sum())
        total += int(k.sum())
        zr_all.append(zr[ok].cpu().numpy())
        zt_all.append(zt[ok].cpu().numpy())

    zr = np.concatenate(zr_all)
    zt = np.concatenate(zt_all)
    rel = (zr - zt) / zt
    slope = float((zr * zt).sum() / (zt * zt).sum())     # least squares through the origin
    corr = float(np.corrcoef(zr, zt)[0, 1])
    print(f"[樣本] {len(probe)} 視角 / {len(zr):,} 個有效點"
          f"（覆蓋率 {100 * covered / max(total, 1):.1f}%，alpha>={a.min_alpha}）")
    if covered < 0.15 * total:
        print("  ⚠ 覆蓋率過低：模型在多數可見點上根本沒有表面，下列數字不可信")
    print(f"  signed median = {np.median(rel) * 100:+7.2f}%      |err| median = "
          f"{np.median(np.abs(rel)) * 100:6.2f}%")
    print(f"  slope (渲染/真實) = {slope:.3f}      corr = {corr:.3f}")
    print(f"  |err|<10% 佔 {100 * (np.abs(rel) < 0.10).mean():.1f}%   "
          f"渲染偏近佔 {100 * (rel < 0).mean():.1f}%")
    verdict = ("✅ 幾何位置正確（毯子不存在）" if abs(slope - 1) < 0.08 else
               "⚠ 介於兩者之間，需更多視角確認" if abs(slope - 1) < 0.15 else
               "❌ 毯子存在（表面系統性偏離真實深度）")
    print(f"\n  判定：{verdict}   ——對照：我方 3×3 合併模型 slope≈0.75")


if __name__ == "__main__":
    main()
