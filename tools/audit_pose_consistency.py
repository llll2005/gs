"""How much of the COLMAP reconstruction is locally inconsistent, and where does it concentrate?

RESULT ON THE CURRENT DATA (2026-08-01): residual is exactly 0.0 px. sparse/0 does not hold
COLMAP-ESTIMATED poses -- it holds the MatrixCity GT poses, with COLMAP only triangulating points
against them. Pose error is therefore ruled out as a cause of the blurry renders, permanently.

Keep this script anyway: it is the check to re-run whenever the dataset is regenerated, and it is
the only thing standing between us and re-opening a closed question.

Motivation: every render on the current data shows correct large-scale structure with no
high-frequency detail. The obvious suspect is the poses -- but "COLMAP disagrees with GT" is NOT
by itself a problem for novel-view synthesis. Splatting never sees GT; it only needs the poses to
agree with EACH OTHER. A smooth global warp renders a warped city, sharply. Only high-frequency,
per-camera error makes views contradict one another, and contradiction is what averages to fog.

The filename->frame correspondence is SEARCHED, not assumed. COLMAP writes image `N` for GT frame
`N+1` in this dataset, and pairing them naively produced a 47 px "drift" with a plausible-looking
18%-of-cameras-are-broken story attached -- entirely an artifact of the off-by-one. Any future
pairing bug now shows up as a non-zero best offset in the printed table instead of as a finding.

So the residual after an optimal Sim3 to GT is split into the two parts:

  global   ||res_i||                      -- drift; harmless, absorbed by the reconstruction
  local    median_j ||res_i - res_j||     -- disagreement with spatial neighbours; this is the
                                             part that cannot be absorbed and must blur

Both are reported in pixels at the training resolution, since that is the unit in which the
photometric loss actually sees the error.

The per-block breakdown is the point: if the bad cameras cluster, the problem is a fixable data
issue in specific blocks rather than a property of the whole dataset.
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.colmap import read_images_binary, read_points3D_binary


def sim3(X, Y):
    """Umeyama: scale/rot/trans mapping X onto Y."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    s = (S * np.array([1, 1, d])).sum() / ((Xc ** 2).sum() / len(X))
    return s, R, my - s * (R @ mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block_dim", default="5_5")
    ap.add_argument("--k", type=int, default=9, help="neighbours (incl. self) for the local test")
    ap.add_argument("--bad_px", type=float, default=10.0, help="local disagreement above this = bad")
    ap.add_argument("--down_sample", type=float, default=1.2)
    ap.add_argument("--out", default="紀錄/pose_consistency.csv")
    a = ap.parse_args()

    tj = json.load(open(f"{a.data}/transforms.json"))
    gt = {f["file_path"].split("/")[-1].split(".")[0].zfill(6): np.array(f["transform_matrix"])
          for f in tj["frames"]}
    imgs = read_images_binary(f"{a.data}/sparse/0/images.bin")
    col = {im.name.split("/")[-1].split(".")[0].zfill(6): -im.qvec2rotmat().T @ im.tvec
           for im in imgs.values()}
    # Search the index offset instead of trusting the filenames (see module docstring).
    def pair(off):
        ks = [k for k in col if str(int(k) + off).zfill(6) in gt]
        return ks, (np.array([gt[str(int(k) + off).zfill(6)][:3, 3] for k in ks]),
                    np.array([col[k] for k in ks]))
    scored = []
    for off in range(-3, 4):
        ks, (X, Y) = pair(off)
        if len(ks) < 0.9 * len(col):
            continue
        s_, R_, t_ = sim3(X, Y)
        scored.append((float(np.median(np.linalg.norm(Y - (s_ * (X @ R_.T) + t_), axis=1))), off))
    scored.sort()
    best = scored[0][1]
    print("[偏移搜尋] " + "  ".join(f"{o:+d}:{v:.4f}" for v, o in sorted(scored, key=lambda x: x[1]))
          + f"   → 採用 {best:+d}")
    keys, (A, B) = pair(best)
    keys = sorted(keys)
    A = np.array([gt[str(int(k) + best).zfill(6)][:3, 3] for k in keys])
    B = np.array([col[k] for k in keys])

    # Sanity: the pairing is by filename, and a wrong pairing would make everything downstream
    # noise. Frame index and position must be strongly related in BOTH frames if the pairing holds.
    print(f"[配對] {len(keys)} 張")

    s, R, t = sim3(A, B)
    res = B - (s * (A @ R.T) + t)

    P = np.array([p.xyz for p in read_points3D_binary(f"{a.data}/sparse/0/points3D.bin").values()])
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    Pc = P[np.all((P > lo) & (P < hi), 1)]
    rng = np.random.default_rng(0)
    depth = np.median(np.linalg.norm(
        Pc[rng.choice(len(Pc), 20000)][:, None, :] - B[None, ::200, :], axis=-1))
    fl = tj["fl_x"]
    to_px = lambda v: fl * np.asarray(v) / depth / a.down_sample

    dd, ii = cKDTree(A).query(A, k=a.k)
    glob = np.linalg.norm(res, axis=1)
    loc = np.array([np.median(np.linalg.norm(res[ii[i, 1:]] - res[i], axis=1)) for i in range(len(A))])
    gpx, lpx = to_px(glob), to_px(loc)
    print(f"[全域 drift ] median={gpx.mean() and np.median(gpx):.1f} px  p90={np.percentile(gpx,90):.1f} px   ← 可被重建吸收")
    print(f"[局部矛盾   ] median={np.median(lpx):.1f} px  p90={np.percentile(lpx,90):.1f} px  "
          f"p99={np.percentile(lpx,99):.1f} px   ← 這才會糊")
    print(f"[壞相機     ] 局部矛盾 > {a.bad_px:.0f} px 的比例 = {100*(lpx>a.bad_px).mean():.2f}%  "
          f"({int((lpx>a.bad_px).sum())} / {len(lpx)} 張)")

    # per-block
    pdir = f"{a.data}/partition/partitions-dim_{a.block_dim}_visibility_0.08"
    name2i = {k: i for i, k in enumerate(keys)}
    rows = []
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".txt"):
            continue
        r, c = fn[:-4].split("_")
        bid = int(r) * int(a.block_dim.split("_")[1]) + int(c)
        names = [ln.strip().split("/")[-1].split(".")[0].zfill(6)
                 for ln in open(f"{pdir}/{fn}") if ln.strip()]
        sel = [name2i[n] for n in names if n in name2i]
        if len(sel) < 10:
            continue
        sub = lpx[sel]
        rows.append((bid, f"{r}_{c}", len(sel), float(np.median(sub)),
                     float(np.percentile(sub, 90)), 100 * float((sub > a.bad_px).mean())))
    rows.sort(key=lambda x: -x[5])
    print(f"\n{'blk':>4} {'grid':>7} {'views':>6} {'局部矛盾中位':>12} {'p90':>8} {'壞相機%':>8}")
    for b, g, n, m, p9, bad in rows:
        print(f"{b:>4} {g:>7} {n:>6} {m:>11.1f}px {p9:>7.1f}px {bad:>7.1f}%")
    with open(a.out, "w") as f:
        f.write("block,grid,views,local_median_px,local_p90_px,bad_pct\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}\n")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
