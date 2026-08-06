"""A shapeless initialisation: N points filling the block's volume uniformly.

The point is to separate the two things an initialisation supplies, which every init we have tried
so far bundles together:

    PRECISION   are the points where the surface actually is
    COVERAGE    is there a point near every place the surface could be

MCMC's claimed robustness to initialisation only covers the first. Its two ways of moving mass are
both anchored to where points already are -- `relocate_gs` samples hosts among the LIVE points by
opacity, and the SGLD noise is a local walk that the paper itself says is "irrecoverable" once a
Gaussian leaves "the previous support regions" (3dgs-mcmc.pdf, Sec. 3). So the prediction is:

    precision can be bad and MCMC will fix it;  coverage cannot, because there is nothing to
    relocate towards in a region with no points.

A uniform volume fill is the extreme test of that: coverage is perfect by construction and
precision is zero -- the cloud carries no information about the scene at all. If MCMC converges
from it, "coverage is what matters" holds in its strongest form.

WHY OPACITY IS THE WHOLE EXPERIMENT
-----------------------------------
`depth_init_blocks.py` seeds at logit(0.99). For a volume fill that is fatal: 3M opaque points form
a fog wall, nothing behind the first layer is visible and no gradient reaches it. 3DGS's own random
init uses 0.1, and MCMC's paper follows it. Default here is 0.05 and it is the first thing to sweep
if the run stalls.

Also note how little of a volume fill lands near a surface. At N=3M over b12's ~408 cubic units the
spacing is ~0.051, and a surface of ~100 square units has only ~75k points within one spacing of it
-- 2.5%. The other 97.5% start in empty air and have to migrate. That IS the mechanism under test.
"""
import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement

FIELDS = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ("opacity", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"),
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_ply", required=True,
                    help="the block's depth-init PLY; its extent defines the volume to fill and "
                         "its point count is the comparison baseline")
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--opacity", type=float, default=0.05,
                    help="ACTIVATED opacity. 0.99 (what depth-init uses) turns a volume fill into "
                         "an opaque fog wall; 3DGS's random init uses 0.1")
    ap.add_argument("--scale_factor", type=float, default=0.5,
                    help="scale = this x the uniform spacing. 0.5 gives 3-sigma coverage of ~1.5 "
                         "spacings, i.e. neighbours overlap without each point being a blob")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="expand the reference extent by this fraction on each side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v = PlyData.read(a.reference_ply)["vertex"]
    ref = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)
    lo, hi = np.percentile(ref, 0.5, 0), np.percentile(ref, 99.5, 0)
    size = hi - lo
    lo, hi = lo - a.pad * size, hi + a.pad * size
    size = hi - lo
    vol = float(np.prod(size))
    spacing = (vol / a.n) ** (1 / 3)

    print(f"[參照] {a.reference_ply}  N={len(ref):,}")
    print(f"[體積] {np.round(lo, 2)} .. {np.round(hi, 2)}   = {vol:.1f} 立方單位")
    print(f"[均勻填充] N={a.n:,}  間距={spacing:.5f}  scale={a.scale_factor * spacing:.5f}"
          f"  opacity={a.opacity}")

    rng = np.random.default_rng(a.seed)
    xyz = lo + rng.random((a.n, 3)) * size

    # how much of this lands anywhere near the real surface -- the fraction that starts useful
    from scipy.spatial import cKDTree
    sub = xyz[rng.choice(a.n, min(200_000, a.n), replace=False)]
    d, _ = cKDTree(ref).query(sub)
    print(f"[有用比例] 離參照表面 <1 個間距的: {100 * (d < spacing).mean():.2f}%"
          f"   <2 個間距: {100 * (d < 2 * spacing).mean():.2f}%")
    print(f"           ⇒ 其餘的點從空氣中出發，要靠 relocation 搬到表面上")

    el = np.empty(a.n, dtype=FIELDS)
    el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # Random unit normals: a volume fill has no surface to derive an orientation from, and any
    # fixed choice would inject a shape prior -- which is exactly what this init must not have.
    nrm = rng.normal(size=(a.n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    el["nx"], el["ny"], el["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    # quaternion taking +z to the sampled normal (same convention as depth_init_blocks)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, nrm)
    an = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.divide(axis, np.maximum(an, 1e-12))
    ang = np.arccos(np.clip(nrm @ z, -1, 1))[:, None]
    el["rot_0"] = np.cos(ang / 2)[:, 0]
    el["rot_1"], el["rot_2"], el["rot_3"] = (axis * np.sin(ang / 2)).T

    el["f_dc_0"] = el["f_dc_1"] = el["f_dc_2"] = 0.0
    el["opacity"] = np.log(a.opacity / (1 - a.opacity))          # logit
    el["scale_0"] = el["scale_1"] = np.log(a.scale_factor * spacing)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(a.out)
    print(f"\n[out] {a.out}  ({a.n:,} 顆, {os.path.getsize(a.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
