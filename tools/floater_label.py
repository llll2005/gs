"""Which primitives are not on any real surface?

Shared by checkpoint_health.py and measure_view_consistency.py so both mean the same thing by
"floater". Getting this wrong twice is what motivated pulling it out:

  1. First attempt thresholded on a GLOBAL height (the scene's p99, 2.14). That is the height of
     the tallest roofs, while the median ground sits at 0.20, so anything hovering at z=1.0 --
     squarely between an overhead camera and the street -- was not counted. It reported 0.1%
     floaters for a model the user could see was full of them.
  2. Second attempt used a LOCAL ground height per 0.05 cell. Better, but it labels every building
     facade a floater: a primitive halfway up a wall is far above the ground of its own cell. It
     reported 63.8% for one model and 22.4% for another, which made the two incomparable and killed
     a measurement that had shown real signal.

Both failures share a cause: height is the wrong quantity. A floater is not "high", it is "not on a
surface". So measure distance to the nearest triangulated point instead.

And then do NOT threshold it. The distance distribution is smooth -- there is no gap between
"on-surface" and "floating", so any cut lands in the middle of a continuum and the resulting
percentage says more about the threshold than the model. Report the distance itself, in units of
how far real points sit from their own neighbours (median 0.0036 here). That ranks models cleanly:

    real points themselves     1x
    the 28.9-PSNR model        6x     primitives sit on the surface
    our b12 models            26-29x
    official b7 rerun        528x     geometry nowhere near any surface; depth slope 0.158

A caveat that applies to all of these: sparse/0 is triangulated, so it has no points where no
features matched -- open water, blank walls. A primitive correctly placed there still measures as
far from the surface. Read the number as an upper bound on error, and compare models on the same
block rather than across blocks with different amounts of texture.

The reference cloud is `sparse/0`, triangulated against the dataset's poses. It projects exactly
onto rooftops and kerbs (verified 2026-08-04), and it is the only metric geometry available.
"""
import numpy as np
from scipy.spatial import cKDTree

_CACHE = {}


def surface_tree(data, quantile=99):
    """KD-tree of the real surface plus the distance scale that defines 'off-surface'.

    The threshold is the `quantile`-th percentile of nearest-neighbour distance WITHIN the point
    cloud, i.e. how far a genuine surface point sits from its own neighbours. Anything beyond that
    is farther from the surface than the surface is from itself.
    """
    key = (data, quantile)
    if key in _CACHE:
        return _CACHE[key]
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from internal.utils.colmap import read_points3D_binary
    P = np.array([p.xyz for p in read_points3D_binary(f"{data}/sparse/0/points3D.bin").values()])
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    P = P[np.all((P > lo) & (P < hi), 1)]
    tree = cKDTree(P)
    # self-distance on a subsample; k=2 because k=1 is the point itself
    rng = np.random.default_rng(0)
    sub = P[rng.choice(len(P), min(200_000, len(P)), replace=False)]
    nn = tree.query(sub, k=2)[0][:, 1]
    thr = float(np.percentile(nn, quantile))
    _CACHE[key] = (tree, thr, P)
    return _CACHE[key]


def surface_distance(xyz, data):
    """Distance from each primitive centre to the nearest real surface point, and the scale to
    read it in: the median nearest-neighbour distance among the real points themselves.

    Returns (distances, unit). `distances / unit` is the headline number -- 1 means as close to the
    surface as real points are to each other.
    """
    tree, _, P = surface_tree(data)
    d, _ = tree.query(np.asarray(xyz, dtype=np.float64))
    rng = np.random.default_rng(0)
    sub = P[rng.choice(len(P), min(200_000, len(P)), replace=False)]
    unit = float(np.median(tree.query(sub, k=2)[0][:, 1]))
    return d, unit


def label(xyz, data, quantile=99, scale=1.0):
    """Binary floater mask. Kept for callers that need one, but prefer surface_distance():
    the underlying distribution is smooth, so this cut is arbitrary."""
    tree, thr, _ = surface_tree(data, quantile)
    d, _ = tree.query(np.asarray(xyz, dtype=np.float64))
    t = thr * scale
    return d > t, d, t
