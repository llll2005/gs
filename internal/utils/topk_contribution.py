"""Mean of the K largest per-primitive transmittances across views."""
import torch


def contribution_accumulator(K: int, reduce: str = "max"):
    """Per-primitive contribution across views.

    `reduce` picks the statistic:
      "topk_mean" -- mean of the K largest. What the code always claimed to compute.
      "max"       -- the single best view. What the broken loop approximated by accident.

    MEASURED 2026-08-06 on post-hoc pruning of `oreg_0p002_b12` (building texture ratio, lower is
    worse): the accident was BETTER. Keeping the top 8.9% of primitives gave 0.266 under the old
    ranking and 0.236 under a true top-K mean; at 17.8% it was 0.303 vs 0.279.

    The reason is coverage. A block's 284 cameras see its centre from many angles and its edges
    from few, so requiring a primitive to score across SEVERAL views systematically discards the
    edges. "Good in one view" keeps everything that is prominent somewhere.

    Post-hoc pruning of `oreg_0p002_b12`, building texture ratio at each retained count:

        kept        max      legacy(broken)   topk_mean
        640,000    0.368        0.368           0.367
        480,000    0.357        0.356           0.351
        320,000    0.339        0.338           0.328
        160,000    0.303        0.296           0.279
         80,000    0.266        0.262           0.236

    `max` wins at every level and the broken loop tracked it closely -- it had the right criterion
    and the wrong arithmetic. So the default is `max`, named and computed as such, rather than a
    buggy approximation of a statistic that measures worse.

    ⚠ Measured on post-hoc pruning of a finished model. During training the trim fires repeatedly
    and the dynamics differ; that has not been separated.
    """
    if reduce not in ("topk_mean", "max", "frustum_topk", "sum", "count"):
        raise ValueError(f"unknown reduce: {reduce}")
    if reduce in ("sum", "count"):
        return _size_aware_accumulator(reduce)
    if reduce == "frustum_topk":
        return _frustum_topk_accumulator(K)
    return _topk_mean_accumulator(1 if reduce == "max" else K)


def _size_aware_accumulator(mode: str):
    """Size-AWARE contribution, in the spirit of MVGSR (arXiv 2503.08093) Sec.4.2 Eq.5:

        C_MV(p) = sum over views, sum over pixels of  [ T*alpha > delta ]

    Ours is the opposite by default: the renderer divides by `num_covered_pixels`
    (`sep_depth_trim_2dgs_renderer.py:132`), so a large primitive and a small one with the same
    per-pixel weight score the same. MVGSR counts pixels, so area counts.

    Eq.5 needs the PER-PIXEL values, which our CUDA does not return -- only their mean and the
    pixel count. Rebuilding the kernel for one experiment is not worth the risk (see the rebuild
    pitfalls in CLAUDE.md), so the two limits that ARE computable stand in:

        "count"  sum of covered pixels over views          -- the delta -> 0 limit of Eq.5
        "sum"    sum of (per-pixel mean * covered pixels)   -- total T*alpha, a weighted version

    Neither is Eq.5 exactly. Both capture the part that differs from ours: area matters.

    `push` takes (transmittance, num_covered_pixels).
    """
    state = {"acc": None}

    def push(trans, covered):
        v = covered.to(trans.dtype) if mode == "count" else trans * covered.to(trans.dtype)
        state["acc"] = v.clone() if state["acc"] is None else state["acc"] + v

    def result():
        return state["acc"]

    return push, result


def _frustum_topk_accumulator(K: int):
    """Top-K mean with K capped by how many views actually contain the primitive.

    Fixed-K top-K punishes anything seen by fewer than K cameras, because slots K_seen+1..K stay
    at zero and drag the mean down. Under block training that is precisely the block's EDGES: the
    centre is seen from 50+ angles, the rim from 3-5. So fixed-K conflates "real geometry near the
    boundary" with "floater", and measurably prunes worse than `max`.

    Capping K at the frustum count removes the zero padding for edge primitives while leaving it in
    place for a floater, which sits inside many frusta and contributes to almost none of them.

    `push` takes (transmittance, num_covered_pixels). `num_covered_pixels > 0` is frustum
    membership: occlusion lowers T, not alpha, so an occluded primitive is still rasterised and
    still counted. Counting "views where transmittance > 0" instead would let the occluded floater
    report a frustum count of 1 and escape.
    """
    state = {"buf": None, "n_frustum": None}

    def push(trans, covered):
        if state["buf"] is None:
            state["buf"] = torch.full((K,) + trans.shape, float("-inf"),
                                      dtype=trans.dtype, device=trans.device)
            state["n_frustum"] = torch.zeros_like(trans)
        state["buf"] = torch.cat([state["buf"], trans[None]], dim=0).topk(K, dim=0).values
        state["n_frustum"] += (covered > 0).to(trans.dtype)

    def result():
        buf, nf = state["buf"], state["n_frustum"]
        k_eff = nf.clamp(1, K)                                   # min(K, N_frustum), at least 1
        rank = torch.arange(K, device=buf.device).view(K, *([1] * (buf.dim() - 1)))
        keep = (rank < k_eff.unsqueeze(0)) & torch.isfinite(buf)
        return buf.masked_fill(~keep, 0.0).sum(0) / keep.sum(0).clamp_min(1)

    return push, result


def _topk_mean_accumulator(K: int):
    """Streaming mean-of-the-K-largest, per primitive.

    Replaces a loop that did not compute that. The old one compared only against slot 0 and wrote
    `top_list[0] = trans` INSIDE the shift, so the shift read the slot it had just overwritten:

        [a, b, c] + record r   ->   [r, r, b]

    which discards the previous maximum and copies the new record into K-1 slots. Steady state is
    `(2*max + prev)/K` -- effectively a running maximum, order-dependent, with the first camera's
    value seeded into every slot. Demonstrated in `tests/topk_contribution_test.py`.

    That distinction is not cosmetic: contribution pruning is reasoned about as a MULTI-VIEW
    statistic ("a floater explains a few views, so summing over views destroys the signal" --
    `tools/measure_view_consistency.py`), but ranking by roughly the single best view is the most
    permissive rule there is, and a floater peaks in exactly the view it was placed for.

    Keeps a [K+1, N] buffer and re-selects, rather than stacking all V views: 284 cameras at N=2M
    would be 2.3 GB on a 6 GB card, while this is (K+1)*N = 48 MB.
    """
    state = {"buf": None}

    def push(trans: torch.Tensor):
        if state["buf"] is None:
            state["buf"] = torch.full((K,) + trans.shape, float("-inf"),
                                      dtype=trans.dtype, device=trans.device)
        state["buf"] = torch.cat([state["buf"], trans[None]], dim=0).topk(K, dim=0).values

    def result() -> torch.Tensor:
        buf = state["buf"]
        seen = torch.isfinite(buf)                     # fewer cameras than K -> ignore the padding
        return (buf.masked_fill(~seen, 0.0).sum(0) / seen.sum(0).clamp_min(1))

    return push, result


# 舊名保留，避免呼叫端一次全改
topk_mean_accumulator = _topk_mean_accumulator
