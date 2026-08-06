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
    if reduce not in ("topk_mean", "max"):
        raise ValueError(f"unknown reduce: {reduce}")
    return _topk_mean_accumulator(1 if reduce == "max" else K)


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
