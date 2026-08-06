"""Mean of the K largest per-primitive transmittances across views."""
import torch


def topk_mean_accumulator(K: int):
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
