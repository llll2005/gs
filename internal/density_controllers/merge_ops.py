"""訓練中合併（2026-10-31，使用者 10-30 提的合併機制；離線不重訓版見 紀錄/new_archived/已關閉的線.md E 段）。

MCMC 只有「搬」與「增」，沒有「併」。這裡在增生期每隔幾個 densify 事件做一次「互為最佳」兩兩合併：
  候選：opacity > min_opacity；鄰域（Morton 近似 k 近鄰的中心）PCA 平面夠平（λ3/λ2 < flat_tol）；
        兩顆的鄰域平面同向（|n_i·n_j| > 0.9）、彼此離對方平面近（< 0.5×長軸和）、面內距離 < 1.5×長軸和；
        顏色（SH DC → RGB）L1 差 < color_tol（預設 1.0＝不看顏色：讓窗戶與窗框也能併成實牆，細節交給後續訓練與 MCMC 增生）
  合併（質量守恆矩匹配，權重 o×面積）：中心＝加權平均；法向＝兩個鄰域平面法向平均（貼牆，不用面元自己的法向）；
        面內兩軸／尺度＝兩片面元 2D 共變異投影到合併平面＋中心離散的二階矩；opacity＝1−(1−o_i)(1−o_j)；SH＝加權平均
寫回 i 的槽位（原始參數），回傳 (i, j)：呼叫端把 i 的 Adam 狀態重設、把 j 併進 dead_mask 交給 MCMC relocation 搬走（N 不變）。
"""
import torch

from internal.metrics.geom_gate import morton_neighbors

_C0 = 0.28209479177387814


def _qmat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def _mat2q(R):
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = torch.sqrt((1 + t).clamp_min(1e-12)) / 2
    q = torch.stack([w, (R[:, 2, 1] - R[:, 1, 2]) / (4 * w), (R[:, 0, 2] - R[:, 2, 0]) / (4 * w), (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)], -1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def merge_step(gm, k: int = 12, max_frac: float = 0.02, color_tol: float = 1.0, flat_tol: float = 0.1,
               min_opacity: float = 0.5, chunk: int = 1 << 19):
    g = gm.gaussians
    X, raw_s, raw_q, raw_o = g["means"], g["scales"], g["rotations"], g["opacities"]
    dc, rest = g["shs_dc"], g["shs_rest"]
    dev = X.device
    O = torch.sigmoid(raw_o).reshape(-1)
    cand = torch.nonzero(O > min_opacity, as_tuple=True)[0]
    empty = torch.empty(0, dtype=torch.long, device=dev)
    if cand.numel() < 100:
        return empty, empty, {"pairs": 0}
    Xc = X[cand].float()
    S = torch.exp(raw_s[cand].float())
    R = _qmat(raw_q[cand].float())
    t1, t2 = R[:, :, 0], R[:, :, 1]
    smax = S.max(dim=-1).values
    rgb = dc[cand, 0, :].float() * _C0 + 0.5
    nb = morton_neighbors(Xc, k).long()                                  # [n, 2k]
    n = Xc.shape[0]
    nrm = torch.empty(n, 3, device=dev); flat = torch.empty(n, device=dev)
    for c0 in range(0, n, chunk):
        P = Xc[nb[c0:c0 + chunk]]; P = P - P.mean(1, keepdim=True)
        ev, evec = torch.linalg.eigh(torch.einsum("nki,nkj->nij", P, P))
        nrm[c0:c0 + chunk] = evec[:, :, 0]; flat[c0:c0 + chunk] = ev[:, 0] / ev[:, 1].clamp_min(1e-18)
    best = torch.full((n,), -1, dtype=torch.long, device=dev); bestc = torch.full((n,), float("inf"), device=dev)
    for c0 in range(0, n, chunk):
        ii = torch.arange(c0, min(c0 + chunk, n), device=dev)
        jj = nb[ii]                                                       # [m, 2k]
        d = Xc[jj] - Xc[ii][:, None]
        ni, nj = nrm[ii][:, None].expand_as(d), nrm[jj]
        off = (d * ni).sum(-1).abs(); inpl = (d - (d * ni).sum(-1, keepdim=True) * ni).norm(dim=-1)
        ssum = smax[ii][:, None] + smax[jj]
        dcol = (rgb[ii][:, None] - rgb[jj]).abs().mean(-1)
        ok = (jj != ii[:, None]) & (flat[ii][:, None] < flat_tol) & (flat[jj] < flat_tol) & ((ni * nj).sum(-1).abs() > 0.9) \
            & (off < 0.5 * ssum) & (inpl < 1.5 * ssum) & (dcol < color_tol)
        cost = torch.where(ok, inpl / ssum + dcol, torch.full_like(inpl, float("inf")))
        cb, ci = cost.min(dim=1)
        best[ii] = torch.where(torch.isfinite(cb), jj.gather(1, ci[:, None])[:, 0], torch.full_like(ci, -1)); bestc[ii] = cb
    has = best >= 0
    ar = torch.arange(n, device=dev)
    bb = best.clamp_min(0)
    mutual = has & has[bb] & (best[bb] == ar) & (ar < bb)
    pi = torch.nonzero(mutual, as_tuple=True)[0]; pj = best[pi]
    lim = int(max_frac * X.shape[0])
    if pi.numel() > lim:
        keep = torch.topk(bestc[pi], lim, largest=False).indices
        pi, pj = pi[keep], pj[keep]
    if pi.numel() == 0:
        return empty, empty, {"pairs": 0}
    I, J = cand[pi], cand[pj]
    oi, oj = O[I], O[J]
    wi = oi * S[pi, 0] * S[pi, 1]; wj = oj * S[pj, 0] * S[pj, 1]; ws = (wi + wj).clamp_min(1e-20)
    nI = nrm[pi]; nJ = nrm[pj] * torch.sign((nrm[pi] * nrm[pj]).sum(-1, keepdim=True))
    nn = nI + nJ; nn = nn / nn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c = (wi[:, None] * Xc[pi] + wj[:, None] * Xc[pj]) / ws[:, None]
    ref = torch.where(nn[:, 2:3].abs() < 0.9, torch.tensor([[0., 0., 1.]], device=dev), torch.tensor([[1., 0., 0.]], device=dev))
    u = torch.cross(nn, ref, dim=-1); u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12); v = torch.cross(nn, u, dim=-1)
    B = torch.stack([u, v], -1)                                           # [m, 3, 2]
    cov = torch.zeros(pi.numel(), 2, 2, device=dev)
    for idx, w in ((pi, wi), (pj, wj)):
        Sig = (S[idx, 0] ** 2)[:, None, None] * torch.einsum("ni,nj->nij", t1[idx], t1[idx]) \
            + (S[idx, 1] ** 2)[:, None, None] * torch.einsum("ni,nj->nij", t2[idx], t2[idx])
        q2 = torch.einsum("nij,ni->nj", B, Xc[idx] - c)
        cov += (w / ws)[:, None, None] * (torch.einsum("nki,nkl,nlj->nij", B, Sig, B) + torch.einsum("ni,nj->nij", q2, q2))
    ev2, evec2 = torch.linalg.eigh(cov)
    sc = torch.sqrt(ev2.flip(-1).clamp_min(1e-14))                        # 大的在前
    a1 = evec2[:, 0, 1:2] * u + evec2[:, 1, 1:2] * v; a1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a2 = torch.cross(nn, a1, dim=-1)
    qn = _mat2q(torch.stack([a1, a2, nn], -1))
    on = (1 - (1 - oi) * (1 - oj)).clamp(1e-4, 0.999)
    fi, fj = (wi / ws)[:, None, None], (wj / ws)[:, None, None]
    X[I] = c.to(X.dtype)
    raw_s[I] = torch.log(sc).to(raw_s.dtype)
    raw_q[I] = qn.to(raw_q.dtype)
    raw_o[I] = torch.log(on / (1 - on)).reshape(-1, *raw_o.shape[1:]).to(raw_o.dtype)
    dc[I] = (fi * dc[I].float() + fj * dc[J].float()).to(dc.dtype)
    rest[I] = (fi * rest[I].float() + fj * rest[J].float()).to(rest.dtype)
    fac = (nn[:, 2].abs() < 0.3).float().mean()
    return I, J, {"pairs": int(pi.numel()), "facade": float(fac), "len_before": float(smax[pi].median()), "len_after": float(sc[:, 0].median())}
