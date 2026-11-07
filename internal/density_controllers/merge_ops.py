"""訓練中合併（2026-10-31，使用者 10-30 提的合併機制；離線不重訓版見 紀錄/new_archived/已關閉的線.md E 段）。

MCMC 只有「搬」與「增」，沒有「併」。這裡在增生期每隔幾個 densify 事件做一次「互為最佳」兩兩合併：
  候選：opacity > min_opacity；鄰域（Morton 近似 k 近鄰的中心）PCA 平面夠平（λ3/λ2 < flat_tol）；
        兩顆的鄰域平面同向（|n_i·n_j| > 0.9）、彼此離對方平面近（< 0.5×長軸和）、面內距離 < 1.5×長軸和；
        顏色（SH DC → RGB）L1 差 < color_tol（預設 1.0＝不看顏色：讓窗戶與窗框也能併成實牆，細節交給後續訓練與 MCMC 增生）
  合併（質量守恆矩匹配，權重 o×面積）：中心＝加權平均；法向＝兩個鄰域平面法向平均（貼牆，不用面元自己的法向）；
        面內兩軸／尺度＝兩片面元 2D 共變異投影到合併平面＋中心離散的二階矩；opacity＝1−(1−o_i)(1−o_j)；SH＝加權平均
寫回 i 的槽位（原始參數），回傳 (i, j)：呼叫端把 i 的 Adam 狀態重設、把 j 併進 dead_mask 交給 MCMC relocation 搬走（N 不變）。

★ 2026-11-07 晚 MCMC 一致版（使用者「MCMC 的機制都基於某個算式，合併能不能也符合、不要限額度讓它自然判斷」）：
  3DGS-MCMC 的 relocation 是「不改變渲染結果的跳躍」——一顆拆成 N 顆時用 Eq.9 調 opacity／尺度讓疊起來的樣子不變，
  搬多少由狀態決定（死了幾顆）而不是額度。合併是它的反向：兩顆換一顆，只有在「換了之後渲染幾乎不變」時才是合法的一步。
  opacity_mode="mass"：合併後 opacity 由不透明質量守恆決定 o_m·|Σ_m|^½ = o_i·|Σ_i|^½ + o_j·|Σ_j|^½（舊的聯集公式只對完全疊在一起的兩顆成立，
  並排的兩顆會讓質量變多）；render_eps > 0：在合併平面上算 f = o_i·G_i + o_j·G_j 與 g = o_m·G_m 的相對 L2 誤差
  ||f−g||²／||f||²（2D 高斯乘積積分有封閉式），誤差 < render_eps 的互為最佳配對全部合併；額度 max_frac >= 1 時不設上限。
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
    """旋轉矩陣 => (w, x, y, z)。Shepperd：依 trace 與三個對角項中最大者分四種情況（只用 trace 的公式在轉角近 180° 時 w→0、
    分母爆掉 => 方向是雜訊；2026-11-02 補洞面元朝下（R＝繞軸 180°）時抓到，訓練中合併寫回的朝向也受影響）。"""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    r21, r12, r02, r20, r10, r01 = R[:, 2, 1], R[:, 1, 2], R[:, 0, 2], R[:, 2, 0], R[:, 1, 0], R[:, 0, 1]
    t = m00 + m11 + m22
    s0 = torch.sqrt((1 + t).clamp_min(1e-12)) * 2
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) * 2
    s2 = torch.sqrt((1 + m11 - m00 - m22).clamp_min(1e-12)) * 2
    s3 = torch.sqrt((1 + m22 - m00 - m11).clamp_min(1e-12)) * 2
    c = torch.stack([
        torch.stack([s0 / 4, (r21 - r12) / s0, (r02 - r20) / s0, (r10 - r01) / s0], -1),
        torch.stack([(r21 - r12) / s1, s1 / 4, (r01 + r10) / s1, (r02 + r20) / s1], -1),
        torch.stack([(r02 - r20) / s2, (r01 + r10) / s2, s2 / 4, (r12 + r21) / s2], -1),
        torch.stack([(r10 - r01) / s3, (r02 + r20) / s3, (r12 + r21) / s3, s3 / 4], -1)], 1)      # [n, 4 種, 4]
    k = torch.stack([t, m00, m11, m22], -1).argmax(-1)
    q = c.gather(1, k[:, None, None].expand(-1, 1, 4))[:, 0]
    q = q * torch.where(q[:, :1] < 0, -1.0, 1.0)                                                 # w >= 0（同一個旋轉）
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def merge_step(gm, k: int = 12, max_frac: float = 0.02, color_tol: float = 1.0, flat_tol: float = 0.1,
               min_opacity: float = 0.5, chunk: int = 1 << 19, exclude=None, prio=None,
               opacity_mode: str = "union", render_eps: float = -1.0):
    """exclude（[N] bool，可省）：不參與合併的顆粒（SfM 錨點）。prio（[N] float，可省）：每顆的成本倍率，
    配對取兩顆中較小的；只在候選對超過 max_frac 額度時改變「先併哪些」（2026-11-05 sfm_merge_eps）。兩者省略＝原行為。
    opacity_mode／render_eps：見檔頭「MCMC 一致版」；預設＝原行為。"""
    g = gm.gaussians
    X, raw_s, raw_q, raw_o = g["means"], g["scales"], g["rotations"], g["opacities"]
    dc, rest = g["shs_dc"], g["shs_rest"]
    dev = X.device
    O = torch.sigmoid(raw_o).reshape(-1)
    ok_c = O > min_opacity
    if exclude is not None and exclude.shape[0] == O.shape[0]:
        ok_c = ok_c & ~exclude
    cand = torch.nonzero(ok_c, as_tuple=True)[0]
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
    n_mutual = int(pi.numel())
    err_med = float("nan")
    if render_eps > 0 and pi.numel() > 0:
        err = _pair_params(O, S, t1, t2, nrm, Xc, cand, pi, pj, opacity_mode)["err"]
        err_med = float(err.median())
        okp = err < render_eps
        pi, pj, bestc_p = pi[okp], pj[okp], err[okp]
    else:
        bestc_p = bestc[pi]
    lim = int(max_frac * X.shape[0]) if max_frac < 1 else pi.numel()
    if pi.numel() > lim:
        key = bestc_p
        if prio is not None and prio.shape[0] == O.shape[0]:
            key = key * torch.minimum(prio[cand[pi]], prio[cand[pj]]).to(key.dtype)
        keep = torch.topk(key, lim, largest=False).indices
        pi, pj = pi[keep], pj[keep]
    extra = {"mutual": n_mutual, "err_med": err_med}
    if pi.numel() == 0:
        return empty, empty, {"pairs": 0, **extra}
    I, J, info = _pair_merge(g, O, S, t1, t2, nrm, Xc, smax, cand, pi, pj, opacity_mode)
    return I, J, {**info, **extra}


def _gauss_overlap(A, a, Bm, b):
    """∫ exp(−½(x−a)ᵀA⁻¹(x−a))·exp(−½(x−b)ᵀB⁻¹(x−b)) dx（2D、不歸一化、峰值 1 的高斯；A、B [m,2,2]，a、b [m,2]）
    ＝ 2π·sqrt(|A||B|／|A+B|)·exp(−½ dᵀ(A+B)⁻¹d)。"""
    C = A + Bm
    d = a - b
    q = (d[:, None, :] @ torch.linalg.solve(C, d[:, :, None]))[:, 0, 0]
    det = lambda M: (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).clamp_min(1e-60)
    return 2 * torch.pi * torch.sqrt(det(A) * det(Bm) / det(C)) * torch.exp(-0.5 * q)


@torch.no_grad()
def _pair_params(O, S, t1, t2, nrm, Xc, cand, pi, pj, opacity_mode="union"):
    """一對一的合併結果（不寫回）：中心 c、尺度 sc、四元數 qn、opacity on、法向 nn、SH 權重 fi／fj、渲染誤差 err（見檔頭）。"""
    dev = Xc.device
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
    S2, Q2 = [], []
    for idx, w in ((pi, wi), (pj, wj)):
        Sig = (S[idx, 0] ** 2)[:, None, None] * torch.einsum("ni,nj->nij", t1[idx], t1[idx]) \
            + (S[idx, 1] ** 2)[:, None, None] * torch.einsum("ni,nj->nij", t2[idx], t2[idx])
        q2 = torch.einsum("nij,ni->nj", B, Xc[idx] - c)
        s2 = torch.einsum("nki,nkl,nlj->nij", B, Sig, B)
        S2.append(s2); Q2.append(q2)
        cov += (w / ws)[:, None, None] * (s2 + torch.einsum("ni,nj->nij", q2, q2))
    ev2, evec2 = torch.linalg.eigh(cov)
    sc = torch.sqrt(ev2.flip(-1).clamp_min(1e-14))                        # 大的在前
    a1 = evec2[:, 0, 1:2] * u + evec2[:, 1, 1:2] * v; a1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a2 = torch.cross(nn, a1, dim=-1)
    qn = _mat2q(torch.stack([a1, a2, nn], -1))
    if opacity_mode == "mass":                                            # 不透明質量守恆：o·s1·s2（2π 約掉）
        on = ((wi + wj) / (sc[:, 0] * sc[:, 1]).clamp_min(1e-30)).clamp(1e-4, 0.999)
    elif opacity_mode == "union":
        on = (1 - (1 - oi) * (1 - oj)).clamp(1e-4, 0.999)
    else:
        raise ValueError(f"opacity_mode={opacity_mode!r}（只接受 union／mass）")
    # 渲染誤差（合併平面上、加法近似；float64）：f = o_i G_i + o_j G_j，g = o_m G_m
    D = torch.float64
    Ai, Aj, Am = S2[0].to(D), S2[1].to(D), cov.to(D)
    qi, qj, z = Q2[0].to(D), Q2[1].to(D), torch.zeros_like(Q2[0], dtype=D)
    oi_, oj_, om_ = oi.to(D), oj.to(D), on.to(D)
    ff = oi_ ** 2 * _gauss_overlap(Ai, qi, Ai, qi) + oj_ ** 2 * _gauss_overlap(Aj, qj, Aj, qj) + 2 * oi_ * oj_ * _gauss_overlap(Ai, qi, Aj, qj)
    fg = oi_ * _gauss_overlap(Ai, qi, Am, z) + oj_ * _gauss_overlap(Aj, qj, Am, z)
    gg = om_ ** 2 * _gauss_overlap(Am, z, Am, z)
    err = ((ff - 2 * om_ * fg + gg).clamp_min(0) / ff.clamp_min(1e-60)).float()
    return {"I": I, "J": J, "c": c, "sc": sc, "qn": qn, "on": on, "nn": nn,
            "fi": (wi / ws)[:, None, None], "fj": (wj / ws)[:, None, None], "err": err}


@torch.no_grad()
def _pair_merge(g, O, S, t1, t2, nrm, Xc, smax, cand, pi, pj, opacity_mode="union"):
    """把 cand[pj] 併進 cand[pi]（質量守恆矩匹配；寫回 cand[pi] 的原始參數）。pi／pj 是 cand 內的局部索引、一一配對、互不重複。
    merge_step（互為最佳的兩兩合併）與 absorb_step（trim 候選併進最佳鄰居）共用。回傳 (留下的全域索引, 併掉的全域索引, 統計)。"""
    X, raw_s, raw_q, raw_o = g["means"], g["scales"], g["rotations"], g["opacities"]
    dc, rest = g["shs_dc"], g["shs_rest"]
    pr = _pair_params(O, S, t1, t2, nrm, Xc, cand, pi, pj, opacity_mode)
    I, J, c, sc, qn, on, nn, fi, fj = (pr[k] for k in ("I", "J", "c", "sc", "qn", "on", "nn", "fi", "fj"))
    X[I] = c.to(X.dtype)
    raw_s[I] = torch.log(sc).to(raw_s.dtype)
    raw_q[I] = qn.to(raw_q.dtype)
    raw_o[I] = torch.log(on / (1 - on)).reshape(-1, *raw_o.shape[1:]).to(raw_o.dtype)
    dc[I] = (fi * dc[I].float() + fj * dc[J].float()).to(dc.dtype)
    rest[I] = (fi * rest[I].float() + fj * rest[J].float()).to(rest.dtype)
    fac = (nn[:, 2].abs() < 0.3).float().mean()
    return I, J, {"pairs": int(pi.numel()), "facade": float(fac), "len_before": float(smax[pi].median()), "len_after": float(sc[:, 0].median())}


@torch.no_grad()
def absorb_step(gm, src, k: int = 12, color_tol: float = 0.1, flat_tol: float = 0.1, min_opacity_dst: float = 0.5,
                min_opacity_src: float = 0.05, chunk: int = 1 << 19):
    """★ 2026-11-02 合併優先的 trim（使用者：結合 trim 判準與合併）：src（bool [N]，本輪 trim 要剪的低 V/C 顆粒）裡，
    旁邊有同平面鄰居（同 merge_step 的平面／相鄰／顏色條件；鄰居不在 src、opacity > min_opacity_dst）的，併進最佳的那一個鄰居
    （每個鄰居一輪最多吸收一顆，取代價最小的）；opacity <= min_opacity_src 的不併（併了也幾乎沒貢獻）。
    只改「留下的鄰居」的參數，src 本身由呼叫端照常剪掉（N 的減少量與 trim 相同）。回傳 (吸收者, 被吸收者, 統計)，皆為全域索引。"""
    g = gm.gaussians
    X, raw_s, raw_q, raw_o = g["means"], g["scales"], g["rotations"], g["opacities"]
    dc = g["shs_dc"]
    dev = X.device
    O = torch.sigmoid(raw_o).reshape(-1)
    empty = torch.empty(0, dtype=torch.long, device=dev)
    cand = torch.nonzero(O > min_opacity_src, as_tuple=True)[0]
    if cand.numel() < 100 or not bool(src[cand].any()):
        return empty, empty, {"pairs": 0}
    is_src = src[cand]
    ok_dst = (~is_src) & (O[cand] > min_opacity_dst)
    Xc = X[cand].float()
    S = torch.exp(raw_s[cand].float())
    R = _qmat(raw_q[cand].float())
    t1, t2 = R[:, :, 0], R[:, :, 1]
    smax = S.max(dim=-1).values
    rgb = dc[cand, 0, :].float() * _C0 + 0.5
    nb = morton_neighbors(Xc, k).long()
    n = Xc.shape[0]
    nrm = torch.empty(n, 3, device=dev); flat = torch.empty(n, device=dev)
    for c0 in range(0, n, chunk):
        P = Xc[nb[c0:c0 + chunk]]; P = P - P.mean(1, keepdim=True)
        ev, evec = torch.linalg.eigh(torch.einsum("nki,nkj->nij", P, P))
        nrm[c0:c0 + chunk] = evec[:, :, 0]; flat[c0:c0 + chunk] = ev[:, 0] / ev[:, 1].clamp_min(1e-18)
    srcl = torch.nonzero(is_src, as_tuple=True)[0]
    best = torch.full((srcl.numel(),), -1, dtype=torch.long, device=dev); bestc = torch.full((srcl.numel(),), float("inf"), device=dev)
    for c0 in range(0, srcl.numel(), chunk):
        ii = srcl[c0:c0 + chunk]
        jj = nb[ii]
        d = Xc[jj] - Xc[ii][:, None]
        ni, nj = nrm[ii][:, None].expand_as(d), nrm[jj]
        off = (d * ni).sum(-1).abs(); inpl = (d - (d * ni).sum(-1, keepdim=True) * ni).norm(dim=-1)
        ssum = smax[ii][:, None] + smax[jj]
        dcol = (rgb[ii][:, None] - rgb[jj]).abs().mean(-1)
        ok = (jj != ii[:, None]) & ok_dst[jj] & (flat[ii][:, None] < flat_tol) & (flat[jj] < flat_tol) & ((ni * nj).sum(-1).abs() > 0.9) \
            & (off < 0.5 * ssum) & (inpl < 1.5 * ssum) & (dcol < color_tol)
        cost = torch.where(ok, inpl / ssum + dcol, torch.full_like(inpl, float("inf")))
        cb, ci = cost.min(dim=1)
        best[c0:c0 + ii.numel()] = torch.where(torch.isfinite(cb), jj.gather(1, ci[:, None])[:, 0], torch.full_like(ci, -1))
        bestc[c0:c0 + ii.numel()] = cb
    has = best >= 0
    if not bool(has.any()):
        return empty, empty, {"pairs": 0, "src": int(srcl.numel())}
    s_loc, d_loc, cst = srcl[has], best[has], bestc[has]
    perm = torch.argsort(cst)                                          # 每個鄰居只吸收代價最小的一顆
    d_sorted = d_loc[perm]
    uniq, inv = torch.unique(d_sorted, return_inverse=True)
    first = torch.full((uniq.numel(),), d_sorted.numel(), dtype=torch.long, device=dev)
    first = first.scatter_reduce(0, inv, torch.arange(d_sorted.numel(), device=dev), reduce="amin")
    keep = perm[first]
    I, J, info = _pair_merge(g, O, S, t1, t2, nrm, Xc, smax, cand, d_loc[keep], s_loc[keep])
    info["src"] = int(srcl.numel())
    return I, J, info
