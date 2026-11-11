"""幾何閘門（2026-10-29 使用者：「把法向正則弄到透明度熵的算法裡，只取我們要的參數並極端化、減少計算」＋「把延後的都排上去」）。

背景（紀錄：記憶 viewer_sea_floaters_roofs_1029）：
- 立面「毛邊」＝面元朝向亂（b3 朝向一致度 現行 0.765、CityGS 0.928）；渲染式法向一致性把 b12 從 0.80 拉到 0.95，但每塊多 50～60% 時間。
- 海面／屋頂浮塊＝光度上有用、幾何上錯的顆粒，熵把它們推到 o≈1 後 sigmoid 飽和，光度梯度推不動，MCMC（o<=0.005 才回收）永遠不收。

這裡兩個**逐顆、不渲染**的訊號，合成一個閘門 g_i：
  共面一致（取代逐像素的法向 loss）
      近鄰：Morton 序（兩種軸序）的前後各 k/2 顆＝近似 kNN，O(N log N) 排序
      w_ij = o_j * exp(-(d_ij . n_i)^2 / (sig*s_i)^2 - |d_ij|^2 / (rad*s_i)^2)     只算落在 i 平面上、距離 rad 個面元長軸內的鄰居
      q_i  = sum_j w_ij |n_i . n_j| / sum_j w_ij                                       1＝和鄰居共面（＝朝向一致度指標的加權版）
      L_plane = lam_p * mean_{i in 抽樣} o_i * sum_j w_ij (1 - |n_i . n_j|) / sum_j w_ij     （w、o_i 不帶梯度；梯度到 n_i、n_j）
      用 |cos| 不用 cos^2：cos^2 的梯度 ∝ cos，90 度附近（最亂的那些）幾乎推不動；|cos| 的梯度在 90 度最大
  多視角深度支撐（單目深度＝parser 已套逐張 scale/offset 的反深度）
      r_ik = (z_ik - D_k) / D_k；視窗內累積 sum r、count、min |r|
      在前方＝mean r < -delta_f（一直在單目表面前方＝浮塊）；從未貼面＝min |r| > delta_b（地底、建築內部）
  g_i = (q_i >= tau 或 q 無效) 且 (沒有被判在前方／從未貼面)
用法（在 MCMCCityGSV2MetricsModule 裡）：熵的往 1 推只給 g_i=1；g_i=0 加 -log(1-o)（對 logit 的梯度＝o，飽和也推得動）；
MCMC 的增生／搬遷抽樣乘上 g（density controller 的 geom_gate_sampling）。

出處：🟨 kNN 法向平滑是點雲處理的常見做法、多視角深度一致性是 MVS 的常見做法；拿來控制熵的方向與 MCMC 抽樣 ⚠ 未查證是否有人做過。
⚠ 單目深度在海面上誤差大（b12 無 SfM 格只有 26% 落在地面 ±0.05）=> 深度支撐用「平均在前方」而不是單一視角判定。
"""
import numpy as np
import torch

_M21 = (1 << 21) - 1


def _part1by2(x: torch.Tensor) -> torch.Tensor:
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def morton_neighbors(xyz: torch.Tensor, k: int = 6) -> torch.Tensor:
    """近似 kNN：兩種軸序（xyz、yzx）的 Morton 序各取前後 k/2 顆 => [N, 2k] 的鄰居索引（含少量重複／遠點，由權重濾掉）。
    回傳 int32（2.5M x 12 顆是 120 MB 而不是 240 MB）；用的時候再 .long()。"""
    n = xyz.shape[0]
    x = xyz.detach().float()
    lo_i = max(1, int(0.001 * n))
    lo = x.kthvalue(lo_i, dim=0).values
    hi = x.kthvalue(max(lo_i, n - lo_i), dim=0).values
    span = (hi - lo).max().clamp_min(1e-9)                      # 等向量化：三軸同一個格距
    q = ((x - lo) / span * _M21).round().clamp(0, _M21).long()
    half = max(1, k // 2)
    offs = torch.cat([torch.arange(-half, 0), torch.arange(1, half + 1)]).to(xyz.device)
    out = []
    for perm in ((0, 1, 2), (1, 2, 0)):
        code = _part1by2(q[:, perm[0]]) | (_part1by2(q[:, perm[1]]) << 1) | (_part1by2(q[:, perm[2]]) << 2)
        order = torch.argsort(code)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n, device=xyz.device)
        pos = (rank[:, None] + offs[None, :]).clamp(0, n - 1)
        out.append(order[pos])
    return torch.cat(out, dim=1).int()


def quat_normals(q: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) 四元數 => 旋轉矩陣第三欄（2DGS 面元法向）。不必先正規化。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def _weights(xyz, nrm, s1, op, i, nb, sig, rad):
    d = xyz[nb] - xyz[i, None]                                   # [c, k, 3]
    sc = s1[i, None]
    dp = (d * nrm[i, None]).sum(-1)
    return op[nb] * torch.exp(-(dp / (sig * sc)) ** 2 - d.pow(2).sum(-1) / (rad * sc) ** 2)


@torch.no_grad()
def plane_q(xyz, nrm, s1, op, nbr, sig=0.5, rad=3.0, chunk=1 << 19):
    """每顆的共面一致 q 與權重和 sw（sw 小＝附近沒有共面鄰居，q 無效）。s1 需已加下限。"""
    n = xyz.shape[0]
    q = torch.ones(n, device=xyz.device)
    sw = torch.zeros(n, device=xyz.device)
    for a in range(0, n, chunk):
        i = torch.arange(a, min(a + chunk, n), device=xyz.device)
        nb = nbr[i].long()
        w = _weights(xyz, nrm, s1, op, i, nb, sig, rad)
        ca = (nrm[nb] * nrm[i, None]).sum(-1).abs()
        s = w.sum(1)
        q[i] = (w * ca).sum(1) / s.clamp_min(1e-12)
        sw[i] = s
    return q, sw


def plane_loss(xyz, nrm, s1, op, nbr, idx, sig=0.5, rad=3.0, w_min=0.05):
    """抽樣顆粒 idx 的共面 loss（梯度只到 nrm；權重不帶梯度）。"""
    nb = nbr[idx].long()
    with torch.no_grad():
        w = _weights(xyz.detach(), nrm.detach(), s1, op, idx, nb, sig, rad)
        s = w.sum(1)
        ok = s > w_min
    ca = (nrm[nb] * nrm[idx, None]).sum(-1).abs()
    per = (w * (1 - ca)).sum(1) / s.clamp_min(1e-12) * op[idx]
    if not bool(ok.any()):
        return per.sum() * 0.0
    return per[ok].mean()


class DepthSupport:
    """逐顆累積「中心投到當前相機、對單目深度的相對殘差」；視窗結束時更新支撐。顆數變了：增加＝補在尾端，減少＝整個重來。
    rule："mean"＝視窗內殘差平均 < −delta_front 判「一直在前方」（原行為）；"vote"＝超過一半的觀測 < −delta_front（＝殘差中位數 < −delta_front）。
    （2026-11-02 全域共識量測：平均會被少數錯很大的視角拉走——遠離 SfM 誤判 37.5% vs 中位 15.3%、海面 20.8% vs 5.1%。）
    update 不做 nonzero／布林索引（2026-11-02：拿掉每步的 GPU 同步點；對全部 N 顆用遮罩累加，數學與舊版相同）。
    ★ 2026-11-11 behind 票（behind_margin 不是 None 時才開；關的時候 support 與之前逐位元相同）：觀測的 r 落在 (behind_margin, behind_max)
    ＝在表面「後面一點點」記一票；視窗結束時看過 >= min_count 次且 behind 票過半 => behind[i]＝True（地下層候選；只給 MCMC 抽母點用，
    不併進 support，SfM 豁免也不套用）。r >= behind_max（被別棟樓擋住、隔著樓看過去）算有效觀測但不投 behind 票
    ——11-08 視線一致性 v4 就是把「從大樓背面看過去」當證據才餓死牆面（`紀錄/實驗分析/33` §6）。兩個門檻都是相對深度（無因次）。"""

    def __init__(self, delta_front=0.05, delta_back=0.05, window=400, min_count=3, near=0.2, rule="mean",
                 behind_margin=None, behind_max=0.05):
        self.df, self.db, self.window, self.cmin, self.near = delta_front, delta_back, window, min_count, near
        if rule not in ("mean", "vote"):
            raise ValueError(f"depth_support_rule 只能是 mean／vote（收到 {rule!r}）")
        self.rule = rule
        self.bm = None if behind_margin is None else float(behind_margin)
        self.bmax = float(behind_max)
        if self.bm is not None and not (0.0 <= self.bm < self.bmax):
            raise ValueError(f"behind 窄帶要 0 <= margin < max（收到 {self.bm!r}、{self.bmax!r}）")
        self.sum = self.cnt = self.mn = self.vote = self.support = None
        self.bvote = self.behind = None
        self.t0 = None
        self.last = {}

    def _resize(self, n, device):
        if self.sum is None or self.sum.shape[0] > n:
            self.sum = torch.zeros(n, device=device)
            self.cnt = torch.zeros(n, device=device)
            self.vote = torch.zeros(n, device=device)
            self.mn = torch.full((n,), float("inf"), device=device)
            self.support = torch.ones(n, dtype=torch.bool, device=device)
            if self.bm is not None:
                self.bvote = torch.zeros(n, device=device)
                self.behind = torch.zeros(n, dtype=torch.bool, device=device)
            self.t0 = None
        elif self.sum.shape[0] < n:
            m = n - self.sum.shape[0]
            self.sum = torch.cat([self.sum, torch.zeros(m, device=device)])
            self.cnt = torch.cat([self.cnt, torch.zeros(m, device=device)])
            self.vote = torch.cat([self.vote, torch.zeros(m, device=device)])
            self.mn = torch.cat([self.mn, torch.full((m,), float("inf"), device=device)])
            self.support = torch.cat([self.support, torch.ones(m, dtype=torch.bool, device=device)])
            if self.bm is not None:
                self.bvote = torch.cat([self.bvote, torch.zeros(m, device=device)])
                self.behind = torch.cat([self.behind, torch.zeros(m, dtype=torch.bool, device=device)])

    @torch.no_grad()
    def update(self, step, means, w2c, fx, fy, cx, cy, width, height, inv_depth):
        n = means.shape[0]
        self._resize(n, means.device)
        if self.t0 is None:
            self.t0 = step
        p = torch.cat([means.detach(), torch.ones_like(means[:, :1])], 1) @ w2c
        z = p[:, 2]
        ok = z > self.near
        u = fx * p[:, 0] / z.clamp_min(1e-6) + cx
        v = fy * p[:, 1] / z.clamp_min(1e-6) + cy
        hd, wd = inv_depth.shape[-2:]
        ud = (u * (wd / float(width))).floor().long()
        vd = (v * (hd / float(height))).floor().long()
        ok &= (ud >= 0) & (ud < wd) & (vd >= 0) & (vd < hd)
        dinv = inv_depth.reshape(hd, wd)[vd.clamp(0, hd - 1), ud.clamp(0, wd - 1)]
        ok &= dinv > 1e-6
        zm = 1.0 / dinv.clamp_min(1e-6)
        r = torch.where(ok, (z - zm) / zm, torch.zeros_like(z))
        self.sum += r
        self.cnt += ok.to(self.cnt.dtype)
        self.vote += (ok & (r < -self.df)).to(self.vote.dtype)
        self.mn = torch.where(ok, torch.minimum(self.mn, r.abs()), self.mn)
        if self.bm is not None:
            self.bvote += (ok & (r > self.bm) & (r < self.bmax)).to(self.bvote.dtype)
        if step - self.t0 >= self.window:
            seen = self.cnt >= self.cmin
            if self.rule == "vote":
                front = seen & (self.vote > 0.5 * self.cnt)
            else:
                front = seen & (self.sum / self.cnt.clamp_min(1) < -self.df)
            never = seen & (self.mn > self.db)
            self.support = ~(front | never)
            self.last = {"seen": float(seen.float().mean()), "front": float(front.float().mean()),
                         "never": float(never.float().mean())}
            if self.bm is not None:
                self.behind = seen & (self.bvote > 0.5 * self.cnt)
                self.last["behind"] = float(self.behind.float().mean())
                self.bvote.zero_()
            self.sum.zero_(); self.cnt.zero_(); self.vote.zero_(); self.mn.fill_(float("inf")); self.t0 = step
            return True
        return False


def median_nn_distance(points, max_query: int = 200_000, seed: int = 0) -> float:
    """SfM 點最近鄰距離的中位數（點多時隨機抽 max_query 個去查整棵樹；固定 seed）。
    2026-11-08：把場景單位的門檻換成「SfM 間距的幾倍」用（Scaffold-GS 的體素＝初始點最近鄰距離中位）。"""
    from scipy.spatial import cKDTree
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.isfinite(P).all(axis=1)]
    if P.shape[0] < 2:
        raise ValueError(f"median_nn_distance 需要至少 2 個點（收到 {P.shape[0]}）")
    Q = P if P.shape[0] <= max_query else P[np.random.default_rng(seed).choice(P.shape[0], max_query, replace=False)]
    d, _ = cKDTree(P).query(Q, k=2)
    return float(np.median(d[:, 1]))


class SfMVoxels:
    """SfM 點的體素集合（體素邊長 r），查詢「顆粒附近有沒有 SfM 點」（2026-11-01：深度支撐的 SfM 豁免）。
    查自己與 26 個鄰格 => r 內的一定算到、最遠約 2√3·r（≈3.5r）也會算到。"""

    def __init__(self, points, r: float, device):
        P = torch.as_tensor(points, dtype=torch.float32, device=device).reshape(-1, 3)
        P = P[torch.isfinite(P).all(dim=-1)]
        self.r = float(r)
        self.n_points = int(P.shape[0])
        self.lo = P.min(dim=0).values - 2 * self.r
        self.keys = torch.unique(self._key(self._cell(P)))

    def _cell(self, X):
        return torch.floor((X.float() - self.lo) / self.r).long()

    @staticmethod
    def _key(c):
        return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]

    @torch.no_grad()
    def near(self, X: torch.Tensor) -> torch.Tensor:
        c0 = self._cell(X)
        out = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
        nk = self.keys.shape[0]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    c = c0 + torch.tensor([dx, dy, dz], device=X.device)
                    ok = ((c >= 0) & (c <= _M21)).all(dim=-1)
                    q = self._key(c.clamp(0, _M21))
                    i = torch.searchsorted(self.keys, q).clamp_max(nk - 1)
                    out |= ok & (self.keys[i] == q)
        return out


def sym3_eig(C: torch.Tensor):
    """[M,3,3] 對稱矩陣 => 特徵值 e1>=e2>=e3 與 e3 的特徵向量（解析式 Smith 1961，逐元素運算、不叫 cusolver）。
    幾乎等向（p≈0）時特徵向量任意 => 呼叫端用 e3/e2 判平坦會把它濾掉。
    11-01 修：叉積改用除過 p 的 (B − β3 I)（無因次、O(1)）。原本用 (A − e3 I) 直接叉積，鄰居幾乎重疊（MCMC 搬移／dup 初始化
    放在同一點）時 cov ~1e-11，叉積平方在 float32 下溢成 0 => 除以 clamp 1e-30 後 |v| 到 3.5e7（lab 朝向項 −221）。
    現在 |v| <= 1；退化（e2≈e3、叉積近 0）時 |v| < 0.5，呼叫端據此排除。入口先除以每個矩陣的最大元素（p 本身在 1e-20 也會下溢）。"""
    sc = C.abs().amax(dim=(1, 2)).clamp_min(1e-38)
    C = C / sc[:, None, None]
    a00, a11, a22 = C[:, 0, 0], C[:, 1, 1], C[:, 2, 2]
    a01, a02, a12 = C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]
    q = (a00 + a11 + a22) / 3
    p = torch.sqrt((((a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * (a01 ** 2 + a02 ** 2 + a12 ** 2)) / 6).clamp_min(1e-30))
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    det = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
    phi = torch.acos((det / 2).clamp(-1.0, 1.0)) / 3
    b3 = 2 * torch.cos(phi + 2.0943951023931953)
    e1 = q + 2 * p * torch.cos(phi)
    e3 = q + p * b3
    e2 = 3 * q - e1 - e3
    r0 = torch.stack([b00 - b3, b01, b02], -1)
    r1 = torch.stack([b01, b11 - b3, b12], -1)
    r2 = torch.stack([b02, b12, b22 - b3], -1)
    cs = torch.stack([torch.cross(r0, r1, dim=-1), torch.cross(r0, r2, dim=-1), torch.cross(r1, r2, dim=-1)], 1)   # [M,3,3]
    nn = cs.norm(dim=-1)
    v = cs.gather(1, nn.argmax(dim=1)[:, None, None].expand(-1, 1, 3))[:, 0]
    nv = v.norm(dim=-1, keepdim=True)
    v = torch.where(nv > 1e-6, v / nv.clamp_min(1e-12), torch.zeros_like(v))
    return e1 * sc, e2 * sc, e3 * sc, v


@torch.no_grad()
def surf_planes(xyz, op, nbr, rad_mult: float = 2.0, chunk: int = 1 << 19):
    """2026-11-01 鄰域平面約束（使用者 11-01：不渲染、用周遭高斯讓深度一致的表面平滑）。
    每顆 i：鄰居（不含自己、距離 <= rad_mult×鄰居距離中位）以 opacity 加權的 PCA 平面
    => 中心 c、法向 nh、面內尺度 r＝sqrt(e1)、平坦度 e3/e2、權重和 w（約等於「有幾顆不透明鄰居」）。
    退化鄰域（法向求不出來，或面內尺度 r 不到中位的 1%：鄰居幾乎疊在同一點，平面只是數值雜訊）平坦度記成 inf => 呼叫端的平坦門檻會排除。"""
    n = xyz.shape[0]
    dev = xyz.device
    C = torch.empty(n, 3, device=dev); NH = torch.empty(n, 3, device=dev)
    R = torch.empty(n, device=dev); F = torch.empty(n, device=dev); W = torch.empty(n, device=dev)
    for c0 in range(0, n, chunk):
        ii = torch.arange(c0, min(c0 + chunk, n), device=dev)
        nb = nbr[c0:c0 + chunk]
        P = xyz[nb].float()
        d = (P - xyz[ii].float()[:, None]).norm(dim=-1)
        w = op[nb].float() * (nb != ii[:, None]) * (d <= rad_mult * d.median(dim=1, keepdim=True).values)
        ws = w.sum(dim=1)
        c = (w[..., None] * P).sum(dim=1) / ws.clamp_min(1e-12)[:, None]
        D = P - c[:, None]
        cov = torch.einsum("mk,mki,mkj->mij", w, D, D) / ws.clamp_min(1e-12)[:, None, None]
        e1, e2, e3, v = sym3_eig(cov)
        C[ii], NH[ii], W[ii] = c, v, ws
        r = torch.sqrt(e1.clamp_min(0))
        R[ii] = r
        good = (v.norm(dim=-1) > 0.5) & (r >= 0.01 * r.median())
        F[ii] = torch.where(good, e3.clamp_min(0) / e2.clamp_min(1e-30), torch.full_like(r, float("inf")))
    return C, NH, R, F, W


def surf_loss(xyz_i, nrm_i, c_i, nh_i, r_i, tau: float):
    """離鄰域平面的距離（以面內尺度 r 為單位）取截斷絕對值 min(d, tau)：tau 以上當真結構、梯度 0（不抹細節）；
    朝向項 1 − |n_i . nh|（|cos|：90 度附近梯度最大）。回傳 (位置項, 朝向項, 被截斷比例)。"""
    d = ((xyz_i - c_i) * nh_i).sum(dim=-1).abs() / r_i.clamp_min(1e-8)
    lp = d.clamp(max=tau).mean()
    lo = (1 - (nrm_i * nh_i).sum(dim=-1).abs()).mean()
    return lp, lo, float((d.detach() >= tau).float().mean())
