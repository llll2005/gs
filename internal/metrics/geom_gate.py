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
    """逐顆累積「中心投到當前相機、對單目深度的相對殘差」；視窗結束時更新支撐。顆數變了：增加＝補在尾端，減少＝整個重來。"""

    def __init__(self, delta_front=0.05, delta_back=0.05, window=400, min_count=3, near=0.2):
        self.df, self.db, self.window, self.cmin, self.near = delta_front, delta_back, window, min_count, near
        self.sum = self.cnt = self.mn = self.support = None
        self.t0 = None
        self.last = {}

    def _resize(self, n, device):
        if self.sum is None or self.sum.shape[0] > n:
            self.sum = torch.zeros(n, device=device)
            self.cnt = torch.zeros(n, device=device)
            self.mn = torch.full((n,), float("inf"), device=device)
            self.support = torch.ones(n, dtype=torch.bool, device=device)
            self.t0 = None
        elif self.sum.shape[0] < n:
            m = n - self.sum.shape[0]
            self.sum = torch.cat([self.sum, torch.zeros(m, device=device)])
            self.cnt = torch.cat([self.cnt, torch.zeros(m, device=device)])
            self.mn = torch.cat([self.mn, torch.full((m,), float("inf"), device=device)])
            self.support = torch.cat([self.support, torch.ones(m, dtype=torch.bool, device=device)])

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
        idx = ok.nonzero(as_tuple=True)[0]
        dinv = inv_depth.reshape(hd, wd)[vd[idx], ud[idx]]
        good = dinv > 1e-6
        idx, dinv = idx[good], dinv[good]
        zm = 1.0 / dinv
        r = (z[idx] - zm) / zm
        self.sum.index_add_(0, idx, r)
        self.cnt.index_add_(0, idx, torch.ones_like(r))
        self.mn[idx] = torch.minimum(self.mn[idx], r.abs())
        if step - self.t0 >= self.window:
            seen = self.cnt >= self.cmin
            front = seen & (self.sum / self.cnt.clamp_min(1) < -self.df)
            never = seen & (self.mn > self.db)
            self.support = ~(front | never)
            self.last = {"seen": float(seen.float().mean()), "front": float(front.float().mean()),
                         "never": float(never.float().mean())}
            self.sum.zero_(); self.cnt.zero_(); self.mn.fill_(float("inf")); self.t0 = step
            return True
        return False
