"""SfM 近鄰平面給面元朝向（2026-11-08 使用者「高處沒轉向」「用 SfM 近鄰擬平面給初始朝向，作用在 MCMC 增生和合併都試試」）。

init 的旋轉是隨機的（`gaussian_2d.py` 用 torch.rand），朝向全靠訓練轉過來；高處立面被看到的次數少（b3 > 250 m 只 63／899 台），
30k 時只有 40% 對齊 CityGS 立面。這裡用可靠 SfM 點的 k 近鄰 PCA 平面當起點：
  - 每個 SfM 點：k 近鄰（含自己）PCA，法向＝最小特徵值的特徵向量，平坦＝λ3/λ2 < flat_tol，鄰域半徑＝到第 k 個鄰居的距離；
  - 查詢點：最近的 SfM 點在它的鄰域半徑內、而且那個鄰域平坦 => 有效，法向取那個 SfM 點的；否則不動。
沒有場景單位的常數：只有近鄰數 k 與無因次的平坦度（同 merge_ops／surf_planes 的 0.1）；「離 SfM 多近」用該點自己的近鄰半徑判。
出處：🟨 k 近鄰 PCA 估法向是點雲處理的標準做法（3DGS 的 init 尺度也用 SfM 近鄰距離）；拿來設 2DGS 面元初始朝向與 MCMC 新點朝向 ⚠ 未查證是否有人做過。
"""
import math

import numpy as np
import torch


class SfMNormals:
    def __init__(self, P, k: int = 16, flat_tol: float = 0.1, chunk: int = 1 << 18):
        from scipy.spatial import cKDTree
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        P = P[np.isfinite(P).all(axis=1)]
        if P.shape[0] <= k:
            raise ValueError(f"SfMNormals 需要多於 k={k} 個點（收到 {P.shape[0]}）")
        self.P, self.k, self.flat_tol = P, int(k), float(flat_tol)
        self.tree = cKDTree(P)
        d, nb = self.tree.query(P, k=self.k)                                  # 含自己
        self.rad = d[:, -1]
        self.n = np.zeros_like(P, dtype=np.float32)
        self.flat = np.zeros(P.shape[0], dtype=bool)
        for s in range(0, P.shape[0], chunk):
            X = P[nb[s:s + chunk]]                                            # [c, k, 3]
            X = X - X.mean(axis=1, keepdims=True)
            C = np.einsum("cki,ckj->cij", X, X) / self.k
            w, v = np.linalg.eigh(C)                                          # 升冪：w0 最小
            self.n[s:s + chunk] = v[:, :, 0]
            self.flat[s:s + chunk] = w[:, 0] < self.flat_tol * np.maximum(w[:, 1], 1e-30)

    def query(self, X):
        """X [m,3] => (法向 [m,3] float32, 有效 [m] bool)。"""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        d, i = self.tree.query(X, k=1)
        ok = (d <= self.rad[i]) & self.flat[i]
        return self.n[i], ok


def quat_with_normal(n: torch.Tensor, q_old: torch.Tensor) -> torch.Tensor:
    """把面元法向（旋轉矩陣第 3 欄）轉到 n，保留舊朝向在平面內的方向（舊的第 1 軸投到新平面；退化時換第 2 軸）。回傳 (w,x,y,z)。"""
    from .merge_ops import _qmat, _mat2q
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    R0 = _qmat(q_old.float())
    t = R0[:, :, 0] - (R0[:, :, 0] * n).sum(-1, keepdim=True) * n
    t2 = R0[:, :, 1] - (R0[:, :, 1] * n).sum(-1, keepdim=True) * n
    t = torch.where(t.norm(dim=-1, keepdim=True) > 1e-3, t, t2)
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = torch.cross(n, t, dim=-1)
    return _mat2q(torch.stack([t, b, n], dim=-1))


@torch.no_grad()
def snap_rotations(gaussian_model, sn: SfMNormals, idx=None) -> dict:
    """把 idx（None＝全部）裡有效的顆粒法向轉到 SfM 近鄰平面；回傳統計（有效數、其中立面、平均轉了幾度）。"""
    g = gaussian_model.gaussians
    rot = g["rotations"]
    if idx is None:
        idx = torch.arange(rot.shape[0], device=rot.device)
    if idx.numel() == 0:
        return {"n": 0, "ok": 0, "facade": 0, "deg": float("nan")}
    X = g["means"][idx].detach().float().cpu().numpy()
    nn, ok = sn.query(X)
    ok_t = torch.from_numpy(ok).to(rot.device)
    sel = idx[ok_t]
    if sel.numel() == 0:
        return {"n": int(idx.numel()), "ok": 0, "facade": 0, "deg": float("nan")}
    n_t = torch.from_numpy(nn[ok]).to(rot.device).float()
    from .merge_ops import _qmat
    old_n = _qmat(rot[sel].float())[:, :, 2]
    deg = float(torch.rad2deg(torch.arccos((old_n * n_t).sum(-1).abs().clamp(max=1.0))).mean())
    rot[sel] = quat_with_normal(n_t, rot[sel]).to(rot.dtype)
    return {"n": int(idx.numel()), "ok": int(sel.numel()), "facade": int((n_t[:, 2].abs() < 0.3).sum()), "deg": deg}
