"""★ 2026-11-08 依覆蓋度加權抽訓練視角（DataModule `view_balance`；使用者 11-08「高處側面擬合失敗、樓頂正常」）。

量測（`紀錄/實驗分析/33` §6；b3、CityGS 不透明表面、不算遮擋）：
  高處立面 > 250 m 只在 63／899 台訓練相機的畫面內（5~50 m 252 台）=> 均勻抽視角時每步被看到的機率只有低處的 1/4；
  我方 > 250 m 立面顆粒朝向對齊 15k／30k／60k＝25／40／50%（50~250 m 65%）=> 隨更新次數變好，瓶頸是「被更新幾次」。
  另外 b3 是角落塊，清單裡不少相機主要拍的是塊外 => 那些步對本塊模型幾乎沒貢獻。
做法：init 點雲（本塊裁切的那份）抽 n 點；c(p)＝p 在幾台訓練相機的畫面內（z > 0.2、投影在影像內；不算遮擋，跟深度支撐閘門同一套投影）；
  視角權重 w_v＝Σ_{p 在 v 內} c(p)^-γ（每個點把一份「需求」分給看得到它的相機；少有人看的點分得多），除以平均後夾到 [lo, hi]。
  γ＝0 => w_v ∝ 畫面內的 init 點數（只做「塊內內容多的視角多抽」）；γ 越大越偏向少有人看的表面。
離線模擬（b3、γ 1、夾 0.25~4）：塊內各類表面每步被看到的機率 ×1.3~1.8、> 250 m 立面 ×1.58；有效視角數 645／899。
"""
import numpy as np
import torch


@torch.no_grad()
def coverage_view_weights(cameras, xyz, gamma: float = 1.0, clip=(0.25, 4.0), n_points: int = 60000, near: float = 0.2, seed: int = 0):
    """cameras：Cameras（批次，有 world_to_camera（轉置）、fx、fy、cx、cy、width、height）；xyz：(P, 3) 點雲。
    回傳 (weights [n_cams] float32、統計 dict)。點雲為空或沒有任何點被看到 => 全 1（等於均勻抽）。"""
    n_cam = int(cameras.R.shape[0])
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) == 0:
        return torch.ones(n_cam), {"ok": False, "reason": "點雲是空的"}
    rng = np.random.default_rng(seed)
    if len(xyz) > n_points:
        xyz = xyz[rng.choice(len(xyz), n_points, replace=False)]
    X = torch.cat([torch.from_numpy(xyz).float(), torch.ones(len(xyz), 1)], 1)
    vis = torch.zeros(n_cam, len(xyz), dtype=torch.bool)
    for i in range(n_cam):
        p = X @ cameras.world_to_camera[i].float()
        z = p[:, 2]
        u = float(cameras.fx[i]) * p[:, 0] / z.clamp_min(1e-6) + float(cameras.cx[i])
        v = float(cameras.fy[i]) * p[:, 1] / z.clamp_min(1e-6) + float(cameras.cy[i])
        vis[i] = (z > near) & (u >= 0) & (u < float(cameras.width[i])) & (v >= 0) & (v < float(cameras.height[i]))
    c = vis.sum(0)
    keep = c > 0
    if not bool(keep.any()):
        return torch.ones(n_cam), {"ok": False, "reason": "沒有任何點在任何相機的畫面內"}
    w = (vis[:, keep].float() * c[keep].float().pow(-float(gamma))).sum(1)
    if float(w.sum()) <= 0:
        return torch.ones(n_cam), {"ok": False, "reason": "權重全為 0"}
    w = w / w.mean()
    lo, hi = float(clip[0]), float(clip[1])
    n_lo, n_hi = int((w < lo).sum()), int((w > hi).sum())
    w = w.clamp(lo, hi)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return w, {"ok": True, "n_points": int(len(xyz)), "seen_points": int(keep.sum()), "c_median": float(c[keep].float().median()),
               "ess": ess, "w_max": float(w.max()), "w_min": float(w.min()), "n_clip_lo": n_lo, "n_clip_hi": n_hi,
               "zero_view": int((vis.sum(1) == 0).sum())}


def epoch_order(n: int, generator, weights=None):
    """一個 epoch 的抽樣順序：沒有權重 => randperm（原行為）；有權重 => 依權重有放回抽 n 次（epoch 長度不變）。"""
    if weights is None:
        return torch.randperm(n, generator=generator).tolist()
    return torch.multinomial(weights.float(), n, replacement=True, generator=generator).tolist()
