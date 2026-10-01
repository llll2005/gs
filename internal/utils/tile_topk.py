"""逐 tile 名次（2026-10-01）：每顆在所有 (視角, tile) 裡拿到的**最好名次**。

用途：「每個 tile 只留前 K 名」（使用者提的 tile 局部背包）。
  - tools/tile_topk_prune.py：事後剪枝曲線／產生剪枝 ckpt
  - SepDepthTrim2DGSRenderer(trim_by_tile_topk=True)：訓練期 trim 判準（剪最好名次最差的那批）

近似（不改光柵器 kernel）：kernel 只回傳「每顆在一個視角的總貢獻」，沒有逐 tile 的值 ⇒
  把該顆在該視角的總貢獻平均分到它覆蓋的 tile（外接盒＝投影中心 ± radii，與光柵器 getRect 同規則），
  在每個 tile 內排名。⚠ 會高估邊緣 tile 的分數；conic 開時光柵器用的是不對稱盒，這裡的對稱盒偏大。

記憶體：先依分數把顆粒排好再展開成 (顆, tile) 對，然後對 tile 鍵做**穩定**排序 ⇒ 同 tile 內自動保持
  分數由高到低，只需一次排序、int32 tile 鍵（訓練期鎖 5.66 GB，Σ對數可達千萬級）。
"""
import torch

TILE = 16
NEVER = 1 << 30


def update_best_rank(best_rank: torch.Tensor, contrib: torch.Tensor, radii: torch.Tensor,
                     xyz: torch.Tensor, camera) -> int:
    """就地更新 best_rank（int64，長度 N，初值 NEVER）。回傳本視角的 (顆, tile) 對數。

    contrib：該視角每顆的總貢獻 Σ T·α（N）；radii：光柵器回傳的半徑（N）；xyz：世界座標（N,3）。
    """
    dev = xyz.device
    W, H = int(camera.width), int(camera.height)
    gx, gy = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    contrib = contrib.reshape(-1)
    radii = radii.reshape(-1)
    vis = torch.nonzero((radii > 0) & (contrib > 0)).squeeze(1)
    if vis.numel() == 0:
        return 0
    ph = torch.cat([xyz[vis], torch.ones(vis.numel(), 1, device=dev, dtype=xyz.dtype)], 1) @ camera.full_projection
    w = ph[:, 3].clamp_min(1e-7)
    px = ((ph[:, 0] / w + 1.0) * W - 1.0) * 0.5
    py = ((ph[:, 1] / w + 1.0) * H - 1.0) * 0.5
    r = radii[vis].float()
    x0 = ((px - r) / TILE).floor().clamp(0, gx).to(torch.int32)
    x1 = ((px + r + TILE - 1) / TILE).floor().clamp(0, gx).to(torch.int32)
    y0 = ((py - r) / TILE).floor().clamp(0, gy).to(torch.int32)
    y1 = ((py + r + TILE - 1) / TILE).floor().clamp(0, gy).to(torch.int32)
    nx, nt = x1 - x0, (x1 - x0) * (y1 - y0)
    keep = nt > 0
    vis, x0, y0, nx, nt = vis[keep], x0[keep], y0[keep], nx[keep], nt[keep]
    if vis.numel() == 0:
        return 0
    score = contrib[vis] / nt.float()
    o = torch.argsort(score, descending=True, stable=True)          # 先依分數排好
    vis, x0, y0, nx, nt = vis[o], x0[o], y0[o], nx[o], nt[o]
    del score, o
    total = int(nt.sum())
    idx = torch.repeat_interleave(torch.arange(vis.numel(), device=dev, dtype=torch.int32), nt)
    start = (torch.cumsum(nt, 0) - nt)
    local = torch.arange(total, device=dev, dtype=torch.int32) - start[idx]
    nxi = nx[idx]
    tile = (y0[idx] + local // nxi) * gx + (x0[idx] + local % nxi)
    del local, nxi, start
    order = torch.argsort(tile, stable=True)                         # 同 tile 內保持分數由高到低
    t_sorted = tile[order]
    del tile
    _, counts = torch.unique_consecutive(t_sorted, return_counts=True)
    seg_start = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    rank = torch.arange(total, device=dev, dtype=torch.int64) - seg_start
    best_rank.scatter_reduce_(0, vis[idx[order].long()], rank, reduce="amin")
    return total
