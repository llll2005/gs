"""誤差低的 tile 少算（2026-11-08 使用者 10-09 的 VRS 想法：「減少 tile 就能減少時間」「判準會很大程度影響結果，多作幾個假設」）。

一步裡 80% 的時間在逐像素混合（`紀錄/實驗分析/28`）=> 每步只渲染一部分 tile 就直接省時間（光柵器的 pix_skip：被跳過的像素一開始就 done，
整塊都跳過時那個 tile 的迴圈立刻結束；backward 的 bwd_zero_grad_skip 也跳過梯度全 0 的像素）。
每個 tile 以機率 π_t 保留；保留的 tile 梯度乘 1/π_t（期望值不變＝無偏），跳過的 loss 與梯度都是 0。
π_t 的判準（都是無因次：分數除以本張平均，再夾到 [min_keep, 1]、平均調到 keep）：
  random   π＝keep（對照：只少算、不挑）
  error    分數＝這張上次被抽到時的逐 tile L1（第一次沒有紀錄 => 全畫）
  texture  分數＝GT 逐 tile 的梯度強度（平坦的海面／屋頂多跳過；不需要訓練訊號）
出處：🟨 可變速率著色（VRS）是即時繪圖的做法、重要度抽樣是蒙地卡羅的標準做法；拿來挑 3DGS 訓練要渲染的 tile ⚠ 未查證是否有人做過。
"""
import torch
import torch.nn.functional as F


def tile_mean(x: torch.Tensor, T: int) -> torch.Tensor:
    """x [H, W] => 逐 tile 平均 [ceil(H/T), ceil(W/T)]（邊緣不足一格的照實際像素數平均）。"""
    H, W = x.shape
    s = F.avg_pool2d(x[None, None].float(), T, T, ceil_mode=True)[0, 0]
    n = F.avg_pool2d(torch.ones_like(x, dtype=torch.float32)[None, None], T, T, ceil_mode=True)[0, 0]
    return s / n.clamp_min(1e-12)


def texture_score(gt: torch.Tensor, T: int) -> torch.Tensor:
    """gt [3, H, W]（0～1）=> 逐 tile 平均梯度強度 |dx|+|dy|。"""
    g = gt.float().mean(0)
    d = torch.zeros_like(g)
    d[:, :-1] += (g[:, 1:] - g[:, :-1]).abs()
    d[:-1, :] += (g[1:, :] - g[:-1, :]).abs()
    return tile_mean(d, T)


def keep_probs(score: torch.Tensor, keep: float, min_keep: float, iters: int = 30) -> torch.Tensor:
    """π＝clamp(k·s/mean(s), min_keep, 1)，二分 k 讓 mean(π)＝keep。score 全 0／非有限 => π＝keep。"""
    keep = float(min(max(keep, min_keep), 1.0))
    s = score.float()
    m = s[torch.isfinite(s)].mean() if torch.isfinite(s).any() else s.new_tensor(0.0)
    if not torch.isfinite(m) or float(m) <= 0:
        return torch.full_like(s, keep)
    r = torch.nan_to_num(s / m, nan=1.0, posinf=1e6).clamp_min(0)
    lo, hi = 0.0, 1.0
    while float((hi * r).clamp(min_keep, 1).mean()) < keep and hi < 1e8:
        hi *= 2
    for _ in range(iters):
        k = 0.5 * (lo + hi)
        if float((k * r).clamp(min_keep, 1).mean()) < keep:
            lo = k
        else:
            hi = k
    return (hi * r).clamp(min_keep, 1)


def tile_to_pix(t: torch.Tensor, H: int, W: int, T: int) -> torch.Tensor:
    """[Ht, Wt] => [H, W]（每格展開成 T x T、裁到影像大小）。"""
    return t.repeat_interleave(T, 0).repeat_interleave(T, 1)[:H, :W]


def scale_grad(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """前向值不變、梯度乘 w（逐像素重要度權重；L1／SSIM／L2 都不用改）。"""
    xd = x.detach()
    return xd + w * (x - xd)
