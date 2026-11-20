#!/usr/bin/env python
"""★ 11-21 純數學化簡 #3＋#6＋#7＋#8 的微基準（不訓練）：逐顆正則（opacity L1、尺度、二元熵）＋塊外遮罩合成，
現行寫法 vs 融合寫法（一個自訂 autograd：熵用 softplus 閉式、權重 0 的 opacity L1 不算、exp 之後不 abs、遮罩合成用 where）。
同大小（N 顆、3xHxW 影像）、同 process 交替、CUDA event 計時 forward＋backward。
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1_500_000)
ap.add_argument("--H", type=int, default=900)
ap.add_argument("--W", type=int, default=1600)
ap.add_argument("--rounds", type=int, default=200)
a = ap.parse_args()
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
raw_o = (torch.randn(a.n, 1, device=dev, generator=g) * 3).requires_grad_(True)
raw_s = (torch.randn(a.n, 2, device=dev, generator=g) - 6).requires_grad_(True)
vis = torch.rand(a.n, device=dev, generator=g) < 0.6
img_leaf = torch.rand(3, a.H, a.W, device=dev, generator=g).requires_grad_(True)
m = torch.randint(0, 3, (a.H, a.W), device=dev, generator=g)          # 0=C、1=A、2=B
gt = torch.rand(3, a.H, a.W, device=dev, generator=g)
bg = torch.rand(3, device=dev, generator=g)
EW, SW = 0.008, 0.007


def current():
    r = img_leaf * 1.0                                                   # 代表渲染輸出（有梯度）
    Bm = (m == 2)[None]; Cm = (m == 0)[None]
    g2 = torch.where(Bm, bg.view(3, 1, 1).expand_as(gt), gt)
    g2 = torch.where(Cm, r.detach(), g2)
    c = Cm.to(r.dtype)
    rr = r * (1 - c) + r.detach() * c
    loss = (rr - g2).abs().mean()
    o = torch.sigmoid(raw_o).squeeze(-1)
    non_immune = o.detach() <= 0.9
    rvf = vis.to(o.dtype)
    niw = non_immune.to(o.dtype) * rvf
    opacity_reg = (o * niw).sum() / niw.sum().clamp_min(1.0)
    sc = torch.exp(raw_s).abs()
    scale_reg = (sc * rvf[:, None]).sum() / (rvf.sum().clamp_min(1.0) * sc.shape[1])
    loss = loss + 0.0 * opacity_reg + SW * scale_reg
    oc = o.clamp(1e-6, 1 - 1e-6)
    Hi = -(oc * torch.log(oc) + (1 - oc) * torch.log(1 - oc))
    Hi = Hi * (oc.detach() < 0.5)
    H = (Hi * rvf).sum() / rvf.sum().clamp_min(1.0)
    loss = loss + EW * H
    loss.backward()


class FusedReg(torch.autograd.Function):
    """loss = EW·Σ_vis H(σ(x))·[x<0] / n_vis + SW·Σ_vis exp(s) / (2 n_vis)；H(σ(x)) = softplus(x) − x·σ(x)、dH/dx = −x·σ(1−σ)。"""
    @staticmethod
    def forward(ctx, x, s, w, ew, sw):
        o = torch.sigmoid(x)
        wsum = w.sum().clamp_min(1.0)
        mw = w * (x < 0).to(x.dtype)
        sc = torch.exp(s)
        val = ew * ((F.softplus(x) - x * o) * mw).sum() / wsum + sw * (sc * w).sum() / (wsum * s.shape[1])
        ctx.save_for_backward(x, o, mw, sc, w, wsum)
        ctx.ew, ctx.sw = ew, sw
        return val

    @staticmethod
    def backward(ctx, gout):
        x, o, mw, sc, w, wsum = ctx.saved_tensors
        gx = (gout * ctx.ew / wsum) * (-x * o * (1 - o)) * mw
        gs = (gout * ctx.sw / (wsum * sc.shape[1])) * sc * w
        return gx, gs, None, None, None


def fused():
    r = img_leaf * 1.0
    Bm = (m == 2)[None]; Cm = (m == 0)[None]
    g2 = torch.where(Bm, bg.view(3, 1, 1).expand_as(gt), gt)
    g2 = torch.where(Cm, r.detach(), g2)
    rr = torch.where(Cm, r.detach(), r)
    loss = (rr - g2).abs().mean()
    loss = loss + FusedReg.apply(raw_o, raw_s, vis.to(raw_o.dtype)[:, None], EW, SW)
    loss.backward()


def timed(fn):
    for p in (raw_o, raw_s, img_leaf):
        p.grad = None
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)


# 數值檢查：梯度一致（熵在夾值範圍內）
for p in (raw_o, raw_s, img_leaf):
    p.grad = None
current(); g_cur = [p.grad.clone() for p in (raw_o, raw_s, img_leaf)]
for p in (raw_o, raw_s, img_leaf):
    p.grad = None
fused(); g_fus = [p.grad.clone() for p in (raw_o, raw_s, img_leaf)]
inside = (torch.sigmoid(raw_o) > 1e-6) & (torch.sigmoid(raw_o) < 1 - 1e-6)
print(f"梯度一致性：opacity（夾值內）max 相對差 {float(((g_cur[0] - g_fus[0]).abs()[inside]).max() / g_cur[0].abs().max()):.2e}｜"
      f"scale {float((g_cur[1] - g_fus[1]).abs().max() / g_cur[1].abs().max()):.2e}｜影像 {'逐位元相同' if torch.equal(g_cur[2], g_fus[2]) else '不同'}")
for _ in range(10):
    timed(current); timed(fused)
T = {0: [], 1: []}
for r in range(a.rounds):
    for k in ((0, 1) if r % 2 == 0 else (1, 0)):
        T[k].append(timed(current if k == 0 else fused))
t0, t1 = np.array(T[0]), np.array(T[1])
print(f"現行 中位 {np.median(t0):.3f} ms｜融合 {np.median(t1):.3f} ms｜成對差 {np.median((t1 - t0)):.3f} ms（{np.median((t1 - t0) / t0) * 100:+.1f}%）")
