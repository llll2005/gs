#!/usr/bin/env python
"""★ 11-21 純數學化簡 #4＋#5（MCMC 位置雜訊）微基準（不訓練）：真實 ckpt 的參數分布，同 process 交替。
現行（fast_noise＋noise_gate_eps 1e-3）：sigmoid(全部 N) -> 門檻 -> nonzero（GPU->CPU 同步）-> 子集建 R -> R (s^2 ⊙ (R^T v)) 兩次 bmm -> index_add
A（同分布、有子集）：raw < −2.526 選子集（不算 sigmoid）-> 只取兩個常態亂數 -> s1^2 v1 t1 + s2^2 v2 t2（t1, t2 只從四元數算兩欄）
B（同分布、無子集、無同步）：全部 N 用遮罩（gate 乘上去），不呼叫 nonzero
"""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch

CK = sys.argv[1] if len(sys.argv) > 1 else "outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt"
dev = torch.device("cuda")
sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
gp = {k.split(".")[-1]: v.to(dev) for k, v in sd.items() if k.startswith("gaussian_model.gaussians.")}
means = gp["means"].clone(); rots = gp["rotations"]; sraw = gp["scales"]; oraw = gp["opacities"]
N = means.shape[0]
LR, NOISE_LR, EPS = 1.6e-5, 5e5, 1e-3
op_sigmoid = lambda x: 1 / (1 + torch.exp(-100 * (x - 0.995)))
TAU_O = 0.005 - math.log(EPS / (1 - EPS)) / 100
TAU_RAW = math.log(TAU_O / (1 - TAU_O))


def build_R(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def two_cols(q):   # R 的前兩欄（切向量 t1, t2）
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    t1 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    t2 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    return t1, t2


def current():
    o = torch.sigmoid(oraw)
    g_all = op_sigmoid(1.0 - o).squeeze(-1)
    sel = torch.nonzero(g_all > EPS, as_tuple=True)[0]
    sc = torch.exp(sraw); sc = torch.cat([sc, torch.zeros_like(sc[:, :1])], -1)
    R = build_R(rots[sel])
    o_s = o[sel]; sc_s = sc[sel]
    noise = torch.randn((o_s.shape[0], 3), device=dev) * op_sigmoid(1 - o_s) * NOISE_LR * LR
    noise = torch.bmm(R.transpose(1, 2), noise.unsqueeze(-1)).squeeze(-1)
    noise = noise * (sc_s ** 2)
    noise = torch.bmm(R, noise.unsqueeze(-1)).squeeze(-1)
    means.index_add_(0, sel, noise)


def math_sub():
    sel = torch.nonzero(oraw.squeeze(-1) < TAU_RAW, as_tuple=True)[0]
    q = rots[sel]; s2 = torch.exp(2 * sraw[sel]); xr = oraw[sel]
    gate = op_sigmoid(1 - torch.sigmoid(xr)) * (NOISE_LR * LR)
    t1, t2 = two_cols(q)
    v = torch.randn((sel.shape[0], 2), device=dev)
    noise = gate * (v[:, :1] * s2[:, :1] * t1 + v[:, 1:] * s2[:, 1:] * t2)
    means.index_add_(0, sel, noise)


def math_dense():
    gate = op_sigmoid(1 - torch.sigmoid(oraw)) * (NOISE_LR * LR) * (oraw < TAU_RAW)
    s2 = torch.exp(2 * sraw)
    t1, t2 = two_cols(rots)
    v = torch.randn((N, 2), device=dev)
    means.add_(gate * (v[:, :1] * s2[:, :1] * t1 + v[:, 1:] * s2[:, 1:] * t2))


def timed(fn, busy):
    # busy＝先排一段 ~20 ms 的 GPU 工作（模擬 backward 還在跑），量「CPU 從呼叫到返回」與「GPU 完成」=> nonzero 的同步會讓 CPU 卡住等前面的工作
    if busy:
        x = torch.randn(3000, 3000, device=dev)
        for _ in range(6):
            x = x @ x * 1e-3
    torch.cuda.synchronize() if not busy else None
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    import time
    t0 = time.perf_counter(); e0.record(); fn(); e1.record(); t1 = time.perf_counter()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1), (t1 - t0) * 1e3


n_sel = int((torch.sigmoid(oraw) < TAU_O).sum())
print(f"N={N:,}｜會動的顆粒（gate > 1e-3）{n_sel:,}（{n_sel / N:.2%}）")
fns = {"現行": current, "化簡：有子集": math_sub, "化簡：無子集無同步": math_dense}
for busy in (False, True):
    for f in fns.values():
        for _ in range(5): timed(f, busy)
    R = {k: [] for k in fns}
    for r in range(60):
        for k, f in (list(fns.items()) if r % 2 == 0 else list(fns.items())[::-1]):
            R[k].append(timed(f, busy))
    tag = "前面有 20 ms 的 GPU 工作在排隊" if busy else "GPU 閒置"
    print(f"── {tag}（GPU 時間＝這段自己的 kernel；CPU 卡住＝呼叫到返回的牆鐘）")
    for k, v in R.items():
        v = np.array(v)
        print(f"   {k:<16} GPU {np.median(v[:, 0]):6.3f} ms｜CPU 卡住 {np.median(v[:, 1]):6.3f} ms")
