#!/usr/bin/env python
"""★ 11-21 純數學化簡 #1：2DGS 射線-面元交點的「反單應」寫法，數值驗證（CPU、不訓練）。

光柵器現行（2DGS Eq. 8–10）每個 (像素, 面元) 配對：
    k = -Tu + x Tw,  l = -Tv + y Tw,  p = k x l,  s = (p.x/p.z, p.y/p.z),  depth = s.x Tw.x + s.y Tw.y + Tw.z
其中 Tu, Tv, Tw 是 transMat 的三列：Tu·h = x Z、Tv·h = y Z、Tw·h = Z，h = (u, v, 1)。
化簡：令 M = [Tu; Tv; Tw]（列），Q = M^{-1}（每顆面元在 preprocess 算一次）。則
    p' = Q (x, y, 1)^T = x q0 + y q1 + q2       （仿射；6 個 FMA）
    Z  = 1 / p'.z                              （深度＝本來就要的那個倒數）
    s  = (p'.x Z, p'.y Z)
證明：k x l = Tu x Tv + x (Tv x Tw) + y (Tw x Tu)（Tw x Tw = 0）＝ det(M) · Q (x, y, 1)；
      Tw · (k x l) = det(M)（另外兩項是 Tw 與自己的外積再點積 = 0）=> depth = det/p.z = 1/p'.z。
梯度：每配對只需 dL/dp'（3 個數），dL/dQ = dL/dp' ⊗ (x, y, 1) 照常逐配對歸約（9 個數，和現在 dL/dT 一樣多）；
      每顆面元一次 dL/dM = -Q^T (dL/dQ) Q^T（反矩陣的微分），取代每配對的兩次外積。

本腳本：用真實 ckpt（b4）＋真實相機，float64 當真值，比較 float32 下「原寫法」與「新寫法」的誤差；再用 autograd 驗證梯度鏈。
用法：python tools/probe/invhomog_verify_1121.py [--ckpt ...] [--n 20000]
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from internal.utils.colmap import read_images_binary, read_cameras_binary, qvec2rotmat

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_4_nogate/checkpoints/view_block_4_nogate.ckpt")
ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08/000_001.txt")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--cams", type=int, default=3)
ap.add_argument("--dsf", type=float, default=1.2)
a = ap.parse_args()
rng = np.random.default_rng(0)

sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
g = {k.split(".")[-1]: v for k, v in sd.items() if k.startswith("gaussian_model.gaussians.")}
mu = g["means"].double().numpy()
sc = np.exp(g["scales"].double().numpy())
q = g["rotations"].double().numpy(); q /= np.linalg.norm(q, axis=1, keepdims=True)
op = 1 / (1 + np.exp(-g["opacities"].double().numpy()[:, 0]))


def quat_to_rot(q):            # 同 glm quat_to_rotmat（w, x, y, z）
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


Rg = quat_to_rot(q)
imgs = read_images_binary(os.path.join(a.data, "sparse/0/images.bin"))
cams = read_cameras_binary(os.path.join(a.data, "sparse/0/cameras.bin"))
names = [l.strip() for l in open(os.path.join(a.data, a.part)) if l.strip()]
byname = {v.name: v for v in imgs.values()}
pick = [byname[n] for n in names[:: max(1, len(names) // a.cams)][: a.cams] if n in byname]

f32 = np.float32
rows = []
for im in pick:
    cam = cams[im.camera_id]
    fx, fy, cx, cy = [p / a.dsf for p in cam.params[:4]]
    Wd, Hd = int(cam.width / a.dsf), int(cam.height / a.dsf)
    Rwc = qvec2rotmat(im.qvec); t = np.asarray(im.tvec)
    pv = mu @ Rwc.T + t
    xs = fx * pv[:, 0] / pv[:, 2] + cx; ys = fy * pv[:, 1] / pv[:, 2] + cy
    vis = np.where((pv[:, 2] > 0.2) & (xs > 0) & (xs < Wd) & (ys > 0) & (ys < Hd) & (op > 1 / 255))[0]
    idx = rng.choice(vis, size=min(a.n, len(vis)), replace=False)
    # 相機座標的面元：c0 = W R0 s0、c1 = W R1 s1、c2 = 中心
    c0 = (Rg[idx][:, :, 0] @ Rwc.T) * sc[idx, 0:1]; c1 = (Rg[idx][:, :, 1] @ Rwc.T) * sc[idx, 1:2]; c2 = pv[idx]
    C = np.stack([c0, c1, c2], axis=2)                                   # 每顆 3x3，欄＝c0 c1 c2
    Tu = fx * C[:, 0, :] + cx * C[:, 2, :]; Tv = fy * C[:, 1, :] + cy * C[:, 2, :]; Tw = C[:, 2, :]
    M = np.stack([Tu, Tv, Tw], axis=1)                                   # 列＝Tu Tv Tw（與 transMat 相同排列）
    # 在面元上取點（|h| < 截斷半徑 R = sqrt(2 ln(255 o))，Ln 支撐內），投到螢幕取像素中心（同 pixf = pix + 0.5）
    R = np.sqrt(np.maximum(2 * np.log(255 * op[idx]), 1e-6))
    ang = rng.uniform(0, 2 * np.pi, len(idx)); rad = R * np.sqrt(rng.uniform(0, 1, len(idx)))
    h = np.stack([rad * np.cos(ang), rad * np.sin(ang), np.ones(len(idx))], 1)
    sxyz = np.einsum("nij,nj->ni", M, h)
    px = np.floor(sxyz[:, 0] / sxyz[:, 2]) + 0.5; py = np.floor(sxyz[:, 1] / sxyz[:, 2]) + 0.5
    # 幾何閉式（float32）：M = K C、C = [s0 t0, s1 t1, mu]（相機座標）=> Q = C^{-1} K^{-1}
    #   C^{-1} 的三列：(t1 x mu)/(s0 n·mu)、(mu x t0)/(s1 n·mu)、n/(n·mu)，n = t0 x t1 —— 單位向量的外積，沒有相消
    t0 = (Rg[idx][:, :, 0] @ Rwc.T).astype(f32); t1 = (Rg[idx][:, :, 1] @ Rwc.T).astype(f32); m32 = pv[idx].astype(f32)
    s0 = sc[idx, 0].astype(f32); s1 = sc[idx, 1].astype(f32)
    nn = np.cross(t0, t1).astype(f32); nm = np.sum(nn * m32, 1).astype(f32)
    r0 = np.cross(t1, m32) / (s0 * nm)[:, None]; r1 = np.cross(m32, t0) / (s1 * nm)[:, None]; r2 = nn / nm[:, None]
    Cinv = np.stack([r0, r1, r2], 1).astype(f32)                         # 列＝r0 r1 r2
    Kinv = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]], dtype=f32)
    Qg = np.einsum("nij,jk->nik", Cinv, Kinv).astype(f32)
    rows.append((M, px, py, sc[idx], op[idx], pv[idx, 2], Qg))

M = np.concatenate([r[0] for r in rows]); px = np.concatenate([r[1] for r in rows]); py = np.concatenate([r[2] for r in rows])
scs = np.concatenate([r[3] for r in rows]); ops = np.concatenate([r[4] for r in rows]); zc = np.concatenate([r[5] for r in rows])
Qg = np.concatenate([r[6] for r in rows])
n = len(M)

# ---- 真值（float64）：直接解 M h = Z (x, y, 1)
Qd = np.linalg.inv(M)
pd = np.einsum("nij,nj->ni", Qd, np.stack([px, py, np.ones(n)], 1))
u_t, v_t, z_t = pd[:, 0] / pd[:, 2], pd[:, 1] / pd[:, 2], 1 / pd[:, 2]

# ---- 原寫法（float32，逐字照 forward.cu）
Tu32, Tv32, Tw32 = M[:, 0].astype(f32), M[:, 1].astype(f32), M[:, 2].astype(f32)
x32, y32 = px.astype(f32), py.astype(f32)
k = -Tu32 + x32[:, None] * Tw32; l = -Tv32 + y32[:, None] * Tw32
p = np.cross(k, l).astype(f32)
u_o, v_o = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
z_o = u_o * Tw32[:, 0] + v_o * Tw32[:, 1] + Tw32[:, 2]

# ---- 新寫法（float32）：Q 在 preprocess 由 float32 的 T 算（伴隨矩陣 / det，全 float32）
def inv32(Mf):
    Tu_, Tv_, Tw_ = Mf[:, 0], Mf[:, 1], Mf[:, 2]
    q0 = np.cross(Tv_, Tw_).astype(f32); q1 = np.cross(Tw_, Tu_).astype(f32); q2 = np.cross(Tu_, Tv_).astype(f32)
    D = np.sum(Tu_ * q0, axis=1).astype(f32)
    inv = (f32(1) / D).astype(f32)
    return q0 * inv[:, None], q1 * inv[:, None], q2 * inv[:, None], D
M32 = M.astype(f32)
q0, q1, q2, D32 = inv32(M32)
pn = (x32[:, None] * q0 + y32[:, None] * q1 + q2).astype(f32)
w = (f32(1) / pn[:, 2]).astype(f32)
u_n, v_n, z_n = pn[:, 0] * w, pn[:, 1] * w, w
# 變體：Q 用 float64 算再存 float32（preprocess 每顆一次，可負擔）
Q64 = Qd.astype(f32)
pn2 = (x32[:, None] * Q64[:, :, 0] + y32[:, None] * Q64[:, :, 1] + Q64[:, :, 2]).astype(f32)
w2 = (f32(1) / pn2[:, 2]).astype(f32)
u_n2, v_n2, z_n2 = pn2[:, 0] * w2, pn2[:, 1] * w2, w2


def err(u, v, z, tag):
    eu = np.hypot(u - u_t, v - v_t)                    # 面元座標誤差（單位＝標準差）
    ez = np.abs(z / z_t - 1)
    rho_t = u_t ** 2 + v_t ** 2; rho = u ** 2 + v ** 2
    ea = np.abs(np.exp(-0.5 * rho) - np.exp(-0.5 * rho_t)) * ops   # alpha 誤差（不含 min 濾波）
    q = lambda x: f"中位 {np.median(x):.2e}／p99 {np.quantile(x, .99):.2e}／max {x.max():.2e}"
    print(f"{tag:<28} |Δ(u,v)| {q(eu)}\n{'':<28} |Δz|/z  {q(ez)}\n{'':<28} |Δalpha| {q(ea)}（alpha 量化步 1/255 = 3.9e-3）")
    return eu, ez, ea


print(f"樣本 {n:,} 個（像素, 面元）配對、{len(pick)} 台相機；面元尺度中位 {np.median(scs):.4f}、中心深度中位 {np.median(zc):.2f}")
eo = err(u_o, v_o, z_o, "原寫法 float32（外積）")
en = err(u_n, v_n, z_n, "新寫法 float32（Q 也 float32）")
en2 = err(u_n2, v_n2, z_n2, "新寫法 float32（Q 用 float64 算）")
pn3 = (x32[:, None] * Qg[:, :, 0] + y32[:, None] * Qg[:, :, 1] + Qg[:, :, 2]).astype(f32)
w3 = (f32(1) / pn3[:, 2]).astype(f32)
en3 = err(pn3[:, 0] * w3, pn3[:, 1] * w3, w3, "新寫法 float32（Q 幾何閉式）")
print(f"幾何閉式 Q 對 float64 反矩陣的相對誤差 中位 {np.median(np.linalg.norm(Qg - Qd, axis=(1, 2)) / np.linalg.norm(Qd, axis=(1, 2))):.2e}")
# ★ 錨定版：原點移到面元中心的投影 (x_mu, y_mu) = (Tu.z/Tw.z, Tv.z/Tw.z)，因 Q (x_mu, y_mu, 1) = (0, 0, 1/Z_c)：
#   p' = (0, 0, 1/Z_c) + (x - x_mu) q0 + (y - y_mu) q1      每顆存 q0 q1 x_mu y_mu 1/Z_c ＝ 9 個 float（同 transMat）
for tag, QQ in (("錨定 float32（Q 幾何閉式）", Qg), ("錨定 float32（Q float32 伴隨）", np.stack([q0, q1, q2], 2))):
    xm = (M32[:, 0, 2] / M32[:, 2, 2]).astype(f32); ym = (M32[:, 1, 2] / M32[:, 2, 2]).astype(f32); izc = (f32(1) / M32[:, 2, 2]).astype(f32)
    dx = (x32 - xm).astype(f32); dy = (y32 - ym).astype(f32)
    pa = (dx[:, None] * QQ[:, :, 0] + dy[:, None] * QQ[:, :, 1]).astype(f32); pa[:, 2] += izc
    wa = (f32(1) / pa[:, 2]).astype(f32)
    err(pa[:, 0] * wa, pa[:, 1] * wa, wa, tag)
bad = np.abs(D32) < 1e-30
print(f"det(M) == 0（真退化）{bad.sum()} 顆；det 的範圍 {np.abs(D32).min():.3e}～{np.abs(D32).max():.3e}")

# ---- 梯度鏈（float64 autograd）：原寫法對 T 的梯度 vs 新寫法（逐配對 dL/dp' -> 每顆 dL/dQ -> dL/dM = -Q^T G Q^T）
m = 2000
Mt = torch.tensor(M[:m], dtype=torch.float64, requires_grad=True)
X = torch.tensor(px[:m], dtype=torch.float64); Y = torch.tensor(py[:m], dtype=torch.float64)
gu, gv, gz = [torch.randn(m, dtype=torch.float64) for _ in range(3)]
Tu_, Tv_, Tw_ = Mt[:, 0], Mt[:, 1], Mt[:, 2]
kk = -Tu_ + X[:, None] * Tw_; ll = -Tv_ + Y[:, None] * Tw_
pp = torch.cross(kk, ll, dim=1)
uu, vv = pp[:, 0] / pp[:, 2], pp[:, 1] / pp[:, 2]
zz = uu * Tw_[:, 0] + vv * Tw_[:, 1] + Tw_[:, 2]
(gu * uu + gv * vv + gz * zz).sum().backward()
G_ref = Mt.grad.numpy()

Qt = np.linalg.inv(M[:m]); P = np.einsum("nij,nj->ni", Qt, np.stack([px[:m], py[:m], np.ones(m)], 1))
W = 1 / P[:, 2]; U = P[:, 0] * W; V = P[:, 1] * W
gU, gV, gZ = gu.numpy(), gv.numpy(), gz.numpy()
gp = np.stack([gU * W, gV * W, -W * (gU * U + gV * V + gZ * W)], 1)           # dL/dp'（逐配對 3 個數）
GQ = np.einsum("ni,nj->nij", gp, np.stack([px[:m], py[:m], np.ones(m)], 1))   # dL/dQ = gp ⊗ (x, y, 1)
G_new = -np.einsum("nji,njk,nlk->nil", Qt, GQ, Qt)                             # -Q^T GQ Q^T
rel = np.linalg.norm(G_new - G_ref, axis=(1, 2)) / np.maximum(np.linalg.norm(G_ref, axis=(1, 2)), 1e-300)
print(f"梯度鏈（float64）：新 vs autograd 相對誤差 中位 {np.median(rel):.2e}／max {rel.max():.2e}")

# ---- ★ 梯度換算的 float32 精度（每顆面元多個像素累加，模擬 render 的歸約；對 float64 真值）
#   v1（11-21 第一版，lab 比對出梯度差 4.7e-3）：把錨定累加量還原成未錨定 Q、G 再 -Q^T G Q^T => xm≈800 的大項相消
#   v2：錨定換算 Qt = [q0, q1, (0,0,1/Z_c)]、D = -Qt^T Gt Qt^T，最後 dL/dTw = D2 - xm D0 - ym D1（與原寫法逐配對 x·(…) 同級）
G_ = 300; PPG = 48
gi = rng.choice(n, G_, replace=False)
def grads(gidx):
    Mg = M[gidx]; Qg64 = np.linalg.inv(Mg)
    xm64 = Mg[0, 2] / Mg[2, 2]; ym64 = Mg[1, 2] / Mg[2, 2]
    R_ = np.sqrt(max(2 * np.log(255 * ops[gidx]), 1e-6))
    ang = rng.uniform(0, 2 * np.pi, PPG); rad = R_ * np.sqrt(rng.uniform(0, 1, PPG))
    hh = np.stack([rad * np.cos(ang), rad * np.sin(ang), np.ones(PPG)], 1)
    sx = hh @ Mg.T; X = np.floor(sx[:, 0] / sx[:, 2]) + 0.5; Y = np.floor(sx[:, 1] / sx[:, 2]) + 0.5
    gs = rng.standard_normal((PPG, 2))                                           # 上游 dL/ds
    # 真值 float64：dL/dM = -Q^T (Σ g ⊗ (x,y,1)) Q^T
    P = np.stack([X, Y, np.ones(PPG)], 1) @ Qg64.T; W_ = 1 / P[:, 2]; S = P[:, :2] * W_[:, None]
    g = np.stack([gs[:, 0] * W_, gs[:, 1] * W_, -W_ * (gs * S).sum(1)], 1)
    Gt = -Qg64.T @ (g.T @ np.stack([X, Y, np.ones(PPG)], 1)) @ Qg64.T
    # 原寫法 float32（逐字照 backward.cu）
    Tu, Tv, Tw = [Mg[i].astype(f32) for i in range(3)]; acc = np.zeros(9, np.float32)
    for xx, yy, (dsx, dsy) in zip(X.astype(f32), Y.astype(f32), gs.astype(f32)):
        k = -Tu + xx * Tw; l = -Tv + yy * Tw; p = np.cross(k, l).astype(f32); s = p[:2] / p[2]
        dsx_pz = f32(dsx) / p[2]; dsy_pz = f32(dsy) / p[2]
        dLdp = np.array([dsx_pz, dsy_pz, -(dsx_pz * s[0] + dsy_pz * s[1])], f32)
        dk = np.cross(l, dLdp).astype(f32); dl = np.cross(dLdp, k).astype(f32)
        acc += np.concatenate([-dk, -dl, xx * dk + yy * dl]).astype(f32)
    G_orig = acc.reshape(3, 3)
    # 反單應 float32：逐配對累加 a0 = Σ dx g、a1 = Σ dy g、a2 = Σ g
    Qa = Qg[gidx]; q0_, q1_ = Qa[:, 0], Qa[:, 1]
    xm = f32(Mg[0, 2] / Mg[2, 2]); ym = f32(Mg[1, 2] / Mg[2, 2]); iz = f32(1 / Mg[2, 2])
    a = np.zeros((3, 3), np.float32)
    for xx, yy, (dsx, dsy) in zip(X.astype(f32), Y.astype(f32), gs.astype(f32)):
        dx_ = f32(xx - xm); dy_ = f32(yy - ym)
        pp = (dx_ * q0_ + dy_ * q1_).astype(f32); pp[2] += iz; wz = f32(1) / pp[2]; s = pp[:2] * wz
        gx = f32(dsx) * wz; gy = f32(dsy) * wz; gz = -(gx * s[0] + gy * s[1])
        gg = np.array([gx, gy, gz], f32)
        a[:, 0] += dx_ * gg; a[:, 1] += dy_ * gg; a[:, 2] += gg                    # 欄＝a0 a1 a2
    # v1
    q2_ = (np.array([0, 0, iz], f32) - xm * q0_ - ym * q1_).astype(f32)
    Qv1 = np.stack([q0_, q1_, q2_], 1).astype(f32)
    Gv1 = np.stack([a[:, 0] + xm * a[:, 2], a[:, 1] + ym * a[:, 2], a[:, 2]], 1).astype(f32)
    G_v1 = (-(Qv1.T @ Gv1 @ Qv1.T)).astype(f32)
    # v2
    Qt = np.stack([q0_, q1_, np.array([0, 0, iz], f32)], 1).astype(f32)
    D = (-(Qt.T @ a @ Qt.T)).astype(f32)
    G_v2 = np.stack([D[0], D[1], D[2] - xm * D[0] - ym * D[1]]).astype(f32)
    nrm = np.linalg.norm(Gt)
    return [np.abs(G_ - Gt).max() / np.abs(Gt).max() for G_ in (G_orig, G_v1, G_v2)]
E = np.array([grads(int(i)) for i in gi])
for j, tag in enumerate(["原寫法（逐配對外積）", "反單應 v1（未錨定換算）", "反單應 v2（錨定換算）"]):
    print(f"dL/dT float32 相對誤差（÷最大分量）{tag:<20} 中位 {np.median(E[:, j]):.2e}／p99 {np.quantile(E[:, j], .99):.2e}／max {E[:, j].max():.2e}")
