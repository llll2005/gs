#!/usr/bin/env python
"""★ 11-21 高斯內參：SH 的可觀測維度（使用者「研究當前高斯的內參組成，有沒有沒用的或可用數學合併的；要求同 39（純數學、不訓練）」）。

每顆面元的顏色 c(d) = Σ_k s_k Y_k(d)（3 階 SH，16 個基底 x 3 色），但訓練只在「看得到它的相機方向」上約束 c(d)。
對塊內每顆累加 M_i = Σ_視角 w_iv Y(d_iv) Y(d_iv)^T（16x16，w＝該視角的貢獻 V＝Σ_像素 T·α，record pass），則：
  - 顏色在訓練視角上的加權能量 = s^T M_i s；M_i 的零空間方向＝訓練完全看不到的 SH 組合（改了不影響任何訓練視角）
  - 全域基底：M̄ = mean_i M_i / tr(M_i) 的前 k 個特徵向量 B_k；每顆投影 a = (B M_i B^T)^-1 B M_i s => s' = B^T a（加權最小平方）
    => 報「把 16 個係數壓成 k 個」時，每顆在自己訓練視角上的加權 RMS 顏色誤差（1/255 = 0.0039）
用法：python tools/probe/sh_observability_1121.py --ckpt <合併 ckpt> --block 2 [--every 3] [--out x.npz]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--block", type=int, required=True)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--out", default="")
ap.add_argument("--render_k", type=int, nargs="*", default=[], help="把盒內顆粒的 SH 投影成 k 維後，渲染看向本塊的 test 幀比 PSNR")
ap.add_argument("--ntest", type=int, default=16)
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
a = ap.parse_args()

from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot

C0 = 0.28209479177387814; C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def shY(d):   # d: [n,3] 單位向量 -> [n,16]（3DGS 的基底順序與常數，同 computeColorFromSH）
    x, y, z = d[:, 0], d[:, 1], d[:, 2]; xx, yy, zz = x * x, y * y, z * z
    return torch.stack([torch.full_like(x, C0), -C1 * y, C1 * z, -C1 * x,
                        C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
                        C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy), C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                        C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)], dim=1)


model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=torch.device("cpu"), eval_mode=True)
for f_ in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "exact_strip_cull", "inv_homog"):
    setattr(renderer, f_, False)
G = {k: v.data for k, v in model.gaussians.items()}
X = G["means"].double().numpy(); N = len(X)
sc = model.get_scaling.detach().double().numpy(); rad = 3.0 * sc.max(axis=1)
S = G["shs"].float()                                    # [N,16,3]（評估模式：dc＋rest 合併，基底順序同 shY）
# 塊內（分區盒，不外擴）
P = torch.load(os.path.join(a.D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
Rp = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = P["partition_coordinates"]["id"].numpy(); xys = P["partition_coordinates"]["xy"].double().numpy(); size = P["scene_config"]["partition_size"].double().numpy()
k_ = int(np.where((ids[:, 0] == a.block % 4) & (ids[:, 1] == a.block // 4))[0][0])
xy = (X @ Rp.T)[:, :2]
inb = (xy[:, 0] >= xys[k_][0]) & (xy[:, 0] < xys[k_][0] + size[k_][0]) & (xy[:, 1] >= xys[k_][1]) & (xy[:, 1] < xys[k_][1] + size[k_][1])
gidx = np.where(inb)[0]; loc = np.full(N, -1, np.int64); loc[gidx] = np.arange(len(gidx)); n = len(gidx)
print(f"N={N:,}｜b{a.block} 盒內 {n:,} 顆", flush=True)

names, cams = eot.load_test_cameras(a.D, 1.2)
want = [l.strip() for l in open(os.path.join(a.D, a.part, "%03d_%03d.txt" % (a.block % 4, a.block // 4))) if l.strip()][:: a.every]
pos = {nm: i for i, nm in enumerate(names)}
dev = torch.device("cuda"); bg = torch.zeros(3, device=dev)
iu = torch.triu_indices(16, 16)
M = torch.zeros(n, iu.shape[1], device=dev)         # 上三角 136 個
Wsum = torch.zeros(n, device=dev)
Xg = torch.from_numpy(X).float()
keys = list(G.keys()); t0 = time.time()
for kk, nm in enumerate(want):
    i = pos[nm]
    R = cams.R[i].double().numpy(); T = cams.T[i].double().numpy()
    W, H = int(cams.width[i]), int(cams.height[i]); fx, fy, cx, cy = (float(cams.fx[i]), float(cams.fy[i]), float(cams.cx[i]), float(cams.cy[i]))
    Xc = X @ R.T + T; z = Xc[:, 2]
    u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy; r = fx * rad / np.maximum(z, 1e-6)
    idx = np.where((z > 0.2) & (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H))[0]
    sub = {k2: torch.nn.Parameter(G[k2][torch.from_numpy(idx)].to(dev), requires_grad=False) for k2 in keys}
    old = model.gaussians
    model.__dict__["_modules"] = dict(model._modules); model.gaussians = torch.nn.ParameterDict(sub)
    try:
        with torch.no_grad():
            mean, cov = renderer(cams[i].to_device(dev), model, bg_color=bg, record_transmittance=True, record_coverage=True)
    finally:
        model.gaussians = old
    w = (mean * cov).float()
    li = torch.from_numpy(loc[idx]).to(dev); m = (li >= 0) & (w > 0)
    if m.any():
        campos = torch.from_numpy(-R.T @ T).float().to(dev)
        d = Xg[torch.from_numpy(idx)].to(dev)[m] - campos; d = d / d.norm(dim=1, keepdim=True)
        Y = shY(d)                                               # [m,16]
        outer = (Y[:, iu[0]] * Y[:, iu[1]]) * w[m][:, None]
        M.index_add_(0, li[m], outer); Wsum.index_add_(0, li[m], w[m])
    del sub; torch.cuda.empty_cache()
    if (kk + 1) % 50 == 0:
        print(f"  {kk + 1}／{len(want)} 台（{time.time() - t0:.0f} s）", flush=True)

# 還原成 16x16（分批，GPU 記憶體有限）
bad = ~torch.isfinite(M).all(1) | ~torch.isfinite(Wsum)
print(f"  非有限值的列：{int(bad.sum())}", flush=True)
seen = (Wsum > 1e-6) & ~bad
sel = torch.where(seen)[0]
Mc = M[sel].cpu(); Wc = Wsum[sel].cpu(); Sc = S[torch.from_numpy(gidx)][sel.cpu()]          # [m,136]、[m]、[m,16,3]
del M; torch.cuda.empty_cache()
CH = 100_000


def full(Mrow):
    F = torch.zeros(Mrow.shape[0], 16, 16, device=dev); F[:, iu[0], iu[1]] = Mrow; F[:, iu[1], iu[0]] = Mrow; return F


Msum = torch.zeros(16, 16, dtype=torch.float64, device=dev); cnt = 0
for b0 in range(0, len(sel), CH):
    F = full(Mc[b0:b0 + CH].to(dev)); tr = F.diagonal(dim1=1, dim2=2).sum(1)
    ok = tr > 1e-12
    Msum += (F[ok] / tr[ok][:, None, None]).double().sum(0); cnt += int(ok.sum())
Mbar = Msum / cnt
print("  M̄ 有限：", bool(torch.isfinite(Mbar).all()), "｜跡", float(Mbar.trace()), flush=True)
_e, _v = np.linalg.eigh(Mbar.cpu().numpy()); ev, evec = torch.from_numpy(_e[::-1].copy()).to(dev), torch.from_numpy(_v[:, ::-1].copy()).to(dev)
print(f"\n══ b{a.block}：{len(want)} 台相機（每 {a.every} 台取 1）｜盒內有貢獻的 {len(sel):,}／{n:,} 顆")
print("  M̄（每顆正規化後平均）特徵值（佔總和 %）：", " ".join(f"{100 * float(x / ev.sum()):.3g}" for x in ev))
KS = tuple(sorted(set((1, 4, 7, 9, 10, 12, 16)) | set(a.render_k)))
PROJ = {}
er3, er4, efull, errs = [], [], [], {k: [] for k in KS}
for b0 in range(0, len(sel), CH):
    F = full(Mc[b0:b0 + CH].to(dev)); w = Wc[b0:b0 + CH].to(dev); Sn = Sc[b0:b0 + CH].to(dev)
    tr = F.diagonal(dim1=1, dim2=2).sum(1)
    evi = torch.linalg.eigvalsh((F / tr[:, None, None]).double().cpu())
    er3.append((evi > 1e-3 * evi[:, -1:]).sum(1).cpu()); er4.append((evi > 1e-4 * evi[:, -1:]).sum(1).cpu())
    efull.append((torch.einsum("nkc,nkl,nlc->n", Sn, F, Sn) / w).clamp_min(0).sqrt().cpu())
    for k in KS:
        B = evec[:, :k].double().T; Fd = F.double(); Sd = Sn.double()
        BM = torch.einsum("kj,njl->nkl", B, Fd); A = torch.einsum("nkl,ql->nkq", BM, B); rhs = torch.einsum("nkl,nlc->nkc", BM, Sd)
        sol = torch.linalg.pinv(A, hermitian=True, rtol=1e-12) @ rhs      # 加權最小平方（A 可能奇異：只被少數方向看到）
        Sp_ = torch.einsum("kj,nkc->njc", B, sol)
        if k in a.render_k:
            PROJ.setdefault(k, []).append(Sp_.float().cpu())
        Dl = Sd - Sp_
        errs[k].append(((torch.einsum("nkc,nkl,nlc->n", Dl, Fd, Dl) / w.double()).clamp_min(0).sqrt() / np.sqrt(3)).float().cpu())
        del Fd, Sd, BM, A, rhs, sol, Dl
    del F, w, Sn; torch.cuda.empty_cache()
for thr, er in ((1e-3, torch.cat(er3).float()), (1e-4, torch.cat(er4).float())):
    print(f"  逐顆有效維度（特徵值 > {thr:g} x 最大）：中位 {float(er.median()):.0f}、p10 {float(er.quantile(0.1)):.0f}、p90 {float(er.quantile(0.9)):.0f}")
ef = torch.cat(efull)
print(f"  顏色（去掉 0.5 偏移）在訓練視角上的加權 RMS：中位 {float(ef.median()):.4f}")
res = {}
for k in KS:
    err = torch.cat(errs[k]); res[k] = err.numpy()
    print(f"  壓成 {k:>2} 個（x3 色＝{3 * k:>2} 個 float，原 48）：每顆在自己訓練視角上的 RMS 顏色誤差 中位 {float(err.median()):.5f}、p90 {float(err.quantile(0.9)):.5f}、"
          f"p99 {float(err.quantile(0.99)):.5f}｜> 1/255 的 {100 * float((err > 1 / 255).float().mean()):.2f}%")
if a.render_k:
    from PIL import Image
    tn, tc = eot.load_test_cameras(a.test_dir, 1.2)
    # 看向本塊：光軸與地面（SfM z 中位）的交點落在分區盒內
    pick = []
    for i in range(len(tn)):
        R = tc.R[i].double().numpy(); T = tc.T[i].double().numpy(); c = -R.T @ T; f = R[2]
        if f[2] >= -1e-3: continue
        t = (0.0 - c[2]) / f[2]; pxy = ((c + t * f) @ Rp.T)[:2]
        if xys[k_][0] <= pxy[0] < xys[k_][0] + size[k_][0] and xys[k_][1] <= pxy[1] < xys[k_][1] + size[k_][1]:
            pick.append(i)
    pick = pick[:: max(1, len(pick) // a.ntest)][: a.ntest]
    gsel = torch.from_numpy(gidx)[sel.cpu()]
    print(f"\n  渲染：看向 b{a.block} 的 test 幀 {len(pick)} 張（{[tn[i] for i in pick][:6]}…）", flush=True)
    shs0 = G["shs"].clone()
    psnr = lambda x, y: float(-10 * torch.log10(((x - y) ** 2).mean().clamp_min(1e-12)))

    def rend(i):
        R = tc.R[i].double().numpy(); T = tc.T[i].double().numpy()
        W, H = int(tc.width[i]), int(tc.height[i]); fx, fy, cx, cy = (float(tc.fx[i]), float(tc.fy[i]), float(tc.cx[i]), float(tc.cy[i]))
        Xc = X @ R.T + T; z = Xc[:, 2]
        u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy; r = fx * rad / np.maximum(z, 1e-6)
        idx = torch.from_numpy(np.where((z > 0.2) & (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H))[0])
        sub = {k2: torch.nn.Parameter(G[k2][idx].to(dev), requires_grad=False) for k2 in keys}
        old = model.gaussians
        model.__dict__["_modules"] = dict(model._modules); model.gaussians = torch.nn.ParameterDict(sub)
        try:
            with torch.no_grad():
                o = renderer(tc[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
        finally:
            model.gaussians = old
        del sub; torch.cuda.empty_cache(); return o
    full = {i: rend(i) for i in pick}
    gts = {i: torch.from_numpy(np.array(Image.open(os.path.join(a.test_dir, "images_1.2", tn[i])).convert("RGB"))).float().permute(2, 0, 1).to(dev) / 255 for i in pick}
    print(f"  完整 SH vs GT：PSNR 平均 {np.mean([psnr(full[i], gts[i]) for i in pick]):.3f}")
    for k in a.render_k:
        G["shs"] = shs0.clone(); G["shs"][gsel] = torch.cat(PROJ[k])
        d_gt, d_full = [], []
        for i in pick:
            o = rend(i); d_gt.append(psnr(o, gts[i]) - psnr(full[i], gts[i])); d_full.append(psnr(o, full[i]))
        print(f"  SH 壓成 {k:>2} 維：對 GT 的 PSNR 變化 平均 {np.mean(d_gt):+.4f}（最差 {min(d_gt):+.4f}）｜與完整 SH 渲染之間的 PSNR 平均 {np.mean(d_full):.1f} dB（最低 {min(d_full):.1f}）")
    G["shs"] = shs0
if a.out:
    np.savez_compressed(a.out, ev=ev.cpu().numpy(), evec=evec.cpu().numpy(), gidx=gidx[sel.cpu().numpy()], **{f"err{k}": v for k, v in res.items()})
    print("=>", a.out)
