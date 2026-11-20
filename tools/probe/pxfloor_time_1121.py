#!/usr/bin/env python
"""★ 11-21 量 px_floor 下限重算（_px_floor_log：每顆 x 全部訓練相機）的成本；不訓練。"""
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl as Impl
CK = sys.argv[1] if len(sys.argv) > 1 else "outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt"
dev = torch.device("cuda")
model, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(CK, device=dev, eval_mode=False, pre_activate=False)
ck = torch.load(CK, map_location="cpu", weights_only=False)
pp = ck["datamodule_hyper_parameters"]["parser"]
if getattr(pp, "points_from", None) == "ply":
    pp.points_from = "random"
cams = pp.instantiate(path=ck["datamodule_hyper_parameters"]["path"], output_path="/tmp", global_rank=0).get_outputs().train_set.cameras
m = model.gaussians["means"].detach()
print(f"N={m.shape[0]:,}  相機 K={len(cams)}")
for _ in range(2):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fl = Impl._px_floor_log(m, cams, 1.0, 1.0)
    torch.cuda.synchronize(); print(f"_px_floor_log：{(time.perf_counter() - t0) * 1e3:.0f} ms")
torch.save(fl.cpu(), "/tmp/claude-1000/-home-LnoArch-Projects----CityGaussian/9aa045bc-1626-460f-9d56-9f86934cf8a4/scratchpad/pxf_ref.pt")

# ── ★ 11-21 線性化：可見性的 5 個條件乘上 z>0 後都是 (X,1) 的線性函數 => 一次 GEMM；最小值 z/fx 也是線性
def pxf_gemm(means, cams, res, kappa, chunk=8192):
    dev = means.device
    W2C = cams.world_to_camera.to(dev).float()                       # p_cam = [p,1] @ W2C（轉置過）
    fx = cams.fx.to(dev).float(); fy = cams.fy.to(dev).float(); cx = cams.cx.to(dev).float(); cy = cams.cy.to(dev).float()
    Wd = cams.width.to(dev).float(); Hd = cams.height.to(dev).float()
    K = W2C.shape[0]
    Mx = W2C[:, :, 0]; My = W2C[:, :, 1]; Mz = W2C[:, :, 2]           # [K,4]：對 (X,1) 的 x_cam、y_cam、z_cam 係數
    Lz = Mz / fx[:, None]                                            # z/fx（值；正比於 z => 也當 z>0 的判準）
    Lu = fx[:, None] * Mx + cx[:, None] * Mz                         # u·z
    LuW = Wd[:, None] * Mz - Lu                                      # (W − u)·z
    Lv = fy[:, None] * My + cy[:, None] * Mz
    LvH = Hd[:, None] * Mz - Lv
    A = torch.cat([Lz, Lu, LuW, Lv, LvH], 0).T.contiguous()          # [4, 5K]
    out = torch.empty(means.shape[0], device=dev)
    prev = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
    for i in range(0, means.shape[0], chunk):
        X = means[i:i + chunk].float()
        Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
        P = Xh @ A                                                   # [n, 5K]
        z, u, uw, v, vh = P.split(K, 1)
        if FAST:   # 5 個條件 => 2 個 minimum：min(u·z, v·z) >= 0 且 min((W−u)z, (H−v)z) > 0（z > 0 隱含在 (W−u)z > 0 與 u·z >= 0 之中？否 => 仍要 z 判準）
            vis = (torch.minimum(u, v) >= 0) & (torch.minimum(uw, vh) > 0) & (z > 1e-6 / fx[None])
            q = torch.where(vis, z, torch.full_like(z, float("inf"))).amin(1)
        else:
            vis = (z > 1e-6 / fx[None]) & (u >= 0) & (uw > 0) & (v >= 0) & (vh > 0)
            q = torch.where(vis, z, torch.full_like(z, float("inf"))).min(1).values
        out[i:i + chunk] = torch.where(torch.isfinite(q), torch.log(kappa * q / max(res, 1e-6)), torch.full_like(q, float("-inf")))
    torch.backends.cuda.matmul.allow_tf32 = prev
    return out
ref = torch.load("/tmp/claude-1000/-home-LnoArch-Projects----CityGaussian/9aa045bc-1626-460f-9d56-9f86934cf8a4/scratchpad/pxf_ref.pt").to(dev)
FAST = False
for FAST, ch in ((False, 8192), (True, 8192), (True, 32768)):
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fl2 = pxf_gemm(m, cams, 1.0, 1.0, ch)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
    fin = torch.isfinite(ref) & torch.isfinite(fl2)
    print(f"GEMM 版{'＋2 個 minimum' if FAST else ''}（chunk {ch}）：{dt:.0f} ms｜可見與否不同 {int((torch.isfinite(ref) != torch.isfinite(fl2)).sum())} 顆｜"
          f"值 max |Δ| {float((ref - fl2)[fin].abs().max()):.2e}（log 空間）、>1e-4 的 {int(((ref - fl2)[fin].abs() > 1e-4).sum())} 顆")
