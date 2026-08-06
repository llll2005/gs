"""
診斷 (GT-free)：模型自己的「多視角深度自一致性」 —— 抓 view-inconsistent floater。

不靠 GT、不靠 Depth Anything 偽深度（上一支 diag_depth_consistency.py 被 CityGSV2 對照證明
偽深度誤差主導、無法分辨好壞模型）。這支只用「模型自己渲染的深度 + 相機位姿」：

  視角 A 的每個像素 → 用 A 的 render 深度反投影成世界點 X → 投影到鄰近視角 B
  → 比較「X 在 B 相機座標的深度 z_proj」 vs 「B 在該像素的 render 深度 D_B」
  一致(z_proj≈D_B) = 真表面；不一致 = floater / 視角矛盾。

  z_proj < D_B（A 的面戳在 B 的面前方）= floater 候選（A 憑空生一片浮在前面）
  z_proj > D_B（A 的面在 B 的面後方）   = 遮擋 or 後方 floater（用鄰近視角降低遮擋干擾）

只用 images+poses → 真實場景能拿到 → 可部署。MatrixCity GT 只拿來「事後驗證」不進機制。

用法（兩個都跑做對照，好模型應 floater 低）：
  python tools/diag_multiview_consistency.py outputs/mcmc_2dgs_60k_b7 --block 7
  python tools/diag_multiview_consistency.py outputs/citygsv2_mc_aerial_sh0_trim --block 7 \
      --ckpt outputs/citygsv2_mc_aerial_sh0_trim/blocks/block_7/checkpoints/epoch=107-step=24000.ckpt
"""
import os
import sys
import glob
import argparse
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthBlockColmap


def find_block_dir(run, block):
    d = os.path.join(run, "blocks", f"block_{block}")
    return d if os.path.isdir(d) else run


def find_ckpt(block_dir, explicit):
    if explicit:
        return explicit
    ckpts = glob.glob(os.path.join(block_dir, "checkpoints", "*.ckpt"))
    assert ckpts, f"no ckpt under {block_dir}/checkpoints"
    return max(ckpts, key=lambda p: int(p[p.rfind("=") + 1:p.rfind(".")]) if "=" in p else -1)


def find_config(block_dir, run):
    for c in [os.path.join(block_dir, "config.yaml"),
              *glob.glob(os.path.join(block_dir, "lightning_logs", "version_*", "config.yaml")),
              os.path.join(run, "config.yaml")]:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"no config.yaml under {block_dir}")


def build_train_set(cfg_path, block, run):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    init_args = dict(data["parser"].get("init_args", {}))
    init_args.setdefault("block_id", block)
    valid = {f.name for f in dataclasses.fields(EstimatedDepthBlockColmap)}
    init_args = {k: v for k, v in init_args.items() if k in valid}
    dp = EstimatedDepthBlockColmap(**init_args).instantiate(path=data["path"], output_path=run, global_rank=0)
    return dp.get_outputs().train_set


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--block", type=int, default=7)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-views", type=int, default=40, help="抽樣 anchor 視角 A 數")
    ap.add_argument("--neighbors", type=int, default=4, help="每個 A 配幾個最近的 B")
    ap.add_argument("--stride", type=int, default=2, help="像素降採樣（省時）")
    ap.add_argument("--rel-thresh", type=float, default=0.05, help="相對深度誤差 > 此 = 不一致")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_dir = find_block_dir(args.run, args.block)
    ckpt = find_ckpt(block_dir, args.ckpt)
    cfg_path = find_config(block_dir, args.run)
    print(f"[load] ckpt={ckpt}\n[load] config={cfg_path}")

    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ckpt, device, eval_mode=True)
    print(f"[load] gaussians={model.get_xyz.shape[0]:,}")

    train_set = build_train_set(cfg_path, args.block, args.run)
    n_total = len(train_set.cameras)
    idxs = list(range(n_total))
    if args.max_views and n_total > args.max_views:
        idxs = np.linspace(0, n_total - 1, args.max_views).astype(int).tolist()
    print(f"[data] {n_total} views, anchor 抽 {len(idxs)}，每個配 {args.neighbors} 鄰居")

    bg = torch.zeros(3, device=device)
    eps = 1e-8
    st = args.stride

    # 1) 渲染並快取每個抽樣視角的深度 + 相機參數（CPU 省 VRAM）
    cache = {}
    centers = []
    for idx in idxs:
        cam = train_set.cameras[idx].to_device(device)
        d = renderer(cam, model, bg_color=bg)["surf_depth"].squeeze().float()  # [H,W]
        cache[idx] = dict(
            D=d[::st, ::st].cpu(),
            fx=float(cam.fx), fy=float(cam.fy), cx=float(cam.cx) / st, cy=float(cam.cy) / st,
            W2C=cam.world_to_camera.float().cpu(),   # [4,4] row 慣例: p_cam = p_world @ W2C
            center=cam.camera_center.float().cpu().numpy(),
        )
        # 內參隨 stride 縮放：fx,fy 不變（焦距以像素計但取樣間距變了→等效縮放）
        cache[idx]["fx"] /= st
        cache[idx]["fy"] /= st
        centers.append(cache[idx]["center"])
    centers = np.stack(centers)  # [V,3]

    # 2) 每個 A 找最近的 neighbors 個 B
    V = len(idxs)
    dmat = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    neigh = np.argsort(dmat, axis=1)[:, :args.neighbors]  # 索引到 idxs 的位置

    def unproject_project(A, B):
        DA = A["D"].to(device)
        DB = B["D"].to(device)
        H, W = DA.shape
        vv, uu = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        uu = uu.float(); vv = vv.float()
        # A pixel -> A camera coords (z forward)
        xc = (uu - A["cx"]) / A["fx"] * DA
        yc = (vv - A["cy"]) / A["fy"] * DA
        zc = DA
        ones = torch.ones_like(zc)
        Pcam = torch.stack([xc, yc, zc, ones], dim=-1)            # [H,W,4]
        W2C_A = A["W2C"].to(device); W2C_B = B["W2C"].to(device)
        Pw = Pcam.reshape(-1, 4) @ torch.linalg.inv(W2C_A)        # row: p_world = p_cam @ inv(W2C_A)
        PcamB = Pw @ W2C_B                                        # [HW,4]  p_cam_B = p_world @ W2C_B
        zB = PcamB[:, 2]
        uB = B["fx"] * PcamB[:, 0] / (zB + eps) + B["cx"]
        vB = B["fy"] * PcamB[:, 1] / (zB + eps) + B["cy"]
        Hb, Wb = DB.shape
        # grid_sample 取 B 的 render 深度
        gx = 2 * uB / (Wb - 1) - 1
        gy = 2 * vB / (Hb - 1) - 1
        grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 1, 2)
        DB_samp = F.grid_sample(DB[None, None], grid, mode="nearest", align_corners=True).reshape(-1)
        inb = (uB >= 0) & (uB <= Wb - 1) & (vB >= 0) & (vB <= Hb - 1)
        valid = inb & (zB > 0) & (DB_samp > 0) & (DA.reshape(-1) > 0)
        if valid.sum() < 50:
            return None
        zB_v = zB[valid]; DB_v = DB_samp[valid]
        rel = (zB_v - DB_v).abs() / (DB_v + eps)
        inc = (rel > args.rel_thresh)
        infront = inc & (zB_v < DB_v)        # A 的面在 B 的面「前方」= floater 戳出來
        return inc.float().mean().item(), infront.float().mean().item()

    # 3) 跑所有 (A,B) pair
    inc_fracs, front_fracs = [], []
    for ai, A_idx in enumerate(idxs):
        A = cache[A_idx]
        for bj in neigh[ai]:
            B = cache[idxs[bj]]
            r = unproject_project(A, B)
            if r is not None:
                inc_fracs.append(r[0]); front_fracs.append(r[1])

    inc = np.asarray(inc_fracs); fr = np.asarray(front_fracs)
    print("\n========= 多視角自一致性診斷 (GT-free) =========")
    print(f"pairs={len(inc)}  rel-thresh={args.rel_thresh:.0%}  stride={st}")
    print(f"不一致像素比例（雙向）   ：mean={inc.mean():.1%}  median={np.median(inc):.1%}  p90={np.percentile(inc,90):.1%}")
    print(f"★floater 比例（A 戳在 B 前）：mean={fr.mean():.1%}  median={np.median(fr):.1%}  p90={np.percentile(fr,90):.1%}")
    print("\n判讀（拿 CityGSV2 當對照）：")
    print("  好模型 floater 比例應「明顯低」。若我們的模型 floater >> CityGSV2 → floater 是真問題、值得修。")
    print("  若兩者差不多 → 自一致性也分不出 → floater 不是主要差距，回去查點數/欠擬合。")
    print("================================================")


if __name__ == "__main__":
    main()
