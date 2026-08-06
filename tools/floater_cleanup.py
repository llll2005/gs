"""
Floater 清理機制 (GT-free, 一次性)：用「逐高斯多視角深度一致性」砍掉 view-inconsistent 浮點。

訊號（diag_multiview_consistency.py 已驗證能分辨好壞模型：我們 floater 4.7% vs CityGSV2 1.1%）：
  對每個高斯中心 mu，投影到 M 個視角，比較「mu 在該視角的深度 z」 vs「該視角 render 深度 D」。
  真表面高斯：z≈D 在多數視角成立（一致）。
  浮點：只在生成它的少數視角 z≈D，其餘視角它浮在真表面前/後 → 一致比例低。

  consistency = (z≈D 的視角數) / (看得到它的視角數)
  砍掉 consistency < thresh 且 seen >= min_views 的（避免只被 1-2 視角看到的誤判）。

避開 RTG-SLAM B4/B5 的坑：一次性（非持續 revert）、用深度（非光度誤差）、可選 relocate 不刪。
全程不碰 GT 也不碰偽深度 → 真實場景可部署。

測試：對「跑到一半」的 ckpt 清一次，量前後 train PSNR/SSIM —— 幫(SSIM升/PSNR守) or 傷。
  python tools/floater_cleanup.py outputs/mcmc_2dgs_60k_b7 --block 7 \
      --ckpt outputs/mcmc_2dgs_60k_b7/blocks/block_7/checkpoints/epoch=156-step=29999.ckpt
"""
import os
import sys
import glob
import argparse
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.ssim import ssim as ssim_fn
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
    import yaml  # noqa
    for c in [os.path.join(block_dir, "config.yaml"),
              *glob.glob(os.path.join(block_dir, "lightning_logs", "version_*", "config.yaml")),
              os.path.join(run, "config.yaml")]:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"no config.yaml under {block_dir}")


def build_train_set(cfg_path, block, run):
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    ia = dict(data["parser"].get("init_args", {}))
    ia.setdefault("block_id", block)
    valid = {f.name for f in dataclasses.fields(EstimatedDepthBlockColmap)}
    ia = {k: v for k, v in ia.items() if k in valid}
    dp = EstimatedDepthBlockColmap(**ia).instantiate(path=data["path"], output_path=run, global_rank=0)
    return dp.get_outputs().train_set


@torch.no_grad()
def render_depth(renderer, model, cam, bg):
    return renderer(cam, model, bg_color=bg)["surf_depth"].squeeze().float()


@torch.no_grad()
def render_rgb(renderer, model, cam, bg):
    return renderer(cam, model, bg_color=bg)["render"].clamp(0, 1).float()


def load_gt(path, W, H, device):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    return (torch.from_numpy(np.asarray(img)).float() / 255.).permute(2, 0, 1).to(device)


@torch.no_grad()
def eval_quality(renderer, model, train_set, eval_idxs, bg, device):
    psnrs, ssims = [], []
    for idx in eval_idxs:
        p = train_set.image_paths[idx]
        if p is None:
            continue
        cam = train_set.cameras[idx].to_device(device)
        pred = render_rgb(renderer, model, cam, bg)
        gt = load_gt(p, pred.shape[-1], pred.shape[-2], device)
        mse = ((pred - gt) ** 2).mean().clamp_min(1e-10)
        psnrs.append((-10 * torch.log10(mse)).item())
        ssims.append(ssim_fn(pred[None], gt[None]).item())
    return float(np.mean(psnrs)), float(np.mean(ssims))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--block", type=int, default=7)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--consist-views", type=int, default=24, help="算一致性用幾個視角")
    ap.add_argument("--min-views", type=int, default=4, help="至少被幾個視角看到才判（避免誤殺）")
    ap.add_argument("--tol", type=float, default=0.05, help="|z-D|/D < tol 視為『在表面上』")
    ap.add_argument("--thresh", type=float, default=0.5, help="一致比例 < 此 = floater 砍掉")
    ap.add_argument("--eval-views", type=int, default=16, help="前後品質評估用幾個 train 視角")
    ap.add_argument("--save-ply", default=None, help="清理後存 ply 路徑（可選）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_dir = find_block_dir(args.run, args.block)
    ckpt = find_ckpt(block_dir, args.ckpt)
    cfg_path = find_config(block_dir, args.run)
    print(f"[load] ckpt={ckpt}")

    # pre_activate=False → 可直接 prune model.properties
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ckpt, device, eval_mode=True, pre_activate=False)
    N0 = model.get_xyz.shape[0]
    print(f"[load] gaussians={N0:,}")

    train_set = build_train_set(cfg_path, args.block, args.run)
    n_total = len(train_set.cameras)
    cviews = np.linspace(0, n_total - 1, min(args.consist_views, n_total)).astype(int).tolist()
    eviews = np.linspace(0, n_total - 1, min(args.eval_views, n_total)).astype(int).tolist()

    # ── 1) 算逐高斯多視角一致性 ───────────────────────────────
    centers = model.get_xyz.detach()                       # [N,3]
    homo = torch.cat([centers, torch.ones_like(centers[:, :1])], dim=1)  # [N,4]
    agree = torch.zeros(N0, device=device)
    seen = torch.zeros(N0, device=device)
    eps = 1e-8
    for idx in cviews:
        cam = train_set.cameras[idx].to_device(device)
        D = render_depth(renderer, model, cam, torch.zeros(3, device=device))  # [H,W]
        H, W = D.shape
        W2C = cam.world_to_camera.float()
        Pcam = homo @ W2C                                  # row: p_cam = p_world @ W2C  [N,4]
        z = Pcam[:, 2]
        u = cam.fx * Pcam[:, 0] / (z + eps) + cam.cx
        v = cam.fy * Pcam[:, 1] / (z + eps) + cam.cy
        inb = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = u.clamp(0, W - 1).long()
        vi = v.clamp(0, H - 1).long()
        Dv = D[vi, ui]                                     # 該像素 render 深度
        on_surf = (z - Dv).abs() / (Dv + eps) < args.tol
        agree += (inb & on_surf).float()
        seen += inb.float()

    consistency = agree / seen.clamp_min(1)
    judged = seen >= args.min_views
    floater = judged & (consistency < args.thresh)         # True = 砍
    n_float = int(floater.sum())
    print(f"\n[consistency] 被判定的高斯（seen>={args.min_views}）：{int(judged.sum()):,}")
    print(f"[consistency] floater（一致<{args.thresh:.0%}）：{n_float:,}  ({n_float/N0:.1%} of all)")
    # 分佈
    cj = consistency[judged].cpu().numpy()
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
        c = int(((cj >= lo) & (cj < hi)).sum())
        print(f"   一致 {lo:.0%}–{hi:.0%}: {c:,}")

    # ── 2) 清理前品質 ─────────────────────────────────────────
    p0, s0 = eval_quality(renderer, model, train_set, eviews, torch.zeros(3, device=device), device)
    print(f"\n[before] N={N0:,}  PSNR={p0:.3f}  SSIM={s0:.4f}")

    # ── 3) 砍掉 floater（直接 prune properties，無需 optimizer）──
    keep = ~floater
    model.properties = {k: val[keep] for k, val in model.properties.items()}
    N1 = model.get_xyz.shape[0]

    # ── 4) 清理後品質 ─────────────────────────────────────────
    p1, s1 = eval_quality(renderer, model, train_set, eviews, torch.zeros(3, device=device), device)
    print(f"[after ] N={N1:,}  PSNR={p1:.3f}  SSIM={s1:.4f}")

    print("\n================ 清理結果 ================")
    print(f"砍掉 {N0-N1:,} ({(N0-N1)/N0:.1%}) → {N1:,} 點")
    print(f"PSNR {p0:.3f} → {p1:.3f}  (Δ {p1-p0:+.3f})")
    print(f"SSIM {s0:.4f} → {s1:.4f}  (Δ {s1-s0:+.4f})")
    print("判讀：PSNR 守住(±0.1內) 且 SSIM 升 → 清的是真 floater，機制成立（之後可加 fine-tune 放大）")
    print("      PSNR 明顯掉 → 殺到有用點，調高 --thresh 或 --tol 保守一點")
    print("========================================")

    if args.save_ply:
        from internal.utils.gaussian_model_loader import GaussianModelLoader as _G  # noqa
        try:
            model.save_ply(args.save_ply)
            print(f"saved → {args.save_ply}")
        except Exception as e:
            print(f"[save skip] {e}")


if __name__ == "__main__":
    main()
