"""
診斷：depth-init 的視角不一致 / floater 規模有多大？（不改訓練，純讀模型）

動機（紀錄/完整Pipeline數學詳解.md §3a + 使用者提案）：depth-init 逐視角反投影 →
單目深度 up-to-scale → 同一表面在不同視角錯位 → floater。voxel 只去重「完全重合」，
視角不一致的錯位點存活 → 傷 SSIM/LPIPS（畫面一團糟）。

本腳本：對訓練視角，渲染當下模型的深度 D_render，跟偽深度 D_pseudo（Depth Anything,
與訓練同一份）比，量「不一致像素」的比例。先量再賭：比例高才值得做修正機制。

用法：
  python tools/diag_depth_consistency.py outputs/mcmc_2dgs_60k_b7 --block 7 --max-views 60 --rel-thresh 0.1
  （--ckpt 可指定特定 ckpt；預設自動找該 block 最新的）
"""
import os
import sys
import glob
import json
import argparse
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
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
    # 取 step 最大的
    def step_of(p):
        try:
            return int(p[p.rfind("=") + 1:p.rfind(".")])
        except Exception:
            return -1
    return max(ckpts, key=step_of)


def find_config(block_dir, run):
    for c in [
        os.path.join(block_dir, "config.yaml"),
        *glob.glob(os.path.join(block_dir, "lightning_logs", "version_*", "config.yaml")),
        os.path.join(run, "config.yaml"),
    ]:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"找不到 config.yaml under {block_dir}")


def build_train_set(cfg_path, block, run):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    path = data["path"]
    init_args = dict(data["parser"].get("init_args", {}))
    init_args.setdefault("block_id", block)
    # 只留 dataclass 認得的欄位，避免多餘 key 爆掉
    valid = {f.name for f in dataclasses.fields(EstimatedDepthBlockColmap)}
    init_args = {k: v for k, v in init_args.items() if k in valid}
    parser_cfg = EstimatedDepthBlockColmap(**init_args)
    dataparser = parser_cfg.instantiate(path=path, output_path=run, global_rank=0)
    outputs = dataparser.get_outputs()
    return outputs.train_set, path


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="outputs/<name> 或 outputs/<name>/blocks/block_X")
    ap.add_argument("--block", type=int, default=7)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-views", type=int, default=60, help="抽樣視角數（省時）")
    ap.add_argument("--rel-thresh", type=float, default=0.1, help="深度相對誤差閾值（>此=不一致像素）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_dir = find_block_dir(args.run, args.block)
    ckpt = find_ckpt(block_dir, args.ckpt)
    cfg_path = find_config(block_dir, args.run)
    print(f"[load] ckpt   = {ckpt}")
    print(f"[load] config = {cfg_path}")

    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ckpt, device, eval_mode=True)
    n_gauss = model.get_xyz.shape[0]
    print(f"[load] gaussians = {n_gauss:,}")

    train_set, _ = build_train_set(cfg_path, args.block, args.run)
    n_total = len(train_set.cameras)
    # 抽樣
    idxs = list(range(n_total))
    if args.max_views and n_total > args.max_views:
        idxs = np.linspace(0, n_total - 1, args.max_views).astype(int).tolist()
    print(f"[data] {n_total} train views, 抽 {len(idxs)} 個")

    bg = torch.zeros(3, device=device)
    eps = 1e-8
    per_view_frac = []      # 每視角「不一致像素比例」（全域尺度對齊後）
    per_view_relmed = []    # 每視角中位相對深度誤差
    skipped = 0

    for k, idx in enumerate(idxs):
        try:
            extra = train_set.extra_data[idx]
            if extra is None:
                skipped += 1
                continue
            cam = train_set.cameras[idx].to_device(device)
            out = renderer(cam, model, bg_color=bg)
            d_render = out["surf_depth"].squeeze().float()                # [H,W] 渲染深度
            gt_inv = train_set.extra_data_processor(extra).to(device).float().squeeze()  # 偽逆深度
            d_pseudo = 1.0 / (gt_inv.clamp_min(eps))                      # 偽深度

            # 解析度對齊（保險）
            if d_pseudo.shape != d_render.shape:
                d_pseudo = torch.nn.functional.interpolate(
                    d_pseudo[None, None], size=d_render.shape, mode="nearest")[0, 0]

            valid = torch.isfinite(d_pseudo) & torch.isfinite(d_render) & (d_pseudo > 0) & (d_render > 0)
            if valid.sum() < 100:
                skipped += 1
                continue
            dr = d_render[valid]
            dp = d_pseudo[valid]

            # ★ 用 median ratio 做全域尺度對齊 → 殘差才是「視角不一致 / floater」，
            #   不是單目深度本來就有的整體尺度差
            scale = torch.median(dr) / (torch.median(dp) + eps)
            dp_aligned = dp * scale

            rel = (dr - dp_aligned).abs() / (dp_aligned + eps)
            frac = (rel > args.rel_thresh).float().mean().item()
            per_view_frac.append(frac)
            per_view_relmed.append(torch.median(rel).item())
        except Exception as e:
            skipped += 1
            if k < 3:
                print(f"  [skip view {idx}] {type(e).__name__}: {e}")
            continue

    if not per_view_frac:
        print("！沒有成功的視角，檢查 depth 檔/路徑")
        return

    f = np.asarray(per_view_frac)
    r = np.asarray(per_view_relmed)
    print("\n================ 深度不一致診斷結果 ================")
    print(f"成功視角 {len(f)} / 抽樣 {len(idxs)}（skip {skipped}）；gaussians={n_gauss:,}")
    print(f"判定閾值：深度相對誤差 > {args.rel_thresh:.0%} 算「不一致像素」（已做全域尺度對齊）")
    print(f"\n每視角不一致像素比例：mean={f.mean():.1%}  median={np.median(f):.1%}  "
          f"p90={np.percentile(f,90):.1%}  max={f.max():.1%}")
    print(f"每視角中位相對深度誤差：mean={r.mean():.1%}  median={np.median(r):.1%}")
    print("\n分佈（不一致比例落在各區間的視角數）：")
    bins = [0, .05, .1, .2, .3, .5, 1.0]
    hist, _ = np.histogram(f, bins=bins)
    for i in range(len(hist)):
        print(f"  {bins[i]:.0%}–{bins[i+1]:.0%} : {hist[i]} views {'#'*hist[i]}")
    print("\n判讀：")
    print(f"  mean 不一致比例 > 20% → 視角不一致嚴重，值得做修正機制（relocate 到深度共識）")
    print(f"  < 5%               → depth reg 已大致處理掉，修正機制 ROI 低")
    print("===================================================")


if __name__ == "__main__":
    main()
