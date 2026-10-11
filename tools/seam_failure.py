"""合併模型的失敗 tile 是不是集中在分區接縫（2026-10-11 使用者：「我怕接口在合併後有重建問題」）

背景：合併模型的失敗 tile 我方 0.25% vs 官方 release 0.17%（16 塊有 12 塊較多，`紀錄/實驗分析/09` §0.4），
      單塊的失敗地圖顯示失敗集中在靠鄰塊那一側的邊緣 => 懷疑接縫（兩邊都只在自己格子邊緣有弱覆蓋）。
做法（只渲染）：官方 741 幀 held-out，逐 tile（48px）判定失敗（GT std >= 0.10 且 corr < 0.6，與 failure_map 同判準）；
      tile 中心用渲染深度（surf_depth，相機 z）反推世界座標 => 轉到我方 4x4 分區座標系 => 到「所在格子的內部邊界」（兩格共用的邊，
      不含場景外緣）的最短距離，以內部格距為單位（0 = 正在接縫上，0.5 = 格子正中央）。依距離分組報失敗率。
      兩個模型都用**我方分區**的接縫量：官方模型當對照（它的單塊從全場景 coarse 起步，邊緣有支撐）。
判讀：我方失敗率在接縫附近（距離 < 0.05~0.1）明顯升高、官方沒有 => 接縫問題；與距離無關 => 不是接縫。
用法：python tools/seam_failure.py --ckpt 我方=<merged.ckpt> release=<官方 merged.ckpt> [--max_views 741]
      python tools/seam_failure.py --ckpt x=<單塊 ckpt> --self_check     （本機：用該 ckpt 的訓練相機驗反投影）
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

BINS = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 10.0]


def tile_stats(gt, out, t=48, min_std=0.10, r_min=0.60):
    """每個 tile：(過對比門檻, 失敗, MSE)；與 eval_official_test.fail_tiles 同灰階與判準。"""
    def gray(x):
        q = (x.clamp(0, 1) * 255).round()
        return ((q[0] * 299 + q[1] * 587 + q[2] * 114) / 1000).floor() / 255.
    g, r = gray(gt), gray(out)
    h, w = g.shape
    ny, nx = h // t, w // t
    G = g[:ny * t, :nx * t].reshape(ny, t, nx, t).permute(0, 2, 1, 3).reshape(ny, nx, -1)
    R = r[:ny * t, :nx * t].reshape(ny, t, nx, t).permute(0, 2, 1, 3).reshape(ny, nx, -1)
    sg, sr = G.std(-1, unbiased=False), R.std(-1, unbiased=False)
    corr = ((G - G.mean(-1, keepdim=True)) * (R - R.mean(-1, keepdim=True))).mean(-1) / torch.clamp_min(sg * sr, 1e-8)
    keep = sg >= min_std
    mse = ((out - gt) ** 2).mean(0)[:ny * t, :nx * t].reshape(ny, t, nx, t).mean((1, 3))
    return keep, keep & (corr < r_min), mse


def seam_frame(parts):
    """我方分區：每格的盒子＋哪些邊是內部邊（與其他格共用）。回傳 (min[N,2], max[N,2], internal[N,4]=左右下上, unit[2])。"""
    from internal.utils.citygs_partitioning_utils import PartitionCoordinates
    coords = PartitionCoordinates(id=parts["partition_coordinates"]["id"], xy=parts["partition_coordinates"]["xy"])
    size = parts["scene_config"]["partition_size"]
    bx = coords.get_bounding_boxes(size, enlarge=0.)
    mn, mx = bx.min.float(), bx.max.float()
    gmin, gmax = mn.min(0).values, mx.max(0).values
    internal = torch.stack([mn[:, 0] > gmin[0] + 1e-6, mx[:, 0] < gmax[0] - 1e-6,
                            mn[:, 1] > gmin[1] + 1e-6, mx[:, 1] < gmax[1] - 1e-6], 1)
    sz = torch.as_tensor(size).float()
    unit = sz.min(0).values if sz.dim() == 2 else sz.expand(2)
    return mn, mx, internal, unit


def seam_distance(xy, mn, mx, internal, unit):
    """xy [M,2]（分區座標系）-> 到所在格子內部邊界的最短距離（以內部格距為單位）；不在任何格子或該格沒有內部邊 => nan。"""
    inside = (xy[:, None, :] >= mn[None]) & (xy[:, None, :] <= mx[None])
    inside = inside.all(-1)                                        # [M, N]
    has = inside.any(1)
    b = inside.float().argmax(1)
    d = torch.stack([(xy[:, 0] - mn[b, 0]) / unit[0], (mx[b, 0] - xy[:, 0]) / unit[0],
                     (xy[:, 1] - mn[b, 1]) / unit[1], (mx[b, 1] - xy[:, 1]) / unit[1]], 1)
    d = torch.where(internal[b], d, torch.full_like(d, float("inf")))
    out = d.min(1).values
    out[~has] = float("nan")
    out[torch.isinf(out)] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="名稱=ckpt 路徑")
    ap.add_argument("--partition", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--max_views", type=int, default=741)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--self_check", action="store_true", help="用 ckpt 自己的訓練相機（本機驗反投影：重投影誤差、地面高度）")
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    import eval_official_test as eot
    dev = torch.device("cuda")
    parts = torch.load(a.partition)
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].float().to(dev)
    assert not parts["scene_config"]["contract"], "收縮空間的分區未實作"
    mn, mx, internal, unit = (x.to(dev) for x in seam_frame(parts))
    print(f"分區 {a.partition}：{mn.shape[0]} 格、內部格距 {unit.tolist()}")
    bg = torch.zeros(3, device=dev)
    t = a.tile

    for spec in a.ckpt:
        nm, path = spec.split("=", 1)
        model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(path, device=dev, eval_mode=True)
        if a.self_check:
            dmh = ck["datamodule_hyper_parameters"]
            ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(path)), global_rank=0).get_outputs().train_set
            names = list(ts.image_names); cams = ts.cameras
            gt_path = lambda i: ts.image_paths[i]
            sel = np.linspace(0, len(names) - 1, min(a.max_views, len(names))).round().astype(int).tolist()
        else:
            names, cams = eot.load_test_cameras(a.test_dir, 1.2)
            gt_path = lambda i: os.path.join(a.test_dir, "images_1.2", names[i])
            sel = list(range(min(a.max_views, len(names))))
        del ck
        acc = np.zeros((len(BINS) - 1, 4))   # 全部 tile、過門檻、失敗、MSE 和
        n_nan, reproj, zs = 0, [], []
        with torch.no_grad():
            for k, i in enumerate(sel):
                cam = cams[i].to_device(dev)
                o = renderer(cam, model, bg_color=bg)
                img = o["render"].clamp(0, 1)
                pil = Image.open(gt_path(i)).convert("RGB")
                if pil.size != (img.shape[2], img.shape[1]):
                    pil = pil.resize((img.shape[2], img.shape[1]), Image.LANCZOS)
                gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
                keep, fail, mse = tile_stats(gt, img, t)
                ny, nx = keep.shape
                dep = o["surf_depth"][0][:ny * t, :nx * t].reshape(ny, t, nx, t).permute(0, 2, 1, 3).reshape(ny, nx, -1)
                alp = o["rend_alpha"].reshape(img.shape[1:])[:ny * t, :nx * t].reshape(ny, t, nx, t).mean((1, 3))
                z = dep.median(-1).values
                vv, uu = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
                u = (uu.float() + 0.5) * t; v = (vv.float() + 0.5) * t
                pc = torch.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], -1).reshape(-1, 3)
                Rc, Tc = cams.R[i].to(dev).float(), cams.T[i].to(dev).float()
                xw = (pc - Tc) @ Rc                                   # x = R^T (p - T)，列向量版
                if a.self_check and k < 4:
                    # 深度語意自檢：surf_depth 若是相機 z，反推點應落在顆粒上（到最近顆粒中心近）；若其實是沿射線的距離，用 z 解讀會偏
                    ok = (z.reshape(-1) > 0) & (alp.reshape(-1) > 0.9)
                    idx = torch.nonzero(ok).squeeze(1)[torch.randperm(int(ok.sum()), device=dev)[:300]]
                    ray = torch.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, torch.ones_like(u)], -1).reshape(-1, 3)
                    x_z = xw[idx]
                    x_r = ((ray[idx] / ray[idx].norm(dim=-1, keepdim=True)) * z.reshape(-1)[idx, None] - Tc) @ Rc
                    P = model.get_xyz
                    def nn(q):
                        return torch.cat([torch.cdist(q[j:j + 50], P).min(1).values for j in range(0, q.shape[0], 50)])
                    reproj.append((float(nn(x_z).median()), float(nn(x_r).median())))
                    zs.append((x_z @ Rp.T)[:, 2].cpu())
                xy = (xw @ Rp.T)[:, :2]
                d = seam_distance(xy, mn, mx, internal, unit)
                valid = (z.reshape(-1) > 0) & (alp.reshape(-1) > 0.5) & ~torch.isnan(d)
                n_nan += int((~valid).sum())
                bi = torch.bucketize(d[valid].contiguous(), torch.tensor(BINS[1:-1], device=dev))
                kv, fv, mv = keep.reshape(-1)[valid], fail.reshape(-1)[valid], mse.reshape(-1)[valid]
                for b in range(len(BINS) - 1):
                    m = bi == b
                    acc[b] += [int(m.sum()), int(kv[m].sum()), int(fv[m].sum()), float(mv[m].sum())]
                if (k + 1) % 100 == 0:
                    print(f"  [{nm}] {k + 1}/{len(sel)}", flush=True)
        print(f"\n=== {nm}（{len(sel)} 幀；無效 tile（沒深度／透明／不在格子內）{n_nan:,}）===")
        if a.self_check:
            zz = torch.cat(zs)
            print(f"  自檢：反推點到最近顆粒中心的距離中位 —— 當成相機 z：{np.median([r[0] for r in reproj]):.4f}；當成射線距離：{np.median([r[1] for r in reproj]):.4f}"
                  f"（前者應明顯較小）；高度 z 中位 {float(zz.median()):.3f}（p10 {float(zz.quantile(.1)):.3f} / p90 {float(zz.quantile(.9)):.3f}）")
        print(f"  {'離接縫（格距）':<14} {'tile 數':>9} {'佔比':>6} {'失敗/全部':>9} {'失敗/過門檻':>11} {'PSNR':>7}")
        tot = acc[:, 0].sum()
        for b in range(len(BINS) - 1):
            n, nk, nf, ms = acc[b]
            if n == 0:
                continue
            lab = f"{BINS[b]:.3f}~{BINS[b + 1]:.3f}" if BINS[b + 1] < 5 else f">{BINS[b]:.2f}"
            print(f"  {lab:<14} {int(n):>9,} {100 * n / tot:>5.1f}% {100 * nf / n:>8.2f}% {100 * nf / max(nk, 1):>10.2f}% {-10 * np.log10(max(ms / n, 1e-12)):>7.2f}")
        del model, renderer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
