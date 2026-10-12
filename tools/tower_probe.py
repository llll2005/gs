"""區塊 10 那棟超高層大樓為什麼糊（2026-10-12 使用者）：
  1 單塊模型在「最接近失敗幀的訓練視角」上，這棟樓是否也糊（也糊 => 長不出來；清楚 => 新視角才糊＝幾何沒對齊）
  2 大樓位置：用單塊模型的深度把失敗幀的失敗 tile 反推到世界座標（分區座標系）
  3 那裡的 SfM 點密度 vs 其他地方（驗「SfM 在高樓缺點」）
  4 Depth Anything 深度圖在那裡有沒有內容（使用者提案：深度有內容但缺 SfM 的地方補點）
輸出拼圖到 --out。純本機（單塊 2.34M 顆、6GB 卡可）。
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


def psnr(a, b, m=None):
    se = ((a - b) ** 2).mean(0)
    if m is not None:
        se = se[m]
    return float(-10 * torch.log10(se.mean().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame", default="0508.png")
    ap.add_argument("--tiles_csv", default=None, help="（未用；失敗 tile 在本工具內用同判準重算）")
    ap.add_argument("--merged_png", default=None, help="合併模型在該幀的 GT|渲染 PNG（對照）")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    import eval_official_test as eot
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.colmap import read_points3D_binary, read_images_binary
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        setattr(renderer, f, False)
    dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; del ck
    if getattr(pc, "points_from", None) == "ply":
        pc.points_from = "random"
    out = pc.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs()
    ts = out.train_set
    parts = torch.load(os.path.join(os.path.dirname(pc.image_list), "partitions.pt"))
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].float().numpy()
    bg = torch.zeros(3, device=dev)

    def load_gt(path, like):
        pil = Image.open(path).convert("RGB")
        if pil.size != (like.shape[2], like.shape[1]):
            pil = pil.resize((like.shape[2], like.shape[1]), Image.LANCZOS)
        return torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.

    # ── 2 大樓位置：測試幀用單塊模型渲染＋深度，失敗 tile 反推 ──
    names, cams = eot.load_test_cameras(a.test_dir, 1.2)
    fi = names.index(a.frame)
    cam = cams[fi].to_device(dev)
    with torch.no_grad():
        o = renderer(cam, model, bg_color=bg)
    img = o["render"].clamp(0, 1)
    gt_png = a.merged_png and np.asarray(Image.open(a.merged_png).convert("RGB"), np.float32) / 255.
    gt_t = torch.from_numpy(gt_png[:, :gt_png.shape[1] // 2]).permute(2, 0, 1).to(dev) if gt_png is not None else None
    t = 48
    W = img.shape[2]; nx = W // t
    # tiles.csv 只記了過門檻 tile 的幀與特徵，沒存 tile 座標 => 這裡重算該幀的失敗 tile 位置（同判準）
    def lum(x):
        q = (x.clamp(0, 1) * 255).round()
        return ((q[0] * 299 + q[1] * 587 + q[2] * 114) / 1000).floor() / 255.
    mrender = torch.from_numpy(gt_png[:, gt_png.shape[1] // 2:]).permute(2, 0, 1).to(dev)
    G, R = lum(gt_t), lum(mrender)
    ny = G.shape[0] // t
    Gt = G[:ny * t, :nx * t].reshape(ny, t, nx, t).permute(0, 2, 1, 3).reshape(ny, nx, -1)
    Rt = R[:ny * t, :nx * t].reshape(ny, t, nx, t).permute(0, 2, 1, 3).reshape(ny, nx, -1)
    sg, sr = Gt.std(-1, unbiased=False), Rt.std(-1, unbiased=False)
    corr = ((Gt - Gt.mean(-1, keepdim=True)) * (Rt - Rt.mean(-1, keepdim=True))).mean(-1) / torch.clamp_min(sg * sr, 1e-8)
    fail = (sg >= 0.10) & (corr < 0.6)
    z = o["surf_depth"][0][:ny * t, :nx * t].reshape(ny, t, nx, t).permute(0, 2, 1, 3).reshape(ny, nx, -1).median(-1).values
    vv, uu = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    u = (uu.float() + 0.5) * t; v = (vv.float() + 0.5) * t
    pcm = torch.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], -1)
    xw = ((pcm - cams.T[fi].to(dev).float()) @ cams.R[fi].to(dev).float()).cpu().numpy()
    xp = xw @ Rp.T                                          # 分區座標系
    fm = fail.cpu().numpy() & (z.cpu().numpy() > 0)
    tw = xp[fm]
    print(f"{a.frame}：失敗 tile {int(fail.sum())}（合併模型判定）；單塊模型此幀 PSNR {psnr(img, gt_t):.2f}（合併 {psnr(mrender, gt_t):.2f}）")
    lo, hi = np.percentile(tw, 10, axis=0), np.percentile(tw, 90, axis=0)
    print(f"大樓（失敗 tile 反推，分區座標系）：x {lo[0]:.2f}~{hi[0]:.2f}  y {lo[1]:.2f}~{hi[1]:.2f}  z（高度）{lo[2]:.2f}~{hi[2]:.2f}（地面約 0.24）")
    cxy = np.median(tw[:, :2], 0)

    # ── 3 SfM 點密度 ──
    pts = read_points3D_binary(os.path.join(dmh["path"], "sparse", "0", "points3D.bin"))
    P = np.array([p.xyz for p in pts.values()], np.float32) @ Rp.T
    rad = 0.35
    def dens(c, zmin=None):
        m = np.linalg.norm(P[:, :2] - c[None], axis=1) < rad
        if zmin is not None:
            m &= P[:, 2] > zmin
        return int(m.sum())
    n_t = dens(cxy); n_t_hi = dens(cxy, 0.24 + 0.5)
    rng = np.random.default_rng(0)
    # 對照：同一區塊（盒內）隨機 200 個位置的同半徑點數
    boxes_xy = parts["partition_coordinates"]["xy"].float().numpy()
    size = np.array(torch.as_tensor(parts["scene_config"]["partition_size"]).float())
    bid = pc.block_id
    sz = size[bid] if size.ndim == 2 else size
    ctrl = boxes_xy[bid] + rng.random((200, 2)) * sz
    cd = np.array([dens(c) for c in ctrl]); cdh = np.array([dens(c, 0.24 + 0.5) for c in ctrl])
    print(f"SfM 點（半徑 {rad}）：大樓處 {n_t:,}（離地 > 0.5 的 {n_t_hi:,}）；同區塊隨機 200 處 中位 {np.median(cd):.0f}（p10 {np.percentile(cd,10):.0f} / p90 {np.percentile(cd,90):.0f}），"
          f"離地 > 0.5 的中位 {np.median(cdh):.0f}（p90 {np.percentile(cdh,90):.0f}）")
    zs = P[np.linalg.norm(P[:, :2] - cxy[None], axis=1) < rad, 2]
    if zs.size:
        print(f"大樓處 SfM 點的高度分布：p10 {np.percentile(zs,10):.2f} / p50 {np.percentile(zs,50):.2f} / p90 {np.percentile(zs,90):.2f} / max {zs.max():.2f}")

    # ── 1 訓練視角：最接近該測試幀的訓練相機 ──
    C = -cams.R[fi].numpy().T @ cams.T[fi].numpy(); d = cams.R[fi].numpy().T @ np.array([0, 0, 1.0])
    tc = np.array([(-ts.cameras.R[i].numpy().T @ ts.cameras.T[i].numpy()) for i in range(len(ts))])
    td = np.array([ts.cameras.R[i].numpy().T @ np.array([0, 0, 1.0]) for i in range(len(ts))])
    sc = np.linalg.norm(tc - C, axis=1) + 2.0 * (1 - td @ d)
    near = np.argsort(sc)[:5]
    tiles_img = []
    for j in near:
        cj = ts.cameras[int(j)].to_device(dev)
        with torch.no_grad():
            oj = renderer(cj, model, bg_color=bg)
        rj = oj["render"].clamp(0, 1); gj = load_gt(ts.image_paths[int(j)], rj)
        print(f"  最近訓練視角 {ts.image_names[int(j)]}：單塊模型 PSNR {psnr(rj, gj):.2f}（距離 {np.linalg.norm(tc[j] - C):.3f}）")
        both = torch.cat([gj, rj], 2).permute(1, 2, 0).cpu().numpy()
        tiles_img.append((both * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(tiles_img[:2], 0)).resize((tiles_img[0].shape[1] // 2, tiles_img[0].shape[0])).save(os.path.join(a.out, "train_views.png"))
    test_both = torch.cat([gt_t, img, mrender], 2).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((test_both * 255).astype(np.uint8)).resize((test_both.shape[1] // 2, test_both.shape[0] // 2)).save(os.path.join(a.out, "test_view.png"))

    # ── 4 Depth Anything 深度圖：最近訓練視角 ──
    jn = ts.image_names[int(near[0])]
    dp = os.path.join(dmh["path"], "estimated_depths", jn + ".npy")
    if os.path.exists(dp):
        dm = np.load(dp).astype(np.float32)
        dm = (dm - np.percentile(dm, 1)) / max(np.percentile(dm, 99) - np.percentile(dm, 1), 1e-6)
        Image.fromarray((np.clip(dm, 0, 1) * 255).astype(np.uint8)).resize((dm.shape[1] // 2, dm.shape[0] // 2)).save(os.path.join(a.out, "depth_anything.png"))
        print(f"Depth Anything 深度圖：{dp} -> depth_anything.png")
    else:
        print(f"⚠ 找不到深度圖 {dp}")
    print(f"圖：{a.out}/test_view.png（GT｜單塊｜合併）、train_views.png（最近 2 個訓練視角：GT｜單塊）")


if __name__ == "__main__":
    main()
