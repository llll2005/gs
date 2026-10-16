"""ADMM 統一框架 P2 的 z-更新與對偶更新（2026-10-14，`紀錄/實驗分析/14` §11）。

一輪結束時對一個區塊做：
  1 A_v x_v：目前模型在本塊全部訓練視角的渲染深度
  2 z-更新：把 (A x + u_prev) 對應的深度（＋可選的鄰塊模型在同一批視角的深度）融合成 TSDF（官方空拍參數 voxel 0.01／sdf_trunc 0.04／depth_trunc 5.0）
    —— 二次懲罰下的共識 z 就是各看法的加權平均＝TSDF 融合（14 §3.2）；抽成網格後對每個視角 raycast => 共識逆深度 z_k(p)
  3 對偶更新（縮放形式，逆深度空間、逐視角逐像素）：u <- u + (A x - z)
  4 下一輪 x-更新的目標＝z - u（x-更新 = f_v + rho/2 ||A x - z + u||^2，用現有的深度 loss 實現，權重＝rho）
     存成 <out>/<影像檔名>.npy（float32，沒打到表面＝NaN => dataparser 遮罩）＋ <out>/consensus_scales.json（scale 1、offset 0）
  depth_trunc 5.0（官方抽網格值）：60k 模型（full44 b10）覆蓋像素中深度 > 5 的比例中位 0.1%、p90 24%（少數斜視視角）
    => 超出的像素沒有共識目標（NaN 遮罩），不加大：與 F1 抽網格同參數，遠處的逆深度目標精度也低。
  可觀測性：原始殘差 r＝|A x - z|、對偶殘差 s＝|z - z_prev|（逆深度相對值）、有效像素比例、raycast 命中率。
⚠ lab 上 open3d 要 `export LD_LIBRARY_PATH=/workspace/data/hdd/11213/o3dlibs/lib`。
用法：python tools/admm_consensus.py --ckpt <本塊目前 ckpt> --out <下一輪深度目錄> [--prev <上一輪目錄>] [--neighbors <ckpt>...] [--stride 1]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TRAIN_FLAGS = ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train")


def load(ckpt, dev):
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ckpt, device=dev, eval_mode=True)
    for f in TRAIN_FLAGS:
        if hasattr(renderer, f):
            setattr(renderer, f, False)
    return model, renderer, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", default=None, help="上一輪的目錄（讀 u 與 z_prev）；第一次不給 => u=0")
    ap.add_argument("--neighbors", nargs="*", default=[], help="P2-b：鄰塊模型也在本塊視角渲染並融合進 z")
    ap.add_argument("--neighbors_all", action="store_true",
                    help="鄰塊用全部顆粒（預設只用鄰塊自己分區盒內的顆粒＝合併時的歸屬規則：鄰塊只對自己負責的區域發言，"
                         "它在本塊區域的塊外顆粒品質差，不該混進共識）")
    ap.add_argument("--stride", type=int, default=1, help="只用每 n 台相機（功能測試用；正式＝1）")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--sdf_trunc", type=float, default=0.04)
    ap.add_argument("--depth_trunc", type=float, default=5.0)
    a = ap.parse_args()
    import open3d as o3d
    from internal.utils.gs2d_mesh_utils import GS2DMeshUtils
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    model, renderer, ck = load(a.ckpt, dev)
    dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; step = int(ck.get("global_step", -1)); del ck
    if getattr(pc, "points_from", None) == "ply":
        pc.points_from = "random"
    ts = pc.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
    idx = list(range(0, len(ts), max(1, a.stride)))
    names = [os.path.basename(ts.image_paths[i]) for i in idx]
    print(f"[z-更新] block {pc.block_id}、ckpt 步數 {step}、視角 {len(idx)}／{len(ts)}、鄰塊 {len(a.neighbors)}", flush=True)
    bg = torch.zeros(3, device=dev)
    # ── 1 A_v x_v：本塊渲染逆深度（之後算殘差用；同時融合） ──
    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=a.voxel, sdf_trunc=a.sdf_trunc,
                                                       color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    Ax = {}
    models = [(model, renderer)]
    for nb in a.neighbors:
        m2, r2, ck2 = load(nb, dev)
        if not a.neighbors_all:
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            from swap_block_eval import block_mask
            pc2 = ck2["datamodule_hyper_parameters"]["parser"]
            msk = block_mask(m2.get_xyz, pc2)[pc2.block_id].to(dev)
            n0 = int(msk.numel())
            m2.properties = {k: v[msk] for k, v in m2.properties.items()}
            print(f"[鄰塊] block {pc2.block_id}：只用盒內 {int(msk.sum()):,}／{n0:,} 顆（{os.path.basename(nb)}）", flush=True)
        del ck2
        models.append((m2, r2))
    with torch.no_grad():
        for k, i in enumerate(idx):
            cam = ts.cameras[i].to_device(dev)
            co = GS2DMeshUtils.to_cam_open3d([cam])[0]
            for j, (m_, r_) in enumerate(models):
                pkg = r_(cam, m_, bg)
                d = pkg["surf_depth"][0]
                alpha = pkg["rend_alpha"].reshape(d.shape)
                dd = d.clone(); dd[alpha <= 0.5] = 0
                if j == 0:
                    inv = torch.where((alpha > 0.5) & (d > 1e-6), 1.0 / d.clamp_min(1e-6), torch.full_like(d, float("nan")))
                    Ax[names[k]] = inv.cpu().numpy().astype(np.float32)
                    # 縮放形式 ADMM 的 z-更新融合的是 (A x + u)，不是只有 A x（u 在逆深度空間、逐視角逐像素）
                    up = os.path.join(a.prev, names[k] + ".u.npy") if a.prev else None
                    if up and os.path.exists(up):
                        u_prev = torch.from_numpy(np.load(up).astype(np.float32)).to(dev)
                        adj = inv + torch.nan_to_num(u_prev, nan=0.0)
                        dd = torch.where(torch.isfinite(adj) & (adj > 1e-6), 1.0 / adj.clamp_min(1e-6), torch.zeros_like(d))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.zeros((d.shape[0], d.shape[1], 3), np.uint8)),
                    o3d.geometry.Image(np.ascontiguousarray(dd.cpu().numpy())), depth_trunc=a.depth_trunc,
                    convert_rgb_to_intensity=False, depth_scale=1.0)
                vol.integrate(rgbd, intrinsic=co.intrinsic, extrinsic=co.extrinsic)
            if (k + 1) % 200 == 0:
                print(f"  渲染＋融合 {k + 1}/{len(idx)}", flush=True)
    t1 = time.perf_counter()
    mesh = vol.extract_triangle_mesh(); del vol
    print(f"[z-更新] TSDF 網格 {len(mesh.triangles):,} 三角形（{t1 - t0:.0f} s 渲染＋融合）", flush=True)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)); del mesh
    # ── 2 raycast 每個視角 => 共識逆深度 z；3 對偶更新；4 目標＝z - u ──
    stats = {"r": [], "s": [], "valid": [], "hit": []}
    scales = {}
    for k, i in enumerate(idx):
        cam = ts.cameras[i]
        W, H = int(cam.width), int(cam.height)
        Kt = o3d.core.Tensor([[float(cam.fx), 0, float(cam.cx)], [0, float(cam.fy), float(cam.cy)], [0, 0, 1]], o3d.core.Dtype.Float64)
        Ext = np.eye(4); Ext[:3, :3] = cam.R.numpy(); Ext[:3, 3] = cam.T.numpy()
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(Kt, o3d.core.Tensor(Ext), W, H)
        t_hit = scene.cast_rays(rays)["t_hit"].numpy()                     # 沿射線的距離（射線方向未正規化 => 方向 z 分量為 1 時就是深度）
        dirs = rays.numpy()[..., 3:]
        dz = (dirs @ cam.R.numpy().T)[..., 2]                               # 射線方向在相機座標的 z 分量
        depth = t_hit * dz
        z = np.where(np.isfinite(depth) & (depth > 1e-6), 1.0 / np.maximum(depth, 1e-6), np.nan).astype(np.float32)
        nm = names[k]
        u = np.zeros_like(z)
        zp = None
        if a.prev and os.path.exists(os.path.join(a.prev, nm + ".u.npy")):
            u = np.load(os.path.join(a.prev, nm + ".u.npy")).astype(np.float32)
            zp = np.load(os.path.join(a.prev, nm + ".z.npy")).astype(np.float32)
        ax = Ax[nm]
        both = np.isfinite(ax) & np.isfinite(z)
        u = np.where(both, u + (ax - z), u).astype(np.float32)
        target = np.where(np.isfinite(z), z - u, np.nan).astype(np.float32)
        np.save(os.path.join(a.out, nm + ".npy"), target)
        np.save(os.path.join(a.out, nm + ".u.npy"), u.astype(np.float16))
        np.save(os.path.join(a.out, nm + ".z.npy"), z.astype(np.float16))
        scales[ts.image_names[i]] = {"scale": 1.0, "offset": 0.0}
        if both.any():
            stats["r"].append(float(np.median(np.abs(ax[both] - z[both]) / np.abs(z[both]))))
        if zp is not None:
            m2 = np.isfinite(z) & np.isfinite(zp)
            if m2.any():
                stats["s"].append(float(np.median(np.abs(z[m2] - zp[m2]) / np.abs(z[m2]))))
        stats["valid"].append(float(np.isfinite(target).mean())); stats["hit"].append(float(np.isfinite(z).mean()))
    with open(os.path.join(a.out, "consensus_scales.json"), "w") as f:
        json.dump(scales, f)
    q = lambda v: (f"中位 {np.median(v):.4f}／p90 {np.percentile(v, 90):.4f}" if v else "—")
    summ = {k: ([float(np.median(v)), float(np.percentile(v, 90))] if v else None) for k, v in stats.items()}
    summ.update(step=step, views=len(idx), seconds=time.perf_counter() - t0)
    with open(os.path.join(a.out, "consensus_stats.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(f"[殘差] 原始 |Ax - z|／z {q(stats['r'])}；對偶 |z - z_prev|／z {q(stats['s'])}；"
          f"raycast 命中 {q(stats['hit'])}；有效目標像素 {q(stats['valid'])}；共 {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
