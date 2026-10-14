"""區塊級幾何 F1（2026-10-14，使用者：每個單變數臂都要有幾何分數；`紀錄/實驗分析/09` §6 全場景 F1 我方 0.356 vs 官方 0.597）。

做法（與換塊評測 tools/swap_block_eval.py 同一個「只換一塊」的想法）：
  1 合併模型（預設 full44）把第 B 塊盒內顆粒換成候選 ckpt 的盒內顆粒（merge 歸屬規則；候選不給＝基準本身）
  2 只用第 B 塊的訓練相機（分區清單）逐張渲染＋TSDF 融合（官方 MatrixCity 空拍參數：voxel 0.01／sdf_trunc 0.04／depth_trunc 5.0、
    SH0、保留最大 50 個連通塊；同 tools/mesh_tsdf_stream.py）
  3 網格均勻表面取樣（點數＝頂點＋面，同 tools/eval_tnt/run.py）；網格點與 GT 都裁到「官方 crop volume ∩ 第 B 塊分區盒」
  4 F1：precision＝網格點離 GT 最近距離 < dTau 的比例；recall＝GT 點離網格最近距離 < dTau 的比例；
    dTau＝全場景 GT 最近鄰平均距離 x 1.5（與 eval_tnt 同公式）
  ⚠ 不做 eval_tnt 的 ICP 配準：兩邊同一座標系（GT 姿態 x1/100），各候選各自配準會把配準雜訊混進比較。
    `--mesh` 給現成網格（不裁到區塊、--block -1）可與 eval_tnt 的全場景結果對照，驗證不配準的差距。
⚠ lab 上 open3d 要先 `export LD_LIBRARY_PATH=/workspace/data/hdd/11213/o3dlibs/lib`。
用法：python tools/block_f1.py --block 6 --cand <ckpt>... --names <名>... [--merged <merged.ckpt>] [--out <dir>]
      python tools/block_f1.py --block -1 --mesh outputs/lab/geom_f1/ours/fuse_post.ply   （驗證）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
TRAIN_FLAGS = ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train")


def sample_mesh(path, seed=0):
    import trimesh
    m = trimesh.load_mesh(path)
    np.random.seed(seed)
    return np.asarray(trimesh.sample.sample_surface_even(m, m.vertices.shape[0] + m.faces.shape[0])[0], np.float64)


def f1(pts, gt, tau):
    import open3d as o3d
    a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)); b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
    dp = np.asarray(a.compute_point_cloud_distance(b)); dr = np.asarray(b.compute_point_cloud_distance(a))
    p = float((dp < tau).mean()) if dp.size else 0.0; r = float((dr < tau).mean()) if dr.size else 0.0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True, help="-1＝不裁到區塊（全場景，驗證用）")
    ap.add_argument("--merged", default="outputs/lab/full44_best/checkpoints/merged.ckpt")
    ap.add_argument("--cand", nargs="*", default=[])
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--mesh", default=None, help="直接評一個現成網格（不抽網格）")
    ap.add_argument("--gt_dir", default="data/geometry_gt/MC_Aerial")
    ap.add_argument("--scene", default="Block_all_ds")
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 4])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="outputs/lab/geom_f1/blocks")
    a = ap.parse_args()
    import open3d as o3d
    os.makedirs(a.out, exist_ok=True)
    GX, GY = a.grid
    # ── GT、官方 crop volume、dTau ──
    gt_pcd = o3d.io.read_point_cloud(os.path.join(a.gt_dir, a.scene + ".ply"))
    tau = float(np.mean(gt_pcd.compute_nearest_neighbor_distance())) * 1.5
    vol = o3d.visualization.read_selection_polygon_volume(os.path.join(a.gt_dir, a.scene + ".json"))
    gt = np.asarray(vol.crop_point_cloud(gt_pcd).points)
    pdir = os.path.join("data/matrix_city/aerial/train/block_all/partition", f"partitions-dim_{GX}_{GY}_visibility_0.08")
    parts = torch.load(os.path.join(pdir, "partitions.pt"))
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    if a.block >= 0:
        from internal.utils.citygs_partitioning_utils import PartitionCoordinates
        boxes = PartitionCoordinates(id=parts["partition_coordinates"]["id"], xy=parts["partition_coordinates"]["xy"]).get_bounding_boxes(
            parts["scene_config"]["partition_size"], enlarge=0.)
        bmin, bmax = boxes.min[a.block].double().numpy(), boxes.max[a.block].double().numpy()
        def inbox(x):
            q = x @ Rp.T
            return np.all((q[:, :2] >= bmin) & (q[:, :2] <= bmax), 1)
    else:
        inbox = lambda x: np.ones(len(x), bool)
    gtb = gt[inbox(gt)]
    print(f"[GT] crop 後 {len(gt):,} 點；本區域 {len(gtb):,} 點；dTau {tau:.4f}", flush=True)

    def score(mesh_path, tag):
        pts = np.asarray(vol.crop_point_cloud(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_mesh(mesh_path)))).points)
        pts = pts[inbox(pts)]
        p, r, f = f1(pts, gtb, tau)
        print(f"  {tag:<34} 網格取樣點 {len(pts):>11,}  P {p:.4f}  R {r:.4f}  F1 {f:.4f}", flush=True)
        return dict(name=tag, block=a.block, n_pts=len(pts), P=p, R=r, F1=f)

    rows = []
    if a.mesh:
        rows.append(score(a.mesh, os.path.basename(os.path.dirname(a.mesh)) or a.mesh))
    else:
        assert a.block >= 0, "抽網格模式要給 --block"
        from internal.utils.gaussian_model_loader import GaussianModelLoader
        from internal.utils.gs2d_mesh_utils import GS2DMeshUtils, post_process_mesh
        from swap_block_eval import block_mask
        dev = torch.device("cuda")
        model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.merged, device=dev, eval_mode=True)
        for f_ in TRAIN_FLAGS:
            if hasattr(renderer, f_):
                setattr(renderer, f_, False)
        dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; del ck
        pc.block_id = None
        if getattr(pc, "points_from", None) == "ply":
            pc.points_from = "random"
        ts = pc.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
        name2i = {n: i for i, n in enumerate(ts.image_names)}
        with open(os.path.join(pdir, f"{a.block % GX:03d}_{a.block // GX:03d}.txt")) as f_:
            cam_idx = [name2i[l.strip()] for l in f_ if l.strip() in name2i][:: max(1, a.stride)]
        base = {k: v.detach().cpu() for k, v in model.properties.items()}
        model.properties = {k: v[:1].to(dev) for k, v in base.items()}; torch.cuda.empty_cache()
        bm = None
        variants = [("基準（合併模型原樣）", None)]
        for c, nm in zip(a.cand, a.names or [os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c)))) for c in a.cand]):
            cck = torch.load(c, map_location="cpu"); cpc = cck["datamodule_hyper_parameters"]["parser"]; del cck
            assert list(cpc.block_dim or []) == [GX, GY] and cpc.block_id == a.block, f"{c}：分區不符"
            if bm is None:
                bm = block_mask(base["means"], cpc)[a.block]
            cm, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(c, device=torch.device("cpu"), eval_mode=True)
            cp = {k: v.detach().cpu() for k, v in cm.properties.items()}; del cm
            if "shs" in cp and cp["shs"].shape[1] < base["shs"].shape[1]:       # SH 較低：補 0（與換塊評測相同）
                r_ = cp["shs"]; cp["shs"] = torch.cat([r_, torch.zeros(r_.shape[0], base["shs"].shape[1] - r_.shape[1], 3)], 1)
            cmask = block_mask(cp["means"], cpc)[a.block]
            variants.append((nm, {k: cp[k][cmask] for k in base}))
            print(f"[{nm}] 盒內 {int(cmask.sum()):,} 顆", flush=True)
        print(f"[相機] block {a.block} 的訓練相機 {len(cam_idx)} 台（stride {a.stride}）", flush=True)
        bg = torch.zeros(3, device=dev)
        for nm, cin in variants:
            props = base if cin is None else {k: torch.cat([base[k][~bm], cin[k]], 0) for k in base}
            model.properties = {k: v.to(dev) for k, v in props.items()}
            model.active_sh_degree = 0
            volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.01, sdf_trunc=0.04,
                                                                  color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            with torch.no_grad():
                for i in cam_idx:
                    cam = ts.cameras[i].to_device(dev)
                    pkg = renderer(cam, model, bg)
                    co = GS2DMeshUtils.to_cam_open3d([cam])[0]
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d.geometry.Image(np.asarray(np.clip(pkg["render"].permute(1, 2, 0).cpu().numpy(), 0, 1) * 255, order="C", dtype=np.uint8)),
                        o3d.geometry.Image(np.asarray(pkg["surf_depth"].permute(1, 2, 0).cpu().numpy(), order="C")),
                        depth_trunc=5.0, convert_rgb_to_intensity=False, depth_scale=1.0)
                    volume.integrate(rgbd, intrinsic=co.intrinsic, extrinsic=co.extrinsic)
            model.properties = {k: v[:1].to(dev) for k, v in base.items()}; torch.cuda.empty_cache()
            mesh = post_process_mesh(volume.extract_triangle_mesh(), cluster_to_keep=50); del volume
            mp = os.path.join(a.out, f"b{a.block}_{nm.replace('（', '_').replace('）', '').replace(' ', '_')}.ply")
            o3d.io.write_triangle_mesh(mp, mesh); del mesh
            rows.append(score(mp, nm))
    import pandas as pd
    out_csv = os.path.join(a.out, f"block_f1_b{a.block}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"CSV：{out_csv}")


if __name__ == "__main__":
    main()
