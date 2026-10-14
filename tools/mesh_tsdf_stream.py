"""逐張融合的 TSDF 抽網格（2026-10-14，幾何評測 A：我方 vs 官方 CityGSV2 的 F1）。

與官方 `utils/gs2d_mesh_extraction.py`（internal/entrypoints/gs2d_mesh_extraction.py）同一套算法與參數：
  有界 TSDF（Open3D ScalableTSDFVolume）、RGB 只用 SH0（active_sh_degree=0）、深度＝renderer 的 surf_depth、
  後處理＝保留最大的 num_cluster 個連通塊（post_process_mesh）；MatrixCity 空拍的官方參數 voxel 0.01／sdf_trunc 0.04／depth_trunc 5.0
  （scripts/citygs/run_citygs_mc_aerial.sh）。
唯一差別：官方先把**全部**視角的 RGB／深度渲染完存在 RAM 再融合（5,621 張 1600x900 約 129 GB），這裡每渲染一張就融合一張
  => 結果相同（融合順序一樣），RAM 只剩 TSDF 體積本身。
相機：預設取 `--cameras_ckpt` 的 dataparser 的 train_set（兩個模型用同一組相機 => 可比）；塊 parser 的 block_id 一律清成 None（全場景）。
輸出：<out>/fuse.ply、fuse_post.ply（官方 eval_tnt 吃 fuse_post.ply）。
⚠ lab 上 open3d 要先 `export LD_LIBRARY_PATH=/workspace/data/hdd/11213/o3dlibs/lib`（無頭容器缺 libX11，裝在本資料夾底下的前綴）。
用法：python tools/mesh_tsdf_stream.py --ckpt <merged ckpt> --out <dir> [--cameras_ckpt <ckpt>] [--stride 1]
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras_ckpt", default=None, help="取相機的 ckpt（預設＝--ckpt）；兩個模型比較時用同一個")
    ap.add_argument("--dataset_path", default=None, help="覆寫相機 ckpt 裡記的資料路徑")
    ap.add_argument("--voxel_size", type=float, default=0.01)
    ap.add_argument("--sdf_trunc", type=float, default=0.04)
    ap.add_argument("--depth_trunc", type=float, default=5.0)
    ap.add_argument("--num_cluster", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1, help="每 n 台相機取一台（1＝全部，同官方）")
    a = ap.parse_args()
    import open3d as o3d
    from tqdm import tqdm
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.gs2d_mesh_utils import GS2DMeshUtils, post_process_mesh
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    model = GaussianModelLoader.initialize_model_from_checkpoint(ck, device=dev)
    model.freeze(); model.pre_activate_all_properties()
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ck, stage="validate", device=dev)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        if hasattr(renderer, f):
            setattr(renderer, f, False)      # 訓練期加速旗標（渲染逐位元相同）
    del ck
    cc = torch.load(a.cameras_ckpt or a.ckpt, map_location="cpu")
    dmh = cc["datamodule_hyper_parameters"]; pc = dmh["parser"]; del cc
    if getattr(pc, "block_id", None) is not None:
        print(f"[相機] parser {type(pc).__name__} block_id {pc.block_id} -> None（全場景）")
        pc.block_id = None
    if getattr(pc, "points_from", None) == "ply":
        pc.points_from = "random"
    path = a.dataset_path or dmh["path"]
    cams = pc.instantiate(path=path, output_path="/tmp", global_rank=0).get_outputs().train_set.cameras
    idx = list(range(0, len(cams), max(1, a.stride)))
    print(f"[模型] {a.ckpt}  N={model.n_gaussians:,}  renderer {type(renderer).__name__} depth_ratio={getattr(renderer, 'depth_ratio', '?')}")
    print(f"[相機] {path}：train {len(cams)} 台，用 {len(idx)} 台（stride {a.stride}）")
    print(f"[TSDF] voxel {a.voxel_size}  sdf_trunc {a.sdf_trunc}  depth_trunc {a.depth_trunc}")

    import resource, time
    t_start = time.perf_counter(); t_render = t_fuse = 0.0
    model.active_sh_degree = 0
    bg = torch.zeros(3, device=dev)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel_size, sdf_trunc=a.sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    with torch.no_grad():
        for i in tqdm(idx, desc="渲染＋融合", mininterval=30):
            t0 = time.perf_counter()
            cam = cams[i].to_device(dev)
            pkg = renderer(cam, model, bg)
            rgb, depth = pkg["render"], pkg["surf_depth"]
            torch.cuda.synchronize(); t1 = time.perf_counter(); t_render += t1 - t0
            co = GS2DMeshUtils.to_cam_open3d([cam])[0]
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")),
                depth_trunc=a.depth_trunc, convert_rgb_to_intensity=False, depth_scale=1.0)
            volume.integrate(rgbd, intrinsic=co.intrinsic, extrinsic=co.extrinsic)
            t_fuse += time.perf_counter() - t1
    t0 = time.perf_counter()
    mesh = volume.extract_triangle_mesh()
    t_ext = time.perf_counter() - t0
    # ★ 成本紀錄（ADMM 框架 14 §3.2：z＝全場景 TSDF 時，每輪 z-更新＝這裡的融合）
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[成本] {len(idx)} 視角：渲染 {t_render:.0f} s、TSDF 融合 {t_fuse:.0f} s（每視角 {1000 * t_fuse / max(len(idx), 1):.0f} ms）、"
          f"抽網格 {t_ext:.0f} s、總計 {time.perf_counter() - t_start:.0f} s；行程 RAM 峰值 {rss:.1f} GiB（含 ckpt 載入）", flush=True)
    o3d.io.write_triangle_mesh(os.path.join(a.out, "fuse.ply"), mesh)
    print(f"網格：{len(mesh.vertices):,} 頂點、{len(mesh.triangles):,} 三角形 -> {a.out}/fuse.ply")
    post = post_process_mesh(mesh, cluster_to_keep=a.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(a.out, "fuse_post.ply"), post)
    print(f"後處理（保留最大 {a.num_cluster} 個連通塊）：{len(post.triangles):,} 三角形 -> {a.out}/fuse_post.ply")


if __name__ == "__main__":
    main()
