"""γ（BYTES_PER_ISECT）的直接量測：binning buffer 的實際位元組 / 相交數。

峰值模型 `[B·N + γ·load]/K` 的 γ 目前寫死 24 = key(8)x2 + value(4)x2（CUB ping-pong），
但 BinningState 還有第五項 `list_sorting_space`（CUB radix sort 的暫存），**沒算進去**。
這支直接呼叫 _C.rasterize_gaussians 拿回 binningBuffer 張量，量它的真實大小。
"""
import sys, os, math, torch
sys.path.insert(0, os.getcwd())
from internal.utils.gaussian_model_loader import GaussianModelLoader
from diff_trim_surfel_rasterization import _C
ck = sys.argv[1]; n_cam = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = torch.device("cuda")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
    ck, device=dev, eval_mode=True, pre_activate=False)
c = torch.load(ck, map_location="cpu"); dmh = c["datamodule_hyper_parameters"]
dp = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(ck)), global_rank=0)
cams = dp.get_outputs().train_set.cameras
bg = torch.zeros((3,), device=dev)
N = model.n_gaussians
means3D = model.get_xyz.detach()
opacity = model.get_opacities().detach()
scales = model.get_scales().detach()
rots = model.get_rotations().detach()
shs = model.get_shs().detach() if hasattr(model, "get_shs") else model.gaussians["shs"].detach()
empty = torch.tensor([], device=dev)
rows = []
step = max(1, len(cams) // n_cam)
for i in range(0, len(cams), step):
    cam = cams[i].to_device(dev)
    args = (bg, means3D, empty, opacity, scales, rots, 1.0, empty,
            cam.world_to_camera, cam.full_projection,
            math.tan(float(cam.fov_x) * 0.5), math.tan(float(cam.fov_y) * 0.5), 0.0,
            int(cam.height), int(cam.width), shs, model.active_sh_degree,
            cam.camera_center, False, False, False, False)
    with torch.no_grad():
        out = _C.rasterize_gaussians(*args)
    num_rendered, binningBuffer, tiles = out[0], out[5], out[9]
    b = binningBuffer.numel()          # uint8 張量 => numel == bytes
    t = int(tiles.sum())
    rows.append((num_rendered, t, b))
    if len(rows) >= n_cam: break
print(f"N = {N:,}   相機 {len(rows)} 台")
print(f"{'num_rendered':>14} {'Σtiles':>14} {'binning bytes':>16} {'B/相交':>9}")
for nr, t, b in rows:
    print(f"{nr:>14,} {t:>14,} {b:>16,} {b/max(nr,1):>9.2f}")
g = [b / max(nr, 1) for nr, _, b in rows]
same = all(nr == t for nr, t, _ in rows)
print(f"\nnum_rendered == Σtiles ？ {'✅ 完全相同' if same else '⛔ 不同（tiles 的定義要重查）'}")
print(f"實際 γ：中位 {sorted(g)[len(g)//2]:.2f} B/相交   範圍 {min(g):.2f}~{max(g):.2f}")
print(f"程式裡寫死的 BYTES_PER_ISECT = 24.0  => 低估 {100*(sorted(g)[len(g)//2]/24-1):+.1f}%")
