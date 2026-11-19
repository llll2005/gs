"""★ 2026-11-19 對 MatrixCity GT 深度的渲染深度誤差（使用者「如果要發論文 缺什麼」=> 作 1：路面在不在正確高度要有正式指標）。

做法：合併模型渲染到抽樣的訓練視角（相機＝--cameras_ckpt 的 parser，全場景；同 tools/mesh_tsdf_stream.py），取 2DGS 的 surf_depth（median），
      與 MatrixCity 的 GT Z 深度（float32 EXR、cm；../mc_depth_f32/aerial/train）逐像素比。GT 深度從沒進過訓練（訓練只用 SfM 與單目）=> 這是乾淨的幾何指標。
像素：GT 有效（有限、> 0）、渲染 alpha > 0.5 的像素（覆蓋率另報）。
指標（e = d / d* − 1）：
  AbsRel 中位／平均 |e|；δ1.01、δ1.05＝max(d/d*, d*/d) < 1.01／1.05 的比例
  在後面（e > +2%）＝渲染表面在真表面後面（地下層、看穿）；在前面（e < −2%）＝浮塊、加厚
  也分「地面類」像素（GT 反投影的法向 |n_z| > 0.8）另報
用法：python tools/eval_depth_gt.py --models 名=ckpt ... --cameras_ckpt <merged ckpt> [--stride 20] [--max_depth 10]
⚠ lab：cwd＝gs；GT 在 ../mc_depth_f32（只有訓練視角）。
"""
import argparse, glob, json, os, re, sys, time
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--cameras_ckpt", required=True)
ap.add_argument("--stride", type=int, default=20)
ap.add_argument("--gt", default="../mc_depth_f32/aerial/train")
ap.add_argument("--fmap", default="data/matrix_city/aerial/global_frame_map_correct.json")
ap.add_argument("--unit", type=float, default=1e-4, help="場景單位／cm（MatrixCity：GT 姿態 x1/100 => 1e-4；下面另用渲染深度中位比值檢查）")
ap.add_argument("--max_depth", type=float, default=10.0, help="只算 GT 深度 < 這個值（場景單位）的像素")
ap.add_argument("--alpha", type=float, default=0.5)
a = ap.parse_args()

from internal.utils.gaussian_model_loader import GaussianModelLoader
dev = torch.device("cuda")

GTIDX = {}
for p in glob.glob(f"{a.gt}/block_*/**/*.exr", recursive=True):
    mb = re.search(r"block_(\d+)", os.path.relpath(p, a.gt)); ms = re.search(r"(\d+)$", os.path.splitext(os.path.basename(p))[0])
    if mb and ms:
        GTIDX[(int(mb.group(1)), int(ms.group(1)))] = p
FMAP = json.load(open(a.fmap))
to_key = lambda name: f"{int(os.path.splitext(os.path.basename(name))[0]) + 1:06d}.png"
print(f"GT 深度 EXR {len(GTIDX):,} 張", flush=True)

cc = torch.load(a.cameras_ckpt, map_location="cpu", weights_only=False)
dmh = cc["datamodule_hyper_parameters"]; pc = dmh["parser"]; del cc
if getattr(pc, "block_id", None) is not None:
    pc.block_id = None
if getattr(pc, "points_from", None) == "ply":
    pc.points_from = "random"
ts = pc.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
cams, names = ts.cameras, ts.image_names
idx = [i for i in range(0, len(cams), max(1, a.stride))]
sel = []
for i in idx:
    m = FMAP.get(to_key(names[i]))
    p = GTIDX.get((m["block"], m["frame_index"])) if m else None
    if p:
        sel.append((i, p))
print(f"[相機] train {len(cams)} 台；stride {a.stride} => {len(idx)} 台，有 GT 深度 {len(sel)} 台", flush=True)

gt_cache = {}
def gt_depth(i, p, W, H):
    if i not in gt_cache:
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        d = d[..., 0] if d.ndim == 3 else d
        d = cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) * a.unit
        gt_cache[i] = d
    return gt_cache[i]

def ground_mask(d, cam):
    """GT 深度反投影的世界法向 |n_z| > 0.8（地面類：路面、屋頂、水面）。"""
    H, W = d.shape
    fx, fy = float(cam.fx), float(cam.fy); cx, cy = float(cam.cx), float(cam.cy)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], -1)
    n = np.cross(np.gradient(X, axis=1), np.gradient(X, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    R = cam.R.cpu().numpy().astype(np.float32)   # world->camera 的旋轉（gspl 的 Camera.R；world_to_camera 欄位是轉置存的）
    nw = n @ R           # 相機 => 世界：n_w = R^T n_c，行向量寫法＝n_c @ R
    return np.abs(nw[..., 2]) > 0.8

for spec in a.models:
    nm, path = spec.split("=", 1)
    t0 = time.time()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GaussianModelLoader.initialize_model_from_checkpoint(ck, device=dev)
    model.freeze(); model.pre_activate_all_properties()
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ck, stage="validate", device=dev)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        if hasattr(renderer, f):
            setattr(renderer, f, False)
    del ck
    bg = torch.zeros(3, device=dev)
    S = {"all": [], "ground": []}; cov = []; ratio = []
    with torch.no_grad():
        for i, p in sel:
            cam = cams[i].to_device(dev)
            pkg = renderer(cam, model, bg)
            d = pkg["surf_depth"][0].float().cpu().numpy(); al = pkg["rend_alpha"][0].float().cpu().numpy()
            H, W = d.shape
            g = gt_depth(i, p, W, H)
            valid = np.isfinite(g) & (g > 0) & (g < a.max_depth)
            vis = valid & (al > a.alpha) & (d > 0)
            cov.append(vis.sum() / max(valid.sum(), 1))
            if vis.sum() < 100:
                continue
            e = d[vis] / g[vis] - 1.0
            ratio.append(np.median(d[vis] / g[vis]))
            S["all"].append(e)
            gm = ground_mask(g, cams[i])
            S["ground"].append(d[vis & gm] / g[vis & gm] - 1.0)
    print(f"== {nm}｜N={model.n_gaussians:,}｜{len(sel)} 視角｜覆蓋（alpha>{a.alpha}）{100 * np.mean(cov):.1f}%｜渲染/GT 深度比中位 {np.median(ratio):.4f}（≈1 => 單位對）｜{time.time() - t0:.0f} s", flush=True)
    for k in ("all", "ground"):
        e = np.concatenate(S[k]) if S[k] else np.zeros(0)
        if not len(e):
            continue
        r = np.maximum(1 + e, 1 / np.maximum(1 + e, 1e-6))
        print(f"   {'全部像素' if k == 'all' else '地面類像素'}（{len(e):,}）：AbsRel 中位 {100 * np.median(np.abs(e)):.3f}%、平均 {100 * np.mean(np.abs(e)):.2f}%｜"
              f"δ1.01 {100 * np.mean(r < 1.01):.1f}%、δ1.05 {100 * np.mean(r < 1.05):.1f}%｜在後面（>+2%）{100 * np.mean(e > 0.02):.2f}%、在前面（<−2%）{100 * np.mean(e < -0.02):.2f}%", flush=True)
    del model, renderer; torch.cuda.empty_cache()
print("DONE eval_depth_gt", flush=True)
