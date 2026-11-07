"""11-07 使用者「把光度低的、沒貢獻的都刪掉，大部分問題就不存在」：離線刪除測試（不訓練）。GPU；cwd＝repo 根。
每顆粒在本塊全部訓練相機（分區清單）的：
  F＝中心投影在畫面內的台數；V＝trim 同一個渲染函式（record_transmittance）的每覆蓋像素平均貢獻，逐台加總；Vmax＝逐台最大值。
  ⚠ 渲染器的覆蓋數不計已飽和（T < 1e-4）像素後面的顆粒 => 完全被擋的顆粒 V＝0、覆蓋 0；訓練中的累加式 trim 把它們當「未觀測」永遠不剪。
刪法（存 ckpt，丟掉 optimizer 狀態，給 held-out／viewer／量測工具用）：
  完全被擋   F >= 3 且 V == 0（任何一台都沒有被算到）
  幾乎沒貢獻 F >= 3 且 Vmax < --vmax（每一台的平均貢獻都 < 1%：含路面正下方一層、T 約 1% 的那些）
另印各區（地下／地面／建築範圍內／其他，同 hidden_probe_1107）、各 opacity 的刪除量。
用法：python tools/probe/zero_contrib_cut_1107.py --block 3 --ckpt <ckpt> --out <資料夾>
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from scipy.ndimage import uniform_filter
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary, read_cameras_binary, read_points3D_binary
from internal.cameras.cameras import Cameras
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--vmax", type=float, default=0.01)
ap.add_argument("--only", nargs="*", default=[], help="只存這幾種刪法的 ckpt（預設全部）")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda")
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.); bb1 = pc.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.05)
Rt = P["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double() @ Rt.T)[:, :2])[B].numpy()

pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
Pp = np.stack([q.xyz for q in pts.values()]); ok = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = Pp[ok & inb(Pp, bb1)]; G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u] = S[order, 2][f + c // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03); bldg = np.isfinite(med) & ~ground
gz = np.where(ground, med, np.nan); num = uniform_filter(np.nan_to_num(gz), size=9); den = uniform_filter(np.isfinite(gz).astype(float), size=9)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nanmedian(gz)))


def regions(x):
    cc = np.clip(np.floor(x[:, :2] / G).astype(int) - o0, 0, np.array(shp) - 1)
    h = x[:, 2] - gfill[cc[:, 0], cc[:, 1]]; g = ground[cc[:, 0], cc[:, 1]]; b = bldg[cc[:, 0], cc[:, 1]]
    R = {"地下": g & (h < -0.005), "地面 ±0.5 m": g & (np.abs(h) <= 0.005), "建築範圍內（離地 > 3 m）": b & (h > 0.03)}
    R["其他"] = ~(R["地下"] | R["地面 ±0.5 m"] | R["建築範圍內（離地 > 3 m）"])
    return R


names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}; cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = [n for n in names if n in ims]


def mk(nm):
    im = ims[nm]; cc = cams[im.camera_id]; W, H = round(cc.width / 1.2), round(cc.height / 1.2); s = W / cc.width
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(im.qvec2rotmat()), T=t(im.tvec), fx=t([cc.params[0] * s])[0], fy=t([cc.params[1] * s])[0], cx=t([cc.params[2] * s])[0],
                   cy=t([cc.params[3] * s])[0], width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1), distortion_params=None,
                   camera_type=torch.zeros(1, dtype=torch.int8))[0]


ck = sorted(glob.glob(a.ckpt))[-1]
model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
for fl in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull"):
    if hasattr(rnd, fl):
        setattr(rnd, fl, False)
model.to(dev)
X = model.get_xyz.detach().float(); N = X.shape[0]; Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
F = torch.zeros(N, device=dev); V = torch.zeros(N, device=dev); Vm = torch.zeros(N, device=dev)
bg = torch.zeros(3, device=dev)
for nm in names:
    cd = mk(nm).to_device(dev)
    with torch.no_grad():
        tr = rnd(cd, model, bg_color=bg, record_transmittance=True)
        if not torch.is_tensor(tr) or tr.shape[0] != N:
            raise RuntimeError(f"record_transmittance 回傳 {type(tr)} {getattr(tr, 'shape', None)}，預期 ({N},)")
        tr = tr.float().clamp_min(0)
        p = Xh @ cd.world_to_camera.float(); zc = p[:, 2]
        uu = float(cd.fx) * p[:, 0] / zc.clamp_min(1e-6) + float(cd.cx); vv = float(cd.fy) * p[:, 1] / zc.clamp_min(1e-6) + float(cd.cy)
        F += ((zc > 1e-3) & (uu >= 0) & (uu < float(cd.width)) & (vv >= 0) & (vv < float(cd.height))).float()
        V += tr; torch.maximum(Vm, tr, out=Vm)
F, V, Vm = F.cpu().numpy(), V.cpu().numpy(), Vm.cpu().numpy()
o = model.get_opacities().detach().reshape(-1).cpu().numpy()                     # 載入器已 pre_activate
Xn = X.cpu().numpy().astype(np.float64); IN = inb(Xn, bb0); R = regions(Xn)
cuts = {"完全被擋": (F >= 3) & (V == 0), "幾乎沒貢獻": (F >= 3) & (Vm < a.vmax),
        # 11-07 拆開：不透明卻每台貢獻都 < vmax => 每台穿透率都 < vmax/o（o > 0.5 => < 2%）＝躲在表面後面的實心塊；透明的淡顆粒另列
        "不透明且被擋": (F >= 3) & (Vm < a.vmax) & (o > 0.5), "透明且幾乎沒貢獻": (F >= 3) & (Vm < a.vmax) & (o <= 0.5)}
if a.only:
    cuts = {k: v for k, v in cuts.items() if k in a.only}
print(f"b{B}｜{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ck)))))}｜{len(names)} 台訓練相機｜{N:,} 顆（盒內 {int(IN.sum()):,}）"
      f"｜畫面內 0 台 {int((F == 0).sum()):,}｜Vmax 分位（畫面內 >= 3）q10/50/90 {np.percentile(Vm[F >= 3], [10, 50, 90]).round(4).tolist()}", flush=True)
print(f"{'刪法':10}{'刪掉':>10}{'其中 o>0.5':>12}" + "".join(f"{k:>24}" for k in R) + "   （各區：刪掉／該區 o>0.5 的盒內顆數）", flush=True)
summary = {}
for cn, cut in cuts.items():
    row = "".join(f"{f'{int((cut & m & IN & (o > 0.5)).sum()):,}／{int((m & IN & (o > 0.5)).sum()):,}':>24}" for m in R.values())
    print(f"{cn:10}{int(cut.sum()):>10,}{int((cut & (o > 0.5)).sum()):>12,}{row}", flush=True)
    summary[cn] = int(cut.sum())
    keep = torch.from_numpy(~cut)
    raw = torch.load(ck, map_location="cpu")
    sd = raw["state_dict"]; Nsd = sd["gaussian_model.gaussians.means"].shape[0]
    assert Nsd == N, (Nsd, N)
    sd2 = {k: (v[keep] if hasattr(v, "shape") and v.dim() and v.shape[0] == N else v) for k, v in sd.items()}
    ck2 = {k: v for k, v in raw.items() if k not in ("state_dict", "optimizer_states")}
    ck2["state_dict"] = sd2; ck2["optimizer_states"] = []
    od = f"{a.out}/{cn}/blocks/block_{B}/checkpoints"; os.makedirs(od, exist_ok=True)
    torch.save(ck2, f"{od}/{os.path.basename(ck)}"); del raw, sd, sd2, ck2
    for extra in ("cameras.json",):
        src = os.path.join(os.path.dirname(os.path.dirname(ck)), extra)
        if os.path.exists(src):
            import shutil; shutil.copy(src, f"{a.out}/{cn}/blocks/block_{B}/{extra}")
json.dump(summary, open(f"{a.out}/summary.json", "w"), ensure_ascii=False)
print("DONE")
