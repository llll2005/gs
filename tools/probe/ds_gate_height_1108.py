"""11-08 使用者「高處側面擬合失敗、樓頂正常」：深度支撐閘門是不是把高處立面擋掉了（CPU；cwd＝repo 根；只讀）。
假說：單目深度把高樓壓扁（11-01：判前方 z 1~2 37%、2~3 47%、≥3 68%）；樓頂有 SfM 點 => 豁免，高處立面 SfM 稀 => 不豁免 =>
     閘門判「一直在前方」或「從未貼面」=> 不能當增生母點／搬遷目標 => 高處立面長不密（糊、破洞）。
可量的預測：閘門擋掉的比例 立面 >> 平面，而且隨高度升高；CityGS 的不透明表面顆粒（當作「真表面」）也一樣被擋。
重現訓練的規則（`internal/metrics/geom_gate.py` DepthSupport，rule vote）：每個視窗 --win 台訓練相機（訓練是 400 步每 8 步＝50 台），
  殘差 r＝(z − z_單目)／z_單目（不管遮擋，跟訓練一樣）；看過 ≥ 3 次、票數過半 r < −5%＝前方；所有觀測 |r| > 5%＝從未貼面；
  離 points3D.bin 的點 0.03 體素鄰域內＝豁免（`SfMVoxels`）。取 --nwin 個隨機視窗平均。
用法：python ds_gate_height_1108.py --models 標籤=ckpt ...
"""
import argparse, glob, json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from internal.utils.colmap import read_images_binary, read_cameras_binary, read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.metrics.geom_gate import SfMVoxels

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--nview", type=int, default=200)
ap.add_argument("--win", type=int, default=50)
ap.add_argument("--nwin", type=int, default=4)
ap.add_argument("--depth_dir", default="estimated_depths_lab")
ap.add_argument("--scale_name", default="estimated_depth_scales_lab")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()

names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
rng = np.random.default_rng(0)
names = [names[i] for i in sorted(rng.choice(len(names), min(a.nview, len(names)), replace=False))]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
SC = json.load(open(f"{D}/{a.scale_name}.json")); MED = float(np.median([v["scale"] for v in SC.values()]))
views = []
for nm in names:
    key = nm.rsplit(".", 1)[0] if nm.rsplit(".", 1)[0] in SC else nm
    sc = SC.get(key) or SC.get(nm)
    if sc is None or not (0.2 * MED <= sc["scale"] <= 5.0 * MED):
        continue                                            # 訓練時這張不載深度
    try:
        raw = np.load(f"{D}/{a.depth_dir}/{nm}.npy").astype(np.float32)
    except FileNotFoundError:
        continue
    im = ims[nm]; c = cams[im.camera_id]; W, H = round(c.width / 1.2), round(c.height / 1.2); s = W / c.width
    w2c = np.eye(4); w2c[:3, :3] = im.qvec2rotmat(); w2c[:3, 3] = im.tvec
    views.append((w2c, c.params[0] * s, c.params[1] * s, c.params[2] * s, c.params[3] * s, W, H, raw * sc["scale"] + sc["offset"]))
print(f"b{B}｜有單目深度的訓練視角 {len(views)}（抽 {a.nview}）｜視窗 {a.win} 台 × {a.nwin} 次", flush=True)

pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
vox = SfMVoxels(np.stack([q.xyz for q in pts.values()]), 0.03, torch.device("cpu")); del pts

def quat_nz(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.abs(1 - 2 * (x * x + y * y))                  # 旋轉矩陣第 3 欄的 z 分量＝面元法向 z

BANDS = [0.05, 0.5, 1.5, 2.5]; BN = ["地面<5m", "5~50m", "50~150m", "150~250m", ">250m"]
for spec in a.models:
    lab, pat = spec.split("=", 1); f = sorted(glob.glob(pat))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy()
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy()
    nz = quat_nz(sd["gaussian_model.gaussians.rotations"].double().numpy()); del sd
    k = inblk(m) & (o > 0.5); m, nz = m[k], nz[k]
    n = len(m); R = np.full((len(views), n), np.nan, np.float32)
    mh = np.c_[m, np.ones(n)]
    for vi, (w2c, fx, fy, cx, cy, W, H, inv) in enumerate(views):
        pcam = mh @ w2c.T; z = pcam[:, 2]
        ok = z > 0.2
        u = fx * pcam[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * pcam[:, 1] / np.maximum(z, 1e-6) + cy
        hd, wd = inv.shape
        ud = np.floor(u * (wd / W)).astype(np.int64); vd = np.floor(v * (hd / H)).astype(np.int64)
        ok &= (ud >= 0) & (ud < wd) & (vd >= 0) & (vd < hd)
        di = np.zeros(n, np.float32); di[ok] = inv[vd[ok], ud[ok]]
        ok &= di > 1e-6
        zm = 1.0 / np.maximum(di, 1e-6)
        R[vi, ok] = ((z - zm) / zm)[ok]
    near = vox.near(torch.from_numpy(m).float()).numpy()
    gated = np.zeros(n); front_ = np.zeros(n); never_ = np.zeros(n)
    for w in range(a.nwin):
        sel = rng.choice(len(views), min(a.win, len(views)), replace=False)
        r = R[sel]; okm = np.isfinite(r); cnt = okm.sum(0)
        seen = cnt >= 3
        fr = seen & ((okm & (np.nan_to_num(r, nan=0.0) < -0.05)).sum(0) > 0.5 * cnt)
        nv = seen & (np.where(okm, np.abs(np.nan_to_num(r, nan=0.0)), np.inf).min(0) > 0.05)
        g = (fr | nv) & ~near
        gated += g; front_ += fr & ~near; never_ += nv & ~fr & ~near
    gated /= a.nwin; front_ /= a.nwin; never_ /= a.nwin
    fac = nz < 0.3; flat = nz > 0.85
    band = np.searchsorted(BANDS, m[:, 2], side="right")
    print(f"\n[{lab}] 塊內不透明 {n:,}｜SfM 豁免 {100 * near.mean():.1f}%")
    print(f"{'':10}" + "".join(f"{b:>30}" for b in BN))
    for cn, cm in (("立面", fac), ("平面", flat)):
        row = ""
        for bi in range(len(BN)):
            s = cm & (band == bi)
            if s.sum() < 100:
                row += f"{'—':>30}"; continue
            row += f"{f'{100 * gated[s].mean():4.1f}%（前{100 * front_[s].mean():.0f} 未貼{100 * never_[s].mean():.0f} 豁{100 * near[s].mean():.0f}）n{s.sum() // 1000}k':>30}"
        print(f"{cn:10}{row}")
print("\n欄：閘門擋掉%（前方%／從未貼面%／SfM 豁免%）n＝顆數（千）")
print("DONE ds_gate_height")
