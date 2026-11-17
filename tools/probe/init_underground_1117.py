"""11-17：平面 v2 起點（densepl2）為什麼讓陸地地下層變多——先看起點本身：陸地地面格裡低於 SfM 中位 0.5 m 以上的起點點數（只讀、CPU）。
地面格定義同 ground_split_sea_1117.py（2 m 格、可靠 SfM 中位 z < 3 m；陸地＝中位 >= −2 m）。
用法：python tools/probe/init_underground_1117.py <塊> 名稱=<PLY> ...
"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from plyfile import PlyData
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.colmap import read_points3D_binary
D = "data/matrix_city/aerial/train/block_all"; B = int(sys.argv[1])
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
Pt = np.stack([x.xyz for x in pts.values()]); trk = np.array([len(x.image_ids) for x in pts.values()]); err = np.array([x.error for x in pts.values()])
Pt = Pt[(trk >= 3) & (err < 1.5)]
CELL = 0.02
key = lambda xy: (np.floor(xy[:, 0] / CELL).astype(np.int64) << 32) + np.floor(xy[:, 1] / CELL).astype(np.int64)
ks = key(Pt[:, :2]); order = np.argsort(ks); ks, zs = ks[order], Pt[order, 2]
uk, st = np.unique(ks, return_index=True); med = np.array([np.median(z) for z in np.split(zs, st[1:])])
gcell = dict(zip(uk[med < 0.03], med[med < 0.03]))
print(f"b{B}｜起點點數（本塊盒內）｜陸地地面格：貼地 |dz|<0.5 m／地上／地下 < −0.5 m（點數；深度分位）")
for a in sys.argv[2:]:
    lab, f = a.split("=", 1)
    v = PlyData.read(f)["vertex"].data; X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(X) @ Rt.T)[:, :2])[B].numpy()
    X = X[inb]; k = key(X[:, :2]); gz = np.array([gcell.get(int(x), np.nan) for x in k]); land = np.isfinite(gz) & (gz >= -0.02)
    dz = (X[:, 2] - gz) * 100
    near = land & (np.abs(dz) < 0.5); above = land & (dz >= 0.5); below = land & (dz <= -0.5)
    q = np.percentile(dz[below], [50, 10, 1]) if below.any() else [np.nan] * 3
    print(f"{lab:10} 盒內 {inb.sum():>9,}｜陸地格內 {land.sum():>9,}：貼地 {near.sum():>8,}｜地上 {above.sum():>8,}｜地下 {below.sum():>7,}"
          f"（{100 * below.sum() / max(land.sum(), 1):.2f}%；dz 中位 {q[0]:.1f}、p10 {q[1]:.1f}、p1 {q[2]:.1f} m）", flush=True)
print("DONE init_underground")
