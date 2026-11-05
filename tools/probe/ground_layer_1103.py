"""11-03 使用者「地面是半透明、會不會是熵／貼表面的參數不夠激進」：地面格（2 m 格裡有 SfM 點、且 SfM 中位 z < 3 m）上的顆粒，
按「中心高度 − 該格 SfM 中位高度」分：貼地 |dz|<0.5 m／地上 >0.5／地下 < −0.5；巨大（3σ 長軸 > 20 m）另列。
覆蓋＝Σ o·2π·su·sv ÷ 地面格總面積（不算遮擋）；貼地覆蓋 >= 1 的格比例＝地面層「蓋滿」的程度。"""
import sys, glob, numpy as np, torch
sys.path.insert(0, ".")
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.colmap import read_points3D_binary
from internal.density_controllers.merge_ops import _qmat
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
gcell = dict(zip(uk[med < 0.03], med[med < 0.03]))          # 地面格 -> SfM 中位高度
print(f"b{B}｜地面格 {len(gcell):,}（全場景 SfM）｜覆蓋單位＝層（Σo·2π·su·sv ÷ 塊內地面格面積）")
print(f"{'':14}{'貼地覆蓋':>8}{'蓋滿格%':>8}{'貼地 o>0.9':>10}{'貼地躺平':>9}{'地上覆蓋':>9}{'地下覆蓋':>9}{'地下 o>0.9':>10}{'巨大覆蓋':>9}{'巨大在地下':>10}")
for a in sys.argv[2:]:
    lab, pat = a.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    g = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m = g("means"); inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(m.double() @ Rt.T)[:, :2])[B]
    m = m[inb].numpy(); o = torch.sigmoid(g("opacities")).reshape(-1)[inb].numpy(); s = torch.exp(g("scales"))[inb].numpy() * 100
    nz = np.abs(_qmat(g("rotations")[inb])[:, 2, 2].numpy()); del sd
    k = key(m[:, :2]); gz = np.array([gcell.get(int(x), np.nan) for x in k]); on = ~np.isnan(gz)
    area = len(np.unique(k[on])) * 4.0
    dz = (m[:, 2] - gz) * 100; cov = o * 2 * np.pi * s[:, 0] * s[:, 1]; big = 3 * s.max(1) > 20
    near = on & ~big & (np.abs(dz) < 0.5); above = on & ~big & (dz >= 0.5); below = on & ~big & (dz <= -0.5); gb = on & big
    kc = k[near]; u, inv = np.unique(kc, return_inverse=True); full = np.bincount(inv, weights=cov[near]) / 4.0
    full_pct = 100 * np.sum(full >= 1) / max(len(np.unique(k[on])), 1)
    f = lambda msk: cov[msk].sum() / area
    print(f"{lab:14}{f(near):>8.2f}{full_pct:>7.1f}%{100 * np.mean(o[near] > 0.9):>9.1f}%{100 * np.mean(nz[near] > 0.8):>8.1f}%"
          f"{f(above):>9.2f}{f(below):>9.2f}{100 * np.mean(o[below] > 0.9):>9.1f}%{f(gb):>9.2f}{100 * np.mean(dz[gb] < -0.5):>9.1f}%", flush=True)
