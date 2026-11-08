"""11-07 自主研究：點精靈（長軸 < 1e-4 場景單位＝1 cm，2DGS 低通濾波讓它在每個距離都畫成約 1 px 的點）在哪裡？
路燈帶（開闊地面上方 3～15 m、孤立柱格）、車輛帶（0.5～2.5 m）、地面（±0.5 m）、地下（< −0.5 m）、其他（建築等）；純 CPU。"""
import glob, os, sys
import numpy as np, torch
from scipy.ndimage import maximum_filter, uniform_filter
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
D = "data/matrix_city/aerial/train/block_all"; B = int(os.environ.get("SPRITE_BLOCK", "3"))
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.); bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ok = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = P[ok & inb(P, bb1)]; G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u] = S[order, 2][f + c // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03); bldg = maximum_filter((np.isfinite(med) & ~ground).astype(np.uint8), size=5) > 0; open_g = ground & ~bldg
gz = np.where(ground, med, np.nan); num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
def bands(x):
    cc = np.clip(np.floor(x[:, :2] / G).astype(int) - o0, 0, np.array(shp) - 1)
    h = x[:, 2] - gfill[cc[:, 0], cc[:, 1]]; og = open_g[cc[:, 0], cc[:, 1]] & np.isfinite(h)
    return cc, h, og
def load(pat):
    sd = torch.load(sorted(glob.glob(pat))[-1], map_location="cpu")["state_dict"]
    g = lambda k: sd[[x for x in sd if x.endswith(k)][0]].float()
    X = g("means").numpy().astype(np.float64); k = inb(X, bb0)
    return X[k], torch.sigmoid(g("opacities")).reshape(-1).numpy()[k], torch.exp(g("scales")).max(1).values.numpy()[k]
Xr, Or, _ = load(sys.argv[1].split("=", 1)[1])
cc0, h0, og0 = bands(Xr); lamp0 = og0 & (h0 > 0.03) & (h0 < 0.15) & (Or > 0.3)
cnt = np.zeros(shp); np.add.at(cnt, (cc0[lamp0, 0], cc0[lamp0, 1]), 1); ring = uniform_filter(cnt, size=3) * 9 - cnt
iso_cell = (cnt >= 15) & (ring < 0.5 * cnt)
print(f"孤立柱格 {int(iso_cell.sum())}（CityGS 定義）", flush=True)
print(f"{'模型':16s} {'塊內不透明':>10s} {'點精靈%':>7s} | 各區的點精靈比例（括號＝該區不透明顆數）")
for spec in sys.argv[1:]:
    lab, pat = spec.split("=", 1); X, O, mx = load(pat)
    cc, h, og = bands(X); op = O > 0.5; sp = mx < 1e-4
    reg = {"路燈孤立柱": og & (h > 0.03) & (h < 0.15) & iso_cell[cc[:, 0], cc[:, 1]], "路燈帶": og & (h > 0.03) & (h < 0.15),
           "車輛帶": og & (h > 0.005) & (h < 0.025), "地面±0.5m": np.isfinite(h) & (np.abs(h) <= 0.005),
           "地下<-0.5m": np.isfinite(h) & (h < -0.005), "其他(建築等)": ~(np.isfinite(h) & (h < 0.025))}
    s = " ".join(f"{k} {100*sp[r&op].mean() if (r&op).any() else float('nan'):4.1f}%({int((r&op).sum()):,})" for k, r in reg.items())
    print(f"{lab:16s} {int(op.sum()):>10,} {100*sp[op].mean():6.1f}% | {s}", flush=True)
