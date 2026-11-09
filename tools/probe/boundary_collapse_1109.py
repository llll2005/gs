"""11-09 使用者 viewer：單塊（b9）細節清楚，但合併後中間幾棟「全崩了」；分塊時就看得到「邊界建築有往地面塌下去的高斯團」
=> 量單塊模型在接縫附近的崩塌範圍，跟訓練的緩衝比（tools/block_mask44.py：A 區＝盒外擴 m_in 5%、初始化裁到 m_c 15%；合併只留盒內，外擴 0）。
純 CPU、只讀。兩步：
  --build_ref <out.npz>：CityGS 16 塊（../cityGS_origin/.../blocks/block_<B>）各自只留盒內 o > 0.5 的顆粒 => 參考表面點雲（只存位置）
  --ref <npz> --run <跑次目錄> [--blocks ...]：我們每塊的原始模型（含盒外）依「到最近接縫的距離 s」分帶（負＝盒內、正＝盒外；
      單位＝內側塊尺寸，與 m_in／m_c 同一把尺；場景外緣不是接縫、不算），每帶統計不透明顆粒裡「離參考表面 > k × SfM 間距中位」的比例（另列只算地面以上 3 m 的：地下層本來就多、會蓋過崩塌）
      （k＝4、8；SfM 點最近鄰距離中位，無場景單位常數），以及地面以上沒貼上的顆粒的離地高度中位。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
from internal.utils.colmap import read_points3D_binary

D = "data/matrix_city/aerial/train/block_all"
ap = argparse.ArgumentParser()
ap.add_argument("--build_ref", default="")
ap.add_argument("--citygs_dir", default="../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/blocks")
ap.add_argument("--ref", default="")
ap.add_argument("--run", default="")
ap.add_argument("--blocks", type=int, nargs="*", default=list(range(16)))
a = ap.parse_args()

p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
BMIN, BMAX = bb.min.double().numpy(), bb.max.double().numpy()
size = p["scene_config"]["partition_size"].double().numpy()
inner = np.median(np.minimum(size[:, 0], size[:, 1]))
REF = np.median(size[(size[:, 0] < 10 * inner) & (size[:, 1] < 10 * inner)], axis=0).min()   # 同 block_mask44 的 ref.min()


def rot_xy(m):
    return (torch.from_numpy(m).double() @ Rt.T)[:, :2].numpy()


def load(f):
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy()
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy()
    s = torch.exp(sd["gaussian_model.gaussians.scales"].float()).numpy()
    del sd
    return m, o, s


if a.build_ref:
    P = []
    for b in range(16):
        f = sorted(glob.glob(f"{a.citygs_dir}/block_{b}/checkpoints/*step=60000.ckpt"))[0]
        m, o, _ = load(f)
        inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=torch.from_numpy(rot_xy(m)))[b].numpy()
        P.append(m[inb & (o > 0.5)].astype(np.float32))
        print(f"CityGS b{b}：盒內不透明 {len(P[-1]):,}", flush=True)
    np.savez(a.build_ref, P=np.concatenate(P))
    print(f"參考點雲 {sum(len(x) for x in P):,} 顆 => {a.build_ref}")
    sys.exit(0)

# 接縫：盒的某一邊與另一塊的對邊重合（容差＝內側塊尺寸的 1e-6）
eps = 1e-6 * REF
seam = np.zeros((16, 4), bool)   # x 下界、x 上界、y 下界、y 上界
for b in range(16):
    for c in range(16):
        if c == b:
            continue
        oy = min(BMAX[b, 1], BMAX[c, 1]) - max(BMIN[b, 1], BMIN[c, 1]) > eps
        ox = min(BMAX[b, 0], BMAX[c, 0]) - max(BMIN[b, 0], BMIN[c, 0]) > eps
        seam[b, 0] |= oy and abs(BMIN[b, 0] - BMAX[c, 0]) < eps
        seam[b, 1] |= oy and abs(BMAX[b, 0] - BMIN[c, 0]) < eps
        seam[b, 2] |= ox and abs(BMIN[b, 1] - BMAX[c, 1]) < eps
        seam[b, 3] |= ox and abs(BMAX[b, 1] - BMIN[c, 1]) < eps


def seam_s(xy, b):
    """到本塊最近接縫的有號距離（÷ REF；負＝盒內）。盒外但越過的是場景外緣、或在盒內但四邊都不是接縫 => nan。"""
    x, y = xy[:, 0], xy[:, 1]
    dx0, dx1, dy0, dy1 = x - BMIN[b, 0], BMAX[b, 0] - x, y - BMIN[b, 1], BMAX[b, 1] - y   # 盒內為正
    d = np.stack([dx0, dx1, dy0, dy1], 1)
    inside = (d >= 0).all(1)
    s = np.full(len(xy), np.nan)
    dd = np.where(seam[b][None, :], d, np.inf)
    s[inside] = -dd[inside].min(1) / REF
    out = ~inside
    viol = d < 0                                    # 越過了哪幾邊
    ok = out & ~(viol & ~seam[b][None, :]).any(1)   # 越過的邊都是接縫
    dist = np.sqrt((np.clip(-d[:, 0:2], 0, None).max(1)) ** 2 + (np.clip(-d[:, 2:4], 0, None).max(1)) ** 2)
    s[ok] = dist[ok] / REF
    s[~np.isfinite(s)] = np.nan
    return s


pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
S = np.stack([q.xyz for q in pts.values() if len(q.image_ids) >= 3 and q.error < 1.5]); del pts
from internal.metrics.geom_gate import median_nn_distance
NN = median_nn_distance(S)
o0 = np.floor(S[:, :2].min(0) / 0.02).astype(int) - 2; ci = np.floor(S[:, :2] / 0.02).astype(int) - o0; shp = tuple(ci.max(0) + 3)
cid = ci[:, 0] * shp[1] + ci[:, 1]; oo = np.lexsort((S[:, 2], cid)); uu, ff, cc = np.unique(cid[oo], return_index=True, return_counts=True)
GZ = np.full(shp[0] * shp[1], np.nan); GZ[uu] = S[oo, 2][ff + cc // 2]; GZ = GZ.reshape(shp); G10 = np.nanpercentile(S[:, 2], 10)
ref = np.load(a.ref)["P"].astype(np.float64)
tree = cKDTree(ref)
EDGES = [-0.5, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
lab = os.path.basename(os.path.normpath(a.run))
print(f"{lab}｜內側塊尺寸 {REF:.3f}（場景單位）；A 區＝盒外 0.05、初始化裁到 0.15、合併切在 0｜SfM 間距中位 {NN * 100:.2f} m；沒貼上＝離 CityGS 盒內不透明顆粒 > 4×／8× 間距")
print("帶（s，負＝盒內）  " + "  ".join(f"[{EDGES[i]:+.2f},{EDGES[i + 1]:+.2f})" for i in range(len(EDGES) - 1)))
tot = {k: np.zeros((len(EDGES) - 1, 3)) for k in ("n", "k4", "k8", "na", "k8a")}
for b in a.blocks:
    fs = sorted(glob.glob(f"{a.run}/blocks/block_{b}/checkpoints/*step=30000.ckpt"))
    if not fs:
        continue
    m, o, sc = load(fs[0])
    keep = o > 0.5
    m = m[keep]
    s = seam_s(rot_xy(m), b)
    d, _ = tree.query(m, k=1, distance_upper_bound=8 * NN * 1.01)
    off4, off8 = d > 4 * NN, d > 8 * NN
    c2 = np.floor(m[:, :2] / 0.02).astype(int) - o0
    okc = (c2[:, 0] >= 0) & (c2[:, 0] < shp[0]) & (c2[:, 1] >= 0) & (c2[:, 1] < shp[1])
    gz = np.full(len(m), G10); gz[okc] = np.where(np.isfinite(GZ[c2[okc, 0], c2[okc, 1]]), GZ[c2[okc, 0], c2[okc, 1]], G10)
    hz = (m[:, 2] - gz) * 100
    row4, row8, row8a, rowh = [], [], [], []
    up = hz > 3
    for i in range(len(EDGES) - 1):
        k = np.isfinite(s) & (s >= EDGES[i]) & (s < EDGES[i + 1])
        n = int(k.sum())
        tot["n"][i, 0] += n; tot["k4"][i, 0] += off4[k].sum(); tot["k8"][i, 0] += off8[k].sum()
        tot["na"][i, 0] += (k & up).sum(); tot["k8a"][i, 0] += (off8 & k & up).sum()
        row8a.append(f"{100 * off8[k & up].mean():5.1f}%" if (k & up).sum() >= 200 else "   — ")
        row4.append(f"{100 * off4[k].mean():5.1f}%" if n >= 200 else "   — ")
        row8.append(f"{100 * off8[k].mean():5.1f}%" if n >= 200 else "   — ")
        rowh.append(f"{np.median(hz[k & off8 & up]):5.0f}m" if (k & off8 & up).sum() >= 50 else "   — ")
    print(f"b{b:<2} 顆數  " + "  ".join(f"{int(((s >= EDGES[i]) & (s < EDGES[i + 1])).sum()):>13,}" for i in range(len(EDGES) - 1)))
    print(f"    >4×   " + "  ".join(f"{x:>13}" for x in row4))
    print(f"    >8×   " + "  ".join(f"{x:>13}" for x in row8))
    print(f"    >8×地上" + "  ".join(f"{x:>13}" for x in row8a))
    print(f"    高度  " + "  ".join(f"{x:>13}" for x in rowh), flush=True)
print("合計 >4× ／ >8× ／ >8× 地面以上 3 m：")
print("          " + "  ".join(f"{100 * tot['k4'][i, 0] / max(tot['n'][i, 0], 1):4.1f}／{100 * tot['k8'][i, 0] / max(tot['n'][i, 0], 1):4.1f}／{100 * tot['k8a'][i, 0] / max(tot['na'][i, 0], 1):4.1f}".rjust(13) for i in range(len(EDGES) - 1)))
print("DONE")
