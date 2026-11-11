"""11-11 behind 票離線檢查沒過（`behind_offline_1111.py`：CityGS 立面誤判 13.0／10.0%、我方真地下只抓到 16／34%）之後，
用它存下的逐顆票數（outputs/lab/behind_offline_1111/<標籤>_b<B>.npz；不重新投影）掃一組判準變體，給使用者選：
  窄帶：(1,5)%＝b0、(0.5,5)%＝b1
  分母：全部有效觀測／不計被擋（扣掉 r >= 5% 的觀測）
  過門檻比例：0.5／0.67
  朝向：不限／只算躺平的顆粒（自己的面元法向 |n_z| > 0.8；無因次，不是場景單位）
類別定義與 behind_offline_1111 相同（顆粒順序也相同：塊內且 o>0.3，ckpt 原順序）。
用法（lab，CPU）：python tools/probe/behind_rules_1111.py --block 1 --models CityGS=<ckpt 樣式> 基準=<ckpt 樣式>
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.ndimage import maximum_filter, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--models", nargs=2, required=True)
ap.add_argument("--npz_dir", default="outputs/lab/behind_offline_1111")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; MU = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); okp = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = P[okp & inb(P, bb1)]
G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u_, f_, c_ = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u_] = S[order, 2][f_ + c_ // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03)
bldg = maximum_filter((np.isfinite(med) & ~ground).astype(np.uint8), size=5) > 0
open_g = ground & ~bldg


def nrm_z(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.abs(1 - 2 * (x * x + y * y))


res = {}
for spec in a.models:
    lab, pat = spec.split("=", 1)
    sd = torch.load(sorted(glob.glob(pat))[-1], map_location="cpu", weights_only=False)["state_dict"]
    gg = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m = gg("means").double().numpy(); o = torch.sigmoid(gg("opacities")).reshape(-1).numpy(); q = gg("rotations").double().numpy()
    del sd
    k = inb(m, bb0) & (o > 0.3)
    m, o, nz = m[k], o[k], nrm_z(q[k])
    z = np.load(f"{a.npz_dir}/{lab}_b{B}.npz")
    assert len(z["o"]) == len(o) and np.allclose(z["o"], o, atol=1e-6), f"{lab}：顆粒順序和 npz 對不上"
    h = z["h"].astype(np.float64)
    cc = np.clip(np.floor(m[:, :2] / G).astype(int) - o0, 0, np.array(shp) - 1)
    isg = ground[cc[:, 0], cc[:, 1]]; iso = open_g[cc[:, 0], cc[:, 1]]
    op5, op3 = o > 0.5, o > 0.3
    hi = op5 & ~iso & (h > 3)
    cats = {"貼地": op5 & isg & (np.abs(h) < 0.5), "真地下": op5 & isg & (h <= -1), "淺地下": op5 & isg & (h <= -0.5) & (h > -1),
            "車輛帶": op3 & iso & (h > 0.5) & (h < 2.5), "路燈帶": op3 & iso & (h > 3) & (h < 15),
            "樓頂": hi & (nz > 0.85), "立面": hi & (nz < 0.3), "斜的": hi & (nz >= 0.3) & (nz <= 0.85)}
    res[lab] = (z, nz, cats)
    print(f"[{lab}] {len(o):,} 顆｜真地下躺平（|n_z|>0.8）{100 * (nz[cats['真地下']] > 0.8).mean():.1f}%、貼地躺平 {100 * (nz[cats['貼地']] > 0.8).mean():.1f}%", flush=True)

ref, ours = a.models[0].split("=")[0], a.models[1].split("=")[0]
print(f"\nb{B}｜每列一條判準：{ref} 各類被判 behind %（誤判，要 <= 約 3%）｜{ours} 真地下被判 %（命中，要 >= 約 50%）與 {ours} 其他類")
hdr = ["貼地", "車輛帶", "路燈帶", "樓頂", "立面", "斜的"]
print(f"  {'窄帶':8}{'分母':6}{'過門檻':>6}{'朝向':>6} | " + "".join(f"{x:>7}" for x in hdr) + f" | {'真地下':>7}{'淺地下':>7}" + "".join(f"{x:>7}" for x in ("樓頂", "立面", "斜的", "貼地")))
for bk, bn in (("b0", "(1,5)%"), ("b1", "(0.5,5)%")):
    for den in ("全部", "不計被擋"):
        for fr in (0.5, 0.67):
            for flat in (False, True):
                row = {}
                for lab in (ref, ours):
                    z, nz, cats = res[lab]
                    cnt = z["cnt"].astype(np.int64); vis = cnt - z["far"] if den == "不計被擋" else cnt
                    sel = (vis >= 3) & (z[bk] > fr * vis)
                    if flat:
                        sel &= nz > 0.8
                    row[lab] = {c: 100 * sel[mk].mean() if mk.sum() else float("nan") for c, mk in cats.items()}
                print(f"  {bn:8}{den:6}{fr:>6.2f}{'躺平' if flat else '不限':>6} | " + "".join(f"{row[ref][c]:7.2f}" for c in hdr)
                      + f" | {row[ours]['真地下']:7.2f}{row[ours]['淺地下']:7.2f}" + "".join(f"{row[ours][c]:7.2f}" for c in ("樓頂", "立面", "斜的", "貼地")), flush=True)
print("DONE")
