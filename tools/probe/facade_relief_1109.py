"""11-09 使用者 viewer（b6 60k）：「好像有機制會把大樓的牆面整個壓平，可能在 30k 時不明顯」=> 量立面的局部厚度隨步數的變化（純 CPU、只讀）。
立面顆粒＝塊內、o > 0.5、|n_z| < 0.3、高 > 50 m。每顆取最近 16 顆塊內不透明鄰居做 PCA：
  局部厚度 = sqrt(λ_min)（場景單位 x100 => m）；扁平度 = λ_min / Σλ（越小越像一張平板）。
同時列立面顆粒的長軸／短軸（3σ，m）、長短比、opacity 中位、立面顆數與佔不透明的比例，依高度帶 50～150／150～250／> 250 m。
用法：python tools/probe/facade_relief_1109.py <block> 標籤=ckpt ...（建議第一個放 CityGS 當參考）
"""
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

D = "data/matrix_city/aerial/train/block_all"
B = int(sys.argv[1])
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
BANDS = [(0.5, 1.5, "50~150m"), (1.5, 2.5, "150~250m"), (2.5, 9.0, ">250m")]


def nrm(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)


print(f"b{B}｜立面顆粒＝塊內 o>0.5、|n_z|<0.3、高 > 50 m；厚度＝16 近鄰 PCA 的 sqrt(λ_min)（m）；扁平度＝λ_min/Σλ（x1000）")
print("模型          帶          立面顆數  佔不透明  厚度中位  扁平度中位  扁平度<1‰  長軸中位  短軸中位  長短比中位  o 中位")
for spec in sys.argv[2:]:
    lab, f = spec.split("=", 1)
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    g = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m = g("means").double()
    inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(m @ Rt.T)[:, :2])[B].numpy()
    m = m.numpy()
    o = torch.sigmoid(g("opacities")).reshape(-1).numpy()
    s = torch.exp(g("scales")).numpy()
    n = nrm(g("rotations").double().numpy())
    del sd
    op = inb & (o > 0.5)
    P = m[op]
    tree = cKDTree(P)
    fac = op & (np.abs(n[:, 2]) < 0.3) & (m[:, 2] > 0.5)
    F = m[fac]
    _, idx = tree.query(F, k=16)
    nb = P[idx] - P[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / 16
    ev = np.linalg.eigvalsh(cov)                       # 由小到大
    thick = np.sqrt(np.clip(ev[:, 0], 0, None)) * 100
    flat = ev[:, 0] / np.clip(ev.sum(1), 1e-18, None) * 1000
    sf = s[fac]
    lng = 3 * sf.max(1) * 100
    sht = 3 * sf.min(1) * 100
    ar = sf.max(1) / np.clip(sf.min(1), 1e-12, None)
    of = o[fac]
    z = F[:, 2]
    for lo, hi, nm in [(0.5, 9.0, "全部")] + BANDS:
        k = (z >= lo) & (z < hi)
        if k.sum() < 50:
            continue
        print(f"{lab:12}  {nm:9}  {k.sum():8,}  {100 * k.sum() / op.sum():6.1f}%  {np.median(thick[k]):7.2f}  {np.median(flat[k]):9.2f}  "
              f"{100 * np.mean(flat[k] < 1):8.1f}%  {np.median(lng[k]):7.2f}  {np.median(sht[k]):7.3f}  {np.median(ar[k]):9.1f}  {np.median(of[k]):.2f}", flush=True)
print("DONE")
