"""現行自適應權重（10-31 價格分配，需要 2.5M 全場景探針）能不能用「訓練前就拿得到」的逐塊訊號預測？只做相關，不改任何東西。"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from internal.utils.colmap import read_points3D_binary
D = "data/matrix_city/aerial/train/block_all"
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu")
R = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = P["partition_coordinates"]["id"].numpy(); xys = P["partition_coordinates"]["xy"].double().numpy()
size = P["scene_config"]["partition_size"].double().numpy()
vis = P["visibilities"].double().numpy(); asg = P["visibility_based_assignments"].numpy().astype(bool)
W = json.load(open("configs/cap_alloc_g44.json"))["weights"]
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
xyz = np.stack([p.xyz for p in pts.values()]); trk = np.array([len(p.image_ids) for p in pts.values()]); err = np.array([p.error for p in pts.values()])
rel = (trk >= 3) & (err < 1.5)
q = xyz @ R.T
rows = []
for k in range(16):
    b = int(ids[k][0] + 4 * ids[k][1])
    lo, hi = xys[k], xys[k] + size[k]
    inb = (q[:, 0] >= lo[0]) & (q[:, 0] < hi[0]) & (q[:, 1] >= lo[1]) & (q[:, 1] < hi[1])
    z = q[inb & rel, 2]
    rows.append(dict(b=b, w=W[str(b)], ncam=int(asg[k].sum()), vis=float(vis[k][asg[k]].sum()),
                     sfm=int((inb & rel).sum()), sfm_all=int(inb.sum()),
                     ztop=float(np.quantile(z, 0.99) - np.quantile(z, 0.05)) if len(z) > 10 else 0.0,
                     trk=float(np.mean(trk[inb & rel])) if (inb & rel).any() else 0.0))
rows.sort(key=lambda r: r["b"])
print(f"{'塊':>3} {'權重':>6} {'相機':>5} {'塊內視野和':>9} {'可靠SfM':>8} {'SfM全部':>8} {'高度差':>7} {'平均track':>8}")
for r in rows:
    print(f"{r['b']:>3} {r['w']:6.3f} {r['ncam']:5d} {r['vis']:9.1f} {r['sfm']:8d} {r['sfm_all']:8d} {r['ztop']:7.3f} {r['trk']:8.2f}")
w = np.array([r["w"] for r in rows])
def rk(x): return np.argsort(np.argsort(x))
for key in ("ncam", "vis", "sfm", "sfm_all", "ztop", "trk"):
    x = np.array([r[key] for r in rows], float)
    print(f"{key:8s} 與權重：Pearson {np.corrcoef(x, w)[0,1]:+.2f}  Spearman {np.corrcoef(rk(x), rk(w))[0,1]:+.2f}")
