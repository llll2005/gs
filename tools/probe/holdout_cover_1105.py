"""留出 K 台訓練相機當控制用驗證視角，塊內 SfM 點的觀測會少多少？（隨機留 vs 挑「看的東西別人也看得到」的相機留）"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from internal.utils.colmap import read_points3D_binary, read_images_binary
D = "data/matrix_city/aerial/train/block_all"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
B = int(sys.argv[1]) if len(sys.argv) > 1 else 3
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu")
R = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
k = [i for i, d in enumerate(P["partition_coordinates"]["id"].numpy()) if int(d[0] + 4 * d[1]) == B][0]
lo = P["partition_coordinates"]["xy"][k].double().numpy(); hi = lo + P["scene_config"]["partition_size"][k].double().numpy()
names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
imgs = read_images_binary(f"{D}/sparse/0/images.bin")
id2name = {i: v.name for i, v in imgs.items()}
train = set(names)
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
cam_idx = {n: j for j, n in enumerate(names)}
rows, cols = [], []
pi = 0
for p in pts.values():
    if len(p.image_ids) < 3 or p.error >= 1.5:
        continue
    q = p.xyz @ R.T
    if not (lo[0] <= q[0] < hi[0] and lo[1] <= q[1] < hi[1]):
        continue
    cs = {cam_idx[id2name[i]] for i in p.image_ids if id2name.get(i) in train}
    if not cs:
        continue
    for c in cs:
        rows.append(pi); cols.append(c)
    pi += 1
import scipy.sparse as sp
M = sp.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(pi, len(names)))
deg = np.asarray(M.sum(1)).ravel()
print(f"b{B}：訓練相機 {len(names)}、塊內可靠 SfM 點 {pi:,}（有被本塊訓練相機看到的）；每點觀測數 中位 {np.median(deg):.0f}、p10 {np.quantile(deg, 0.1):.0f}；"
      f"本來就 <= 2 台的 {np.mean(deg <= 2):.2%}")
def loss(hold):
    m = np.zeros(len(names), bool); m[list(hold)] = True
    lost = np.asarray(M[:, m].sum(1)).ravel()
    after = deg - lost
    touched = lost > 0
    return dict(touched=touched.mean(), zero=np.mean((after == 0) & (deg > 0)), lt3=np.mean((after < 3) & (deg >= 3)),
                rel=np.mean(lost[touched] / deg[touched]) if touched.any() else 0)
rng = np.random.default_rng(0)
R_ = [loss(rng.choice(len(names), K, replace=False)) for _ in range(50)]
def show(tag, rs):
    f = lambda k: (np.mean([r[k] for r in rs]), np.max([r[k] for r in rs]))
    print(f"{tag}：被影響的點 {f('touched')[0]:.1%}｜變成 0 台看到 {f('zero')[0]:.4%}（最差 {f('zero')[1]:.4%}）｜從 >=3 掉到 <3 {f('lt3')[0]:.3%}（最差 {f('lt3')[1]:.3%}）"
          f"｜被影響的點平均少 {f('rel')[0]:.1%} 觀測")
show(f"隨機留 {K} 台（50 次平均）", R_)
# 挑選：每台相機的「最脆弱觀測」= 它看到的點裡，觀測數最少的那些 => 分數＝它看到的點中 deg<=4 的比例，越低越安全
weak = np.asarray(M[deg <= 4].sum(0)).ravel() / np.maximum(np.asarray(M.sum(0)).ravel(), 1)
cand = np.argsort(weak)[: max(K * 8, K)]
R2 = [loss(rng.choice(cand, K, replace=False)) for _ in range(50)]
show(f"挑「看的點別人也看得到」的相機（最安全的 {len(cand)} 台裡隨機 {K} 台）", R2)
