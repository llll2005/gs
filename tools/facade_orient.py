#!/usr/bin/env python
"""立面面元貼牆度（2026-10-30 使用者「離開空拍視角就扭曲、立面變透空格線，CityGS 任何視角都完整」；`紀錄/實驗分析/22`）。
held-out 全是空拍視角 => 量不到；這個指標不需要 GT。CityGS release b6／b12 貼牆中位 0.98、平躺 5～7%；A＋dist4 0.70～0.86、平躺 11～32%。
各跑次的第 B 塊 ckpt（最大步數）裁到「我方 4x4 分區的第 B 塊盒」（同 merge 判定），只看 o>0.5：
  立面判定＝16 近鄰 PCA 法向接近水平（|n_pca·z|<0.3）、局部平（λ3/λ2<0.1）、高出地面 0.15 以上（世界上方向 +z）
  報 |n·n_pca| 中位（1＝貼牆）、側視比例（<0.3）、平躺比例（|n_z|>0.7）、長軸中位、塊內不透明顆數
2DGS 面元法向＝旋轉矩陣第三欄（scales 只有兩軸）。
用法：python tools/facade_orient.py --block 3 --runs <跑次名（outputs/lab/ 底下）或目錄> ...
      python tools/facade_orient.py --block 3 --ckpt <ckpt>          （post_run_data.sh 的 facade 項目）
"""
import argparse, glob, os, re, sys
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--runs", nargs="*", default=[])
ap.add_argument("--ckpt", default=None, help="直接給 ckpt（取代 --runs）")
ap.add_argument("--part44", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
a = ap.parse_args()
P = torch.load(a.part44, map_location="cpu")
Rr = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = P["partition_coordinates"]["id"].numpy()
k = int(np.where((ids[:, 0] == a.block % 4) & (ids[:, 1] == a.block // 4))[0][0])
lo = P["partition_coordinates"]["xy"][k].double().numpy(); hi = lo + P["scene_config"]["partition_size"][k].double().numpy()
print(f"b{a.block} 盒（分區座標）x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}]", flush=True)
rng = np.random.default_rng(0)
todo = [(a.ckpt, a.ckpt)] if a.ckpt else []
for r in a.runs:
    d = r if os.path.isdir(r) else os.path.join("outputs/lab", r)
    cs = [c for c in glob.glob(os.path.join(d, "blocks", f"block_{a.block}", "checkpoints", "*.ckpt")) if re.search(r"step=(\d+)", c)]
    if not cs:
        print(f"{r:<58} ⚠ 沒有 block_{a.block} ckpt", flush=True); continue
    todo.append((r, max(cs, key=lambda x: int(re.search(r"step=(\d+)", x).group(1)))))
for r, c in todo:
    sd = torch.load(c, map_location="cpu")["state_dict"]
    g = "gaussian_model.gaussians."
    X = sd[g + "means"].double().numpy()
    o = torch.sigmoid(sd[g + "opacities"].float()).numpy().reshape(-1)
    q = (X @ Rr.T)[:, :2]
    keep = (o > 0.5) & np.all((q >= lo) & (q < hi), 1)
    X = X[keep]
    qt = torch.nn.functional.normalize(sd[g + "rotations"].float()[torch.from_numpy(keep)], dim=-1).double().numpy()
    sc = np.exp(sd[g + "scales"].float()[torch.from_numpy(keep)].double().numpy())
    del sd
    w, x, y, z = qt.T
    n = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)      # R[:, 2]
    zg = np.percentile(X[:, 2], 5)
    s = rng.choice(len(X), min(300000, len(X)), replace=False)
    _, nb = cKDTree(X).query(X[s], k=16)
    Pn = X[nb] - X[nb].mean(1, keepdims=True)
    ev, evec = np.linalg.eigh(np.einsum("nki,nkj->nij", Pn, Pn))
    npca = evec[:, :, 0]; flat = ev[:, 0] / np.maximum(ev[:, 1], 1e-12)
    fac = (np.abs(npca[:, 2]) < 0.3) & (flat < 0.1) & (X[s, 2] > zg + 0.15)
    al = np.abs((n[s][fac] * npca[fac]).sum(1)); nz = np.abs(n[s][fac][:, 2])
    st = re.search(r"step=(\d+)", c).group(1)
    print(f"{r:<58} step {st:>6}  塊內不透明 {len(X):>9,}  立面 {fac.mean() * 100:4.1f}%  "
          f"貼牆中位 {np.median(al):.2f}  側視 {np.mean(al < 0.3) * 100:4.1f}%  平躺 {np.mean(nz > 0.7) * 100:4.1f}%  "
          f"長軸中位 {np.median(sc.max(1)[s][fac]):.4f}", flush=True)
print("DONE facade", flush=True)
