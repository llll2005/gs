"""11-17 使用者 viewer：規格 6 S1 的殘留浮塊集中在港口（碼頭之間的水面）。查它們是在海面格上（有錨定層、上限也在，還是長出來）
還是落在海面格之外（海面物件沒蓋到碼頭之間的水 => 沒有錨定層也沒有海面上限）。只讀、CPU。
浮塊＝本塊盒內、o > 0.5、在共識平面上方 > --h（預設 5 m）。投影到平面上的鋪磚格 (i, j)：
  在海面格上／離最近的海面格 1～3 格／4～10 格／更遠；另報高度分布與 Σo·面積。
另報「水面但不是海面格」：模型裡貼在平面上（|高度| < 1 m）的不透明面元所在格，有多少不在海面格裡（＝訓練學到那裡是水、海面物件沒收）。
用法：python tools/probe/sea_residual_1117.py --block 3 --obj <block_3.npz> --models S1=<ckpt> ...
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--obj", required=True); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--h", type=float, default=0.05)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()
ob = np.load(a.obj)
n, c, org, e1, e2, g, base = ob["n"], float(ob["c"]), ob["origin"], ob["e1"], ob["e2"], float(ob["g"]), int(ob["base"])
keys = np.sort(ob["keys"]); ti = np.round(keys / base).astype(np.int64); tj = keys - ti * base
ctree = cKDTree(np.stack([ti, tj], 1))
print(f"b{B}｜海面格 {len(keys):,}｜鋪磚邊長 {g * M:.2f} m｜共識平面法向與世界 z 夾角 {np.degrees(np.arccos(abs(n[2]))):.2f}°", flush=True)


def cells(X):
    q = X - org; return np.floor(q @ e1 / g).astype(np.int64), np.floor(q @ e2 / g).astype(np.int64)


for spec in a.models:
    lab, pat = spec.split("=", 1); f = sorted(x for x in glob.glob(pat) if "xyz" not in x)[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]; gg = "gaussian_model.gaussians."
    m = sd[gg + "means"].double().numpy(); o = torch.sigmoid(sd[gg + "opacities"].float()).reshape(-1).numpy()
    s = torch.exp(sd[gg + "scales"].float()).numpy()[:, :2]; del sd
    k = (o > 0.5) & inblk(m); m, o, s = m[k], o[k], s[k]
    h = m @ n - c
    i, j = cells(m - np.outer(h, n))
    d, _ = ctree.query(np.stack([i, j], 1))
    insea = np.isin(i * base + j, keys)
    area = o * 2 * np.pi * s[:, 0] * s[:, 1] * M * M
    fl = h > a.h
    print(f"[{lab}] {f}", flush=True)
    for nm, sel in (("在海面格上", fl & insea), ("離海面格 1～3 格", fl & ~insea & (d <= 3)), ("4～10 格", fl & ~insea & (d > 3) & (d <= 10)), ("更遠", fl & ~insea & (d > 10))):
        hh = h[sel] * M
        print(f"   平面上方 > {a.h * M:.0f} m｜{nm:10} {int(sel.sum()):>7,} 顆｜Σo·面積 {area[sel].sum():>10,.0f} m²"
              + (f"｜高度中位 {np.median(hh):.1f} m、p90 {np.percentile(hh, 90):.1f} m、>20 m {int((hh > 20).sum()):,}" if sel.any() else ""), flush=True)
    flat = np.abs(h) < 0.01
    ci = np.unique(i[flat] * base + j[flat]); cin = np.isin(ci, keys)
    dd, _ = ctree.query(np.stack([np.round(ci[~cin] / base), ci[~cin] - np.round(ci[~cin] / base) * base], 1)) if (~cin).any() else (np.zeros(0), None)
    print(f"   貼在平面上（|高度| < 1 m）的不透明面元佔 {len(ci):,} 格：海面格 {int(cin.sum()):,}｜不是海面格 {int((~cin).sum()):,}"
          f"（其中離海面格 1～3 格 {int((dd <= 3).sum()):,}、更遠 {int((dd > 3).sum()):,}）", flush=True)
print("DONE sea_residual")
