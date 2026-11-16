"""11-16 使用者看完新底 5 全場景：「海面 floater 有些塊幾乎正常、特定塊嚴重」=> 16 塊各量一次海面浮塊，並列每塊的特徵找共通點。只讀、CPU。
海面浮塊的判準與 tools/probe/sea_opacity_nosfm_1106.py 完全相同（4 cm 格＝4 m；3x3 鄰域沒有可靠 SfM、或格內 SfM 中位在海平面 ±1 m；扣掉高樓格；
顆粒在本塊分區盒內、高於海平面 5 m、o > 0.02）。另列每塊特徵：
  海面格數（有 SfM 的海＝格內 SfM 中位在海平面 ±1 m；沒 SfM 的海＝3x3 鄰域沒有可靠 SfM，只數在可靠 SfM 的 AABB 內的）、外圈塊、訓練相機數、
  起點（稠密起點 PLY）在海面上方 > 5 m 的點數，其中空中隨機補點（make_sfm_fill_init 用 SfM 平均色＝最常見的 RGB）幾個。
用法（lab，cwd＝repo 根）：python tools/probe/sea_blocks_1116.py --run <跑次> [--run2 <對照跑次> --lab2 <名稱>]
"""
import argparse, glob, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--run2", default=""); ap.add_argument("--lab2", default="對照")
ap.add_argument("--citygs", default="../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/blocks")
ap.add_argument("--init", default="data/matrix_city/aerial/train/block_all/sfmfill_tri44_crop/dup1")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, B: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb0, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
gd = (TRK >= 3) & (ERR < 1.5)
zg = P[gd, 2]; e = np.arange(-0.10, 0.0501, 0.001); h, _ = np.histogram(zg, bins=e); hs = np.convolve(h, np.ones(5) / 5, mode="same")
pk = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
ZS = e[pk[0]] + 0.0005
S = P[gd]


def cellmed(cell):
    o0 = np.floor(S[:, :2].min(0) / cell).astype(int) - 2
    ci = np.floor(S[:, :2] / cell).astype(int) - o0; shp = tuple(ci.max(0) + 3)
    cid = ci[:, 0] * shp[1] + ci[:, 1]; o = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[o], return_index=True, return_counts=True)
    med = np.full(shp[0] * shp[1], np.nan); med[u] = S[o, 2][f + c // 2]
    return o0, shp, med.reshape(shp)


def look(x, cell, o0, shp, grid):
    ci = np.floor(x[:, :2] / cell).astype(int) - o0
    ok = (ci[:, 0] >= 0) & (ci[:, 0] < shp[0]) & (ci[:, 1] >= 0) & (ci[:, 1] < shp[1])
    r = np.full(len(x), np.nan); r[ok] = grid[ci[ok, 0], ci[ok, 1]]; return r


G4 = cellmed(0.04); _o0, _shp, _med = G4
NOSFM = (~(maximum_filter(np.isfinite(_med).astype(np.uint8), size=3) > 0)).astype(float)
o10 = np.floor(S[:, :2].min(0) / 0.1).astype(int) - 2; ci10 = np.floor(S[:, :2] / 0.1).astype(int) - o10
shp10 = tuple(ci10.max(0) + 3); top = np.full(shp10, -1e9); np.maximum.at(top, (ci10[:, 0], ci10[:, 1]), S[:, 2])
TALL = maximum_filter((top > 1.0).astype(np.uint8), size=3).astype(float)
# 每個 4 cm 格的中心（只拿來數海面格）：有 SfM 的海＝中位在 ±1 m；沒 SfM 的海＝NOSFM（G4 本來就只蓋可靠 SfM 的 AABB）
gx, gy = np.meshgrid(np.arange(_shp[0]), np.arange(_shp[1]), indexing="ij")
cen = np.stack([(gx + _o0[0] + 0.5) * 0.04, (gy + _o0[1] + 0.5) * 0.04, np.zeros_like(gx, dtype=float)], -1).reshape(-1, 3)
sea_sfm = (np.isfinite(_med) & (np.abs(_med - ZS) < 0.01)).reshape(-1)
sea_nos = (NOSFM > 0).reshape(-1)
tall_c = look(cen, 0.1, o10, shp10, TALL) > 0
print(f"海平面估 {ZS * 100:.1f} m（場景 1＝100 m）；海面格 4 m 見方", flush=True)


def load_model(f):
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy(); o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy()
    sc = torch.exp(sd["gaussian_model.gaussians.scales"].float()).numpy(); del sd
    return m, o, sc


def sea_sel(m, B):
    sm = look(m, 0.04, *G4); nos = look(m, 0.04, _o0, _shp, NOSFM) > 0; tl = look(m, 0.1, o10, shp10, TALL) > 0
    return inb(m, B) & ~tl & ((np.isfinite(sm) & (np.abs(sm - ZS) < 0.01)) | nos) & (m[:, 2] > ZS + 0.05)


def stats(m, o, sc, B):
    s = sea_sel(m, B) & (o > 0.02)
    ms, os_, ss = m[s], o[s], sc[s]
    hgt = (ms[:, 2] - ZS) * 100
    area = (os_ * 2 * np.pi * ss[:, 0] * ss[:, 1]).sum() * 1e4
    return int((os_ > 0.5).sum()), area, (np.median(hgt) if len(hgt) else float("nan")), int((hgt > 20).sum()), int((hgt > 50).sum())


rows = []
for B in range(16):
    bx, by = [int(v) for v in pc.id[B]]
    ncam = len(open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{bx:03d}_{by:03d}.txt").read().split())
    outer = not (1 <= bx <= 2 and 1 <= by <= 2)
    inbox = inb(cen, B) & ~tall_c
    n_sfm, n_nos = int((sea_sfm & inbox).sum()), int((sea_nos & inbox).sum())
    # 起點：空中隨機補點＝最常見顏色
    from plyfile import PlyData
    v = PlyData.read(f"{a.init}/block_{B}.ply")["vertex"].data
    xi = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64); key = v["red"].astype(np.int64) * 65536 + v["green"].astype(np.int64) * 256 + v["blue"]
    uk, cnt = np.unique(key, return_counts=True); fill = key == uk[cnt.argmax()]
    si = sea_sel(xi, B)
    init_all, init_fill = int(si.sum()), int((si & fill).sum())
    r = dict(B=B, outer=outer, ncam=ncam, sea_sfm=n_sfm * 16, sea_nos=n_nos * 16, init_all=init_all, init_fill=init_fill)
    for lab, pat in (("ours", f"outputs/lab/{a.run}/blocks/block_{B}/checkpoints/*.ckpt"), ("cg", f"{a.citygs}/block_{B}/checkpoints/*step=60000.ckpt"),
                     ("r2", f"outputs/lab/{a.run2}/blocks/block_{B}/checkpoints/*.ckpt" if a.run2 else "")):
        fs = [f for f in sorted(glob.glob(pat)) if "xyz" not in f] if pat else []
        r[lab] = stats(*load_model(fs[-1]), B) if fs else None
    rows.append(r)
    print(f"b{B} 完成", flush=True)

print("\n=== 16 塊海面浮塊（塊內、高於海平面 5 m、o>0.02；Σo·面積 m²）＋每塊特徵 ===")
hdr = (f"{'塊':>4}{'外圈':>5}{'相機':>6}{'有SfM海 m²':>12}{'沒SfM海 m²':>13}{'起點海上>5m':>12}{'其中空中補點':>12}"
       f"{'我方 o>0.5':>11}{'我方 Σo·面積':>13}{'高度中位':>9}{'>20m':>8}{'>50m':>7}{'CityGS o>0.5':>13}{'CityGS Σo·面積':>15}")
if a.run2:
    hdr += f"{a.lab2 + ' o>0.5':>12}{a.lab2 + ' Σo·面積':>14}"
print(hdr)
for r in rows:
    ours, cg, r2 = r["ours"], r["cg"], r["r2"]
    line = (f"{'b' + str(r['B']):>4}{'是' if r['outer'] else '':>5}{r['ncam']:>6}{r['sea_sfm']:>12,}{r['sea_nos']:>13,}{r['init_all']:>12,}{r['init_fill']:>12,}"
            + (f"{ours[0]:>11,}{ours[1]:>13,.0f}{ours[2]:>9.1f}{ours[3]:>8,}{ours[4]:>7,}" if ours else f"{'—':>11}{'—':>13}{'—':>9}{'—':>8}{'—':>7}")
            + (f"{cg[0]:>13,}{cg[1]:>15,.0f}" if cg else f"{'—':>13}{'—':>15}"))
    if a.run2:
        line += (f"{r2[0]:>12,}{r2[1]:>14,.0f}" if r2 else f"{'—':>12}{'—':>14}")
    print(line)
# 共通點：我方 Σo·面積 對各特徵的等級相關（Spearman）
from scipy.stats import spearmanr
ys = np.array([r["ours"][1] if r["ours"] else np.nan for r in rows])
for name, k in (("有 SfM 的海面積", "sea_sfm"), ("沒 SfM 的海面積", "sea_nos"), ("起點海上 >5 m 點數", "init_all"), ("起點空中補點（海上）", "init_fill"), ("相機數", "ncam")):
    xs = np.array([r[k] for r in rows], float); ok = np.isfinite(ys)
    rho, pv = spearmanr(xs[ok], ys[ok]); print(f"  Σo·面積 對 {name}：Spearman {rho:+.2f}（p {pv:.3f}）")
print("DONE sea_blocks")
