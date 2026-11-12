"""11-05 使用者「b3 藍色高樓幾乎都重建失敗、診斷也作」：b3 的高樓（CPU；cwd＝repo 根；只讀）。
① 用可靠 SfM 點找出 b3 的高樓（10 m 格、頂高 > 100 m 的連通區），列每棟的頂高、顏色（找藍色玻璃塔）、
   各高度帶的 SfM 點數／平均 track 長度（＝幾台相機配到它）。
② 每個模型在每棟、每個高度帶的顆粒：不透明（o>0.5）／半透明數，以及「表面覆蓋」：
   參考表面點（可靠 SfM 點、CityGS 的不透明顆粒中心）周圍 R 內有沒有本模型的不透明顆粒 => 沒有＝破洞。
用法：python tower_probe_1105.py --models 標籤=ckpt ...（第一個 CityGS=... 也當參考表面）
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.ndimage import label, maximum_filter
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--top", type=float, default=1.0, help="頂高門檻（場景單位；1＝100 m）")
ap.add_argument("--R", type=float, default=0.01, help="覆蓋半徑（場景單位；0.01＝1 m）")
ap.add_argument("--detail_top", type=float, default=1.5, help="★ 11-12 逐棟細表的頂高門檻（場景單位；預設 1.5＝150 m，量測用）")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()

pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); RGB = np.stack([q.rgb for q in pts.values()]).astype(float)
ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
good = (TRK >= 3) & (ERR < 1.5) & inblk(P)
P, RGB, TRK = P[good], RGB[good], TRK[good]
ground = np.percentile(P[:, 2], 5)

# ① 高樓：10 m 格的最高可靠 SfM 點 > top => 連通區
CELL = 0.1
o0 = np.floor(P[:, :2].min(0) / CELL).astype(int); ci = np.floor(P[:, :2] / CELL).astype(int) - o0
shp = tuple(ci.max(0) + 1); top = np.full(shp, -1e9)
np.maximum.at(top, (ci[:, 0], ci[:, 1]), P[:, 2])
lab_, n = label(top > a.top)
bid = lab_[ci[:, 0], ci[:, 1]]
BANDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 9.9]
bn = lambda z: np.clip(np.searchsorted(BANDS, z, side="right") - 1, 0, len(BANDS) - 2)
print(f"b{B}｜可靠 SfM {len(P):,}｜地面（z 5 分位）{ground * M:.1f} m｜高樓＝10 m 格頂高 > {a.top * M:.0f} m 的連通區：{n} 棟")
print(f"{'棟':>3}{'格':>5}{'頂高 m':>8}{'中心 x,y':>18}{'  上半部顏色 R/G/B':>20}{'藍度':>6}" + "".join(f"{f'{BANDS[i] * M:.0f}~{BANDS[i + 1] * M:.0f}m':>16}" for i in range(len(BANDS) - 1)))
bld = []
for k in range(1, n + 1):
    s = bid == k
    if s.sum() < 50:
        continue
    zt = np.percentile(P[s, 2], 99); up = s & (P[:, 2] > 0.5 * zt)
    c = RGB[up].mean(0); blue = (c[2] - c[0]) / max(c.mean(), 1)
    cells = int((lab_ == k).sum()); cen = P[s, :2].mean(0)
    row = ""
    for i in range(len(BANDS) - 1):
        sb = s & (bn(P[:, 2]) == i)
        row += f"{(f'{int(sb.sum()):,}/{TRK[sb].mean():.0f}' if sb.sum() else '—'):>16}"
    bld.append((k, zt, blue))
    print(f"{k:>3}{cells:>5}{zt * M:>8.0f}{f'{cen[0]:.2f},{cen[1]:.2f}':>18}{f'{c[0]:.0f}/{c[1]:.0f}/{c[2]:.0f}':>20}{blue:>6.2f}{row}")
print("（各高度帶：可靠 SfM 點數／平均 track 長度）")

# ② 各模型在每棟每個高度帶
def load(spec):
    lab, pat = spec.split("=", 1); f = sorted(glob.glob(pat))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy()
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy(); del sd
    k = inblk(m)
    return lab, m[k], o[k]
models = [load(s) for s in a.models]
ref_cg = models[0][1][models[0][2] > 0.5] if models[0][0].lower().startswith("citygs") else None
tall = [b for b in bld if b[1] > a.detail_top]
print(f"\n=== 高樓（頂高 > {a.detail_top * M:.0f} m）逐棟、逐高度帶（中心在該棟 10 m 格膨脹 1 格內）===")
print("欄：不透明顆粒／半透明顆粒｜SfM 參考點 R 內無本模型不透明顆粒（破洞）%｜CityGS 不透明中心 R 內無本模型不透明（破洞）%")
foot = {k: maximum_filter(lab_ == k, size=3) for k, _, _ in tall}
for k, zt, blue in sorted(tall, key=lambda x: -x[2]):
    fp = foot[k]
    def on(m):
        c = np.floor(m[:, :2] / CELL).astype(int) - o0
        ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
        r = np.zeros(len(m), bool); r[ok] = fp[c[ok, 0], c[ok, 1]]; return r
    sfm = P[on(P)]
    cg = ref_cg[on(ref_cg)] if ref_cg is not None else None
    print(f"\n棟 {k}（頂高 {zt * M:.0f} m、藍度 {blue:.2f}）｜SfM 參考點 {len(sfm):,}" + (f"｜CityGS 不透明 {len(cg):,}" if cg is not None else ""))
    print(f"{'':14}" + "".join(f"{f'{BANDS[i] * M:.0f}~{BANDS[i + 1] * M:.0f}m':>24}" for i in range(len(BANDS) - 1) if BANDS[i] < zt))
    for lab, m, o in models:
        s = on(m); mo = m[s & (o > 0.5)]; mt = m[s & (o > 0.05) & (o <= 0.5)]
        tr = cKDTree(mo) if len(mo) else None
        row = ""
        for i in range(len(BANDS) - 1):
            if BANDS[i] >= zt:
                continue
            nb = int((bn(mo[:, 2]) == i).sum()); nt = int((bn(mt[:, 2]) == i).sum())
            def hole(ref):
                r = ref[bn(ref[:, 2]) == i]
                if tr is None or len(r) < 20:
                    return "—"
                d, _ = tr.query(r, k=1, distance_upper_bound=a.R)
                return f"{100 * np.mean(~np.isfinite(d)):.0f}"
            h1 = hole(sfm); h2 = hole(cg) if cg is not None else "—"
            row += f"{f'{nb // 1000}k/{nt // 1000}k {h1}%/{h2}%':>24}"
        print(f"{lab:14}{row}", flush=True)
print("\nDONE tower_probe")
