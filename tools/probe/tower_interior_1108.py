"""11-08 使用者「藍色大樓破洞、高處側面擬合失敗（樓頂正常）」：反光是不是被做成塔內部的假景物（CPU；cwd＝repo 根；只讀）。
假說：玻璃的鏡面反光隨視角變，三階 SH 擬合不了 => 優化器把反射的景物放到玻璃「後面」（塔內部）、玻璃本身變半透明或被挖洞，
     空拍視角看起來對，低角度一看就是破洞。可量的預測：玻璃塔內部（離外殼 > d_in）的不透明顆粒遠多於其他高樓、外殼的不透明顆粒較少。
外殼＝CityGS 不透明顆粒在該高度帶的足跡（2 m 格、補洞）；內部＝足跡往內侵蝕 d_in 之後剩下的格；外殼帶＝足跡邊緣 d_in 內。
用法：python tower_interior_1108.py --models CityGS=ckpt 標籤=ckpt ...（第一個必須是 CityGS＝外殼參考）
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.ndimage import label, maximum_filter, binary_fill_holes, binary_erosion, binary_closing
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--top", type=float, default=1.5, help="只看頂高 > top 的高樓（1.5＝150 m）")
ap.add_argument("--g", type=float, default=0.02, help="足跡格（0.02＝2 m）")
ap.add_argument("--d_in", type=float, default=0.06, help="內部＝離外殼 > d_in（0.06＝6 m）")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()

# 高樓：同 tower_probe_1105（10 m 格最高可靠 SfM 點 > 100 m 的連通區），棟號一致
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); RGB = np.stack([q.rgb for q in pts.values()]).astype(float)
ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
good = (TRK >= 3) & (ERR < 1.5) & inblk(P)
P, RGB = P[good], RGB[good]
CELL = 0.1
o0 = np.floor(P[:, :2].min(0) / CELL).astype(int); ci = np.floor(P[:, :2] / CELL).astype(int) - o0
shp = tuple(ci.max(0) + 1); top = np.full(shp, -1e9)
np.maximum.at(top, (ci[:, 0], ci[:, 1]), P[:, 2])
lab_, n = label(top > 1.0)
bid = lab_[ci[:, 0], ci[:, 1]]
tall = []
for k in range(1, n + 1):
    s = bid == k
    if s.sum() < 50:
        continue
    zt = np.percentile(P[s, 2], 99)
    if zt > a.top:
        up = s & (P[:, 2] > 0.5 * zt); c = RGB[up].mean(0)
        tall.append((k, zt, (c[2] - c[0]) / max(c.mean(), 1)))
foot10 = {k: maximum_filter(lab_ == k, size=3) for k, _, _ in tall}
def on(m, k):
    c = np.floor(m[:, :2] / CELL).astype(int) - o0
    ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
    r = np.zeros(len(m), bool); r[ok] = foot10[k][c[ok, 0], c[ok, 1]]; return r

def load(spec):
    lab, pat = spec.split("=", 1); f = sorted(glob.glob(pat))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy()
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy(); del sd
    k = inblk(m)
    return lab, m[k], o[k]
models = [load(s) for s in a.models]
assert models[0][0].lower().startswith("citygs"), "第一個模型要是 CityGS（外殼參考）"
cg_m, cg_o = models[0][1], models[0][2]

BANDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
ner = max(1, int(round(a.d_in / a.g)))
print(f"b{B}｜頂高 > {a.top * M:.0f} m 的高樓 {len(tall)} 棟｜足跡格 {a.g * M:.0f} m、內部＝外殼往內 {a.d_in * M:.0f} m 以上")
print("欄：內部不透明顆粒（o>0.5）／外殼帶不透明顆粒｜內部比例＝內部 ÷（內部＋外殼帶）；括號＝內部半透明（0.05<o≤0.5）")
summ = {}
for k, zt, blue in sorted(tall, key=lambda x: -x[2]):
    print(f"\n棟 {k}（頂高 {zt * M:.0f} m、藍度 {blue:.2f}{'，藍色玻璃塔' if blue == max(t[2] for t in tall) else ''}）")
    print(f"{'':12}" + "".join(f"{f'{BANDS[i] * M:.0f}~{BANDS[i + 1] * M:.0f}m':>26}" for i in range(len(BANDS) - 1) if BANDS[i] < zt))
    sel_cg = on(cg_m, k) & (cg_o > 0.5)
    masks = []
    for i in range(len(BANDS) - 1):
        if BANDS[i] >= zt:
            masks.append(None); continue
        q = cg_m[sel_cg & (cg_m[:, 2] >= BANDS[i]) & (cg_m[:, 2] < BANDS[i + 1])]
        if len(q) < 50:
            masks.append(None); continue
        g0 = np.floor(q[:, :2].min(0) / a.g).astype(int) - 3 * ner
        gi = np.floor(q[:, :2] / a.g).astype(int) - g0
        sh = tuple(gi.max(0) + 3 * ner + 1); occ = np.zeros(sh, bool); occ[gi[:, 0], gi[:, 1]] = True
        fp = binary_fill_holes(binary_closing(occ, iterations=2))
        inner = binary_erosion(fp, iterations=ner)
        masks.append((g0, fp, inner))
    for lab, m, o in models:
        s = on(m, k)
        row = ""
        for i, mk in enumerate(masks):
            if BANDS[i] >= zt:
                continue
            if mk is None:
                row += f"{'—':>26}"; continue
            g0, fp, inner = mk
            sb = s & (m[:, 2] >= BANDS[i]) & (m[:, 2] < BANDS[i + 1])
            gi = np.floor(m[sb, :2] / a.g).astype(int) - g0
            ok = (gi[:, 0] >= 0) & (gi[:, 0] < fp.shape[0]) & (gi[:, 1] >= 0) & (gi[:, 1] < fp.shape[1])
            fin = np.zeros(len(gi), bool); inn = np.zeros(len(gi), bool)
            fin[ok] = fp[gi[ok, 0], gi[ok, 1]]; inn[ok] = inner[gi[ok, 0], gi[ok, 1]]
            ob = o[sb]
            ni = int((inn & (ob > 0.5)).sum()); ns = int((fin & ~inn & (ob > 0.5)).sum()); nt = int((inn & (ob > 0.05) & (ob <= 0.5)).sum())
            summ.setdefault((lab, k == max(tall, key=lambda t: t[2])[0]), [0, 0]); summ[(lab, k == max(tall, key=lambda t: t[2])[0])][0] += ni
            summ[(lab, k == max(tall, key=lambda t: t[2])[0])][1] += ns
            row += f"{f'{ni:,}／{ns:,} {100 * ni / max(ni + ns, 1):3.0f}%（{nt:,}）':>26}"
        print(f"{lab:12}{row}")
print("\n=== 總結：50 m 以上各帶加總的內部比例（內部 ÷（內部＋外殼帶）） ===")
for lab, _, _ in models:
    gl = summ.get((lab, True), [0, 0]); ot = summ.get((lab, False), [0, 0])
    print(f"{lab:12} 玻璃塔 {gl[0]:,}／{gl[1]:,}＝{100 * gl[0] / max(sum(gl), 1):.1f}%｜其他高樓 {ot[0]:,}／{ot[1]:,}＝{100 * ot[0] / max(sum(ot), 1):.1f}%")
print("DONE tower_interior")
