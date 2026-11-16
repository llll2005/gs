"""11-16 使用者看完新底 5 全場景：高樓高層平視破洞、接縫的大樓群連低樓層都崩 => 分開「接縫問題」與「塔頂問題」。只讀、CPU。
GT＝MatrixCity 幾何 GT 點雲（data/geometry_gt/MC_Aerial/Block_all_ds.ply，帶顏色、法向用 16 近鄰 PCA 估）；模型＝合併後的 ckpt。
每個 GT 點（抽樣）：
  覆蓋   有不透明面元（o > 0.5）蓋到它：離面元平面 < 1 m 且面內 (u/s1)^2+(v/s2)^2 < 9（同 tower_gt_1113.py）
  朝向   蓋到它的面元裡離平面最近那顆 |n·n_gt| > 0.9
  分組   離「塊與塊之間的內部邊界線」的距離（分區轉換後的 xy；場景外緣不算接縫）x 離地高度（地面＝可靠 SfM z 的 5 百分位）
另出圖（--png_dir）：俯視的 GT 顏色圖、10 m 格覆蓋率圖（都畫分區線與塊號），用來對使用者 viewer 截圖找位置。
用法：python tools/probe/seam_coverage_1116.py --models 新底5=<merged ckpt> [CityGS=<merged ckpt>] [--n 3000000] [--png_dir <dir>]
"""
import argparse, glob, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--n", type=int, default=3000000)
ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
ap.add_argument("--png_dir", default="")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
Rp = p["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = p["partition_coordinates"]["id"].numpy(); xys = p["partition_coordinates"]["xy"].double().numpy(); size = p["scene_config"]["partition_size"].double().numpy()
boxes = {}
for k in range(len(ids)):
    boxes[int(ids[k][1] * 4 + ids[k][0])] = (xys[k], xys[k] + size[k])
# 內部邊界線：某塊的上界＝另一塊的下界
def _shared(hi_, lo_):   # 浮點容差比對（11-16 第一版用 round(…,6) 精確比對，漏掉 x＝−3.61、y＝−2.54 兩條）
    out = []
    for v in sorted(set(hi_)):
        if any(abs(v - w) < 1e-4 for w in lo_) and not any(abs(v - o) < 1e-4 for o in out):
            out.append(v)
    return out
XL = _shared([float(boxes[b][1][0]) for b in boxes], [float(boxes[b][0][0]) for b in boxes])
YL = _shared([float(boxes[b][1][1]) for b in boxes], [float(boxes[b][0][1]) for b in boxes])
print(f"內部邊界線 x={XL} y={YL}", flush=True)

pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
ZG = float(np.percentile(P[(TRK >= 3) & (ERR < 1.5)][:, 2], 5)); del P

with open(a.gt, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        hdr += f.readline()
    props, nv = [], 0
    for ln in hdr.decode().splitlines():
        t = ln.split()
        if t[:2] == ["element", "vertex"]:
            nv = int(t[2])
        elif t and t[0] == "property":
            props.append((t[2], {"double": "<f8", "float": "<f4", "uchar": "u1"}[t[1]]))
    G = np.fromfile(f, dtype=np.dtype(props), count=nv)
gx = np.stack([G["x"], G["y"], G["z"]], 1); gc = np.stack([G["red"], G["green"], G["blue"]], 1); del G
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(gx), min(a.n, len(gx)), replace=False))
tree_all = cKDTree(gx)
_, nn = tree_all.query(gx[sel], k=16)
Q = gx[nn] - gx[nn].mean(1, keepdims=True); del nn
_, _, evec = np.linalg.svd(Q, full_matrices=False); del Q
X, N, C = gx[sel], evec[:, 2, :], gc[sel]; del tree_all
U = X @ Rp.T
hgt = (X[:, 2] - ZG) * M
dseam = np.full(len(X), np.inf)
for xl in XL:
    dseam = np.minimum(dseam, np.abs(U[:, 0] - xl))
for yl in YL:
    dseam = np.minimum(dseam, np.abs(U[:, 1] - yl))
dseam *= M
blk = np.full(len(X), -1)
for b, (lo, hi) in boxes.items():
    m = (U[:, 0] >= lo[0]) & (U[:, 0] <= hi[0]) & (U[:, 1] >= lo[1]) & (U[:, 1] <= hi[1]); blk[m & (blk < 0)] = b
fac = np.abs(N[:, 2]) < 0.3; flat = np.abs(N[:, 2]) > 0.8
# ★ 11-16 轉角（使用者圖 7「牆面轉折處糊掉」）：立面點 1.5 m 內有另一個立面點的法向夾角 > 45°（|n·n'| < 0.7）
_fi = np.where(fac)[0]; _ft = cKDTree(X[_fi])
_dd, _jj = _ft.query(X[_fi], k=16, distance_upper_bound=0.015)
_ok = np.isfinite(_dd); _jj = np.where(_ok, _jj, 0)
_mincos = np.where(_ok, np.abs((N[_fi][:, None, :] * N[_fi][_jj]).sum(-1)), 1.0).min(1)
corner = np.zeros(len(X), bool); corner[_fi] = _mincos < 0.7; del _ft, _dd, _jj, _ok
print(f"GT 抽 {len(X):,} 點（全部 {len(gx):,}）｜立面 {100 * fac.mean():.0f}%（其中轉角 {100 * corner[fac].mean():.0f}%）、平面 {100 * flat.mean():.0f}%｜地面參考 z {ZG:.4f}", flush=True)


def qrot(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def _load_opaque(f, own_block=None):
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    g = "gaussian_model.gaussians."
    o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy()
    m = sd[g + "means"].double().numpy()
    keep = o > 0.5
    if own_block is not None:     # 分塊 ckpt：只留自己分區盒內的（同 merge_citygs_ckpts 的裁法）
        lo, hi = boxes[own_block]; u = m @ Rp.T
        keep &= (u[:, 0] >= lo[0]) & (u[:, 0] <= hi[0]) & (u[:, 1] >= lo[1]) & (u[:, 1] <= hi[1])
    out = (m[keep], torch.exp(sd[g + "scales"].float()).numpy()[keep][:, :2], sd[g + "rotations"].float().numpy()[keep].astype(np.float64))
    del sd
    return out


def cover(f):
    if "{b}" in f:   # ★ 分塊模式：pattern 裡的 {b} 換成 0～15，每塊只留自己盒內（CityGS 的合併檔太大時用）
        parts = []
        for b in range(16):
            fs = [x for x in sorted(glob.glob(f.replace("{b}", str(b)))) if "xyz" not in x]
            parts.append(_load_opaque(fs[-1], own_block=b))
        m = np.concatenate([p_[0] for p_ in parts]); s = np.concatenate([p_[1] for p_ in parts]); q = np.concatenate([p_[2] for p_ in parts]); del parts
    else:
        m, s, q = _load_opaque(f)
    keep = np.ones(len(m), bool)
    R = qrot(q); del q
    t1, t2, nrm = R[:, :, 0], R[:, :, 1], R[:, :, 2]; del R
    tr = cKDTree(m)
    cov = np.zeros(len(X), bool); al = np.zeros(len(X), bool)
    for s0 in range(0, len(X), 100000):
        Xs = X[s0:s0 + 100000]
        dd, jj = tr.query(Xs, k=32, distance_upper_bound=0.15)
        valid = np.isfinite(dd); jj = np.where(valid, jj, 0)
        d = Xs[:, None, :] - m[jj]
        u = (d * t1[jj]).sum(-1) / s[jj, 0]; v = (d * t2[jj]).sum(-1) / s[jj, 1]
        hh = np.abs((d * nrm[jj]).sum(-1))
        ok = valid & (u * u + v * v < 9) & (hh < 0.01)
        best = np.where(ok, hh, np.inf).argmin(1)
        cov[s0:s0 + 100000] = ok.any(1)
        nb = nrm[jj[np.arange(len(Xs)), best]]
        al[s0:s0 + 100000] = ok.any(1) & (np.abs((nb * N[s0:s0 + 100000]).sum(1)) > 0.9)
    print(f"[載入] {f}｜不透明 {int(keep.sum()):,} 顆", flush=True)
    return cov, al


DB = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 1e9)]
HB = [(-1e9, 50), (50, 150), (150, 250), (250, 1e9)]
res = {}
for spec in a.models:
    lab, pat = spec.split("=", 1)
    f = pat if "{b}" in pat else [x for x in sorted(glob.glob(pat)) if "xyz" not in x][-1]
    res[lab] = cover(f)
for kind, km in (("立面（|n_z|<0.3）", fac), ("平面（|n_z|>0.8：屋頂、地面）", flat), ("立面－轉角（1.5 m 內有夾角 > 45° 的牆）", fac & corner), ("立面－平牆", fac & ~corner)):
    print(f"\n=== {kind}：覆蓋％／朝向對齊％（GT 點數）；列＝離內部接縫線的距離，欄＝離地高度 ===")
    print(f"{'':22}" + "".join(f"{f'{lo:.0f}~{hi:.0f}m' if hi < 1e8 else f'>{lo:.0f}m':>26}" for lo, hi in HB))
    for lab, (cov, al) in res.items():
        for lo, hi in DB:
            cells = []
            for hl, hh_ in HB:
                mm = km & (dseam >= lo) & (dseam < hi) & (hgt >= hl) & (hgt < hh_)
                n = int(mm.sum())
                cells.append(f"{100 * cov[mm].mean():.0f}％／{100 * al[mm & cov].sum() / max((mm & cov).sum(), 1):.0f}％（{n:,}）" if n >= 200 else "—")
            dl = f"{lo:.0f}~{hi:.0f}m" if hi < 1e8 else f">{lo:.0f}m"
            print(f"{(lab + ' 接縫 ' + dl)[:22]:22}" + "".join(f"{c:>26}" for c in cells))

if a.png_dir:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as _fm, subprocess
    try:   # 中文字型（同 tools/plot_analysis.py）
        _ff = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK TC"], capture_output=True, text=True).stdout.strip()
        _fm.fontManager.addfont(_ff); plt.rcParams["font.sans-serif"] = [_fm.FontProperties(fname=_ff).get_name(), "DejaVu Sans"]
    except Exception:
        pass
    os.makedirs(a.png_dir, exist_ok=True)
    lo_u, hi_u = np.percentile(U[:, 0], [0.2, 99.8]); lo_v, hi_v = np.percentile(U[:, 1], [0.2, 99.8])
    # GT 顏色俯視（每個 2 m 格取最高點的顏色）
    cell = 0.02
    iu = ((U[:, 0] - lo_u) / cell).astype(int); iv = ((U[:, 1] - lo_v) / cell).astype(int)
    ok = (iu >= 0) & (iv >= 0) & (iu < int((hi_u - lo_u) / cell)) & (iv < int((hi_v - lo_v) / cell))
    W, H = int((hi_u - lo_u) / cell), int((hi_v - lo_v) / cell)
    img = np.zeros((H, W, 3), np.uint8); zb = np.full((H, W), -1e9)
    order = np.argsort(X[ok, 2])
    iuo, ivo, co = iu[ok][order], iv[ok][order], C[ok][order]
    img[H - 1 - ivo, iuo] = co
    for lab, (cov, al) in res.items():
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(img, extent=[lo_u, hi_u, lo_v, hi_v]); axs[0].set_title("GT 顏色俯視（分區轉換後的 xy）")
        c10 = 0.1; ju = ((U[:, 0] - lo_u) / c10).astype(int); jv = ((U[:, 1] - lo_v) / c10).astype(int)
        Wc, Hc = int((hi_u - lo_u) / c10) + 1, int((hi_v - lo_v) / c10) + 1
        okc = (ju >= 0) & (jv >= 0) & (ju < Wc) & (jv < Hc)
        num = np.zeros((Hc, Wc)); den = np.zeros((Hc, Wc))
        np.add.at(den, (Hc - 1 - jv[okc], ju[okc]), 1); np.add.at(num, (Hc - 1 - jv[okc], ju[okc]), cov[okc])
        rate = np.where(den >= 20, num / np.maximum(den, 1), np.nan)
        im = axs[1].imshow(rate, extent=[lo_u, hi_u, lo_v, hi_v], cmap="RdYlGn", vmin=0.3, vmax=1.0)
        plt.colorbar(im, ax=axs[1], fraction=0.03); axs[1].set_title(f"{lab}：10 m 格 GT 點覆蓋率（≥ 20 點的格）")
        for ax in axs:
            for xl in XL: ax.axvline(xl, color="cyan", lw=0.8)
            for yl in YL: ax.axhline(yl, color="cyan", lw=0.8)
            for b, (blo, bhi) in boxes.items():
                cu = np.clip((max(blo[0], lo_u) + min(bhi[0], hi_u)) / 2, lo_u, hi_u); cv = np.clip((max(blo[1], lo_v) + min(bhi[1], hi_v)) / 2, lo_v, hi_v)
                ax.text(cu, cv, f"b{b}", color="yellow", fontsize=14, ha="center", va="center", weight="bold")
            ax.set_xlim(lo_u, hi_u); ax.set_ylim(lo_v, hi_v)
        fig.tight_layout(); out = os.path.join(a.png_dir, f"coverage_{lab}.png"); fig.savefig(out, dpi=90); plt.close(fig)
        print(f"[圖] {out}")
    np.savez_compressed(os.path.join(a.png_dir, "seam_cov_points.npz"), U=U.astype(np.float32), hgt=hgt.astype(np.float32),
                        fac=fac, flat=flat, blk=blk, corner=corner, **{f"cov_{k}": v[0] for k, v in res.items()}, **{f"al_{k}": v[1] for k, v in res.items()})
print("DONE seam_coverage")
