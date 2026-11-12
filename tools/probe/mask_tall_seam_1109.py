"""11-09 使用者 viewer：b1／b2／b6 交界的高樓在三塊都重建得破爛、合併也爛——「這棟的本體在哪？沒有本體嗎？」
疑點：塊外遮罩（tools/block_mask44.py）的 A／B／C 是用**單目深度**反投影的第一個表面判的；單目在高樓誤差大（塔頂 68% 判錯、整體 ±10%）
=> 盒內高樓的像素可能被判成 C（不監督）或 B（強制透明）。遮罩工具自己的 --check 是全部盒內 SfM 點的平均（A 97.5～99.4%），被地面稀釋。
這裡把「盒內、本塊訓練影像觀測到的可靠 SfM 點」（＝真的第一個表面）依 離地高度 × 到最近接縫的距離 分組，查它在該影像遮罩裡的類別比例。
純 CPU、只讀（lab：遮罩在 <data>/block_masks44/block_<B>/<影像名>）。
用法：python tools/probe/mask_tall_seam_1109.py --blocks 1 2 5 6
★ 11-12 規格 4 §3：--masks block_masks44_tri 量校正深度遮罩；--gt_stride N 改用 MatrixCity 真深度在遮罩上每 N px 抽一個像素反投影當樣本
  （SfM 點上校正深度本來就準，用 SfM 點當樣本會偏袒新遮罩；真深度只診斷，不進任何機制）。
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary

D = "data/matrix_city/aerial/train/block_all"
ap = argparse.ArgumentParser()
ap.add_argument("--blocks", type=int, nargs="+", default=list(range(16)))
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--masks", default="block_masks44")
ap.add_argument("--gt_stride", type=int, default=0, help=">0：樣本改成遮罩上每 N px 一個像素、用真深度反投影（0＝SfM 觀測）")
a = ap.parse_args()
if a.gt_stride > 0:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import monodepth_sea_1102 as MGT

p = torch.load(f"{D}/{a.part}/partitions.pt", map_location="cpu", weights_only=False)
Rp = p["extra_data"]["rotation_transform"][:3, :3].double().numpy()
xy0 = p["partition_coordinates"]["xy"].double().numpy(); size = p["scene_config"]["partition_size"].double().numpy()
BMIN, BMAX = xy0, xy0 + size
inner = np.median(np.minimum(size[:, 0], size[:, 1]))
REF = np.median(size[(size[:, 0] < 10 * inner) & (size[:, 1] < 10 * inner)], axis=0).min()
eps = 1e-6 * REF
seam = np.zeros((16, 4), bool)
for b in range(16):
    for c in range(16):
        if c == b:
            continue
        oy = min(BMAX[b, 1], BMAX[c, 1]) - max(BMIN[b, 1], BMIN[c, 1]) > eps
        ox = min(BMAX[b, 0], BMAX[c, 0]) - max(BMIN[b, 0], BMIN[c, 0]) > eps
        seam[b, 0] |= oy and abs(BMIN[b, 0] - BMAX[c, 0]) < eps
        seam[b, 1] |= oy and abs(BMAX[b, 0] - BMIN[c, 0]) < eps
        seam[b, 2] |= ox and abs(BMIN[b, 1] - BMAX[c, 1]) < eps
        seam[b, 3] |= ox and abs(BMAX[b, 1] - BMIN[c, 1]) < eps

pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
ids = np.array(list(pts.keys())); mx = ids.max() + 1
XYZ = np.zeros((mx, 3)); OK = np.zeros(mx, bool)
for k, q in pts.items():
    XYZ[k] = q.xyz; OK[k] = len(q.image_ids) >= 3 and q.error < 1.5
del pts
S = XYZ[OK]
o0 = np.floor(S[:, :2].min(0) / 0.02).astype(int) - 2; ci = np.floor(S[:, :2] / 0.02).astype(int) - o0; shp = tuple(ci.max(0) + 3)
cid = ci[:, 0] * shp[1] + ci[:, 1]; oo = np.lexsort((S[:, 2], cid)); uu, ff, cc = np.unique(cid[oo], return_index=True, return_counts=True)
GZ = np.full(shp[0] * shp[1], np.nan); GZ[uu] = S[oo, 2][ff + cc // 2]; GZ = GZ.reshape(shp)
# 地面＝10 m 窗內 2 m 格中位的最小值（描述用的分組，不進任何規則）
from scipy.ndimage import minimum_filter
GZ10 = minimum_filter(np.nan_to_num(GZ, nan=np.inf), size=5); GZ10[~np.isfinite(GZ10)] = np.nanpercentile(S[:, 2], 10)
ims = read_images_binary(f"{D}/sparse/0/images.bin"); cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
by_name = {os.path.basename(im.name): im for im in ims.values()}
HB = [(-1e9, 20, "< 20 m"), (20, 50, "20~50"), (50, 100, "50~100"), (100, 200, "100~200"), (200, 1e9, "> 200 m")]
SB = [(-1e9, -0.2, "內部 < −0.2"), (-0.2, -0.1, "−0.2~−0.1"), (-0.1, -0.05, "−0.1~−0.05"), (-0.05, 0.0, "−0.05~0（接縫）")]
TOT = np.zeros((len(HB), len(SB), 3), np.int64)
for b in a.blocks:
    names = [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip()]
    md = os.path.join(D, a.masks, f"block_{b}")
    cnt = np.zeros((len(HB), len(SB), 3), np.int64)
    nimg = 0
    for nm in names:
        im = by_name.get(nm); mp = os.path.join(md, nm)
        if im is None or not os.path.exists(mp):
            continue
        cls = np.asarray(Image.open(mp)); H, W = cls.shape
        cam = cams[im.camera_id]
        if a.gt_stride > 0:                      # 真深度像素樣本
            dg = MGT.load_gt(nm)
            if dg is None:
                continue
            H0, W0 = dg.shape[:2]
            vv, uu = np.meshgrid(np.arange(0, H, a.gt_stride), np.arange(0, W, a.gt_stride), indexing="ij")
            vv, uu = vv.ravel(), uu.ravel()
            V0 = np.clip(((vv + 0.5) * H0 / H).astype(int), 0, H0 - 1); U0 = np.clip(((uu + 0.5) * W0 / W).astype(int), 0, W0 - 1)
            z = dg[V0, U0].astype(np.float64)
            g = np.isfinite(z) & (z > 0) & (z < 1e8)
            vv, uu, z = vv[g], uu[g], z[g] * MGT.K_UNIT
            fx, fy, cx, cy = cam.params[:4]
            Uc = (uu + 0.5) * cam.width / W; Vc = (vv + 0.5) * cam.height / H
            Xc = np.stack([z * (Uc - cx) / fx, z * (Vc - cy) / fy, z], 1)
            from internal.utils.colmap import qvec2rotmat
            Rm = qvec2rotmat(im.qvec)
            X = (Xc - np.asarray(im.tvec)) @ Rm
            q = X @ Rp.T
            ui, vi = uu, vv
        else:
            pi = im.point3D_ids; ok = (pi >= 0) & (pi < mx)
            ok[ok] &= OK[pi[ok]]
            if not ok.any():
                continue
            X = XYZ[pi[ok]]; q = X @ Rp.T
        x, y = q[:, 0], q[:, 1]
        d = np.stack([x - BMIN[b, 0], BMAX[b, 0] - x, y - BMIN[b, 1], BMAX[b, 1] - y], 1)
        inb = (d >= 0).all(1)
        if not inb.any():
            continue
        dd = np.where(seam[b][None, :], d, np.inf)
        s = -dd.min(1) / REF
        c2 = np.floor(X[:, :2] / 0.02).astype(int) - o0
        okc = (c2[:, 0] >= 0) & (c2[:, 0] < shp[0]) & (c2[:, 1] >= 0) & (c2[:, 1] < shp[1])
        gz = np.full(len(X), np.nanpercentile(S[:, 2], 10)); gz[okc] = GZ10[c2[okc, 0], c2[okc, 1]]
        hz = (X[:, 2] - gz) * 100
        if a.gt_stride <= 0:
            xy = im.xys[ok]
            ui = np.clip((xy[:, 0] * W / cam.width).astype(int), 0, W - 1); vi = np.clip((xy[:, 1] * H / cam.height).astype(int), 0, H - 1)
        lab = cls[vi, ui]
        for i, (h0, h1, _) in enumerate(HB):
            for j, (s0, s1, _) in enumerate(SB):
                k = inb & np.isfinite(s) & (hz >= h0) & (hz < h1) & (s >= s0) & (s < s1)
                if k.any():
                    cnt[i, j] += np.bincount(lab[k], minlength=3)[:3]
        nimg += 1
    TOT += cnt
    print(f"\n== b{b}（{nimg} 張遮罩 {a.masks}）｜盒內{'真深度像素（每 ' + str(a.gt_stride) + ' px）' if a.gt_stride > 0 else ' SfM 觀測'}被標成 A（監督）／B（強制透明）／C（不監督）%（樣本數）")
    print("  離地高度    " + "".join(f"{nm:>28}" for _, _, nm in SB))
    for i, (_, _, hn) in enumerate(HB):
        cells = []
        for j in range(len(SB)):
            n = cnt[i, j].sum()
            cells.append(f"{100 * cnt[i, j, 1] / n:4.0f}／{100 * cnt[i, j, 2] / n:3.0f}／{100 * cnt[i, j, 0] / n:3.0f}（{n:,}）" if n >= 100 else "—")
        print(f"  {hn:10}" + "".join(f"{c:>28}" for c in cells), flush=True)
print(f"\n== 合計 {a.blocks}")
for i, (_, _, hn) in enumerate(HB):
    cells = []
    for j in range(len(SB)):
        n = TOT[i, j].sum()
        cells.append(f"{100 * TOT[i, j, 1] / n:4.0f}／{100 * TOT[i, j, 2] / n:3.0f}／{100 * TOT[i, j, 0] / n:3.0f}（{n:,}）" if n >= 100 else "—")
    print(f"  {hn:10}" + "".join(f"{c:>28}" for c in cells))
print("DONE")
