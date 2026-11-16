#!/usr/bin/env python
"""★ 2026-11-16 單目法向先驗的離線產生器（Fable H3；使用者 11-16「都做」）。CPU；lab 上跑（estimated_depths 是 lab 的命名）。
每張訓練影像：單目反深度 x 逐張 scale＋offset（estimated_depth_scales.json）=> 相機座標點 => 中央差分叉積的法向 => 轉世界座標；
只留「可以當先驗」的像素（其餘 A＝0）：
  校正深度不可靠（estimated_depths_tri_rel 的斜率級 > slope_max 或大小級 > size_max，或校正深度無效）——可靠處已經有深度上限與 SfM 撐著
  局部平面：步長 1 與 2 算出的法向夾角 < planar_deg（單目在邊緣、細物上法向會亂）
  不在深度斷層：3x3 內深度最大／最小 < 1 + disc（相對量）
  深度在本張影像可靠 SfM 點深度的 [min / range_k, max x range_k] 內（排除單目當成無限遠的海面、天空；相對量）
  立面（--facade_only，預設開）：世界法向 |n_z| < facade_nz
沒有場景單位門檻：全部是角度、相對深度比或可靠度級別。
輸出：<dataset>/<out>/<影像名去副檔名>.png（RGBA：RGB＝世界法向 x127.5＋127.5、A＝255 可用）、stats.json
依據：H3 離線檢查（tools/probe/h3_mono_normal_1116.py）塔頂校正深度不可靠的立面像素上，單目法向誤差中位 11～12°、70～75% < 20°。
用法（lab，cwd＝repo 根）：python tools/make_mono_normals.py data/matrix_city/aerial/train/block_all --blocks 2 3 --nproc 12
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A = None


def normals(z, fx, fy, cx, cy, s):
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, s:-s] = P[:, 2 * s:] - P[:, :-2 * s]; dy[s:-s] = P[2 * s:] - P[:-2 * s]
    n = np.cross(dx, dy)
    return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)


def one(job):
    name, cam, R, zmin, zmax = job
    import cv2
    from PIL import Image
    a = A
    out = os.path.join(a["out"], os.path.splitext(name)[0] + ".png")
    if os.path.exists(out) and not a["force"]:
        return name, None
    fi = os.path.join(a["D"], a["depth_dir"], name + ".npy")
    sc = a["scales"].get(name)
    if sc is None or not os.path.exists(fi):
        return name, "缺單目深度或 scale"
    W, H = a["W"], a["H"]
    inv = np.load(fi).squeeze().astype(np.float32)
    inv = cv2.resize(inv, (W, H), interpolation=cv2.INTER_AREA) * sc["scale"] + sc["offset"]
    z = np.where(inv > 1e-6, 1.0 / np.clip(inv, 1e-6, None), np.nan)
    fr_ = os.path.join(a["D"], "estimated_depths_tri_rel", name + ".npy"); ft_ = os.path.join(a["D"], "estimated_depths_tri", name + ".npy")
    if not (os.path.exists(fr_) and os.path.exists(ft_)):
        return name, "缺校正深度"
    rel = np.load(fr_)
    rel = cv2.resize(rel.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    tri = np.load(os.path.join(a["D"], "estimated_depths_tri", name + ".npy")).squeeze().astype(np.float32)
    tri = cv2.resize(tri, (W, H), interpolation=cv2.INTER_NEAREST)
    unrel = ((rel & 7) > a["slope_max"]) | ((rel >> 3) > a["size_max"]) | ~(tri > 1e-6)
    fx, fy, cx, cy, w0, h0 = cam
    kx, ky = W / w0, H / h0
    n1 = normals(z, fx * kx, fy * ky, cx * kx, cy * ky, 1)
    n2 = normals(z, fx * kx, fy * ky, cx * kx, cy * ky, 2)
    planar = np.abs((n1 * n2).sum(-1)) > np.cos(np.radians(a["planar_deg"]))
    from scipy.ndimage import maximum_filter, minimum_filter
    zz = np.nan_to_num(z, nan=0.0)
    smooth = (maximum_filter(zz, 3) / np.clip(minimum_filter(zz, 3), 1e-9, None)) < 1 + a["disc"]
    inrange = (z >= zmin / a["range_k"]) & (z <= zmax * a["range_k"])
    nw = n1 @ R                                   # 世界法向＝R^T n_cam（COLMAP：X_cam = R X_world + t）
    ok = np.isfinite(z) & unrel & planar & smooth & inrange & np.isfinite(nw).all(-1)
    ok[:2] = ok[-2:] = False; ok[:, :2] = ok[:, -2:] = False
    if a["facade_only"]:
        ok &= np.abs(nw[..., 2]) < a["facade_nz"]
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = np.clip(np.round(np.nan_to_num(nw) * 127.5 + 127.5), 0, 255).astype(np.uint8)
    rgba[..., 3] = ok.astype(np.uint8) * 255
    Image.fromarray(rgba, "RGBA").save(out)
    return name, (float(ok.mean()), float(unrel.mean()))


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset"); ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--out", default="normal_prior_mono")
    ap.add_argument("--width", type=int, default=640, help="輸出寬（高照影像比例）；metric 用 nearest 放大到渲染解析度")
    ap.add_argument("--slope_max", type=int, default=1); ap.add_argument("--size_max", type=int, default=3)
    ap.add_argument("--planar_deg", type=float, default=10.0); ap.add_argument("--disc", type=float, default=0.05)
    ap.add_argument("--range_k", type=float, default=1.2); ap.add_argument("--facade_nz", type=float, default=0.3)
    ap.add_argument("--all_surfaces", action="store_true", help="不只收立面")
    ap.add_argument("--nproc", type=int, default=12); ap.add_argument("--force", action="store_true")
    ap.add_argument("--depth_dir", default="estimated_depths", help="單目反深度目錄（本機的舊命名是 estimated_depths_lab）")
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 張（測試用）")
    a = ap.parse_args()
    from internal.utils.colmap import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat
    D = a.dataset
    cams = read_cameras_binary(os.path.join(D, "sparse/0/cameras.bin")); imgs = read_images_binary(os.path.join(D, "sparse/0/images.bin"))
    pts = read_points3D_binary(os.path.join(D, "sparse/0/points3D.bin"))
    good = {k for k, q in pts.items() if len(q.image_ids) >= 3 and q.error < 1.5}
    by_name = {v.name: v for v in imgs.values()}
    names = []
    for b in a.blocks:
        names += [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip()]
    names = sorted(set(n for n in names if n in by_name))
    if a.limit:
        names = names[:: max(1, len(names) // a.limit)][: a.limit]
    c0 = cams[by_name[names[0]].camera_id]
    W = a.width; H = int(round(W * c0.height / c0.width))
    out = os.path.join(D, a.out); os.makedirs(out, exist_ok=True)
    jobs = []
    for n in names:
        im = by_name[n]; cm = cams[im.camera_id]; R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec)
        ids = [i for i in im.point3D_ids if i in good]
        if len(ids) < 10:
            continue
        Xc = np.stack([pts[i].xyz for i in ids]) @ R.T + t
        zc = Xc[:, 2][Xc[:, 2] > 0]
        fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
        cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
        jobs.append((n, (fx, fy, cx, cy, cm.width, cm.height), R, float(np.percentile(zc, 1)), float(np.percentile(zc, 99))))
    A = dict(D=D, out=out, W=W, H=H, scales=json.load(open(os.path.join(D, "estimated_depth_scales.json"))), slope_max=a.slope_max,
             size_max=a.size_max, planar_deg=a.planar_deg, disc=a.disc, range_k=a.range_k, facade_nz=a.facade_nz,
             facade_only=not a.all_surfaces, force=a.force, depth_dir=a.depth_dir)
    t0 = time.time(); fr = []; bad = 0
    with Pool(a.nproc) as pool:
        for i, (n, r) in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if isinstance(r, str):
                bad += 1
            elif r is not None:
                fr.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}（{time.time() - t0:.0f} s）", flush=True)
    fr = np.array(fr) if fr else np.zeros((0, 2))
    st = dict(blocks=a.blocks, images=len(jobs), new=len(fr), failed=bad, size=[W, H],
              prior_px_mean=float(fr[:, 0].mean()) if len(fr) else None, unreliable_px_mean=float(fr[:, 1].mean()) if len(fr) else None)
    json.dump(st, open(os.path.join(out, f"stats_{'_'.join(map(str, a.blocks))}.json"), "w"), indent=1)
    print(f"完成：{len(jobs)} 張（新產生 {len(fr)}、失敗 {bad}）｜先驗像素平均 {st['prior_px_mean']}｜校正深度不可靠像素平均 {st['unreliable_px_mean']}｜"
          f"{time.time() - t0:.0f} s => {out}")


if __name__ == "__main__":
    main()
