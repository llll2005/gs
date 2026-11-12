"""★ 2026-11-12 規格 3 §3.1／§3.2 深度上限的遮罩（metric `depth_cap_mask_dir` 讀；調參元件 `cmflat`）。

做什麼：每張訓練影像一張 uint8 PNG（255＝這個像素設上限、0＝不設），檔名同影像、解析度同校正深度（預設 960x540）。
  --kind flat：校正深度反投影成相機座標點圖，鄰點取 ±step px（預設 2）的差分做外積得法向，轉到世界座標後 avg x avg（預設 5x5）平均再正規化；
               |n_z| > thr（預設 0.8；世界 +z 朝上，分區旋轉是單位矩陣）才留。沒深度、或 ±step 的四個鄰點有任何一個沒深度＝0。
  --kind rel／photo：規格 3 §3.2 寫「要 §1 的離線拆解指到才做」=> 還沒實作。
為什麼（規格 3 §0 線索 1）：B 臂的抽母點計價不限朝向時多掉 −0.134／−0.082 dB、只算躺平時 ≈ 0；只開上限的掉分 40% 在立面（`37` §3.1、§7）。
  => 上限只罰躺平的先驗表面（路面、屋頂；地下層就在它們底下），不碰立面和斜面。
沒有場景單位：只用法向的 z 分量（無因次）、px。法向方向對深度的整體縮放不變（`tests/cap_mask_test.py`：場景放大 10 倍逐像素相同）。
只吃訓練輸入：校正深度（`estimated_depths_tri`）＋姿態（sparse/0）。可靠度只用來印統計（訓練時上限本來就只設在可靠像素上）。
輸出：<out>/<影像名>（PNG）、<out>/stats_block_<B>.json（張數、缺檔、保留比例；task_tune25 的 cmflat 用它判斷遮罩在不在）。
用法（lab，cwd＝repo 根目錄，CPU）：
  conda run -n gspl --no-capture-output python tools/make_cap_masks.py --kind flat --blocks 1 4 3
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import cv2
sys.path.insert(0, ".")
cv2.setNumThreads(1)


def world_normals(inv, fx, fy, cx, cy, sx, sy, R, step=2, avg=5):
    """inv [h,w] 逆深度（<=0＝無效）；fx..cy＝COLMAP 相機（全解析度像素）；sx, sy＝深度圖寬高 / 相機寬高；R＝世界→相機旋轉。
    回傳 (n [h,w,3] 世界座標單位法向（無效處為 0）, ok [h,w] bool)。"""
    inv = np.asarray(inv, dtype=np.float64)
    h, w = inv.shape
    valid = inv > 0
    z = np.where(valid, 1.0 / np.where(valid, inv, 1.0), 0.0)
    u = (np.arange(w) + 0.5) / sx
    v = (np.arange(h) + 0.5) / sy
    uu, vv = np.meshgrid(u, v)
    P = np.stack([z * (uu - cx) / fx, z * (vv - cy) / fy, z], -1)
    s = int(step)
    ok = np.zeros((h, w), bool)
    du = np.zeros((h, w, 3)); dv = np.zeros((h, w, 3))
    du[:, s:w - s] = P[:, 2 * s:] - P[:, :w - 2 * s]
    dv[s:h - s, :] = P[2 * s:, :] - P[:h - 2 * s, :]
    ok[s:h - s, s:w - s] = (valid[s:h - s, s:w - s] & valid[s:h - s, 2 * s:] & valid[s:h - s, :w - 2 * s]
                            & valid[2 * s:, s:w - s] & valid[:h - 2 * s, s:w - s])
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1)
    ok &= nn > 0
    n = np.where(ok[..., None], n / np.where(nn > 0, nn, 1.0)[..., None], 0.0)
    n = n @ np.asarray(R, dtype=np.float64)            # 世界 = R^T n_cam（列向量）=> 行向量寫法 n_cam @ R
    if avg and avg > 1:
        k = (int(avg), int(avg))
        n = np.stack([cv2.boxFilter(n[..., i], cv2.CV_64F, k, normalize=False, borderType=cv2.BORDER_CONSTANT) for i in range(3)], -1)
        nn = np.linalg.norm(n, axis=-1)
        ok &= nn > 0
        n = np.where(ok[..., None], n / np.where(nn > 0, nn, 1.0)[..., None], 0.0)
    return n, ok


def flat_mask(inv, fx, fy, cx, cy, sx, sy, R, thr=0.8, step=2, avg=5):
    """uint8 [h,w]：255＝躺平（|n_z| > thr）且法向算得出來，其餘 0。"""
    n, ok = world_normals(inv, fx, fy, cx, cy, sx, sy, R, step, avg)
    return np.where(ok & (np.abs(n[..., 2]) > thr), 255, 0).astype(np.uint8)


def _one(args):
    name, = args
    from PIL import Image
    from internal.utils.colmap import qvec2rotmat
    out = os.path.join(A.out, name)
    npy = os.path.join(A.data, A.depth_dir, name + ".npy")
    im = BYNAME.get(name)
    if im is None or not os.path.exists(npy):
        return name, None
    rel_p = os.path.join(A.data, A.depth_dir + "_rel", name + ".npy")
    inv = np.load(npy)
    cam = CAMS[im.camera_id]; fx, fy, cx, cy = cam.params[:4]
    h, w = inv.shape
    if A.force or not os.path.exists(out):
        m = flat_mask(inv, fx, fy, cx, cy, w / cam.width, h / cam.height, qvec2rotmat(im.qvec), A.thr, A.step, A.avg)
        Image.fromarray(m).save(out)
    else:
        m = np.array(Image.open(out))
    keep = m > 0
    valid = inv > 0
    st = [float(valid.mean()), float((keep & valid).sum() / max(valid.sum(), 1))]
    if os.path.exists(rel_p):
        cls = np.load(rel_p)
        relok = valid & ((cls & 7) <= A.rel_slope_max) & ((cls >> 3) <= A.rel_size_max)
        st += [float(relok.mean()), float((keep & relok).sum() / max(relok.sum(), 1))]
    else:
        st += [np.nan, np.nan]
    return name, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--kind", default="flat", choices=["flat", "rel", "photo"])
    ap.add_argument("--depth_dir", default="estimated_depths_tri")
    ap.add_argument("--out", default=None, help="預設 <data>/cap_masks_<kind>")
    ap.add_argument("--thr", type=float, default=0.8, help="躺平＝世界法向 z 分量絕對值 > thr（同 bpf）")
    ap.add_argument("--step", type=int, default=2, help="差分的鄰點距離（px）")
    ap.add_argument("--avg", type=int, default=5, help="法向平均的視窗（px）")
    ap.add_argument("--rel_slope_max", type=int, default=1, help="只用來印統計：訓練時的可靠度過濾（斜率級）")
    ap.add_argument("--rel_size_max", type=int, default=3, help="只用來印統計：訓練時的可靠度過濾（大小級）")
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    A = ap.parse_args()
    if A.kind != "flat":
        sys.exit(f"⛔ --kind {A.kind} 還沒實作（規格 3 §3.2：要 §1 的離線拆解指到才做）")
    A.out = A.out or os.path.join(A.data, f"cap_masks_{A.kind}")
    os.makedirs(A.out, exist_ok=True)
    from internal.utils.colmap import read_cameras_binary, read_images_binary
    from multiprocessing import Pool
    t0 = time.time()
    CAMS = read_cameras_binary(os.path.join(A.data, "sparse/0/cameras.bin"))
    BYNAME = {im.name: im for im in read_images_binary(os.path.join(A.data, "sparse/0/images.bin")).values()}
    for b in A.blocks:
        names = [l.strip() for l in open(os.path.join(A.data, A.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip()]
        res, miss = {}, []
        with Pool(A.nproc) as pool:
            for name, st in pool.imap_unordered(_one, [(n,) for n in names], chunksize=8):
                (miss.append(name) if st is None else res.__setitem__(name, st))
        S = np.array(list(res.values()), dtype=np.float64) if res else np.zeros((0, 4))
        summary = {"block": b, "kind": A.kind, "thr": A.thr, "step": A.step, "avg": A.avg, "depth_dir": A.depth_dir,
                   "images": len(names), "written": len(res), "missing": len(miss), "missing_names": miss[:20],
                   "valid_depth_frac": float(np.nanmean(S[:, 0])) if len(S) else None,
                   "kept_of_valid": float(np.nanmean(S[:, 1])) if len(S) else None,
                   "reliable_frac": float(np.nanmean(S[:, 2])) if len(S) else None,
                   "kept_of_reliable": float(np.nanmean(S[:, 3])) if len(S) else None}
        json.dump(summary, open(os.path.join(A.out, f"stats_block_{b}.json"), "w"), indent=1, ensure_ascii=False)
        print(f"block {b}：{len(res):,}／{len(names):,} 張（缺深度 {len(miss)}）｜有深度的像素 {100 * summary['valid_depth_frac']:.1f}%，其中躺平留下 "
              f"{100 * summary['kept_of_valid']:.1f}%｜可靠像素（訓練時設上限的那些）{100 * (summary['reliable_frac'] or 0):.1f}%，其中留下 "
              f"{100 * (summary['kept_of_reliable'] or 0):.1f}%｜{time.time() - t0:.0f} s => {A.out}", flush=True)
        if miss:
            print(f"  ⚠⚠ 缺深度 {len(miss)} 張（例：{miss[:3]}）=> 這些影像訓練時會被當成缺檔（不設上限）", flush=True)
