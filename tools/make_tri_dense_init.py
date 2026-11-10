"""★ 2026-11-11 用 SfM 三角形校正深度做稠密初始化（每塊一個 xyz+rgb 的 PLY）。

做什麼：SfM＋補點（sfmfill_sweep44_crop/dup1，裁盒後的原始點，一顆不動）之外，把本塊訓練影像裡「可靠三角形內」的像素沿校正深度
        反投影成 3D 點，落進世界座標的體素；**至少 min_views 個不同視角都有點落在同一格**才算數（多視角一致，擋掉單張深度的離群值），
        每格取質心與平均顏色。已經有 SfM 點的格不再加點。體素邊長用二分法定，讓總顆數＝target_frac x 本塊 cap（cap＝平均 cap x SfM 點數分配權重）。
為什麼：現行初始化＝SfM 點複製 K 份（同一位置抖動），沒有新資訊；無紋理的地方（柏油路面、素色牆）沒有 SfM 點，要靠 MCMC 增生慢慢長過去，
        增生期因此不能短（11-10 3x3：增生期 40% 全面最差）；路面那一層沒蓋滿，是地下層的來源（紀錄/實驗分析/31）。
        校正深度在可靠三角形內誤差中位 0.05%（tools/probe/tri_depth_1111.py）=> 一開始就把看得到的表面鋪滿一層、顏色直接取影像。
        🟨 同類：EDGS（稠密對應點初始化、拿掉增生）、Mini-Splatting2（深度重新初始化）；我方的深度來源是 SfM 三角形校正單目。
沒有場景單位常數：體素邊長由目標顆數解出來；可靠度用三角形的無因次斜率與像素邊長。
輸出：<dataset>/<out>/dup1/block_<B>.ply（task_cmp.sh：CITYGS_CROP_INIT=<out> CITYGS_DUP44=1）
用法（lab，cwd＝repo 根目錄，CPU）：
  conda run -n gspl --no-capture-output python tools/make_tri_dense_init.py data/matrix_city/aerial/train/block_all --blocks 1 4 3
"""
import os, sys, json, time, argparse
import numpy as np
import cv2
sys.path.insert(0, ".")


def write_xyzrgb_ply(path, xyz, rgb):
    """與 tools/make_sfm_fill_init.py 同一個格式（只有 xyz＋rgb；尺度由 setup_from_pcd 從點雲算）。"""
    from plyfile import PlyData, PlyElement
    el = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el["red"], el["green"], el["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def voxel_keys(xyz, origin, g):
    """世界座標 -> 體素整數鍵（int64；每軸 21 bit）。"""
    q = np.floor((xyz - origin) / g).astype(np.int64)
    if q.min() < 0 or q.max() >= (1 << 21):
        raise ValueError(f"體素索引超出 21 bit（g={g}，範圍 {q.min()}～{q.max()}）")
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def supported_voxels(xyz, rgb, view, origin, g, min_views, block_keys=None):
    """每個體素的質心／平均顏色，只留「不同視角數 >= min_views」而且不在 block_keys（已被 SfM 點佔用）裡的。回傳 (xyz [m,3], rgb [m,3])。"""
    k = voxel_keys(xyz, origin, g)
    uk, inv = np.unique(k, return_inverse=True)
    pair = np.unique(inv.astype(np.int64) * (int(view.max()) + 1) + view.astype(np.int64))
    nv = np.bincount(pair // (int(view.max()) + 1), minlength=len(uk))
    keep = nv >= min_views
    if block_keys is not None and len(block_keys):
        keep &= ~np.isin(uk, block_keys)
    cnt = np.bincount(inv, minlength=len(uk)).astype(np.float64)
    cen = np.stack([np.bincount(inv, weights=xyz[:, a], minlength=len(uk)) / cnt for a in range(3)], 1)
    col = np.stack([np.bincount(inv, weights=rgb[:, a].astype(np.float64), minlength=len(uk)) / cnt for a in range(3)], 1)
    return cen[keep], col[keep]


def count_supported(xyz, view, origin, g, min_views, block_xyz=None):
    k = voxel_keys(xyz, origin, g)
    uk, inv = np.unique(k, return_inverse=True)
    pair = np.unique(inv.astype(np.int64) * (int(view.max()) + 1) + view.astype(np.int64))
    nv = np.bincount(pair // (int(view.max()) + 1), minlength=len(uk))
    keep = nv >= min_views
    if block_xyz is not None and len(block_xyz):
        keep &= ~np.isin(uk, np.unique(voxel_keys(block_xyz, origin, g)))
    return int(keep.sum())


def solve_voxel(xyz, view, origin, n_want, min_views, block_xyz, g_hi, n_scan=40, iters=12):
    """找體素邊長 g，讓「有支撐、沒被 SfM 佔用」的格數 ≈ n_want。
    格數對 g 不單調：g 變小先變多（表面被切得更細），小到每格湊不滿 min_views 個視角後又變少。
    => 從大往小掃（每次 x 0.84），停在第一個格數 >= n_want 的 g（在「還有支撐」的那一支上），再和前一個 g 之間二分。
    掃完都不到 n_want（候選不夠密）=> 回傳格數最多的那個 g。回傳 (g, 格數)。"""
    g, prev, best = g_hi, None, (g_hi, -1)
    for _ in range(n_scan):
        n = count_supported(xyz, view, origin, g, min_views, block_xyz)
        if n > best[1]:
            best = (g, n)
        if n >= n_want:
            if prev is None:
                return g, n
            lo, hi = g, prev                      # count(lo) >= n_want > count(hi)
            for _ in range(iters):
                mid = np.sqrt(lo * hi)
                nm = count_supported(xyz, view, origin, mid, min_views, block_xyz)
                if nm >= n_want:
                    lo, n = mid, nm
                else:
                    hi = mid
            return lo, n
        if n < 0.5 * best[1]:                     # 已經過了峰值、越切越少 => 不用再掃
            break
        prev, g = g, g * 0.84
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--depth", default="estimated_depths_tri")
    ap.add_argument("--base", default="sfmfill_sweep44_crop/dup1", help="SfM＋補點（裁盒後），一顆不動")
    ap.add_argument("--out", default="sfmfill_tri44_crop")
    ap.add_argument("--cap_avg", type=float, default=1e6); ap.add_argument("--alloc", default="configs/cap_alloc_sfm_g44.json")
    ap.add_argument("--target_frac", type=float, default=0.9, help="起始顆數＝此比例 x cap（init 跟 cap 走）")
    ap.add_argument("--m_c", type=float, default=0.15, help="裁盒外擴（x 內側塊尺寸；與 sfmfill_sweep44_crop 相同）")
    ap.add_argument("--view_step", type=int, default=2); ap.add_argument("--stride", type=int, default=3, help="深度圖像素的取樣步長")
    ap.add_argument("--slope_max", type=int, default=1); ap.add_argument("--size_max", type=int, default=2)
    ap.add_argument("--min_views", type=int, default=2)
    a = ap.parse_args()
    import torch
    from plyfile import PlyData
    from internal.utils.colmap import read_cameras_binary, read_images_binary, qvec2rotmat
    D = a.dataset
    parts = torch.load(os.path.join(D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    ids = parts["partition_coordinates"]["id"].numpy(); xys = parts["partition_coordinates"]["xy"].double().numpy()
    size = parts["scene_config"]["partition_size"].double().numpy()
    inner = np.median(np.minimum(size[:, 0], size[:, 1]))
    ref = np.median(size[(size[:, 0] < 10 * inner) & (size[:, 1] < 10 * inner)], axis=0)
    mc = a.m_c * ref.min()
    cams = read_cameras_binary(os.path.join(D, "sparse/0/cameras.bin")); imgs = read_images_binary(os.path.join(D, "sparse/0/images.bin"))
    by_name = {v.name: v for v in imgs.values()}
    W8 = json.load(open(a.alloc))["weights"]
    rng = np.random.default_rng(0)
    for b in a.blocks:
        t0 = time.time()
        k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0])
        lo, hi = xys[k] - mc, xys[k] + size[k] + mc
        cap = a.cap_avg * float(W8[str(b)]); target = int(a.target_frac * cap)
        v = PlyData.read(os.path.join(D, a.base, f"block_{b}.ply"))["vertex"].data
        bx = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64); bc = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.uint8)
        names = [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip()][:: a.view_step]
        XS, CS, VS = [], [], []
        for vi, nm in enumerate(names):
            im = by_name.get(nm); f = os.path.join(D, a.depth, nm + ".npy")
            if im is None or not os.path.exists(f):
                continue
            inv = np.load(f); rel = np.load(os.path.join(D, a.depth + "_rel", nm + ".npy"))
            H, W = inv.shape
            cam = cams[im.camera_id]; fx, fy, cx, cy = cam.params[:4]
            sx, sy = W / cam.width, H / cam.height
            oy, ox = rng.integers(0, a.stride, 2)
            vv, uu = np.meshgrid(np.arange(oy, H, a.stride), np.arange(ox, W, a.stride), indexing="ij")
            i_ = inv[vv, uu]; r_ = rel[vv, uu]
            ok = (i_ > 1e-6) & ((r_ & 7) <= a.slope_max) & ((r_ >> 3) <= a.size_max)
            if not ok.any():
                continue
            z = 1.0 / i_[ok].astype(np.float64)
            u = (uu[ok] + 0.5) / sx; w_ = (vv[ok] + 0.5) / sy
            Xc = np.stack([(u - cx) / fx * z, (w_ - cy) / fy * z, z], 1)
            R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec, dtype=np.float64)
            Xw = (Xc - t) @ R
            xy = (Xw @ Rp.T)[:, :2]
            inb = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
            if not inb.any():
                continue
            img = cv2.imread(os.path.join(D, "input", nm))
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1]
            XS.append(Xw[inb].astype(np.float32)); CS.append(img[vv[ok][inb], uu[ok][inb]]); VS.append(np.full(int(inb.sum()), vi, np.int32))
        X = np.concatenate(XS).astype(np.float64); C = np.concatenate(CS); V = np.concatenate(VS)
        origin = np.minimum(X.min(0), bx.min(0)) - 1e-3
        ext = float((np.maximum(X.max(0), bx.max(0)) - origin).max())
        n_want = max(target - len(bx), 0)
        g, n_got = solve_voxel(X, V, origin, n_want, a.min_views, bx, ext / 64)
        dx, dc = supported_voxels(X, C, V, origin, g, a.min_views, np.unique(voxel_keys(bx, origin, g)))
        out = os.path.join(D, a.out, "dup1", f"block_{b}.ply")
        write_xyzrgb_ply(out, np.concatenate([bx, dx]).astype(np.float32), np.concatenate([bc, np.clip(np.round(dc), 0, 255).astype(np.uint8)]))
        # 報告：體素邊長對 SfM 近鄰間距、對像素足跡的倍數（都只是描述，不是門檻）
        from scipy.spatial import cKDTree
        sub = bx[rng.choice(len(bx), min(len(bx), 50000), replace=False)]
        nn = np.median(cKDTree(bx).query(sub, k=2)[0][:, 1])
        print(f"b{b}：cap {cap:,.0f} => 目標 {target:,}｜SfM＋補點 {len(bx):,}｜視角 {len(names)}、候選 {len(X):,}｜體素邊長 {g:.5f}（SfM 近鄰間距中位 {nn:.5f} 的 {g / nn:.2f} 倍）｜"
              f"稠密點 {len(dx):,}（要 {n_want:,}）=> 共 {len(bx) + len(dx):,}（{(len(bx) + len(dx)) / cap:.2f} cap）｜{time.time() - t0:.0f} s => {out}", flush=True)
    print("DONE make_tri_dense_init", flush=True)


if __name__ == "__main__":
    main()
