#!/usr/bin/env python
"""★ 2026-11-17 規格 6 §2：3D 海面物件（共識平面＋海面格＋每張影像的反投影遮罩）；同時做 §1 的前提量測 P1～P5。CPU；lab 上跑（P2／P3 要 GT 深度）。
輸入：`tools/make_plane_depth.py` 的每張影像海面區（<影像名>_k.png 的 kind 2）與平面參數（<影像名>_planes.npz，相機座標 n·X = c、像素數、殘差）、
      校正深度與可靠度、相機。
共識平面：本塊訓練影像的所有海面區，轉成世界平面（n_w＝R^T n、c_w＝c − n·t，法向朝 +z），依像素數加權：
      法向＝分量加權中位再正規化；丟掉夾角或截距偏離超過 3 x 加權 MAD 的區，剩下的加權平均重算（P1 報一致性與分群）。
海面格：平面上的規則格子，邊長＝P 像素 x（海面格最近可見深度的中位）/ fx_train（P 預設 16；像素單位，沒有公尺常數）。
      每張影像沿像素射線（步長 stride）投到平面：落在該張海面區＝贊成；該像素校正深度可靠、而且表面在平面上或後面
      （深度 >= 平面深度 x (1 − margin)）＝反對；其他棄權。海面格＝贊成 >= min_yes 而且 贊成 > 反對（每張影像每格最多一票）。
      範圍＝稠密起點的裁切盒（塊盒＋m_c x 內側塊尺寸，同 make_tri_dense_init.py）。
反投影遮罩（--masks）：每張影像每個像素沿射線投到平面、落在海面格＝255；但該像素校正深度可靠且在平面前面（被擋）＝0。解析度同校正深度。
輸出：<dataset>/<out>/block_<b>.npz（平面、格子原點與面內軸、邊長、海面格索引、票數）、<out>/masks/<影像名>.png、stats_block_<b>.json
前提量測（--premise，lab：GT 深度經 tools/probe/monodepth_sea_1102）：
  P1 各區對共識平面的一致性（區重心像素的相對深度差、法向夾角）對各區自己的殘差；截距分群數
  P2 GT 海面像素上：共識平面深度 vs 單張平面深度（平面深度圖）對 GT 的相對誤差中位、p90
  P3 反投影遮罩對 GT 海面像素的涵蓋；遮罩內不是海的：GT 在平面前面（有東西擋）／真的不是海
  P4 海面格數、鋪磚顆數佔本塊 cap 的比例
  P5（--models）現成模型：中心投到平面落在海面格、離平面 > 3 x 自己像素長度的不透明顆粒（上方／下方）的顆數與 Σ o·面積
GT 只用來量，不進輸出。
用法（lab，cwd＝repo 根）：python tools/make_sea_object.py data/matrix_city/aerial/train/block_all --blocks 1 3 12 --premise --masks [--models 名=<glob> ...]
"""
import argparse, glob, json, os, sys, time
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def wmedian(x, w):
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset"); ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--plane_dir", default="estimated_depths_plane"); ap.add_argument("--out", default="sea_object")
    ap.add_argument("--P", type=float, default=16.0); ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--min_yes", type=int, default=3); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--m_c", type=float, default=0.15); ap.add_argument("--down", type=float, default=1.2)
    ap.add_argument("--slope_max", type=int, default=1); ap.add_argument("--size_max", type=int, default=3)
    ap.add_argument("--cap_avg", type=float, default=1e6); ap.add_argument("--alloc", default="configs/cap_alloc_sfm_g44.json")
    ap.add_argument("--masks", action="store_true"); ap.add_argument("--premise", action="store_true"); ap.add_argument("--ncam", type=int, default=60)
    ap.add_argument("--models", nargs="*", default=[])
    # ★ 11-17 前提第一次：規格的反對票「可靠深度在平面上或後面」把 SfM 看得到的海面（深度就在平面上）也算反對（b12 有贊成票的格只 25% 成為海面格）
    #   => behind：只有「可靠深度在平面後面超過 margin」（穿過平面看到更低的東西）才反對；在平面 ±margin 內的可靠像素算贊成（有 SfM 的海）
    #   behind 量出來會把岸邊陸地算成海（陸地比海高約 4 m，300～500 m 外 1% 深度就是 3～5 m，分不開）
    #   => abstain：在平面 ±margin 內的可靠像素棄權（不贊成也不反對），只有在平面後面才反對；贊成票仍只來自影像定義的海面區
    ap.add_argument("--no_rule", choices=["spec", "behind", "abstain"], default="abstain")
    # ★ 11-17 規格 6 v2 A（使用者「A 跟 B 都作」）：S1 的殘留集中在港口碼頭之間的水面——那裡沒拿到「是海」的票（不是影像定義的海面區），
    #   所以沒有錨定層也沒有海面上限。--grow：從海面格往相連的格長，要同時滿足
    #   ① 像海的影像證據：打到該格、不在平面前面也不在後面（可靠深度沒有或在平面上）的像素，5x5 RGB 標準差 < col_std 且顏色和
    #      同一張影像的海面像素中位差 < col_tol（逐張比，避開逐幀曝光差）=> 「像海」票；像海票 >= min_yes 且多於（不像海＋前面有東西＋後面）票；
    #   ② SfM 沒有結構：格內沒有可靠 SfM 點高出平面 > g/2（碼頭、路面、船）；③ 和既有海面格 8 連通（廣度優先，長到不能再長）。
    #   只用影像、校正深度與 SfM；門檻是像素／RGB／格距的倍數，沒有場景公尺。
    ap.add_argument("--max_tilt", type=float, default=10.0, help="共識平面和水平面夾角超過這個度數 => 不是海面、整塊跳過")
    ap.add_argument("--grow", action="store_true"); ap.add_argument("--col_std", type=float, default=6.0); ap.add_argument("--col_tol", type=float, default=20.0)   # 11-17 使用者同意改用 abstain（前提三種規則對照後）
    a = ap.parse_args()
    import torch
    from internal.utils.colmap import read_cameras_binary, read_images_binary, qvec2rotmat
    D = a.dataset; PD = os.path.join(D, a.plane_dir); OUT = os.path.join(D, a.out); os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    parts = torch.load(os.path.join(D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    ids = parts["partition_coordinates"]["id"].numpy(); xys = parts["partition_coordinates"]["xy"].double().numpy()
    size = parts["scene_config"]["partition_size"].double().numpy()
    inner = np.median(np.minimum(size[:, 0], size[:, 1])); ref = np.median(size[(size[:, 0] < 10 * inner) & (size[:, 1] < 10 * inner)], axis=0)
    mc = a.m_c * ref.min()
    cams = read_cameras_binary(os.path.join(D, "sparse/0/cameras.bin")); imgs = read_images_binary(os.path.join(D, "sparse/0/images.bin"))
    by_name = {v.name: v for v in imgs.values()}
    W8 = json.load(open(a.alloc))["weights"]
    M = None
    if a.premise:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe")); _argv = sys.argv; sys.argv = sys.argv[:1]
        import monodepth_sea_1102 as M
        sys.argv = _argv
    for b in a.blocks:
        t0 = time.time()
        k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0]); lo, hi = xys[k] - mc, xys[k] + size[k] + mc
        names = [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip() and l.strip() in by_name]
        CAM = {}
        for nm in names:
            im = by_name[nm]; cm = cams[im.camera_id]
            fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
            cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
            CAM[nm] = (qvec2rotmat(im.qvec), np.asarray(im.tvec, dtype=np.float64), fx, fy, cx, cy, cm.width, cm.height)
        FXT = next(iter(CAM.values()))[2] / a.down
        # ── 各區世界平面（P1）──
        R_ = []
        for nm in names:
            f = os.path.join(PD, nm + "_planes.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f); s = z["kind"] == 2
            if not s.any():
                continue
            R, t = CAM[nm][0], CAM[nm][1]
            for n_c, c, npx, rs, cu, cv in zip(z["n"][s], z["c"][s], z["npix"][s], z["resid"][s], z["cu"][s], z["cv"][s]):
                n_w = n_c @ R; c_w = c - n_c @ t
                if n_w[2] < 0:
                    n_w, c_w = -n_w, -c_w
                R_.append((nm, n_w, c_w, npx, rs, cu, cv, n_c, c, tuple(z["size"])))
        if len(R_) < 3:
            print(f"b{b}：海面區只有 {len(R_)} 個（缺 _planes.npz？先跑 make_plane_depth.py）=> 跳過", flush=True); continue
        Nw = np.stack([r[1] for r in R_]); Cw = np.array([r[2] for r in R_]); Wt = np.array([r[3] for r in R_], np.float64); Rs = np.array([r[4] for r in R_])
        n0 = np.array([wmedian(Nw[:, i], Wt) for i in range(3)]); n0 /= np.linalg.norm(n0)
        ang = np.degrees(np.arccos(np.clip(np.abs(Nw @ n0), 0, 1))); c0 = wmedian(Cw, Wt)
        madA = wmedian(np.abs(ang - wmedian(ang, Wt)), Wt) + 1e-9; madC = wmedian(np.abs(Cw - c0), Wt) + 1e-12
        inl = (ang <= wmedian(ang, Wt) + 3 * 1.4826 * madA) & (np.abs(Cw - c0) <= 3 * 1.4826 * madC)
        n1 = (Nw[inl] * Wt[inl, None]).sum(0); n1 /= np.linalg.norm(n1); c1 = float((Cw[inl] * Wt[inl]).sum() / Wt[inl].sum())
        # 各區重心像素：自己的平面深度 vs 共識平面深度
        rel = []
        for (nm, n_w, c_w, npx, rs, cu, cv, n_c, c, sz) in R_:
            R, t, fx, fy, cx, cy, Wc, Hc = CAM[nm]; sx, sy = sz[0] / Wc, sz[1] / Hc
            d = np.array([(cu + 0.5) / sx - cx, (cv + 0.5) / sy - cy, 0.0]); d = np.array([d[0] / fx, d[1] / fy, 1.0])
            z_own = c / (n_c @ d); nc1 = R @ n1; cc1 = c1 + nc1 @ t; z_con = cc1 / (nc1 @ d)
            rel.append(z_con / z_own - 1.0)
        rel = np.abs(np.array(rel))
        # 截距分群：法向相近的區，c_w 排序後找大間隙
        cs = np.sort(Cw[ang < 5]); gaps = np.diff(cs) if len(cs) > 1 else np.zeros(0)
        ngrp = 1 + int((gaps > 10 * (np.median(np.abs(cs - np.median(cs))) + 1e-12)).sum()) if len(cs) > 3 else 1
        print(f"b{b}｜海面區 {len(R_)}（收 {int(inl.sum())}）｜共識平面法向與世界 z 夾角 {np.degrees(np.arccos(abs(n1[2]))):.2f}°、截距 {c1:.5f}｜"
              f"P1：各區重心對共識的相對深度差中位 {100 * np.median(rel):.2f}%（p90 {100 * np.percentile(rel, 90):.2f}%）vs 各區自己的殘差中位 {100 * np.median(Rs):.2f}%｜"
              f"法向夾角中位 {np.median(ang):.2f}°｜截距分群 {ngrp} 群", flush=True)
        # ★ 11-18 全場景：沒有海的塊也可能有幾個被誤判的「海面區」，擬出歪掉的平面（b6：與世界 z 夾 31°、海面格 0）=> 跳過、不寫檔
        _tilt = float(np.degrees(np.arccos(min(1.0, abs(float(n1[2]))))))
        if _tilt > a.max_tilt:
            print(f"b{b}：共識平面和水平面夾 {_tilt:.1f}° > {a.max_tilt:g}° => 不是海面，跳過", flush=True); continue
        # ── 平面基底與裁切 ──
        e1 = np.cross(n1, [1.0, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.array([0, 1.0, 0]); e2 = np.cross(n1, e1)
        org = n1 * c1
        def plane_hits(nm, us, vs, sz):
            R, t, fx, fy, cx, cy, Wc, Hc = CAM[nm]; sx, sy = sz[0] / Wc, sz[1] / Hc
            d = np.stack([((us + 0.5) / sx - cx) / fx, ((vs + 0.5) / sy - cy) / fy, np.ones_like(us, float)], -1)
            nc = R @ n1; cc = c1 + nc @ t; den = d @ nc
            z = np.where(np.abs(den) > 1e-12, cc / np.where(np.abs(den) > 1e-12, den, 1), np.nan)
            ok = np.isfinite(z) & (z > 0)
            Xw = ((d * z[:, None]) - t) @ R
            return Xw, z, ok
        def inbox(X):
            xy = (X @ Rp.T)[:, :2]; return (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
        def cell_of(X, g):
            q = X - org; return np.floor(q @ e1 / g).astype(np.int64), np.floor(q @ e2 / g).astype(np.int64)
        def load_img(nm):
            kp = os.path.join(PD, nm + "_k.png"); ft = os.path.join(D, "estimated_depths_tri", nm + ".npy"); fr = os.path.join(D, "estimated_depths_tri_rel", nm + ".npy")
            if not (os.path.exists(kp) and os.path.exists(ft) and os.path.exists(fr)):
                return None
            kd = cv2.imread(kp, cv2.IMREAD_UNCHANGED); tri = np.load(ft).squeeze(); rl = np.load(fr)
            rel_ok = ((rl & 7) <= a.slope_max) & ((rl >> 3) <= a.size_max) & (tri > 1e-6)
            return kd, tri, rel_ok
        # 第一遍：海面像素最近可見深度 => 邊長
        dmin = {}
        g0 = 0.04
        for nm in names[::4]:
            L = load_img(nm)
            if L is None:
                continue
            kd, tri, rel_ok = L; H, Wd = kd.shape
            vv, uu = np.nonzero(kd[:: a.stride * 2, :: a.stride * 2] == 2); vv, uu = vv * a.stride * 2, uu * a.stride * 2
            if len(vv) == 0:
                continue
            X, z, ok = plane_hits(nm, uu.astype(float), vv.astype(float), (Wd, H)); ok &= inbox(X)
            i, j = cell_of(X[ok], g0)
            for key, zz in zip((i * 1000003 + j).tolist(), z[ok].tolist()):
                if zz < dmin.get(key, np.inf):
                    dmin[key] = zz
        if not dmin:
            print(f"b{b}：沒有海面像素落在裁切盒內 => 跳過", flush=True); continue
        g = a.P * float(np.median(list(dmin.values()))) / FXT
        # 第二遍：投票
        YES, NO, CK, CC = [], [], [], []
        LK, UL, FR, GK, GC = [], [], [], [], []           # --grow：像海／不像海／前面有東西 的票、像海像素的顏色
        for nm in names:
            L = load_img(nm)
            if L is None:
                continue
            kd, tri, rel_ok = L; H, Wd = kd.shape
            img = cv2.imread(os.path.join(D, "input", nm))
            img = cv2.resize(img, (Wd, H), interpolation=cv2.INTER_AREA)[..., ::-1] if img is not None else None
            vv, uu = np.mgrid[0:H:a.stride, 0:Wd:a.stride]; vv, uu = vv.ravel(), uu.ravel()
            X, z, ok = plane_hits(nm, uu.astype(float), vv.astype(float), (Wd, H)); ok &= inbox(X)
            ks = kd[vv, uu] == 2
            zt = np.where(tri[vv, uu] > 1e-6, 1.0 / np.maximum(tri[vv, uu], 1e-9), np.nan)
            if a.no_rule == "spec":
                no = rel_ok[vv, uu] & (zt >= z * (1 - a.margin)) & ~ks
            elif a.no_rule == "behind":
                no = rel_ok[vv, uu] & (zt > z * (1 + a.margin)) & ~ks
                ks = ks | (rel_ok[vv, uu] & (np.abs(zt / z - 1) <= a.margin))
            else:
                no = rel_ok[vv, uu] & (zt > z * (1 + a.margin)) & ~ks
            i, j = cell_of(X, g); key = i * 1000003 + j
            if a.grow and img is not None and (ok & ks).sum() >= 50:
                imf = img.astype(np.float32)
                mu = cv2.blur(imf, (5, 5)); sd_ = np.sqrt(np.maximum(cv2.blur(imf * imf, (5, 5)) - mu * mu, 0)).mean(-1)
                sea_med = np.median(imf[vv[ok & ks], uu[ok & ks]], 0)
                front = rel_ok[vv, uu] & (zt < z * (1 - a.margin))
                rest = ok & ~ks & ~front & ~no
                like = rest & (sd_[vv, uu] < a.col_std) & (np.abs(imf[vv, uu] - sea_med).mean(-1) < a.col_tol)
                lk_, li_ = np.unique(key[like], return_index=True); LK.append(lk_); UL.append(np.unique(key[rest & ~like])); FR.append(np.unique(key[ok & ~ks & front]))
                if len(lk_):
                    sl_ = np.nonzero(like)[0][li_]; GK.append(lk_); GC.append(img[vv[sl_], uu[sl_]])
            yk_, yi_ = np.unique(key[ok & ks], return_index=True); YES.append(yk_); NO.append(np.unique(key[ok & no]))
            if img is not None and len(yk_):
                sel_ = np.nonzero(ok & ks)[0][yi_]; CK.append(yk_); CC.append(img[vv[sel_], uu[sel_]])
        yk, yc = np.unique(np.concatenate(YES), return_counts=True); nk, nc_ = np.unique(np.concatenate(NO), return_counts=True)
        noc = np.zeros(len(yk), np.int64); pos = np.searchsorted(nk, yk); hit = (pos < len(nk)) & (nk[np.minimum(pos, len(nk) - 1)] == yk); noc[hit] = nc_[pos[hit]]
        sea = (yc >= a.min_yes) & (yc > noc); seak = yk[sea]
        n_grown = 0
        if a.grow and LK:
            cnt = lambda L_: (lambda u_c: dict(zip(u_c[0].tolist(), u_c[1].tolist())))(np.unique(np.concatenate(L_), return_counts=True)) if L_ else {}
            cL, cU, cF = cnt(LK), cnt(UL), cnt(FR); cN = dict(zip(nk.tolist(), nc_.tolist()))
            if "SFM_REL" not in globals():
                from internal.utils import colmap_fast
                _d = colmap_fast.points3D_arrays(os.path.join(D, "sparse/0/points3D.bin"))
                globals()["SFM_REL"] = np.ascontiguousarray(_d["xyz"][(np.diff(_d["off"]) >= 3) & (_d["err"] < 1.5)], dtype=np.float64)
            hs = SFM_REL @ n1 - c1; si, sj = cell_of(SFM_REL - np.outer(hs, n1), g)
            struct = set((si[hs > g / 2] * 1000003 + sj[hs > g / 2]).tolist())
            ok_cell = {k_ for k_, v_ in cL.items() if v_ >= a.min_yes and v_ > cU.get(k_, 0) + cF.get(k_, 0) + cN.get(k_, 0) and k_ not in struct}
            cur = set(seak.tolist()); front_ = list(cur)
            while front_:
                nxt = []
                for k_ in front_:
                    ii_ = int(round(k_ / 1000003)); jj_ = k_ - ii_ * 1000003
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            q_ = (ii_ + di) * 1000003 + (jj_ + dj)
                            if q_ in ok_cell and q_ not in cur:
                                cur.add(q_); nxt.append(q_)
                front_ = nxt
            grown = np.array(sorted(cur - set(seak.tolist())), dtype=np.int64); n_grown = len(grown)
            print(f"b{b}｜--grow：像海票格 {len(cL):,}、符合條件 {len(ok_cell):,}（SfM 結構格 {len(struct):,} 排除）｜和海面格相連而長進來 {n_grown:,} 格"
                  f"（原海面格 {len(seak):,} => {len(seak) + n_grown:,}）", flush=True)
            if n_grown:
                seak = np.sort(np.concatenate([seak, grown]))
                yc = np.concatenate([yc[sea], np.array([cL[k_] for k_ in grown.tolist()], np.int64)]); noc = np.concatenate([noc[sea], np.array([cN.get(k_, 0) for k_ in grown.tolist()], np.int64)])
                o_ = np.argsort(np.concatenate([yk[sea], grown])); yc, noc = yc[o_], noc[o_]
                sea = np.ones(len(seak), bool); yk = seak
                CK += GK; CC += GC
        if len(seak) == 0:
            print(f"b{b}：海面格 0 => 跳過、不寫檔", flush=True); continue
        cap = a.cap_avg * float(W8[str(b)])
        # 鋪磚：每個海面格的格心一顆；顏色＝贊成票影像在該格的顏色中位
        ti = np.round(seak / 1000003).astype(np.int64); tj = seak - ti * 1000003
        tiles = org + np.outer((ti + 0.5) * g, e1) + np.outer((tj + 0.5) * g, e2)
        ckk = np.concatenate(CK) if CK else np.zeros(0, np.int64); ccc = np.concatenate(CC) if CC else np.zeros((0, 3), np.uint8)
        inn = np.isin(ckk, seak); ckk, ccc = ckk[inn], ccc[inn]
        od = np.argsort(ckk, kind="stable"); ckk, ccc = ckk[od], ccc[od]
        uk_, st_ = np.unique(ckk, return_index=True); col = np.full((len(seak), 3), 128, np.uint8)
        if len(uk_):
            meds = np.stack([np.array([np.median(x) for x in np.split(ccc[:, ch], st_[1:])]) for ch in range(3)], 1)
            col[np.searchsorted(seak, uk_)] = np.clip(np.round(meds), 0, 255).astype(np.uint8)
        np.savez(os.path.join(OUT, f"block_{b}.npz"), n=n1, c=c1, origin=org, e1=e1, e2=e2, g=g, keys=seak, base=1000003, yes=yc[sea], no=noc[sea],
                 tiles=tiles, colors=col)
        print(f"b{b}｜鋪磚邊長 {g * 100:.2f} m（P {a.P:.0f} px x 最近可見深度中位 {np.median(list(dmin.values())) * 100:.0f} m）｜"
              f"有贊成票的格 {len(yk):,}、海面格 {sea.sum():,}｜P4：鋪磚 {sea.sum():,} 顆＝cap {cap:,.0f} 的 {100 * sea.sum() / cap:.1f}%｜{time.time() - t0:.0f} s", flush=True)
        seaset = seak
        def is_sea_cell(i, j):
            kk = i * 1000003 + j; p_ = np.searchsorted(seaset, kk); return (p_ < len(seaset)) & (seaset[np.minimum(p_, len(seaset) - 1)] == kk)
        # ★ 規格 6 §4：每塊一個目錄 cap_b<b>/：<影像名>.npy（共識平面逆深度，float16，寬或窄遮罩處）、_k.png（寬遮罩＝2，給 depth_cap_plane_*）、
        #   _n.png（窄遮罩＝該張自己的海面區 ∧ 共識平面有交點，給 sea_front_*）
        if a.masks:
            from PIL import Image
            CD = os.path.join(OUT, f"cap_b{b}"); os.makedirs(CD, exist_ok=True)
            for nm in names:
                L = load_img(nm)
                if L is None:
                    continue
                kd, tri, rel_ok = L; H, Wd = kd.shape
                vv, uu = np.mgrid[0:H, 0:Wd]; vv, uu = vv.ravel(), uu.ravel()
                X, z, ok = plane_hits(nm, uu.astype(float), vv.astype(float), (Wd, H))
                i, j = cell_of(X, g); wide = ok & is_sea_cell(i, j)
                zt = np.where(tri[vv, uu] > 1e-6, 1.0 / np.maximum(tri[vv, uu], 1e-9), np.inf)
                wide &= ~(rel_ok[vv, uu] & (zt < z * (1 - a.margin)))
                narrow = ok & (kd[vv, uu] == 2)
                inv = np.where(wide | narrow, 1.0 / np.where(ok, z, 1.0), 0.0).astype(np.float16)
                np.save(os.path.join(CD, nm + ".npy"), inv.reshape(H, Wd))
                Image.fromarray((wide.reshape(H, Wd) * 2).astype(np.uint8)).save(os.path.join(CD, nm + "_k.png"))
                Image.fromarray((narrow.reshape(H, Wd) * 255).astype(np.uint8)).save(os.path.join(CD, nm + "_n.png"))
        # P2／P3（GT）
        if a.premise and M is not None:
            rng = np.random.default_rng(0); samp = [names[i] for i in sorted(rng.choice(len(names), min(a.ncam, len(names)), replace=False))]
            e_con, e_own, cov_n, cov_d, nfront, nnotsea, nmask = [], [], 0, 0, 0, 0, 0
            cov_nb, cov_db, cov_sb = 0, 0, 0                 # 只算裁切盒內的 GT 海面像素（規格 §7：盒外的遠海不歸這份規格）
            for nm in samp:
                gt = M.load_gt(nm); L = load_img(nm)
                if gt is None or L is None:
                    continue
                kd, tri, rel_ok = L; H, Wd = kd.shape
                zg = cv2.resize((gt * M.K_UNIT).astype(np.float32), (Wd, H), interpolation=cv2.INTER_NEAREST).astype(np.float64)
                vv, uu = np.mgrid[0:H, 0:Wd]; vv, uu = vv.ravel(), uu.ravel()
                X, z, ok = plane_hits(nm, uu.astype(float), vv.astype(float), (Wd, H))
                R, t, fx, fy, cx, cy, Wc, Hc = CAM[nm]; sx, sy = Wd / Wc, H / Hc
                dd = np.stack([((uu + 0.5) / sx - cx) / fx, ((vv + 0.5) / sy - cy) / fy, np.ones_like(uu, float)], -1)
                zgv = zg.ravel(); Xg = ((dd * zgv[:, None]) - t) @ R
                gsea = np.isfinite(zgv) & (zgv > 0) & (np.abs(Xg[:, 2] - M.Z_SEA) < 0.01)
                pinv = np.load(os.path.join(PD, nm + ".npy")).astype(np.float64).ravel()
                own = np.where(pinv > 1e-6, 1.0 / np.maximum(pinv, 1e-9), np.nan)
                s2 = gsea & (kd.ravel() == 2) & ok & np.isfinite(own)
                e_con.append(np.abs(z[s2] / zgv[s2] - 1)); e_own.append(np.abs(own[s2] / zgv[s2] - 1))
                i, j = cell_of(X, g); m = ok & is_sea_cell(i, j)
                zt = np.where(tri.ravel() > 1e-6, 1.0 / np.maximum(tri.ravel(), 1e-9), np.inf); m &= ~(rel_ok.ravel() & (zt < z * (1 - a.margin)))
                cov_n += int((m & gsea).sum()); cov_d += int(gsea.sum())
                gin = gsea & inbox(Xg); cov_nb += int((m & gin).sum()); cov_db += int(gin.sum()); cov_sb += int((gin & (kd.ravel() == 2)).sum())
                ns = m & ~gsea & np.isfinite(zgv) & (zgv > 0); nmask += int(m.sum())
                nfront += int((ns & (zgv < z * (1 - a.margin))).sum()); nnotsea += int((ns & ~(zgv < z * (1 - a.margin))).sum())
            ec = np.concatenate(e_con) if e_con else np.zeros(1); eo = np.concatenate(e_own) if e_own else np.zeros(1)
            print(f"b{b}｜P2（GT 海面像素、單張區內 {len(ec):,} px）：共識平面 中位 {100 * np.median(ec):.2f}% p90 {100 * np.percentile(ec, 90):.2f}%｜單張平面 中位 {100 * np.median(eo):.2f}% p90 {100 * np.percentile(eo, 90):.2f}%", flush=True)
            print(f"b{b}｜P3（{a.no_rule}）：反投影遮罩涵蓋 GT 海面像素 {100 * cov_n / max(cov_d, 1):.1f}%｜只算裁切盒內：遮罩 {100 * cov_nb / max(cov_db, 1):.1f}% vs 單張海面區 {100 * cov_sb / max(cov_db, 1):.1f}%｜遮罩內不是海：前面有東西擋 {100 * nfront / max(nmask, 1):.2f}%、真的不是海 {100 * nnotsea / max(nmask, 1):.2f}%", flush=True)
        # P5：現成模型
        if a.models:
            FX0 = FXT
            def px_len(Xq):
                best = np.full(len(Xq), np.inf)
                for R, t, fx, fy, cx, cy, Wc, Hc in CAM.values():
                    Xc = Xq @ R.T + t; zz = Xc[:, 2]
                    u_ = fx * Xc[:, 0] / np.maximum(zz, 1e-9) + cx; v_ = fy * Xc[:, 1] / np.maximum(zz, 1e-9) + cy
                    okk = (zz > 1e-6) & (u_ >= 0) & (u_ < Wc) & (v_ >= 0) & (v_ < Hc); best = np.where(okk & (zz < best), zz, best)
                return best / FX0
            for spec in a.models:
                lab, pat = spec.split("=", 1); fs_ = sorted(x for x in glob.glob(pat.replace("{B}", str(b))) if "xyz" not in x)
                if not fs_:
                    print(f"b{b}｜P5 {lab}：沒有模型（{pat}）", flush=True); continue
                f = fs_[-1]
                sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]; gp = "gaussian_model.gaussians."
                m_ = sd[gp + "means"].double().numpy(); o_ = torch.sigmoid(sd[gp + "opacities"].float()).reshape(-1).numpy(); s_ = torch.exp(sd[gp + "scales"].float()).numpy()[:, :2]; del sd
                kk = (o_ > 0.5) & inbox(m_); m_, o_, s_ = m_[kk], o_[kk], s_[kk]
                dist = m_ @ n1 - c1; i, j = cell_of(m_ - np.outer(dist, n1), g); onsea = is_sea_cell(i, j)
                cand = onsea & (np.abs(dist) > 0)
                pl = np.full(len(m_), np.inf); pl[cand] = px_len(m_[cand])
                off = cand & (np.abs(dist) > 3 * pl); area = o_ * 2 * np.pi * s_[:, 0] * s_[:, 1] * 1e4
                up, dn = off & (dist > 0), off & (dist < 0)
                huge = 3 * s_.max(1) > 8 * g
                print(f"b{b}｜P5 {lab}：海面格上方離平面 > 3 px 的不透明顆粒 {up.sum():,}（Σo·面積 {area[up].sum() / 1e3:,.0f}k m²）｜下方 {dn.sum():,}（{area[dn].sum() / 1e3:,.0f}k m²）"
                      f"｜海面格上、在平面附近 {int((onsea & ~off).sum()):,}｜巨大面元（3σ > 8 x 鋪磚邊長）海面格內 {int((huge & onsea).sum()):,}、格外 {int((huge & ~onsea).sum()):,}", flush=True)
        json.dump(dict(block=b, g=g, n=n1.tolist(), c=c1, sea_cells=int(sea.sum()), regions=len(R_)), open(os.path.join(OUT, f"stats_block_{b}.json"), "w"), indent=1)
    print("DONE make_sea_object", flush=True)


if __name__ == "__main__":
    main()
