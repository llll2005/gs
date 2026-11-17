#!/usr/bin/env python
"""★ 11-18 全場景 b3 遠海變黑的診斷：test 幀上「新模型黑、新底 5 不黑」的像素，沿射線投到海平面，
看那些點在哪（離相機多遠、在不在 b3 分區盒）、b3 訓練影像怎麼看它們（看得到的張數、該像素的平面種類 kind、像素大小層級、校正深度可不可靠）。
目的：決定遠海鋪磚該用什麼票（v4 --far_levels 只收到 19 格 => 那些像素大多不是「海面區」kind 2）。CPU。
用法（lab）：python tools/probe/far_sea_black_1118.py --frames 0226 0269 --new_dir <新跑次>/test_official --old_dir <新底5>/test_official --block 3
"""
import argparse, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--frames", nargs="+", required=True)
ap.add_argument("--new_dir", required=True); ap.add_argument("--old_dir", required=True)
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--sea_level", type=float, default=-0.026)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--plane_dir", default="estimated_depths_plane")
ap.add_argument("--sea_obj", default="sea_object_v4")
ap.add_argument("--nsamp", type=int, default=4000)
a = ap.parse_args()
import torch
import eval_official_test as eot
from internal.utils.colmap import read_cameras_binary, read_images_binary, qvec2rotmat

names_t, cams_t = eot.load_test_cameras(a.test_dir, 1.2)
D = a.D; b = a.block
p = torch.load(os.path.join(D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
Rp = p["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = p["partition_coordinates"]["id"].numpy(); xys = p["partition_coordinates"]["xy"].double().numpy(); size = p["scene_config"]["partition_size"].double().numpy()
k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0]); lo, hi = xys[k], xys[k] + size[k]
def inbox(X):
    xy = (X @ Rp.T)[:, :2]; return (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
cams = read_cameras_binary(os.path.join(D, "sparse/0/cameras.bin")); imgs = read_images_binary(os.path.join(D, "sparse/0/images.bin"))
by_name = {v.name: v for v in imgs.values()}
tn = [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip() and l.strip() in by_name]
TC = []
for nm in tn:
    im = by_name[nm]; cm = cams[im.camera_id]; pr = cm.params
    fx, fy, cx, cy = (pr[0], pr[1], pr[2], pr[3]) if len(pr) > 3 else (pr[0], pr[0], pr[1], pr[2])
    TC.append((nm, qvec2rotmat(im.qvec), np.asarray(im.tvec, float), fx, fy, cx, cy, cm.width, cm.height))
so = None
f_so = os.path.join(D, a.sea_obj, f"block_{b}.npz")
if os.path.exists(f_so):
    so = np.load(f_so)
n = np.array([0, 0, 1.0]); c = a.sea_level
for fr in a.frames:
    i = [j for j, x in enumerate(names_t) if os.path.splitext(x)[0] == fr][0]
    A = cv2.imread(os.path.join(a.new_dir, fr + "_gt_vs_render.png"))[..., ::-1].astype(np.float32) / 255
    B_ = cv2.imread(os.path.join(a.old_dir, fr + "_gt_vs_render.png"))[..., ::-1].astype(np.float32) / 255
    W = A.shape[1] // 2; H = A.shape[0]
    rn, ro, gt = A[:, W:], B_[:, W:], A[:, :W]
    black = (rn.sum(-1) < 0.06) & (ro.sum(-1) > 0.15)
    vv, uu = np.nonzero(black)
    R = cams_t.R[i].numpy().astype(np.float64); T = cams_t.T[i].numpy().astype(np.float64)
    fx, fy, cx, cy = float(cams_t.fx[i]), float(cams_t.fy[i]), float(cams_t.cx[i]), float(cams_t.cy[i])
    # cams_t 是 world->camera（x_c = R x_w + T）的轉置慣例？用 eval 工具同一套：中心 = -R^T T（見 merged_heldout_by_block）
    C = -R.T @ T
    print(f"== {fr}｜{H}x{W}｜新黑舊不黑 {len(vv):,} px（{100 * len(vv) / (H * W):.1f}%）｜相機中心 {C.round(3)}｜高度 {C[2] - c:.3f}", flush=True)
    if len(vv) == 0:
        continue
    sel = np.random.default_rng(0).choice(len(vv), min(a.nsamp, len(vv)), replace=False); vv, uu = vv[sel], uu[sel]
    dcam = np.stack([(uu + 0.5 - cx) / fx, (vv + 0.5 - cy) / fy, np.ones(len(uu))], -1)
    dw = dcam @ R            # 相機 -> 世界方向（R 是 w2c 的旋轉 => d_w = R^T d_c；列向量寫法 d_c @ R）
    den = dw @ n; tt = (c - C @ n) / np.where(np.abs(den) > 1e-12, den, np.nan)
    hit = np.isfinite(tt) & (tt > 0)
    X = C[None] + tt[:, None] * dw
    print(f"   射線打到海平面 {100 * hit.mean():.1f}%（其餘朝上＝天空方向）", flush=True)
    if not hit.any():
        continue
    X = X[hit]; dist = np.linalg.norm(X - C[None], axis=1); ib = inbox(X)
    print(f"   打到點：離相機距離 中位 {np.median(dist):.2f}、p10 {np.percentile(dist, 10):.2f}、p90 {np.percentile(dist, 90):.2f}（場景單位）｜在 b{b} 分區盒內 {100 * ib.mean():.1f}%", flush=True)
    gt_col = gt[vv[hit], uu[hit]]
    print(f"   GT 顏色中位 {np.median(gt_col, 0).round(3)}、舊模型 {np.median(ro[vv[hit], uu[hit]], 0).round(3)}", flush=True)
    if so is not None:
        g = float(so["g"]); org = so["origin"]; e1 = so["e1"]; e2 = so["e2"]
        q = X - org; ii = np.floor(q @ e1 / g).astype(np.int64); jj = np.floor(q @ e2 / g).astype(np.int64)
        on0 = np.isin(ii * 1000003 + jj, so["keys"])
        onf = np.zeros(len(X), bool)
        for L in (so["far_L"].tolist() if "far_L" in so.files else []):
            gL = g * 2 ** L; i2 = np.floor(q @ e1 / gL).astype(np.int64); j2 = np.floor(q @ e2 / gL).astype(np.int64)
            onf |= np.isin(i2 * 1000003 + j2, so[f"far_keys_{L}"])
        onx = np.zeros(len(X), bool)
        if "tile_frozen" in so.files and so["tile_frozen"].any():      # 外推磚（--far_fill）：格心 ± 邊長/2 的正方形
            fz = so["tile_frozen"]; tc = so["tiles"][fz]; tg_ = so["tile_g"][fz]
            qu, qv = (X - org) @ e1, (X - org) @ e2; cu, cv = (tc - org) @ e1, (tc - org) @ e2
            for gg in np.unique(tg_):
                m_ = tg_ == gg; ki = set(zip(np.floor(cu[m_] / gg).astype(np.int64).tolist(), np.floor(cv[m_] / gg).astype(np.int64).tolist()))
                onx |= np.array([(a_, b_) in ki for a_, b_ in zip(np.floor(qu / gg).astype(np.int64).tolist(), np.floor(qv / gg).astype(np.int64).tolist())])
        print(f"   落在 {a.sea_obj} 海面格：第 0 層 {100 * on0.mean():.1f}%、遠海層 {100 * onf.mean():.1f}%、外推磚 {100 * onx.mean():.1f}%｜三者合計 {100 * (on0 | onf | onx).mean():.1f}%", flush=True)
    # b 訓練影像怎麼看這些點
    nvis = np.zeros(len(X), int); kinds = np.zeros((len(X), 4), int); relok = np.zeros(len(X), int); infront = np.zeros(len(X), int)
    for (nm, Rt, tt_, fx2, fy2, cx2, cy2, Wc, Hc) in TC:
        Xc = X @ Rt.T + tt_; z = Xc[:, 2]
        u = fx2 * Xc[:, 0] / np.maximum(z, 1e-9) + cx2; v = fy2 * Xc[:, 1] / np.maximum(z, 1e-9) + cy2
        ok = (z > 1e-6) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        if not ok.any():
            continue
        kp = os.path.join(D, a.plane_dir, nm + "_k.png")
        if not os.path.exists(kp):
            continue
        kd = cv2.imread(kp, cv2.IMREAD_UNCHANGED); h2, w2 = kd.shape
        ui = np.clip((u * w2 / Wc).astype(int), 0, w2 - 1); vi = np.clip((v * h2 / Hc).astype(int), 0, h2 - 1)
        nvis += ok
        kv = kd[vi, ui]
        for kk in range(4):
            kinds[:, kk] += ok & (kv == kk)
        ft = os.path.join(D, "estimated_depths_tri", nm + ".npy"); fr_ = os.path.join(D, "estimated_depths_tri_rel", nm + ".npy")
        if os.path.exists(ft) and os.path.exists(fr_):
            tri = np.load(ft).squeeze(); rl = np.load(fr_); h3, w3 = tri.shape
            u3 = np.clip((u * w3 / Wc).astype(int), 0, w3 - 1); v3 = np.clip((v * h3 / Hc).astype(int), 0, h3 - 1)
            r_ok = ((rl[v3, u3] & 7) <= 1) & ((rl[v3, u3] >> 3) <= 3) & (tri[v3, u3] > 1e-6)
            relok += ok & r_ok
            zt = np.where(tri[v3, u3] > 1e-6, 1 / np.maximum(tri[v3, u3], 1e-9), np.inf)
            infront += ok & r_ok & (zt < z * 0.99)
    tot = np.maximum(nvis, 1)
    print(f"   b{b} 訓練影像 {len(TC)} 張：每點看得到的張數 中位 {np.median(nvis):.0f}（0 張的點 {100 * (nvis == 0).mean():.1f}%）", flush=True)
    seen = nvis > 0
    if seen.any():
        print(f"   看得到時該像素平面種類（逐點比例的中位）：無 {np.median(kinds[seen, 0] / tot[seen]):.2f}、立面 {np.median(kinds[seen, 1] / tot[seen]):.2f}、海面 {np.median(kinds[seen, 2] / tot[seen]):.2f}"
              f"｜海面票 >= 3 的點 {100 * (kinds[seen, 2] >= 3).mean():.1f}%｜校正深度可靠 {np.median(relok[seen] / tot[seen]):.2f}、可靠且在平面前面 {np.median(infront[seen] / tot[seen]):.2f}", flush=True)
print("DONE far_sea_black", flush=True)
