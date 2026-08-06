"""起始 trim（sep_depth_trim_2dgs_renderer.py:185-218）到底刪掉了什麼。

量三件事：被砍點的離表面距離、被砍錨點的 track 長度、被砍點是否在存活點「後方」。
⚠ 厚殼假說最後被 PCA 厚度 0.81 推翻（見 紀錄/研究總覽.md §7.1）；本工具的「後方 69.1%」
是拿相距 6.86 格距的點在比，那距離下表面起伏就能解釋。
"""
import argparse
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from floater_label import surface_distance

DATA = "data/matrix_city/aerial/train/block_all"


def block_cameras(data, block, block_dim):
    names, cams = load_test_cameras(data, 1.2)
    by, bx = block // block_dim[0], block % block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        data, "partition", f"partitions-dim_{block_dim[0]}_{block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    idx = [i for i, n in enumerate(names) if n in want]
    return np.array([cams[i].camera_center.numpy() for i in idx], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_ply", required=True)
    ap.add_argument("--ckpt", required=True, help="earliest post-trim checkpoint (step 499)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--match_tol", type=float, default=0.5,
                    help="a seed counts as surviving if a primitive is within this many grid "
                         "spacings of it. 499 steps at means_lr 2e-4 move points far less than "
                         "one spacing, so the match is unambiguous")
    a = ap.parse_args()

    v = PlyData.read(a.seed_ply)["vertex"]
    seed = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)
    wpath = a.seed_ply + ".w.npy"
    w = np.load(wpath) if os.path.exists(wpath) else np.zeros(len(seed), np.float32)
    tpath = a.seed_ply + ".track.npy"
    track = np.load(tpath) if os.path.exists(tpath) else np.zeros(len(seed), np.int32)
    anchor = w > 0.5

    st = cKDTree(seed)
    rng = np.random.default_rng(0)
    sub = seed[rng.choice(len(seed), min(200_000, len(seed)), replace=False)]
    grid = float(np.median(st.query(sub, k=2)[0][:, 1]))

    cur = torch.load(a.ckpt, map_location="cpu", weights_only=False)[
        "state_dict"]["gaussian_model.gaussians.means"].float().numpy().astype(np.float64)
    d, _ = cKDTree(cur).query(seed)
    alive = d < a.match_tol * grid
    dead = ~alive

    print(f"[種子] {a.seed_ply}\n  N={len(seed):,}  錨點 {int(anchor.sum()):,}  格距 {grid:.5f}")
    print(f"[ckpt] {os.path.basename(a.ckpt)}  N={len(cur):,}")
    print(f"\n[存活] 整體 {100 * alive.mean():.1f}%   "
          f"錨點 {100 * alive[anchor].mean():.1f}%   填充 {100 * alive[~anchor].mean():.1f}%")
    print(f"[被砍] {int(dead.sum()):,} 顆（{100 * dead.mean():.1f}%）")

    # ── 判準 1：被砍的點離真實表面比較遠嗎？（若是 ⇒ trim 剪對了）
    sd, unit = surface_distance(seed, a.data)
    print(f"\n[判準1 離表面] 單位={unit:.5f}")
    print(f"  存活 p50={np.median(sd[alive]) / unit:6.1f}x    被砍 p50={np.median(sd[dead]) / unit:6.1f}x")
    if anchor.any():
        print(f"  錨點：存活 {int((alive & anchor).sum()):,} / 被砍 {int((dead & anchor).sum()):,}"
              f"    被砍錨點的 track 中位 = {int(np.median(track[dead & anchor])) if (dead & anchor).any() else 0}"
              f"（存活錨點 {int(np.median(track[alive & anchor])) if (alive & anchor).any() else 0}）")

    # ── 判準 2：被砍的點在存活點「後面」嗎？（厚殼假說）
    cc = block_cameras(a.data, a.block, a.block_dim)
    ctree = cKDTree(cc)
    at = cKDTree(seed[alive])
    idx_dead = np.where(dead)[0]
    samp = idx_dead[rng.choice(len(idx_dead), min(200_000, len(idx_dead)), replace=False)]
    nn_d, nn_i = at.query(seed[samp])
    partner = seed[alive][nn_i]
    _, ci = ctree.query(seed[samp])
    cam = cc[ci]
    r_dead = np.linalg.norm(seed[samp] - cam, axis=1)
    r_alive = np.linalg.norm(partner - cam, axis=1)
    behind = r_dead > r_alive
    print(f"\n[判準2 厚殼] 取樣 {len(samp):,} 個被砍點，比對其最近的存活點（相對最近相機）")
    print(f"  被砍點在存活點『後方』的比例 = {100 * behind.mean():.1f}%   （隨機應為 50%）")
    print(f"  兩者的徑向距離差 中位 = {np.median(np.abs(r_dead - r_alive)):.4f}"
          f"  （深度誤差尺度 0.0795、體素 0.03）")
    print(f"  被砍點到最近存活點的距離 p50 = {np.median(nn_d) / grid:.2f} 格距")

    verdict = "厚殼假說成立 ⇒ 初始點雲是多層互相矛盾的表面估計，trim 只留最前層" \
        if behind.mean() > 0.65 else \
        ("被砍點不在後方 ⇒ 厚殼假說證偽，另尋原因" if behind.mean() < 0.55 else "訊號不明確")
    print(f"\n[判定] {verdict}")


if __name__ == "__main__":
    main()
