"""★ 2026-11-15 本機換塊評測用（無損版）：把合併模型裁成「至少出現在一張第 B 塊 test 視角畫面裡」的顆粒，去掉優化器狀態。
為什麼：tools/swap_block_eval.py 只渲染第 B 塊視角（相機中心落在該塊訓練相機 AABB 內的 MatrixCity test 幀），
        不在任何一張畫面內的顆粒本來就畫不到 => 拿掉後「第 B 塊視角」的渲染逐像素不變（全 741 幀的 --all 就不適用）。
        11-15 裁「相鄰 8 塊」仍有 867 萬顆、上 GPU 3.2 GB，本機 6 GB 卡（桌面約 1.6 GB）渲染時 OOM。
判定：中心投影到畫面（含外擴邊界）內、在相機前方；外擴＝該顆 3 x 長軸投影成的像素半徑＋16 px（面元伸進畫面的部分也算）。
      同一個視角選法與相機（tools/eval_official_test.py 的 load_test_cameras(1.2)、block_bounds）。
用法：python tools/crop_merged_frustum.py --src outputs/lab/full44_best --dst outputs/local/full44_best_fr_b1 --block 1
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_official_test as eot

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--block", type=int, required=True)
ap.add_argument("--grid", type=int, nargs=2, default=[4, 4])
ap.add_argument("--neighbors_from", default=None, help="給一個 4x4 候選 ckpt（提供分區設定）=> 另外只留第 B 塊與相鄰塊（Chebyshev <= 1）的顆粒；遠處的塊會從畫面裡消失（各候選相同）")
ap.add_argument("--train_dir", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
a = ap.parse_args()
torch.set_num_threads(max(1, os.cpu_count() - 2))
names, cams = eot.load_test_cameras(a.test_dir, 1.2)
centres = np.array([(-cams.R[i].numpy().T @ cams.T[i].numpy()) for i in range(len(names))])
lo, hi, _ = eot.block_bounds(a.train_dir, a.block, list(a.grid))
sel = [i for i in range(len(names)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)]
ck = torch.load(os.path.join(a.src, "checkpoints", "merged.ckpt"), map_location="cpu", weights_only=False)
sd = ck["state_dict"]; pre = "gaussian_model.gaussians."
X = sd[pre + "means"].float(); n0 = X.shape[0]
rad = 3.0 * torch.exp(sd[pre + "scales"].float()).max(dim=1).values            # 世界單位的 3 sigma 半徑
keep = torch.zeros(n0, dtype=torch.bool)
for i in sel:
    R, T = cams.R[i].float(), cams.T[i].float()
    fx, fy, cx, cy = float(cams.fx[i]), float(cams.fy[i]), float(cams.cx[i]), float(cams.cy[i])
    W, H = float(cams.width[i]), float(cams.height[i])
    for c0 in range(0, n0, 4_000_000):
        Xc = X[c0:c0 + 4_000_000] @ R.T + T
        z = Xc[:, 2]
        front = z > 1e-3
        zz = z.clamp_min(1e-3)
        u = fx * Xc[:, 0] / zz + cx; v = fy * Xc[:, 1] / zz + cy
        m = fx * rad[c0:c0 + 4_000_000] / zz + 16.0
        keep[c0:c0 + 4_000_000] |= front & (u > -m) & (u < W + m) & (v > -m) & (v < H + m)
if a.neighbors_from:
    from swap_block_eval import block_mask
    pc = torch.load(a.neighbors_from, map_location="cpu", weights_only=False)["datamodule_hyper_parameters"]["parser"]
    ids = torch.load(os.path.join(os.path.dirname(pc.image_list), "partitions.pt"), weights_only=False)["partition_coordinates"]["id"]
    nb = [j for j in range(len(ids)) if int((ids[j] - ids[a.block]).abs().max()) <= 1]
    k_fr = int(keep.sum())
    keep &= block_mask(X, pc)[nb].any(0)
    print(f"只留第 {a.block} 塊與相鄰塊 {nb}：畫面內 {k_fr:,} -> {int(keep.sum()):,} 顆", flush=True)
for k in list(sd):
    if k.startswith(pre) and sd[k].shape[:1] == (n0,):
        sd[k] = sd[k][keep].clone()
out = {k: ck[k] for k in ck if k not in ("optimizer_states", "lr_schedulers", "loops", "callbacks")}
out["state_dict"] = sd
os.makedirs(os.path.join(a.dst, "checkpoints"), exist_ok=True)
p = os.path.join(a.dst, "checkpoints", "merged.ckpt")
torch.save(out, p)
print(f"第 {a.block} 塊 test 視角 {len(sel)} 幀｜{n0:,} -> {int(keep.sum()):,} 顆（{100 * keep.float().mean():.1f}%）｜{p} {os.path.getsize(p) / 2**30:.2f} GB", flush=True)
