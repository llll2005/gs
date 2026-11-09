"""11-09 現行配方全場景 27.490 vs A＋dist4 SfM 分配 27.822（−0.33 dB）：掉在哪些幀、哪些塊？
讀兩個全場景 `tools/eval_official_test.py --save_dir` 存的 741 張「GT｜渲染」並排圖，逐幀算 PSNR（8 位元 PNG，跟原始浮點值差一點，但兩邊一樣），
依兩種方式分組：MatrixCity 來源塊（transforms_test.json 的 file_path）、我們的 4x4 分塊（相機中心落在哪塊的 AABB，落在外面算最近的塊中心）。
另外把每幀切成 4x4 個子區，看差距集中在畫面哪裡（上方＝遠處／天際線、下方＝近處地面）。純 CPU、只讀。
用法（cwd＝repo 根）：python tools/probe/fullscene_frame_diff_1109.py <基準跑次目錄> <比較跑次目錄> [--test_dir ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def frame_psnr(path, grid=4):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    w = a.shape[1] // 2
    gt, out = a[:, :w], a[:, w:2 * w]
    se = (gt - out) ** 2
    full = float(-10 * np.log10(max(se.mean(), 1e-12)))
    H, W = se.shape[:2]
    sub = np.zeros((grid, grid))
    for r in range(grid):
        for c in range(grid):
            m = se[r * H // grid:(r + 1) * H // grid, c * W // grid:(c + 1) * W // grid].mean()
            sub[r, c] = -10 * np.log10(max(m, 1e-12))
    return full, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cmp")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--train_dir", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--pose_json", default="data/matrix_city/aerial/pose/block_all/transforms_test.json")
    a = ap.parse_args()
    from internal.utils.colmap import read_images_binary
    from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

    fa = sorted(glob.glob(os.path.join(a.base, "test_official", "*_gt_vs_render.png")))
    names = [os.path.basename(f).split("_gt_vs_render")[0] for f in fa]
    fb = [os.path.join(a.cmp, "test_official", f"{n}_gt_vs_render.png") for n in names]
    assert all(os.path.exists(f) for f in fb), "兩邊的幀對不起來"
    print(f"幀數 {len(names)}｜基準 {a.base}｜比較 {a.cmp}", flush=True)

    src = {f"{i:04d}": fr["file_path"].split("/")[0] for i, fr in enumerate(json.load(open(a.pose_json))["frames"])}
    imgs = read_images_binary(os.path.join(a.test_dir, "sparse", "0", "images.bin"))
    cen = {}
    for im in imgs.values():
        q = im.qvec; w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        cen[im.name.rsplit(".", 1)[0]] = -R.T @ im.tvec
    p = torch.load(f"{a.train_dir}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
    pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
    bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
    Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
    C = torch.tensor(np.stack([cen[n] for n in names])).double()
    xy = (C @ Rt.T)[:, :2]
    inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=xy).numpy()   # [塊, 幀]
    mid = ((bb.min + bb.max) / 2).double() if hasattr(bb, "min") else None
    blk = np.full(len(names), -1)
    for i in range(len(names)):
        hit = np.where(inb[:, i])[0]
        if len(hit):
            blk[i] = hit[0]
        elif mid is not None:
            blk[i] = int(torch.argmin(((mid - xy[i]) ** 2).sum(1)))

    pa = np.zeros(len(names)); pb = np.zeros(len(names)); sa = np.zeros((len(names), 4, 4)); sb = np.zeros_like(sa)
    for i, (x, y) in enumerate(zip(fa, fb)):
        pa[i], sa[i] = frame_psnr(x); pb[i], sb[i] = frame_psnr(y)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}", flush=True)
    d = pb - pa
    print(f"\n全部：基準 {pa.mean():.3f}｜比較 {pb.mean():.3f}｜Δ {d.mean():+.3f}（中位 {np.median(d):+.3f}）"
          f"｜變差 > 0.5 dB 的幀 {np.mean(d < -0.5):.1%}、> 1 dB {np.mean(d < -1):.1%}｜變好 > 0.5 dB {np.mean(d > 0.5):.1%}")
    print("Δ 分位（p5／p25／p50／p75／p95）：" + "／".join(f"{np.quantile(d, q):+.2f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95)))
    print("\n依 MatrixCity 來源塊：幀數｜基準｜比較｜Δ｜Δ 中位")
    for s in sorted(set(src[n] for n in names), key=lambda t: int(t.split("_")[1])):
        k = np.array([src[n] == s for n in names])
        print(f"  {s:14} {k.sum():4d}｜{pa[k].mean():6.2f}｜{pb[k].mean():6.2f}｜{d[k].mean():+.3f}｜{np.median(d[k]):+.3f}")
    print("\n依 4x4 分塊（相機中心；塊外算最近的塊）：幀數｜基準｜比較｜Δ｜Δ 中位｜Δ 對總差的貢獻")
    for b in range(inb.shape[0]):
        k = blk == b
        if k.sum() == 0:
            continue
        print(f"  b{b:<3} {k.sum():4d}｜{pa[k].mean():6.2f}｜{pb[k].mean():6.2f}｜{d[k].mean():+.3f}｜{np.median(d[k]):+.3f}｜{d[k].sum() / len(d):+.3f}")
    ds = (sb - sa).mean(0)
    print("\n畫面 4x4 子區的平均 Δ（列＝上到下）：")
    for r in range(4):
        print("  " + "  ".join(f"{ds[r, c]:+.2f}" for c in range(4)))
    o = np.argsort(d)
    print("\n最差 15 幀：名稱｜來源塊｜4x4 塊｜基準｜比較｜Δ")
    for i in o[:15]:
        print(f"  {names[i]}｜{src[names[i]]}｜b{blk[i]}｜{pa[i]:.2f}｜{pb[i]:.2f}｜{d[i]:+.2f}")
    print("最好 5 幀：")
    for i in o[::-1][:5]:
        print(f"  {names[i]}｜{src[names[i]]}｜b{blk[i]}｜{pa[i]:.2f}｜{pb[i]:.2f}｜{d[i]:+.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
