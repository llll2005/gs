"""零成本的逐塊顆數權重：塊內可靠 SfM 點數（2026-11-05，使用者「自適應 cap 要用通用算法、不能靠既有結果」）。

訓練前就有：分區（partitions.pt）＋ SfM（sparse/0/points3D.bin）。可靠＝track >= 3 且重投影誤差 < 1.5（同 SfM 錨點）。
  w_b = 塊 b 盒內可靠 SfM 點數 / 16 塊平均        => 每塊 cap＝平均 cap x w_b（Σcap 不變；格式同 configs/cap_alloc_g44.json）
11-05 在 MatrixCity 4x4 與 10-31 價格分配（要 2.5M 全場景探針）的相關：Pearson +0.69／Spearman +0.68（scratchpad cap_predictors_1105.py）。
它回答「這塊東西多不多」，不回答「這塊多給顆粒會不會更好」=> 當定價自決容量（density price_*）的起點與對照。

用法：python tools/sfm_cap_alloc.py <dataset> [--part partition/partitions-dim_4_4_visibility_0.08] [--out configs/cap_alloc_sfm_g44.json]
全場景：MASKPLAN_ALLOC=configs/cap_alloc_sfm_g44.json bash scripts/lab/task_fullscene.sh maskplan <STEPS> <平均 cap>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--max-err", type=float, default=1.5)
    ap.add_argument("--out", default="configs/cap_alloc_sfm_g44.json")
    a = ap.parse_args()
    from internal.utils.colmap import read_points3D_binary
    P = torch.load(os.path.join(a.dataset, a.part, "partitions.pt"), map_location="cpu")
    assert not P["scene_config"]["contract"], "contract 模式未支援"
    R = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    ids = P["partition_coordinates"]["id"].numpy()
    xys = P["partition_coordinates"]["xy"].double().numpy()
    size = P["scene_config"]["partition_size"].double().numpy()
    nx = int(P["scene_config"]["block_dim"][0])
    pts = read_points3D_binary(os.path.join(a.dataset, "sparse/0/points3D.bin"))
    xyz = np.stack([p.xyz for p in pts.values()])
    rel = np.array([len(p.image_ids) >= a.min_track and p.error < a.max_err for p in pts.values()])
    q = xyz[rel] @ R.T
    cnt = {}
    for k in range(len(ids)):
        b = int(ids[k][0] + nx * ids[k][1])        # 與 task_cmp.sh 的分區檔名 <BLK % nx>_<BLK / nx> 同一個編號
        lo, hi = xys[k], xys[k] + size[k]
        cnt[b] = int(((q[:, 0] >= lo[0]) & (q[:, 0] < hi[0]) & (q[:, 1] >= lo[1]) & (q[:, 1] < hi[1])).sum())
    m = float(np.mean(list(cnt.values())))
    w = {str(b): round(cnt[b] / m, 6) for b in sorted(cnt)}
    out = {"說明": "逐塊顆數權重＝塊內可靠 SfM 點數 / 平均（tools/sfm_cap_alloc.py；零成本、訓練前就有）。每塊 cap = 平均 cap x weight。",
           "來源": f"{a.dataset} {a.part}；可靠＝track >= {a.min_track}、誤差 < {a.max_err}",
           "sfm_counts": {str(b): cnt[b] for b in sorted(cnt)}, "weights": w}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    for b in sorted(cnt):
        print(f"block {b:>2}：可靠 SfM {cnt[b]:>8,}  權重 {w[str(b)]:.3f}")
    print(f"=> {a.out}")


if __name__ == "__main__":
    main()
