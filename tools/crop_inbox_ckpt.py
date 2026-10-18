#!/usr/bin/env python
"""把單塊跑次的 ckpt 裁成「塊內」（使用者 10-18：「把 g44s15k_best1sh1 切成塊內」）。

盒內判定與 `utils/merge_citygs_ckpts.py` 完全相同（同一個 partitions.pt、rotation_transform[:3,:3] 後取 xy、enlarge 0）
=> 裁出來的就是合併時這塊會貢獻的那一批顆粒。
所有第一維 = N 的張量（參數、Adam exp_avg／exp_avg_sq、逐顆狀態）一起裁（`prune_resume_ckpt.prune_tree`；
只裁參數不裁動量會靜默壞掉）。原檔不動，輸出到 `<跑次>_inbox/blocks/block_<B>/`（小檔一併複製）。

用法：python tools/crop_inbox_ckpt.py outputs/lab/g44s15k_best1sh1 [--blocks 6 12] [--enlarge 0]
"""
import argparse
import glob
import os
import re
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def inbox_mask(means, parser_cfg, enlarge):
    from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
    parts = torch.load(os.path.join(os.path.dirname(parser_cfg.image_list), "partitions.pt"))
    coords = PartitionCoordinates(id=parts["partition_coordinates"]["id"], xy=parts["partition_coordinates"]["xy"])
    boxes = coords.get_bounding_boxes(parts["scene_config"]["partition_size"], enlarge=enlarge)
    x = means.float() @ parts["extra_data"]["rotation_transform"][:3, :3].T
    if parts["scene_config"]["contract"]:
        x = CityGSPartitioning.contract_to_unisphere(x[:, :2], parts["scene_config"]["aabb"], ord=torch.inf)
    return CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=x[:, :2])[parser_cfg.block_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--blocks", type=int, nargs="*", default=None)
    ap.add_argument("--enlarge", type=float, default=0.0, help="同 merge_citygs_ckpts.py --enlarge（預設 0）")
    ap.add_argument("--suffix", default="_inbox")
    a = ap.parse_args()
    from prune_resume_ckpt import prune_tree
    run = a.run.rstrip("/")
    out_run = run + a.suffix
    for bd in sorted(glob.glob(os.path.join(run, "blocks", "block_*"))):
        b = int(bd.rsplit("_", 1)[1])
        if a.blocks and b not in a.blocks:
            continue
        cks = glob.glob(os.path.join(bd, "checkpoints", "*.ckpt"))
        if not cks:
            print(f"block {b}：沒有 ckpt，略過"); continue
        ck_path = max(cks, key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
        ck = torch.load(ck_path, map_location="cpu")
        means = ck["state_dict"]["gaussian_model.gaussians.means"]
        N = means.shape[0]
        m = inbox_mask(means, ck["datamodule_hyper_parameters"]["parser"], a.enlarge)
        new, cnt = prune_tree(ck, m, N)
        od = os.path.join(out_run, "blocks", f"block_{b}")
        os.makedirs(os.path.join(od, "checkpoints"), exist_ok=True)
        for f in os.listdir(bd):          # 小檔（cfg_args、cameras.json、appearance_group_ids…）照抄，方便 viewer／評測沿用
            p = os.path.join(bd, f)
            if os.path.isfile(p) and os.path.getsize(p) < 50 * 2 ** 20:
                shutil.copy2(p, os.path.join(od, f))
        op = os.path.join(od, "checkpoints", os.path.basename(ck_path))
        torch.save(new, op)
        print(f"block {b}：{os.path.basename(ck_path)}  盒內 {int(m.sum()):,}／{N:,}（{100 * float(m.float().mean()):.1f}%）"
              f"  裁了 {cnt} 個逐顆張量  {os.path.getsize(ck_path) / 2**20:.0f} MiB -> {os.path.getsize(op) / 2**20:.0f} MiB  => {op}", flush=True)


if __name__ == "__main__":
    main()
