"""單塊 held-out 的落差，合併後還會在嗎？（2026-10-09 使用者問：bestvt 等 v/c 類判準 val 好、塊內 held-out 差，
是不是剪掉的是塊外內容、合併後就會改善）

合併（utils/merge_citygs_ckpts.py）只保留中心落在該塊 partition 範圍內的顆粒，其餘丟掉、由鄰塊補上。
本工具對同一塊的幾個 ckpt，用**與 merge 完全相同的規則**切出「合併後會留下的部分」，在塊內官方 held-out 視角上量：
  全模型          現行塊內 held-out（含塊外顆粒）
  塊內區域像素    只算「塊內顆粒蓋得到」的像素（所有 ckpt 的塊內部分 alpha > 0.5 的交集）—— 合併後這些像素主要由本塊負責
  塊外區域像素    其餘像素 —— 合併後由鄰塊負責，本塊在這裡的好壞會被換掉
判讀：兩個 ckpt 的落差若集中在「塊外區域」，合併後大致會消失；若在「塊內區域」，就是真的。

用法：python tools/merge_view_heldout.py --ckpt <A.ckpt> <B.ckpt> [--names 基準 bestvt]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
    import eval_official_test as eot
    dev = torch.device("cuda")
    names = a.names or [os.path.basename(c) for c in a.ckpt]

    def merge_mask(ckpt_path, xyz):
        ck = torch.load(ckpt_path, map_location="cpu")
        pc = ck["datamodule_hyper_parameters"]["parser"]
        parts = torch.load(os.path.join(os.path.dirname(pc.image_list), "partitions.pt"))
        coords = PartitionCoordinates(id=parts["partition_coordinates"]["id"], xy=parts["partition_coordinates"]["xy"])
        boxes = coords.get_bounding_boxes(parts["scene_config"]["partition_size"], enlarge=0.)
        x = xyz.cpu() @ parts["extra_data"]["rotation_transform"][:3, :3].T
        if parts["scene_config"]["contract"]:
            x = CityGSPartitioning.contract_to_unisphere(x[:, :2], parts["scene_config"]["aabb"], ord=torch.inf)
        m = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=x[:, :2])[pc.block_id]
        return m.to(xyz.device), pc

    models = []
    for c in a.ckpt:
        model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(c, device=dev, eval_mode=True)
        m, pc = merge_mask(c, model.get_xyz.detach())
        models.append((model, renderer, m, {k: v for k, v in model.properties.items()}))
        print(f"[{names[len(models) - 1]}] N={model.n_gaussians:,}  合併後保留（塊內）{int(m.sum()):,}（{100 * float(m.float().mean()):.1f}%）")
    blk, bdim = pc.block_id, list(pc.block_dim or [5, 5])
    ck = torch.load(a.ckpt[0], map_location="cpu"); train_dir = ck["datamodule_hyper_parameters"]["path"]; del ck
    tn, tc = eot.load_test_cameras(a.test_dir, 1.2)
    centres = np.array([(-tc.R[i].numpy().T @ tc.T[i].numpy()) for i in range(len(tn))])
    lo, hi, _ = eot.block_bounds(train_dir, blk, bdim, getattr(pc, "content_threshold", 0.08))
    sel = [i for i in range(len(tn)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)]
    print(f"block {blk}：塊內官方 held-out 視角 {len(sel)}")

    bg = torch.zeros(3, device=dev)
    acc = {n: {"full": [0.0, 0], "in": [0.0, 0], "out": [0.0, 0], "inonly": [0.0, 0], "psnr_full": []} for n in names}
    frac_in = []
    with torch.no_grad():
        for i in sel:
            cam = tc[i].to_device(dev)
            gt = None
            renders, masks = [], []
            for (model, renderer, m, orig) in models:
                o = renderer(cam, model, bg_color=bg)
                full = o["render"].clamp(0, 1)
                model.properties = {k: v[m] for k, v in orig.items()}
                oi = renderer(cam, model, bg_color=bg)
                model.properties = orig
                renders.append((full, oi["render"].clamp(0, 1)))
                masks.append(oi["rend_alpha"].reshape(oi["render"].shape[1:]) > 0.5)
                if gt is None:
                    pil = Image.open(os.path.join(a.test_dir, "images_1.2", tn[i])).convert("RGB")
                    gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
            mk = torch.stack(masks).all(0)
            frac_in.append(float(mk.float().mean()))
            for n, (full, inonly) in zip(names, renders):
                se = ((full - gt) ** 2).mean(0)
                acc[n]["full"][0] += float(se.sum()); acc[n]["full"][1] += se.numel()
                acc[n]["in"][0] += float(se[mk].sum()); acc[n]["in"][1] += int(mk.sum())
                acc[n]["out"][0] += float(se[~mk].sum()); acc[n]["out"][1] += int((~mk).sum())
                sei = ((inonly - gt) ** 2).mean(0)
                acc[n]["inonly"][0] += float(sei[mk].sum()); acc[n]["inonly"][1] += int(mk.sum())
                acc[n]["psnr_full"].append(float(-10 * torch.log10(se.mean().clamp_min(1e-12))))
    ps = lambda s, c: -10 * np.log10(max(s / max(c, 1), 1e-12))
    print(f"\n塊內區域像素佔畫面 平均 {100 * np.mean(frac_in):.1f}%（所有 ckpt 的塊內部分都蓋到的交集）")
    print(f"{'':<12} {'逐張平均 PSNR':>14} {'全像素（像素加權）':>18} {'塊內區域':>10} {'塊外區域':>10} {'只用塊內顆粒・塊內區域':>22}")
    for n in names:
        d = acc[n]
        print(f"{n:<12} {np.mean(d['psnr_full']):>14.3f} {ps(*d['full']):>18.3f} {ps(*d['in']):>10.3f} {ps(*d['out']):>10.3f} {ps(*d['inonly']):>22.3f}")
    if len(names) >= 2:
        b0 = acc[names[0]]
        for n in names[1:]:
            d = acc[n]
            print(f"  {n} − {names[0]}：全像素 {ps(*d['full']) - ps(*b0['full']):+.3f}  塊內區域 {ps(*d['in']) - ps(*b0['in']):+.3f}  "
                  f"塊外區域 {ps(*d['out']) - ps(*b0['out']):+.3f}  只用塊內顆粒 {ps(*d['inonly']) - ps(*b0['inonly']):+.3f}")


if __name__ == "__main__":
    main()
