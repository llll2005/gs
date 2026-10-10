"""塊外顆粒的成本：單塊訓練時，合併會丟掉的那批顆粒佔了多少時間／VRAM／渲染成本（2026-10-10 使用者：「排 塊外成本量測」）

背景：4x4 全場景 16 塊共訓練 37.4M 顆，合併只留 17.6M（47%）；best0 b6 只留 27%。
      塊外顆粒不是純浪費：它們解釋訓練視角裡的塊外像素（相機分配規則：SfM 觀測點有 > content_threshold 在塊內就收，
      所以很多訓練視角大半畫面在塊外）；但它們和塊內內容用同一個價格（同 SH3、同佔 cap）。
本工具回答「如果塊外變便宜，最多能省多少」—— 只量成本，不量品質：
  變體   全部顆粒 ／ 只留分區盒內（與 merge 同規則）／ 盒內＋margin（get_bounding_boxes(enlarge=m)，m＝盒邊長的比例）
  量     N；常駐 VRAM 估計（參數＋梯度＋Adam = 16 B/float）；
         訓練視角（均勻抽 --ncam 台）上的 forward／forward＋backward 時間（CUDA event，每台取中位、再平均）；
         精確 Σtiles（光柵器回傳的 tiles，＝每步 binning 工作量）；forward＋backward 的峰值配置增量；
         盒內顆粒單獨渲染 alpha>0.5 的像素比例（= 訓練畫面裡「塊內內容」佔多少）
⚠ 只拿掉塊外顆粒不是可行的訓練法（塊外像素沒人解釋，梯度會推壞塊內）——這是「塊外成本降到 0」的上界。

用法：python tools/outbox_cost.py --ckpt <block.ckpt> [<block2.ckpt> ...] [--margins 0 0.1] [--ncam 24]
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def cuda_ms(fn, repeat=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--ncam", type=int, default=24)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
    dev = torch.device("cuda")

    for ck_path in a.ckpt:
        model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            ck_path, device=dev, eval_mode=False, pre_activate=False)
        ck = torch.load(ck_path, map_location="cpu"); dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; del ck
        train = pc.instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(ck_path)), global_rank=0).get_outputs().train_set
        parts = torch.load(os.path.join(os.path.dirname(pc.image_list), "partitions.pt"))
        coords = PartitionCoordinates(id=parts["partition_coordinates"]["id"], xy=parts["partition_coordinates"]["xy"])
        orig = {k: v.detach() for k, v in model.properties.items()}
        if getattr(model, "_outside_pool", None) is not None:      # 塊外池跑次：本工具量的是「可訓練集合裡的塊外顆粒」，池先卸下並另報
            print(f"（此 ckpt 有塊外池 {model._outside_pool.n:,} 顆，{model._outside_pool.bytes_per_point()} B／顆；以下只量可訓練集合）")
            model._outside_pool = None
        N0 = orig["means"].shape[0]
        x = orig["means"].cpu() @ parts["extra_data"]["rotation_transform"][:3, :3].T
        if parts["scene_config"]["contract"]:
            x = CityGSPartitioning.contract_to_unisphere(x[:, :2], parts["scene_config"]["aabb"], ord=torch.inf)
        masks = {"全部": torch.ones(N0, dtype=torch.bool)}
        for m in a.margins:
            boxes = coords.get_bounding_boxes(parts["scene_config"]["partition_size"], enlarge=m)
            masks[f"盒內＋{m:.2f}" if m > 0 else "盒內（merge 規則）"] = CityGSPartitioning.is_in_bounding_boxes(
                bounding_boxes=boxes, coordinates=x[:, :2])[pc.block_id]
        C = len(train)
        idx = np.linspace(0, C - 1, min(a.ncam, C)).round().astype(int)
        cams = [train.cameras[int(i)].to_device(dev) for i in idx]
        bg = torch.zeros(3, device=dev)
        print(f"\n######## {ck_path}\nblock {pc.block_id}（{list(pc.block_dim or [])}）N={N0:,}  訓練相機 {C}（抽 {len(cams)} 台）")

        # 訓練畫面裡塊內內容佔多少：盒內顆粒單獨渲染 alpha > 0.5
        with torch.no_grad():
            m0 = masks["盒內（merge 規則）"].to(dev)
            model.properties = {k: torch.nn.Parameter(v[m0].clone(), requires_grad=False) for k, v in orig.items()}
            cov = [float((renderer(c, model, bg_color=bg)["rend_alpha"] > 0.5).float().mean()) for c in cams]
        print(f"訓練視角上盒內顆粒蓋住的畫面（alpha>0.5）：平均 {100 * np.mean(cov):.1f}%（中位 {100 * np.median(cov):.1f}%，最少 {100 * np.min(cov):.1f}%）")

        rows = []
        for vn, mk in masks.items():
            mk = mk.to(dev)
            model.properties = {k: torch.nn.Parameter(v[mk].clone()) for k, v in orig.items()}
            torch.cuda.empty_cache()
            n = int(mk.sum()); nfl = sum(p[0].numel() for p in model.properties.values()) * n
            fw, fb, tl, pk = [], [], [], []
            for c in cams:
                def f():
                    with torch.no_grad():
                        return renderer(c, model, bg_color=bg)
                o = f()
                t = o.get("tiles")
                tl.append(float(t.double().sum()) if t is not None else float("nan"))
                gt = torch.rand_like(o["render"])

                def fbk():
                    oo = renderer(c, model, bg_color=bg)
                    torch.abs(oo["render"] - gt).mean().backward()
                    for p in model.properties.values():
                        p.grad = None
                fw.append(cuda_ms(f, a.repeat))
                fb.append(cuda_ms(fbk, a.repeat))
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b0 = torch.cuda.memory_allocated()
                fbk(); torch.cuda.synchronize(); pk.append((torch.cuda.max_memory_allocated() - b0) / 2 ** 30)
            rows.append((vn, n, nfl * 16 / 2 ** 30, np.mean(fw), np.mean(fb), np.median(tl) / 1e6, np.mean(tl) / 1e6, np.max(pk)))
        model.properties = {k: torch.nn.Parameter(v.clone()) for k, v in orig.items()}
        print(f"{'變體':<14} {'N':>10} {'常駐GB':>7} {'fwd ms':>7} {'fwd+bwd ms':>10} {'Σtiles中位M':>11} {'Σtiles均M':>9} {'fwd+bwd峰值增量GB':>16}")
        r0 = rows[0]
        for r in rows:
            print(f"{r[0]:<14} {r[1]:>10,} {r[2]:>7.2f} {r[3]:>7.1f} {r[4]:>10.1f} {r[5]:>11.3f} {r[6]:>9.3f} {r[7]:>16.2f}")
        for r in rows[1:]:
            print(f"  {r[0]} 相對全部：N {100 * (r[1] / r0[1] - 1):+.0f}%  常駐 {100 * (r[2] / r0[2] - 1):+.0f}%  fwd {100 * (r[3] / r0[3] - 1):+.0f}%  "
                  f"fwd+bwd {100 * (r[4] / r0[4] - 1):+.0f}%  Σtiles 中位 {100 * (r[5] / r0[5] - 1):+.0f}%  峰值增量 {100 * (r[7] / r0[7] - 1):+.0f}%")
        del model, renderer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
