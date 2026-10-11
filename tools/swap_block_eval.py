"""換塊評測：合併模型只換掉一塊，量「這一塊換配方」在合併後的真實影響（2026-10-10）。

背景：單塊 held-out 的畫面只有 11~15% 是塊內內容（mergeview），val⊂train 也是整張畫面 => 兩把尺都混了
「單塊模型帶了多少塊外內容」，而合併（utils/merge_citygs_ckpts.py）會把塊外顆粒全部丟掉。
b12 實例：我方單塊 −3.84 dB vs 官方，合併後 +0.73（紀錄/實驗分析/09 §0.2b）。

做法（只渲染）：
  基準   = 現成的 4x4 合併模型（預設 full44_best）原樣
  換塊後 = 基準去掉「落在第 B 塊分區盒內」的顆粒（就是當初由第 B 塊貢獻的那批），
           換成候選 ckpt 中落在同一個盒內的顆粒 —— 與 merge 完全相同的歸屬規則（rotation_transform → contract → is_in_bounding_boxes）
  => 換塊後的模型就是「其他 15 塊不變、只有第 B 塊換配方」的真實合併模型；鄰塊遮擋、塊外內容被丟掉，都和真正合併一樣。
評分：官方 741 幀 test，報「第 B 塊視角」（相機落在該塊訓練相機 AABB 內）與「全部 741 幀」的 PSNR／SSIM／LPIPS＋失敗 tile。
⚠ 候選必須是 4x4 格子（block_dim [4,4]）訓練的；5x5 的跑次沒有可換進去的合併模型。
⚠ 自我檢查：把基準自己的第 B 塊 ckpt 換回去（--sanity）應該與基準幾乎相同（只差盒邊界上的並列顆粒）。

用法：python tools/swap_block_eval.py --block 6 --cand <候選.ckpt> [<候選2.ckpt> ...] [--names a b] [--sanity] [--all]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def block_mask(xyz, parser_cfg):
    from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
    parts = torch.load(os.path.join(os.path.dirname(parser_cfg.image_list), "partitions.pt"))
    coords = PartitionCoordinates(id=parts["partition_coordinates"]["id"], xy=parts["partition_coordinates"]["xy"])
    boxes = coords.get_bounding_boxes(parts["scene_config"]["partition_size"], enlarge=0.)
    x = xyz.detach().cpu() @ parts["extra_data"]["rotation_transform"][:3, :3].T
    if parts["scene_config"]["contract"]:
        x = CityGSPartitioning.contract_to_unisphere(x[:, :2], parts["scene_config"]["aabb"], ord=torch.inf)
    return CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=x[:, :2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--cand", nargs="*", default=[])
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--merged_run", default="outputs/lab/full44_best")
    ap.add_argument("--sanity", action="store_true", help="另外把基準自己的第 B 塊 ckpt 換回去，應與基準幾乎相同")
    ap.add_argument("--all", action="store_true", help="另外評全部 741 幀（較慢）")
    ap.add_argument("--train_dir", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 4], help="候選的分區格子（本機測試可用 5 5）")
    ap.add_argument("--views", choices=["heldout", "train"], default="heldout", help="train＝第一個候選自己的訓練相機（本機功能測試用）")
    ap.add_argument("--max_views", type=int, default=100000)
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import eval_official_test as eot
    dev = torch.device("cuda")
    B = a.block

    mck = sorted(glob.glob(os.path.join(a.merged_run, "checkpoints", "*.ckpt")))
    mck = [c for c in mck if os.path.basename(c) == "merged.ckpt"] or mck
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(mck[-1], device=dev, eval_mode=True)
    base = {k: v.detach().cpu() for k, v in model.properties.items()}   # 17.6M 顆約 4 GB：放 CPU，每個變體組好才搬上 GPU（10-10 三個任務同卡時 OOM）
    model.properties = {k: v[:1].to(dev) for k, v in base.items()}; torch.cuda.empty_cache()
    print(f"[基準] {mck[-1]}  N={base['means'].shape[0]:,}")

    cands = list(a.cand)
    names = list(a.names or [os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c)))) for c in cands])
    assert a.cand or a.sanity, "至少給一個 --cand 或 --sanity"
    if a.sanity:
        sc = sorted(glob.glob(os.path.join(a.merged_run, "blocks", f"block_{B}", "checkpoints", "*step=60000.ckpt")))
        assert sc, f"找不到基準的 block {B} ckpt"
        cands.insert(0, sc[0]); names.insert(0, "自我檢查（基準自己的塊）")

    variants = [("基準（原樣）", None)]   # 換塊後的屬性在渲染時才組（每個 17M+ 顆約 4 GB，不同時常駐）
    mmask = None
    for c, nm in zip(cands, names):
        ck = torch.load(c, map_location="cpu"); pc = ck["datamodule_hyper_parameters"]["parser"]; del ck
        bd = list(pc.block_dim or [])
        assert bd == list(a.grid) and pc.block_id == B, f"{c}：block_dim {bd}／block_id {pc.block_id}，要 {list(a.grid)}／{B}"
        pc0 = pc
        if mmask is None:
            mmask = block_mask(base["means"], pc)[B]
            print(f"[基準] 第 {B} 塊盒內 {int(mmask.sum()):,} 顆（換掉這批）")
        cm, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(c, device=dev, eval_mode=True)
        cp = {k: v for k, v in cm.properties.items()}
        # SH 階數較低的候選（best0sh2／1／0）：shs_rest 補 0 到基準的長度 —— 高階係數為 0 與低階渲染完全等價（10-11）
        if "shs_rest" in cp and cp["shs_rest"].shape[1] < base["shs_rest"].shape[1]:
            r = cp["shs_rest"]
            cp["shs_rest"] = torch.cat([r, torch.zeros((r.shape[0], base["shs_rest"].shape[1] - r.shape[1], r.shape[2]), dtype=r.dtype, device=r.device)], 1)
            print(f"[{nm}] SH 較低：shs_rest {r.shape[1]} -> {cp['shs_rest'].shape[1]} 補 0")
        assert set(cp) == set(base) and all(cp[k].shape[1:] == base[k].shape[1:] for k in base), f"{nm}：屬性與基準不同（SH 階數？）"
        cmask = block_mask(cp["means"], pc)[B].to(cp["means"].device)
        variants.append((nm, {k: cp[k][cmask].detach().cpu().clone() for k in base}))
        print(f"[{nm}] 塊內 {int(cmask.sum()):,}／{cm.n_gaussians:,} 顆 => 換塊後 N={int((~mmask).sum()) + int(cmask.sum()):,}")
        del cm, cp

    if a.views == "heldout":
        names_t, cams = eot.load_test_cameras(a.test_dir, 1.2)
        gt_path = lambda i: os.path.join(a.test_dir, "images_1.2", names_t[i])
        centres = np.array([(-cams.R[i].numpy().T @ cams.T[i].numpy()) for i in range(len(names_t))])
        lo, hi, _ = eot.block_bounds(a.train_dir, B, list(a.grid))
        selB = [i for i in range(len(names_t)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)][:a.max_views]
    else:
        ck = torch.load(cands[0], map_location="cpu"); dmh = ck["datamodule_hyper_parameters"]; del ck
        if getattr(dmh["parser"], "points_from", None) == "ply":
            dmh["parser"].points_from = "random"
        ts = dmh["parser"].instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
        names_t, cams = list(ts.image_names), ts.cameras
        gt_path = lambda i: ts.image_paths[i]
        selB = np.linspace(0, len(names_t) - 1, min(a.max_views, len(names_t))).round().astype(int).tolist()
    sets = [(f"第 {B} 塊視角（{len(selB)} 幀）", selB)]
    if a.all:
        sets.append((f"全部 {len(names_t)} 幀", list(range(len(names_t)))))
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)
    bg = torch.zeros(3, device=dev)
    res = {}
    # 10-11 使用者：本塊區域再依「離分區接縫的距離」分組（tile 中心用基準合併模型的深度反推世界座標；函式沿用 tools/seam_failure.py，已自檢 z 深度）
    #   => 換配方（例如塊外池去重）若只在接縫附近傷害、離接縫遠的地方不變，可以直接看出來
    import seam_failure as sf
    _parts = torch.load(os.path.join(os.path.dirname(pc0.image_list), "partitions.pt"))
    _Rp = _parts["extra_data"]["rotation_transform"][:3, :3].float().to(dev)
    _mn, _mx, _int, _unit = (x.to(dev) for x in sf.seam_frame(_parts))
    SB = [0.0, 0.05, 0.1, 0.2, 0.3, 10.0]
    T = 48
    tdist = {}
    rbin = {}
    need = sorted(set(i for _, s in sets for i in s))
    # 「本塊負責區域」：基準第 B 塊的顆粒單獨渲染時 alpha > 0.5 的像素（合併後這些像素主要由第 B 塊決定）；
    #   整張畫面的 PSNR 會被其他 15 塊稀釋（塊視角上第 B 塊只佔畫面一部分），這個區域的 PSNR 才是第 B 塊換配方的直接效果。
    region = {}
    with torch.no_grad():
        model.properties = {k: v[mmask].to(dev) for k, v in base.items()}
        for i in need:
            o = renderer(cams[i].to_device(dev), model, bg_color=bg)
            region[i] = (o["rend_alpha"].reshape(o["render"].shape[1:]) > 0.5).cpu()
        frac = np.mean([float(region[i].float().mean()) for i in need])
        print(f"[區域] 第 {B} 塊負責的像素佔畫面平均 {100 * frac:.1f}%", flush=True)
        for vn, cin in variants:
            props = {k: (base[k] if cin is None else torch.cat([base[k][~mmask], cin[k]], 0)).to(dev) for k in base}
            model.properties = props
            per = {}
            for i in need:
                cam = cams[i].to_device(dev)
                oo = renderer(cam, model, bg_color=bg)
                out = oo["render"].clamp(0, 1)
                ny, nx = out.shape[1] // T, out.shape[2] // T
                if cin is None:          # 基準：算每個 tile 離接縫的距離（之後各變體共用）
                    z = oo["surf_depth"][0][:ny * T, :nx * T].reshape(ny, T, nx, T).permute(0, 2, 1, 3).reshape(ny, nx, -1).median(-1).values
                    vv, uu = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
                    u = (uu.float() + 0.5) * T; v = (vv.float() + 0.5) * T
                    pcm = torch.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], -1).reshape(-1, 3)
                    xw = (pcm - cams.T[i].to(dev).float()) @ cams.R[i].to(dev).float()
                    dd = sf.seam_distance((xw @ _Rp.T)[:, :2], _mn, _mx, _int, _unit)
                    dd[z.reshape(-1) <= 0] = float("nan")
                    tdist[i] = torch.bucketize(torch.nan_to_num(dd, nan=-1.0), torch.tensor(SB[1:-1], device=dev)).masked_fill(torch.isnan(dd), -1).cpu()
                pil = Image.open(gt_path(i)).convert("RGB")
                if pil.size != (out.shape[2], out.shape[1]):
                    pil = pil.resize((out.shape[2], out.shape[1]), Image.LANCZOS)
                gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
                ft = eot.fail_tiles(gt, out, 48, 0.10, 0.60)
                se = ((out - gt) ** 2).mean(0); m = region[i].to(dev)
                per[i] = (float(-10 * torch.log10(se.mean().clamp_min(1e-12))), float(ssim_fn(out, gt)),
                          float(lp(out.unsqueeze(0), gt.unsqueeze(0))), ft, float(se[m].sum()), int(m.sum()))
                if i in selB:
                    tse = (se * m)[:ny * T, :nx * T].reshape(ny, T, nx, T).sum((1, 3)).reshape(-1)
                    tcn = m[:ny * T, :nx * T].reshape(ny, T, nx, T).sum((1, 3)).reshape(-1).float()
                    bi = tdist[i].to(dev)
                    acc = rbin.setdefault(vn, np.zeros((len(SB) - 1, 2)))
                    for b in range(len(SB) - 1):
                        mb = bi == b
                        acc[b] += [float(tse[mb].sum()), float(tcn[mb].sum())]
            res[vn] = per
            model.properties = {k: v[:1] for k, v in props.items()}; del props; torch.cuda.empty_cache()
            print(f"  渲染完 {vn}（{len(need)} 幀）", flush=True)
    rps = lambda r, s: -10 * np.log10(max(sum(r[i][4] for i in s) / max(sum(r[i][5] for i in s), 1), 1e-12))
    for sn, s in sets:
        print(f"\n=== {sn} ===")
        print(f"{'':<26} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'失敗tile%':>9} {'ΔPSNR':>7} {'贏的幀':>8} {'本塊區域PSNR':>12} {'Δ區域':>7}")
        b0 = np.array([res["基準（原樣）"][i][0] for i in s]); rb = rps(res["基準（原樣）"], s)
        for vn, _ in variants:
            r = res[vn]
            p = np.array([r[i][0] for i in s]); t = np.array([list(r[i][3]) for i in s]).sum(0)
            print(f"{vn:<26} {p.mean():>8.3f} {np.mean([r[i][1] for i in s]):>7.4f} {np.mean([r[i][2] for i in s]):>7.4f} "
                  f"{100 * t[2] / max(t[0], 1):>9.2f} {p.mean() - b0.mean():>+7.3f} {int((p > b0 + 1e-6).sum()):>4}/{len(s)} "
                  f"{rps(r, s):>12.3f} {rps(r, s) - rb:>+7.3f}")
    # 本塊區域依離接縫距離分組（第 B 塊視角）
    labs = [f"{SB[b]:.2f}~{SB[b + 1]:.2f}" if SB[b + 1] < 5 else f">{SB[b]:.2f}" for b in range(len(SB) - 1)]
    b0 = rbin["基準（原樣）"]
    print(f"\n=== 本塊區域 PSNR 依離接縫距離（內部格距；0＝接縫上）===")
    print(f"{'':<26} " + " ".join(f"{l:>11}" for l in labs))
    print(f"{'（像素佔比）':<26} " + " ".join(f"{100 * b0[b, 1] / max(b0[:, 1].sum(), 1):>10.1f}%" for b in range(len(SB) - 1)))
    for vn, _ in variants:
        r = rbin[vn]
        cells = []
        for b in range(len(SB) - 1):
            ps = -10 * np.log10(max(r[b, 0] / max(r[b, 1], 1), 1e-12)); p0 = -10 * np.log10(max(b0[b, 0] / max(b0[b, 1], 1), 1e-12))
            cells.append(f"{ps:>6.2f}({ps - p0:+.2f})" if r[b, 1] > 0 else f"{'—':>11}")
        print(f"{vn:<26} " + " ".join(cells))


if __name__ == "__main__":
    main()
