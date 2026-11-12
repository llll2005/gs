"""tile 同步損失的上限量測（2026-11-05，使用者「整個 tile 同步前進能不能做成非同步」→「都排進去」；不訓練）。

光柵器 forward／backward 一個 tile（16x16＝256 條執行緒）一起走同一份配對清單：每 256 個配對同步兩次，
整塊要等最慢的像素飽和才結束；同一個 warp（2 列 x 16 像素＝32 條）本來就步調一致（SIMT）。
用稽核 kernel（forward.cu auditCUDA）讀每個像素「提早結束前實際跑的迴圈次數」n_proc，以它當工作量的代理：
  W_use   = sum_像素 n_proc                         真的要跑的
  W_warp  = sum_warp 32 x max_warp(n_proc)          每個 warp 各走各的（warp 層非同步）時要付的
  W_tile  = sum_tile 256 x max_tile(n_proc)         現在（整個 tile 一起走）要付的
  => 整塊同步損失 (W_tile - W_warp) / W_tile        ＝改成 warp 層非同步「最多」能省的部分（逐配對成本都算一樣）
     warp 內損失 (W_warp - W_use) / W_tile          ＝SIMT 本身的閒置，改 kernel 結構也拿不回來
⚠ 是上限：等最慢像素的執行緒閒著時，同一個 SM 上其他 tile 的 warp 會補上（佔用率），實際損失比這小；
  另外不同配對的成本不同（被跳過的 alpha<1/255 配對便宜、真的混合的貴）。
backward 的逐像素工作量是「最後貢獻者」以前的配對，與 n_proc 同級；這裡只用 n_proc。
--mask_dir：另外報「C 區像素不跑」（skip_unscored_pixels）之後的同一組數字。
★ 11-13 規格 5 §1：另印每視角 Load（Σ 顆粒蓋到的 tile 數）、每像素平均迴圈次數／混合次數，有 --mask_dir 時再分 A 塊內／B 塊外／C 接縫帶不計分（1／2／0）
   報像素比例、工作量（迴圈次數）佔比、混合次數佔比與每像素平均。

用法：PYTHONPATH=<有 audit_tiles 的光柵器> python tools/tile_sync_audit.py --ckpt <ckpt> [--nview 24] [--mask_dir <block_masks44/block_B>]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = 16


def _load_mask(mask_dir, name, h, w):
    from PIL import Image
    p = os.path.join(mask_dir, os.path.basename(str(name)))
    if not os.path.exists(p):
        return None
    t = torch.from_numpy(np.array(Image.open(p)))[None, None].float()
    return torch.nn.functional.interpolate(t, size=(h, w), mode="nearest")[0, 0].to(torch.uint8)


def sync_stats(n_proc):
    """n_proc [H, W] -> (W_use, W_warp, W_tile)；圖外的執行緒工作量 0（它們從一開始就 done）。"""
    H, W = n_proc.shape
    Ht, Wt = math.ceil(H / T), math.ceil(W / T)
    pad = torch.zeros((Ht * T, Wt * T), dtype=torch.float64, device=n_proc.device)
    pad[:H, :W] = n_proc.double()
    tiles = pad.reshape(Ht, T, Wt, T)                                   # [Ht, 16 列, Wt, 16 行]
    w_use = float(pad.sum())
    w_tile = float(tiles.amax(dim=(1, 3)).sum()) * T * T
    warps = pad.reshape(Ht, T // 2, 2, Wt, T)                           # warp＝tile 內連續 32 條執行緒＝2 列 x 16 行
    w_warp = float(warps.amax(dim=(2, 4)).sum()) * 32
    return w_use, w_warp, w_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nview", type=int, default=24)
    ap.add_argument("--mask_dir", default="")
    a = ap.parse_args()
    import diff_trim_surfel_rasterization as R
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    assert "audit_tiles" in R.GaussianRasterizationSettings._fields, f"光柵器沒有 audit_tiles：{R.__file__}"
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"
    ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                                   global_rank=0).get_outputs().train_set
    del ck
    idx = np.linspace(0, len(ts.cameras) - 1, a.nview).round().astype(int).tolist()
    bg = torch.zeros(3, device=dev)
    acc = {"all": np.zeros(3), "noC": np.zeros(3)}
    blend = proc = 0.0
    per_view = []
    loads, npix = [], 0
    reg = {r: np.zeros(3) for r in ("A", "B", "C")}          # [像素數, Σ迴圈次數, Σ混合次數]
    with torch.no_grad():
        for i in idx:
            cam = ts.cameras[i].to_device(dev)
            _r, _u, _radii, others, _tiles = renderer(cam, model, bg_color=bg, _audit=True)
            n_proc, n_blend = others[0], others[2]
            s = np.array(sync_stats(n_proc))
            acc["all"] += s
            per_view.append(s)
            proc += float(n_proc.double().sum()); blend += float(n_blend.double().sum())
            npix += n_proc.numel()
            if torch.is_tensor(_tiles):
                loads.append(float(_tiles.double().sum()))
            if a.mask_dir:
                m = _load_mask(a.mask_dir, ts.image_names[i], n_proc.shape[0], n_proc.shape[1])
                if m is not None:
                    acc["noC"] += np.array(sync_stats(n_proc * (m.to(dev) != 0)))
                    md = m.to(dev)
                    for r, v in (("A", 1), ("B", 2), ("C", 0)):
                        k = md == v
                        reg[r] += [float(k.sum()), float(n_proc[k].double().sum()), float(n_blend[k].double().sum())]
    print(f"N={model.n_gaussians:,}  視角 {len(idx)}  ckpt {a.ckpt}")
    print(f"逐像素迴圈裡真的有混合的比例 {blend / max(proc, 1):.3f}（其餘是 alpha<1/255 或沒交到）")
    print(f"每像素平均：迴圈次數 {proc / max(npix, 1):.1f}、混合次數 {blend / max(npix, 1):.1f}")
    if loads:
        print(f"每視角 Load（Σ 顆粒蓋到的 tile 數）：平均 {np.mean(loads) / 1e6:.2f}M、中位 {np.median(loads) / 1e6:.2f}M、p90 {np.quantile(loads, 0.9) / 1e6:.2f}M")
    _tp = sum(v[0] for v in reg.values())
    if _tp > 0:
        _tw = sum(v[1] for v in reg.values()); _tb = sum(v[2] for v in reg.values())
        print("A／B／C（A 塊內計分／B 塊外、目標＝隨機背景（學透明）／C 接縫帶不計分；skip_unscored_pixels 略過的是 C）：")
        for r, (n, w, b) in reg.items():
            print(f"  {r}：像素 {100 * n / _tp:5.1f}%｜迴圈次數佔 {100 * w / max(_tw, 1):5.1f}%｜混合次數佔 {100 * b / max(_tb, 1):5.1f}%"
                  f"｜每像素 迴圈 {w / max(n, 1):.1f}、混合 {b / max(n, 1):.1f}")
    for k, (u, w, t) in acc.items():
        if t <= 0:
            continue
        print(f"[{'全部像素' if k == 'all' else 'C 區不跑'}] 以現在（整塊一起走）的工作量為 1："
              f"真的要跑 {u / t:.3f}｜warp 內閒置（SIMT）{(w - u) / t:.3f}｜整塊同步損失 {(t - w) / t:.3f}")
    if acc["noC"][2] > 0:
        print(f"C 區不跑之後，現在的工作量變成 {acc['noC'][2] / acc['all'][2]:.3f} 倍"
              f"（warp 層非同步再疊上去：{acc['noC'][1] / acc['all'][2]:.3f} 倍）")
    pv = np.array(per_view)
    loss = (pv[:, 2] - pv[:, 1]) / np.maximum(pv[:, 2], 1)
    print(f"逐視角整塊同步損失：中位 {np.median(loss):.3f}、p10 {np.quantile(loss, 0.1):.3f}、p90 {np.quantile(loss, 0.9):.3f}")


if __name__ == "__main__":
    main()
