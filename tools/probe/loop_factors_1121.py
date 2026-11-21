#!/usr/bin/env python
"""★ 11-21 光柵器「迴圈數」由什麼決定（使用者「問題出在迴圈數，那迴圈跟什麼有關」；純量測、不訓練）。

用 audit 模式（forward.cu auditCUDA：幾何／遮擋邏輯與 renderCUDA 相同）拆 forward 的工作量：
  配對數（binning 的 (tile, 顆粒)）＝ Σ_i 顆粒 i 覆蓋的 tile 數 —— 由顆數與螢幕上的大小決定
  逐像素迴圈（每個像素從清單頭走到自己飽和 T < 1e-4 或清單尾）—— 由「要疊幾層才蓋滿」決定（不透明度、空洞）
  warp 迴圈（16x2 像素一起走，最慢的像素決定）—— 由同一條像素條裡像素的差異決定
  真正混合（alpha >= 1/255 且未飽和）—— 其餘是 alpha < 1/255 或幾何不相交、白走的
用法：python tools/probe/loop_factors_1121.py [--ckpt ...] [--nview 24]
"""
import argparse, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt")
ap.add_argument("--nview", type=int, default=24)
a = ap.parse_args()

import torch
import diff_trim_surfel_rasterization as R
from internal.utils.gaussian_model_loader import GaussianModelLoader
assert "audit_tiles" in R.GaussianRasterizationSettings._fields
dev = torch.device("cuda")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
pp = ck["datamodule_hyper_parameters"]["parser"]
if getattr(pp, "points_from", None) == "ply":
    pp.points_from = "random"
cams = pp.instantiate(path=ck["datamodule_hyper_parameters"]["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                      global_rank=0).get_outputs().train_set.cameras
del ck
idx = np.linspace(0, len(cams) - 1, a.nview).round().astype(int).tolist()
bg = torch.zeros(3, device=dev)
N = model.n_gaussians
S = dict(pairs=0, useful=0, reached=0, proc=0.0, blend=0.0, warp=0.0, tilemax=0.0, npix=0, ntile=0,
         proc_unsat=0.0, pix_unsat=0, top1=0.0, top10=0.0, vis=0)
per_view = []
with torch.no_grad():
    for i in idx:
        cam = cams[i].to_device(dev)
        o = renderer(cam, model, bg_color=bg)
        alpha = o["rend_alpha"][0] if "rend_alpha" in o else None
        reached, useful, radii, others, tiles = renderer(cam, model, bg_color=bg, _audit=True)
        n_proc, n_blend = others[0], others[2]
        H, W = n_proc.shape
        t = tiles.long()
        vis = t > 0
        tv = t[vis].double().sort(descending=True).values
        k1, k10 = max(1, int(0.01 * tv.numel())), max(1, int(0.10 * tv.numel()))
        S["top1"] += float(tv[:k1].sum()); S["top10"] += float(tv[:k10].sum()); S["vis"] += int(vis.sum())
        S["pairs"] += int(t.sum()); S["useful"] += int(useful.long().sum()); S["reached"] += float(reached.double().sum())
        S["proc"] += float(n_proc.double().sum()); S["blend"] += float(n_blend.double().sum())
        # warp＝16x2 像素條：走到條裡最慢的像素為止；tile（16x16）同理（一批 256 個配對要等整個 tile）
        Hp, Wp = math.ceil(H / 16) * 16, math.ceil(W / 16) * 16
        P = torch.zeros(Hp, Wp, device=dev, dtype=torch.float64); P[:H, :W] = n_proc.double()
        S["warp"] += float(P.view(Hp // 2, 2, Wp // 16, 16).amax(dim=(1, 3)).sum()) * 32
        for (wy, wx) in ((4, 8), (8, 4), (1, 32)):   # 同樣 32 個像素、不同形狀的 warp（只換執行緒對應像素的方式）
            S[f"warp{wx}x{wy}"] = S.get(f"warp{wx}x{wy}", 0.0) + float(P.view(Hp // wy, wy, Wp // wx, wx).amax(dim=(1, 3)).sum()) * 32
        S["tilemax"] += float(P.view(Hp // 16, 16, Wp // 16, 16).amax(dim=(1, 3)).sum()) * 256
        S["npix"] += H * W; S["ntile"] += (Hp // 16) * (Wp // 16)
        if True:
            un = n_proc >= others[1] - 0.5          # 走到清單尾＝沒有在中途飽和（out_others[1]＝該 tile 的配對數）
            S["proc_unsat"] += float(n_proc.double()[un].sum()); S["pix_unsat"] += int(un.sum())
        per_view.append((int(t.sum()), float(n_proc.double().sum())))
        del o, reached, useful, radii, others, tiles

nv = len(idx)
print(f"模型 N={N:,}｜{nv} 個訓練視角 {W}x{H}｜每視角平均")
print(f"  在畫面內的顆粒 {S['vis'] / nv:,.0f}（{100 * S['vis'] / nv / N:.0f}%）")
print(f"  配對（binning 的 tile x 顆粒）{S['pairs'] / nv:,.0f}＝每顆平均覆蓋 {S['pairs'] / S['vis']:.1f} 個 tile；"
      f"最大的 1% 顆粒佔配對 {100 * S['top1'] / S['pairs']:.1f}%、最大的 10% 佔 {100 * S['top10'] / S['pairs']:.1f}%")
print(f"  每個 tile 的清單長度平均 {S['pairs'] / S['ntile']:,.0f}")
print(f"  配對裡：整個 tile 都沒有像素 alpha>=1/255 的（tile 精確剔除能刪的）{100 * (1 - S['useful'] / S['pairs']):.1f}%；"
      f"整個 tile 都飽和後才輪到的（遮擋尾巴）{100 * (1 - S['reached'] / S['pairs']):.1f}%")
print(f"  逐像素迴圈 {S['proc'] / S['npix']:.1f} 次／像素；真正混合 {S['blend'] / S['npix']:.1f} 次／像素（佔 {100 * S['blend'] / S['proc']:.1f}%）")
print(f"  warp（16x2）層級：lane 槽位 {S['warp'] / S['npix']:.1f} 次／像素 => 被同條最慢像素拖住的浪費 {100 * (1 - S['proc'] / S['warp']):.1f}%")
for k in ("warp8x4", "warp4x8", "warp32x1"):
    print(f"    若 warp 改成 {k[4:]}：{S[k] / S['npix']:.1f} 次／像素（浪費 {100 * (1 - S['proc'] / S[k]):.1f}%）")
print(f"  tile（16x16）層級：{S['tilemax'] / S['npix']:.1f} 次／像素（整個 tile 走到最慢像素為止的上界）")
if S["pix_unsat"]:
    print(f"  沒有中途飽和（走到清單尾）的像素 {100 * S['pix_unsat'] / S['npix']:.1f}%，佔逐像素迴圈 {100 * S['proc_unsat'] / S['proc']:.1f}%"
          f"（這些像素的迴圈數＝清單長度）")
pv = np.array(per_view)
r = np.corrcoef(pv[:, 0], pv[:, 1])[0, 1]
print(f"  各視角「配對數」與「逐像素迴圈總數」的相關 r = {r:.2f}")
