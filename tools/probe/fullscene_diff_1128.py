"""11-28 使用者「先看問題出在哪」：兩個（以上）全場景合併模型在 MatrixCity test 741 幀上，掉分落在哪一塊、哪類物件。

做法（只渲染）：每幀先用基準模型渲染，取它的表面深度把每個像素反推回世界座標 =>
  塊＝落在哪個 4x4 分區盒（和合併歸屬同一套盒；盒外＝「外」）；
  類別＝基準的渲染法線＋世界高度（同 swap_block_eval.py --cat：水平且 z<=−0.02 海面、−0.02~0.03 地面、更高屋頂；|n_z|<0.3 立面；其餘其他）；
  只算基準 alpha>0.5 且深度有效的像素。其他模型在同一幀、同一批像素上算平方誤差。
輸出：各塊／各類別／（塊 x 類別）的 PSNR 與差；以及每格對整體 MSE 變化的貢獻（排序 => 掉分主要來自哪裡）。
整體數字是逐像素 MSE 合起來的 PSNR（不是逐幀 PSNR 平均），只拿來拆解。
用法（cwd＝gs）：python tools/probe/fullscene_diff_1128.py --models 最終版=<merged.ckpt> cap分段放=<merged.ckpt> [...] [--max_views 741]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "."); sys.path.insert(0, "tools")
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, help="名稱=合併 ckpt；第一個是基準")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
ap.add_argument("--partition", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
ap.add_argument("--max_views", type=int, default=100000)
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--zero_rest", action="store_true", help="★ 11-28 另加一個變體＝第一個模型拿掉視角項（SH 1～3 階歸零、只留 DC）=> 各類物件用掉多少視角相依")
ap.add_argument("--edge_q", type=float, default=-1.0, help="★ 11-28 >0：GT 亮度梯度前 (1-q) 的像素算邊緣，另報各模型在邊緣／非邊緣的誤差")
a = ap.parse_args()

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
import eval_official_test as eot

dev = torch.device("cuda")
CATS = ["海面", "地面", "屋頂", "立面", "其他"]
P = torch.load(a.partition, map_location="cpu")
assert not P["scene_config"]["contract"], "收縮空間的分區未實作"
coords = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
boxes = coords.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.)
Rp = P["extra_data"]["rotation_transform"][:3, :3].float()
NB = int(len(P["partition_coordinates"]["id"]))
print(f"分區 {a.partition}：{NB} 塊", flush=True)

models = []
for spec in a.models:
    nm, path = spec.split("=", 1)
    m, r, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(path, device=dev, eval_mode=True)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        if hasattr(r, f):
            setattr(r, f, False)
    models.append((nm, m, r))
    print(f"[{nm}] {path}  N={m.n_gaussians:,}", flush=True)
if a.zero_rest:
    nm0, path0 = a.models[0].split("=", 1)
    m2, r2, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(path0, device=dev, eval_mode=True)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        if hasattr(r2, f):
            setattr(r2, f, False)
    pr = dict(m2.properties)
    if "shs" in pr:
        sh = pr["shs"].clone(); sh[:, 1:, :] = 0; pr["shs"] = sh
    elif "shs_rest" in pr:
        pr["shs_rest"] = torch.zeros_like(pr["shs_rest"])
    m2.properties = pr
    models.append((nm0 + "只留DC", m2, r2))
    print(f"[{nm0}只留DC] 視角項歸零（SH 1～3 階 = 0）", flush=True)

names_t, cams = eot.load_test_cameras(a.test_dir, 1.2)
sel = list(range(min(a.max_views, len(names_t))))
bg = torch.zeros(3, device=dev)
NBX = NB + 1                       # 最後一格＝盒外
acc = np.zeros((len(models), NBX, len(CATS), 2))   # [模型, 塊, 類別, (se 總和, 像素數)]
fr_psnr = np.zeros((len(models), len(sel)))
eacc = np.zeros((len(models), 2, 2))    # [模型, (非邊緣, 邊緣), (se, n)]
SOB = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=dev)
with torch.no_grad():
    for k, i in enumerate(sel):
        cam = cams[i].to_device(dev)
        pil = Image.open(os.path.join(a.test_dir, "images_1.2", names_t[i])).convert("RGB")
        for j, (nm, m, r) in enumerate(models):
            o = r(cam, m, bg_color=bg)
            out = o["render"].clamp(0, 1)
            H, W = out.shape[1:]
            if j == 0:
                if pil.size != (W, H):
                    pil = pil.resize((W, H), Image.LANCZOS)
                gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
                zz = o["surf_depth"][0].reshape(H, W)
                v2, u2 = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
                pc = torch.stack([(u2 + 0.5 - cam.cx) / cam.fx * zz, (v2 + 0.5 - cam.cy) / cam.fy * zz, zz], -1).reshape(-1, 3)
                xw = (pc - cams.T[i].to(dev).float()) @ cams.R[i].to(dev).float()
                valid = ((zz > 0) & (o["rend_alpha"].reshape(H, W) > 0.5)).reshape(-1)
                inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=(xw.cpu() @ Rp.T)[:, :2])  # [NB, HW]
                bid = torch.where(inb.any(0), inb.float().argmax(0), torch.full((inb.shape[1],), NB)).to(dev)
                nr = o["rend_normal"].reshape(3, H, W)
                nz = (nr[2] / nr.norm(dim=0).clamp_min(1e-6)).abs().reshape(-1)
                zw = xw[:, 2]
                hor = nz > 0.8
                cc = torch.full_like(bid, 4)
                cc[nz < 0.3] = 3
                cc[hor & (zw >= 0.03)] = 2
                cc[hor & (zw < 0.03) & (zw > -0.02)] = 1
                cc[hor & (zw <= -0.02)] = 0
                key = (bid * len(CATS) + cc)[valid]
                if a.edge_q > 0:
                    lum = gt.mean(0)[None, None]
                    gx = torch.nn.functional.conv2d(lum, SOB[None, None], padding=1)[0, 0]
                    gy = torch.nn.functional.conv2d(lum, SOB.T[None, None], padding=1)[0, 0]
                    gmag = (gx * gx + gy * gy).sqrt().reshape(-1)
                    edge = (gmag >= torch.quantile(gmag[::7], a.edge_q))
            se_full = ((out - gt) ** 2).mean(0)
            fr_psnr[j, k] = float(-10 * torch.log10(se_full.mean().clamp_min(1e-12)))
            se = se_full.reshape(-1)[valid]
            if a.edge_q > 0:
                sf = se_full.reshape(-1)
                for t_, mk in ((0, ~edge), (1, edge)):
                    eacc[j, t_] += [float(sf[mk].sum()), int(mk.sum())]
            s = torch.zeros(NBX * len(CATS), device=dev, dtype=torch.float64).index_add_(0, key, se.double())
            c = torch.bincount(key, minlength=NBX * len(CATS)).double()
            acc[j, :, :, 0] += s.view(NBX, len(CATS)).cpu().numpy()
            acc[j, :, :, 1] += c.view(NBX, len(CATS)).cpu().numpy()
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(sel)}", flush=True)

ps = lambda se, n: -10 * np.log10(max(se / max(n, 1), 1e-12))
tot_n = acc[0, :, :, 1].sum()
print(f"\n=== 逐幀 PSNR 平均（{len(sel)} 幀；和 eval_official_test 同定義）===")
for j, (nm, _, _) in enumerate(models):
    print(f"  {nm:<16} {fr_psnr[j].mean():.3f}" + ("" if j == 0 else f"（{fr_psnr[j].mean() - fr_psnr[0].mean():+.3f}；贏的幀 {int((fr_psnr[j] > fr_psnr[0]).sum())}/{len(sel)}）"))
if a.edge_q > 0:
    print(f"\n=== 邊緣（GT 亮度梯度前 {100 * (1 - a.edge_q):.0f}% 的像素；整張畫面）===")
    for j, (nm, _, _) in enumerate(models):
        tot = eacc[j, :, 0].sum()
        line = f"  {nm:<16} 邊緣佔平方誤差 {100 * eacc[j, 1, 0] / max(tot, 1e-12):5.1f}%｜邊緣 PSNR {ps(eacc[j, 1, 0], eacc[j, 1, 1]):.3f}｜非邊緣 {ps(eacc[j, 0, 0], eacc[j, 0, 1]):.3f}"
        if j > 0:
            line += f"（對{models[0][0]}：邊緣 {ps(eacc[j, 1, 0], eacc[j, 1, 1]) - ps(eacc[0, 1, 0], eacc[0, 1, 1]):+.3f}、非邊緣 {ps(eacc[j, 0, 0], eacc[j, 0, 1]) - ps(eacc[0, 0, 0], eacc[0, 0, 1]):+.3f}）"
        print(line)
print(f"\n=== 各塊（像素歸屬＝基準表面落在哪個分區盒；「外」＝盒外）===")
hdr = f"{'塊':>4} {'像素佔比':>8} {models[0][0]:>10}" + "".join(f" {nm + 'Δ':>12} {'貢獻%':>7}" for nm, _, _ in models[1:])
print(hdr)
dm = [acc[j, :, :, 0].sum() - acc[0, :, :, 0].sum() for j in range(len(models))]
for b in range(NBX):
    n0 = acc[0, b, :, 1].sum()
    if n0 == 0:
        continue
    p0 = ps(acc[0, b, :, 0].sum(), n0)
    line = f"{('外' if b == NB else str(b)):>4} {100 * n0 / tot_n:>7.1f}% {p0:>10.3f}"
    for j in range(1, len(models)):
        pj = ps(acc[j, b, :, 0].sum(), n0)
        cj = (acc[j, b, :, 0].sum() - acc[0, b, :, 0].sum()) / dm[j] * 100 if dm[j] != 0 else float("nan")
        line += f" {pj - p0:>+12.3f} {cj:>6.0f}%"
    print(line)
print(f"\n=== 各類別（全場景）===")
for c in range(len(CATS)):
    n0 = acc[0, :, c, 1].sum()
    p0 = ps(acc[0, :, c, 0].sum(), n0)
    line = f"  {CATS[c]:<4} {100 * n0 / tot_n:>6.1f}% {p0:>8.3f}"
    for j in range(1, len(models)):
        pj = ps(acc[j, :, c, 0].sum(), n0)
        cj = (acc[j, :, c, 0].sum() - acc[0, :, c, 0].sum()) / dm[j] * 100 if dm[j] != 0 else float("nan")
        line += f"  {models[j][0]} {pj - p0:+.3f}（貢獻 {cj:.0f}%）"
    print(line)
for j in range(1, len(models)):
    print(f"\n=== {models[j][0]}：對整體 MSE 變化貢獻最大的（塊 x 類別）格（正＝讓它變差）===")
    d = acc[j, :, :, 0] - acc[0, :, :, 0]
    order = np.argsort(-d.reshape(-1))
    print(f"  整體 MSE 變化 {dm[j] / tot_n:+.3e}（基準 {acc[0, :, :, 0].sum() / tot_n:.3e}）")
    for idx in list(order[:a.top]) + list(order[::-1][:5]):
        b, c = divmod(int(idx), len(CATS))
        n0 = acc[0, b, c, 1]
        if n0 == 0:
            continue
        print(f"  {('外' if b == NB else '塊' + str(b)):>4} {CATS[c]:<4} 像素 {100 * n0 / tot_n:5.2f}%  {ps(acc[0, b, c, 0], n0):7.3f} -> {ps(acc[j, b, c, 0], n0):7.3f}"
              f"（{ps(acc[j, b, c, 0], n0) - ps(acc[0, b, c, 0], n0):+.3f}）  貢獻 {d[b, c] / dm[j] * 100:+6.1f}%")
