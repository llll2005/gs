"""11-28 使用者「面元記主要顏色＋漸變範式＋透明度」構想的前提量測（只渲染、不訓練）。對一塊：
  (1) 面元在訓練視角的畫面大小：record pass 的 Σ T·α（有效像素）與實際評估的像素數，按每個看得到它的視角平均；
      按類別（面元法向＋高度：海面／地面／屋頂／立面／其他）分開，並列「渲染貢獻質量」落在各大小區間的比例。
      => 面內色場要有東西可畫，面元得夠大。
  (2) 相鄰共面的互為最佳配對：用訓練中合併的同一套規則（merge_ops.merge_step，as_trim＝不改參數），
      顏色門檻 0.1（現行）vs 1.0（不看顏色）=> 「只因顏色不同而沒併」的配對有多少（面內色場能讓它們合併）。
  (3) 邊緣：GT 亮度梯度前 10% 的像素＝邊緣；面元中心投影落在邊緣上的比例（在看得到它的視角裡過半＝邊緣面元）、
      邊緣面元的大小 vs 其他、邊緣像素佔整體平方誤差的比例 => 面內切邊（不透明度改成面內的場）有多少可省。
用法（lab，cwd＝gs）：python tools/probe/surfel_texture_premise_1128.py --run outputs/lab/fs1122_base10_f32 --block 1 [--views 40]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "."); sys.path.insert(0, "tools")
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="outputs/lab/fs1122_base10_f32")
ap.add_argument("--block", type=int, default=1)
ap.add_argument("--views", type=int, default=40)
ap.add_argument("--edge_q", type=float, default=0.9)
a = ap.parse_args()

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.density_controllers.merge_ops import merge_step

dev = torch.device("cuda")
B = a.block
D = "data/matrix_city/aerial/train/block_all"
CATS = ["海面", "地面", "屋頂", "立面", "其他"]
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pcd = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pcd.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].float()

mck = os.path.join(a.run, "checkpoints", "merged.ckpt")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(mck, device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
X = model.get_xyz.detach()
N = X.shape[0]
inbox = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb0, coordinates=(X.cpu() @ Rt.T)[:, :2])[B].to(dev)
q = model.get_rotation.detach()
q = q / q.norm(dim=-1, keepdim=True)
w, x, y, z = q.unbind(-1)
nz = (1 - 2 * (x * x + y * y)).abs()          # 旋轉矩陣第三欄（面元法向）的 z 分量
cat = torch.full((N,), 4, dtype=torch.long, device=dev)
cat[nz < 0.3] = 3
hor = nz > 0.8
cat[hor & (X[:, 2] >= 0.03)] = 2
cat[hor & (X[:, 2] < 0.03) & (X[:, 2] > -0.02)] = 1
cat[hor & (X[:, 2] <= -0.02)] = 0
print(f"[模型] {mck} N={N:,}｜第 {B} 塊盒內 {int(inbox.sum()):,}", flush=True)

bck = sorted(glob.glob(os.path.join(a.run, "blocks", f"block_{B}", "checkpoints", "*.ckpt")))[-1]
ck = torch.load(bck, map_location="cpu", weights_only=False); dmh = ck["datamodule_hyper_parameters"]; del ck
if getattr(dmh["parser"], "points_from", None) == "ply":
    dmh["parser"].points_from = "random"
ts = dmh["parser"].instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
sel = np.linspace(0, len(ts.image_names) - 1, min(a.views, len(ts.image_names))).round().astype(int).tolist()
bg = torch.zeros(3, device=dev)
V = torch.zeros(N, device=dev, dtype=torch.float64); C = torch.zeros(N, device=dev, dtype=torch.float64)
NV = torch.zeros(N, device=dev); EH = torch.zeros(N, device=dev)
se_edge = se_all = 0.0
SOB = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=dev)
with torch.no_grad():
    for k, i in enumerate(sel):
        cam = ts.cameras[i].to_device(dev)
        tm, cov = renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True)
        cov = cov.reshape(-1).double(); tm = tm.reshape(-1).double()
        V += tm * cov; C += cov
        vis = cov > 0
        NV += vis.float()
        o = renderer(cam, model, bg_color=bg)
        img = o["render"].clamp(0, 1); H, W = img.shape[1:]
        gt = torch.from_numpy(np.asarray(Image.open(ts.image_paths[i]).convert("RGB").resize((W, H), Image.LANCZOS))).float().permute(2, 0, 1).to(dev) / 255.
        lum = gt.mean(0, keepdim=True)[None]
        gx = F.conv2d(lum, SOB[None, None], padding=1)[0, 0]; gy = F.conv2d(lum, SOB.T[None, None], padding=1)[0, 0]
        gm = (gx * gx + gy * gy).sqrt()
        edge = gm >= torch.quantile(gm.reshape(-1)[::7], a.edge_q)
        se = ((img - gt) ** 2).mean(0)
        se_all += float(se.sum()); se_edge += float(se[edge].sum())
        # 面元中心投到這個視角：中心落在邊緣像素上？
        Rm = cam.R.float(); Tm = cam.T.float()
        pc = X @ Rm.T + Tm
        u = (pc[:, 0] / pc[:, 2].clamp_min(1e-6) * cam.fx + cam.cx).long(); v = (pc[:, 1] / pc[:, 2].clamp_min(1e-6) * cam.fy + cam.cy).long()
        m = vis & (pc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        EH[m] += edge[v[m], u[m]].float()
        if (k + 1) % 10 == 0:
            print(f"  視角 {k + 1}/{len(sel)}", flush=True)

seen = inbox & (NV > 0)
eff = (V / NV.clamp_min(1)).float(); sup = (C / NV.clamp_min(1)).float()
print(f"\n=== (1) 第 {B} 塊盒內、被 {len(sel)} 個訓練視角看到的面元 {int(seen.sum()):,} 顆：每個看得到它的視角裡的畫面大小（像素）===")
print(f"  {'類別':<6} {'顆數佔比':>8} {'有效像素(Σ T·α) p10/中位/p90':>30} {'評估像素 中位':>12}")
for c in list(range(len(CATS))) + [None]:
    mm = seen if c is None else (seen & (cat == c))
    if int(mm.sum()) == 0:
        continue
    e = eff[mm]; qv = torch.quantile(e[torch.randperm(e.numel(), device=dev)[:1_000_000]], torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()
    print(f"  {('全部' if c is None else CATS[c]):<6} {100 * float(mm.sum()) / float(seen.sum()):>7.1f}% {qv[0]:>10.2f} / {qv[1]:>6.2f} / {qv[2]:>7.2f}"
          f" {float(sup[mm].median()):>12.1f}")
print("  渲染貢獻質量（Σ T·α）落在各大小區間的比例（有效像素／視角）：")
mass = V[seen].float(); es = eff[seen]
for lo, hi in ((0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 1e9)):
    mm = (es >= lo) & (es < hi)
    print(f"    {lo:>4}~{('∞' if hi > 1e8 else int(hi)):<4} px：顆數 {100 * float(mm.float().mean()):5.1f}%｜質量 {100 * float(mass[mm].sum() / mass.sum()):5.1f}%")

print(f"\n=== (3) 邊緣（GT 亮度梯度前 {100 * (1 - a.edge_q):.0f}% 的像素）===")
print(f"  邊緣像素佔整體平方誤差 {100 * se_edge / max(se_all, 1e-12):.1f}%（像素只佔 {100 * (1 - a.edge_q):.0f}%）")
ef = (EH / NV.clamp_min(1)) > 0.5
for lab, mm in (("邊緣面元", seen & ef), ("其他面元", seen & ~ef)):
    if int(mm.sum()):
        print(f"  {lab}：佔 {100 * float(mm.sum()) / float(seen.sum()):5.1f}% 顆數｜有效像素中位 {float(eff[mm].median()):.2f}｜不透明度 > 0.99 的 {100 * float((model.get_opacity.detach().reshape(-1)[mm] > 0.99).float().mean()):.1f}%")
del model; torch.cuda.empty_cache()

# ── (2) 合併規則的配對（用該塊自己的 ckpt；原始參數、不改）──
print(f"\n=== (2) 相鄰共面的互為最佳配對（merge_ops.merge_step 同規則，as_trim＝只數不改）：{os.path.basename(bck)} ===", flush=True)
ckb = torch.load(bck, map_location="cpu", weights_only=False)
gmb = GaussianModelLoader.initialize_model_from_checkpoint(ckb, device=dev); del ckb
nb = gmb.gaussians["means"].shape[0]
for tol in (0.1, 0.2, 1.0):
    with torch.no_grad():
        I, J, info = merge_step(gmb, k=12, max_frac=1.0, color_tol=tol, chunk=1 << 16, as_trim=True)
    print(f"  顏色門檻 {tol:g}：互為最佳配對 {info.get('mutual', 0):,}（佔 N {100 * info.get('mutual', 0) / nb:.1f}%）", flush=True)
    if tol >= 1.0 and I.numel():
        # ★ 「主要顏色＋漸變」能不能只記亮度：配對兩顆的 DC 顏色差，有多少落在「亮度方向」（Δc 平行於兩顆平均色）
        C0 = 0.28209479177387814
        ci = gmb.gaussians["shs_dc"][I, 0, :].float() * C0 + 0.5; cj = gmb.gaussians["shs_dc"][J, 0, :].float() * C0 + 0.5
        dc_ = cj - ci; mc = 0.5 * (ci + cj); u_ = mc / mc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        par = (dc_ * u_).sum(-1, keepdim=True) * u_
        l1 = dc_.abs().mean(-1)
        big = l1 >= 0.1
        if int(big.sum()):
            fr = (par[big].norm(dim=-1) ** 2 / (dc_[big].norm(dim=-1) ** 2).clamp_min(1e-12))
            print(f"  色差 >= 0.1 的配對 {int(big.sum()):,}：色差能量落在亮度方向的比例 中位 {float(fr.median()):.2f}（p25 {float(fr.quantile(0.25)):.2f}）"
                  f"｜>= 0.9 的佔 {100 * float((fr >= 0.9).float().mean()):.1f}%", flush=True)
print("  => 門檻 1.0 減 0.1 的差＝只因顏色不同而沒併的配對（現行合併用 0.1）")
