"""★ 2026-11-13 沒用的顆粒佔多少（使用者「你量量看」：吸收態的延伸問題——無用顆粒佔容量、加運算？）。只渲染、不訓練；cwd＝repo 根。
對一個（或幾個）我方單塊 ckpt，用本塊全部訓練相機（--stride）逐張渲染（完整路徑），靠光柵器的逐顆帳（depth_cap_acc）累加每顆的貢獻：
  V_all[i]＝Σ_{全部視角, 全部像素} T·α（混合權重和）；V_A[i]＝只算塊外遮罩 A（計分）像素；V_B[i]＝只算 B（塊外、目標＝隨機背景）像素。
  （做法：上限圖設成極小的正數 => 沒有任何貢獻落在上限後面，tot＝所有設了上限的像素上的 T·α；上限圖只在 A／B 像素非 0 => 分開累加。）
分群（全部用相對量，沒有場景單位門檻）：
  Z0＝V_all＝0（在每個訓練視角都完全看不到：被擋死或不在視野）｜ZB＝V_A＝0 但 V_all > 0（只出現在 B／C 像素）｜
  T3＝0 < V_A < 1e-3 x（V_A > 0 的中位數）（幾乎沒貢獻）｜其餘＝有用。
每群：顆數與佔 N、塊內（分區盒）／塊外、不透明（o > 0.5）、離 SfM 地面的高度帶（地下 < −0.5 m／貼地 ±0.5 m／地上；只是量測分類）。
拿掉測試（opacity 設 0，不重訓）：訓練視角 A 像素 ΔPSNR（Z0、ZB 應該是 0＝自我檢查）、換塊 held-out ΔPSNR（同 tools/swap_block_eval.py 的換法）、
  渲染時間（全部訓練視角 forward 的總時間；o＝0 的顆粒不進 binning => 量到的就是它們佔的渲染成本；⚠ 卡上要沒有別的行程才準）。
用法（lab，佇列 [solo]）：conda run -n gspl --no-capture-output python tools/probe/dead_weight_1113.py --block 1 \
   --models 新底='outputs/lab/g44tn_s30g50np16tricap0p5cmflat_best1maskdistrfed10b4/blocks/block_1/checkpoints/*.ckpt'
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.gaussian_model_loader import GaussianModelLoader

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--models", nargs="+", required=True, help="標籤=ckpt 樣式（第一個的 ckpt 提供渲染器、訓練相機、塊外遮罩）")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--max_cams", type=int, default=0)
ap.add_argument("--tiny", type=float, default=1e-3, help="T3＝V_A < tiny x 中位數")
ap.add_argument("--merged_run", default="outputs/lab/full44_best")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
ap.add_argument("--no_heldout", action="store_true")
a = ap.parse_args()
B = a.block; D = "data/matrix_city/aerial/train/block_all"; MU = 100.0; dev = torch.device("cuda")
t0 = time.time()

# ---- 分區盒、SfM 地面（同 behind_cause_1112；只用來分類）----
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc_ = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc_.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc_.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double().cpu() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); okp = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = P[okp & inb(P, bb1)]
G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u_, f_, c_ = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u_] = S[order, 2][f_ + c_ // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03)
gz = np.where(ground, med, np.nan)
num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
GL = float(np.nanmedian(gz))


def height_and_ground(xyz):
    cc = np.floor(xyz[:, :2] / G).astype(int) - o0
    okc = (cc[:, 0] >= 0) & (cc[:, 0] < shp[0]) & (cc[:, 1] >= 0) & (cc[:, 1] < shp[1])
    cc = np.clip(cc, 0, np.array(shp) - 1)
    g = gfill[cc[:, 0], cc[:, 1]]
    return (xyz[:, 2] - np.where(okc & np.isfinite(g), g, GL)) * MU, okc & ground[cc[:, 0], cc[:, 1]]


# ---- 渲染器、訓練相機、塊外遮罩（取自第一個 ckpt）----
specs = []
for sp in a.models:
    lab, pat = sp.split("=", 1); fs = sorted(glob.glob(pat))
    if not fs:
        raise SystemExit(f"⛔ {lab}：找不到 {pat}")
    specs.append((lab, fs[-1]))
base_ck = specs[0][1]
_, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(base_ck, device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "skip_unscored_pixels", "trim_running_vc"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
ck = torch.load(base_ck, map_location="cpu", weights_only=False)
dmh = ck["datamodule_hyper_parameters"]; mcfg = ck["hyper_parameters"].get("metric")
mask_dir = getattr(getattr(mcfg, "init_args", mcfg), "block_mask_dir", "") or ""
del ck
parser = dmh["parser"]
if getattr(parser, "points_from", None) == "ply":
    parser.points_from = "random"
ts = parser.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
cams = ts.cameras; names = list(ts.image_names)
sel = list(range(0, len(names), max(1, a.stride)))
if a.max_cams > 0:
    sel = sel[: a.max_cams]
print(f"b{B}｜本塊訓練相機 {len(names)} 台（用 {len(sel)}）｜遮罩 {mask_dir or '⚠⚠ 沒有（全畫面當 A）'}", flush=True)


def mask_of(i, H, W):
    """塊外遮罩：1＝A（計分）、2＝B（塊外、目標＝隨機背景）、0＝C（不計分）。沒有遮罩 => 全部當 A。"""
    if mask_dir:
        mp = os.path.join(mask_dir, os.path.basename(str(names[i])))
        if os.path.exists(mp):
            mt = torch.from_numpy(np.array(Image.open(mp)))[None, None].float()
            return torch.nn.functional.interpolate(mt, size=(H, W), mode="nearest")[0, 0].to(dev).long()
    return torch.ones(H, W, dtype=torch.long, device=dev)


bg = torch.zeros(3, device=dev)
EPS_ICAP = 1e-6          # 上限極遠 => 沒有任何貢獻落在後面；tot＝設了上限的像素上的 T·α


def gt_of(i, H, W):
    pil = Image.open(ts.image_paths[i]).convert("RGB")
    if pil.size != (W, H):
        pil = pil.resize((W, H), Image.LANCZOS)
    return torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.


psnr = lambda s, n: -10 * np.log10(max(s / max(n, 1), 1e-12))

for lab, ckf in specs:
    m, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ckf, device=dev, eval_mode=True)
    n = m.get_xyz.shape[0]
    print(f"\n== {lab}｜{ckf}｜N {n:,}", flush=True)
    V = {k: torch.zeros(n, device=dev) for k in ("all", "A", "B")}
    dummy = torch.zeros(n, device=dev)
    with torch.no_grad():
        for jj, i in enumerate(sel):
            cam = cams[i].to_device(dev); H, W = int(cam.height), int(cam.width)
            mk = mask_of(i, H, W)
            for key, sel_px in (("all", torch.ones_like(mk, dtype=torch.bool)), ("A", mk == 1), ("B", mk == 2)):
                cap = (sel_px.float() * EPS_ICAP).contiguous()
                if int(sel_px.sum()) == 0:
                    continue
                renderer(cam, m, bg_color=bg, _depth_cap=cap, _depth_cap_acc=(dummy, V[key]))
            if (jj + 1) % 200 == 0:
                print(f"  逐顆帳 相機 {jj + 1}／{len(sel)}（{time.time() - t0:.0f} s）", flush=True)
    va = V["A"]; pos = va[va > 0]
    medA = float(pos.median()) if pos.numel() else 0.0
    grp = {"Z0 每個視角都看不到": V["all"] <= 0,
           "ZB 只在 B／C 像素": (V["all"] > 0) & (va <= 0),
           f"T3 對 A 幾乎沒貢獻（< {a.tiny:g} x 中位）": (va > 0) & (va < a.tiny * medA)}
    grp["合計（Z0＋ZB＋T3）"] = grp["Z0 每個視角都看不到"] | grp["ZB 只在 B／C 像素"] | grp[f"T3 對 A 幾乎沒貢獻（< {a.tiny:g} x 中位）"]
    xyz = m.get_xyz.detach().double().cpu().numpy(); o = m.get_opacities().reshape(-1).detach().cpu().numpy()
    ib = inb(xyz, bb0); h, isg = height_and_ground(xyz)
    under = isg & (h < -0.5); onground = isg & (np.abs(h) <= 0.5)
    print(f"  V_A > 0 的中位數 {medA:.4g}；塊內 {int(ib.sum()):,} 顆（{100 * ib.mean():.1f}%）")
    print(f"  {'群':28}{'顆數':>10}{'佔 N':>8}{'塊內':>10}{'塊內佔塊內%':>12}{'塊內 o>0.5':>12}{'地下':>9}{'貼地':>9}{'地上／非地面格':>16}")
    for g, k in grp.items():
        kk = k.cpu().numpy(); ki = kk & ib
        print(f"  {g:28}{int(kk.sum()):>10,}{100 * kk.mean():>7.1f}%{int(ki.sum()):>10,}{100 * ki.sum() / max(ib.sum(), 1):>11.1f}%"
              f"{int((ki & (o > 0.5)).sum()):>12,}{int((ki & under).sum()):>9,}{int((ki & onground).sum()):>9,}{int((ki & ~under & ~onground).sum()):>16,}", flush=True)
    # 有用的那群做對照：塊內不透明顆粒裡地下的比例
    useful = (~grp["合計（Z0＋ZB＋T3）"]).cpu().numpy() & ib
    print(f"  對照：塊內有用的 {int(useful.sum()):,} 顆裡 地下 {int((useful & under).sum()):,}、貼地 {int((useful & onground).sum()):,}")

    # ---- 拿掉測試：訓練視角 ΔPSNR（A 像素）＋渲染時間 ----
    props = dict(m.properties); op0 = props["opacities"].detach().clone()
    rm = {"Z0": grp["Z0 每個視角都看不到"], "Z0＋ZB": grp["Z0 每個視角都看不到"] | grp["ZB 只在 B／C 像素"], "Z0＋ZB＋T3": grp["合計（Z0＋ZB＋T3）"]}
    se = np.zeros(1 + len(rm)); npx = 0; tm = np.zeros(1 + len(rm))
    with torch.no_grad():
        for jj, i in enumerate(sel):
            cam = cams[i].to_device(dev); H, W = int(cam.height), int(cam.width)
            Am = mask_of(i, H, W) == 1; gt = gt_of(i, H, W)
            for ri, (rn, kk) in enumerate([("原樣", None)] + list(rm.items())):
                opl = op0 if kk is None else torch.where(kk.reshape(op0.shape), torch.zeros_like(op0), op0)
                m.properties = {**props, "opacities": opl}
                torch.cuda.synchronize(); ts0 = time.perf_counter()
                img = renderer(cam, m, bg_color=bg)["render"]
                torch.cuda.synchronize(); tm[ri] += time.perf_counter() - ts0
                s_ = ((img.clamp(0, 1) - gt) ** 2).mean(0)
                se[ri] += float(s_[Am].double().sum())
            npx += int(Am.sum())
    m.properties = {**props, "opacities": op0}
    print(f"  拿掉測試（opacity 設 0、不重訓；訓練視角 {len(sel)} 台、A 像素 {npx:,}；時間＝每台 forward 的總和，⚠ 卡上有別的行程時不準）")
    print(f"  {'拿掉':12}{'顆數':>10}{'訓練視角 A ΔPSNR':>18}{'渲染時間':>10}{'Δ%':>8}")
    print(f"  {'（原樣）':12}{'—':>10}{'—':>18}{tm[0]:>9.1f}s{'':>8}")
    for ri, (rn, kk) in enumerate(rm.items(), 1):
        print(f"  {rn:12}{int(kk.sum()):>10,}{psnr(se[ri], npx) - psnr(se[0], npx):>+18.4f}{tm[ri]:>9.1f}s{100 * (tm[ri] / tm[0] - 1):>+7.1f}%", flush=True)

    # ---- 換塊 held-out（同 behind_cause_1112 的換法）----
    if not a.no_heldout:
        import eval_official_test as eot
        from swap_block_eval import block_mask
        mck = sorted(glob.glob(os.path.join(a.merged_run, "checkpoints", "*.ckpt")))
        mck = [c for c in mck if os.path.basename(c) == "merged.ckpt"] or mck
        mm, rr, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(mck[-1], device=dev, eval_mode=True)
        for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            setattr(rr, f, False)
        basep = {k: v.detach() for k, v in mm.properties.items()}
        pcfg = torch.load(ckf, map_location="cpu", weights_only=False)["datamodule_hyper_parameters"]["parser"]
        keep_other = ~block_mask(basep["means"], pcfg)[B].to(dev)
        cp = dict(props)
        for _k in ("shs_rest",):
            if _k in cp and _k in basep and cp[_k].shape[1] < basep[_k].shape[1]:
                r_ = cp[_k]; cp[_k] = torch.cat([r_, torch.zeros((r_.shape[0], basep[_k].shape[1] - r_.shape[1], r_.shape[2]), dtype=r_.dtype, device=r_.device)], 1)
        cmask = block_mask(cp["means"], pcfg)[B].to(dev)
        merged = {k: torch.cat([basep[k][keep_other], cp[k][cmask].detach()], 0) for k in basep}
        n_other = int(keep_other.sum())
        del basep, cp; torch.cuda.empty_cache()
        opm0 = merged["opacities"].clone()
        names_t, tcams = eot.load_test_cameras(a.test_dir, 1.2)
        centres = np.array([(-tcams.R[j].numpy().T @ tcams.T[j].numpy()) for j in range(len(names_t))])
        lo, hi, _ = eot.block_bounds(D, B, [4, 4])
        selB = [j for j in range(len(names_t)) if np.all(centres[j] >= lo) and np.all(centres[j] <= hi)]
        se_f = np.zeros(1 + len(rm)); nf = 0
        with torch.no_grad():
            for jj, j in enumerate(selB):
                cam = tcams[j].to_device(dev)
                pil = Image.open(os.path.join(a.test_dir, "images_1.2", names_t[j])).convert("RGB")
                for ri, (rn, kk) in enumerate([("原樣", None)] + list(rm.items())):
                    opl = opm0.clone()
                    if kk is not None:
                        idx = torch.nonzero(kk[cmask]).reshape(-1) + n_other
                        opl[idx] = 0.0
                    mm.properties = {**merged, "opacities": opl}
                    img = rr(cam, mm, bg_color=bg)["render"].clamp(0, 1)
                    if ri == 0:
                        H_, W_ = img.shape[1:]
                        if pil.size != (W_, H_):
                            pil = pil.resize((W_, H_), Image.LANCZOS)
                        gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
                    se_f[ri] += float(((img - gt) ** 2).mean(0).double().sum())
                nf += int(gt.shape[1] * gt.shape[2])
        print(f"  換塊 held-out（{a.merged_run} 的第 {B} 塊換成 {lab}；{len(selB)} 幀；全畫面 PSNR＝合併 MSE）：原樣 {psnr(se_f[0], nf):.3f}")
        for ri, (rn, kk) in enumerate(rm.items(), 1):
            print(f"    拿掉 {rn:10}（塊內 {int((kk & cmask).sum()):,} 顆）：Δ {psnr(se_f[ri], nf) - psnr(se_f[0], nf):+.4f}", flush=True)
        del mm, merged; torch.cuda.empty_cache()
    del m; torch.cuda.empty_cache()
print(f"\n{time.time() - t0:.0f} s")
print("DONE")
