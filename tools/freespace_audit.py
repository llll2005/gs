"""空區雕刻稽核（2026-10-02，使用者同意）：SfM 的「相機 -> 點」可見光線 = 這段空間是空的。

問題：落在「很多光線穿過、又沒有任何 SfM 點」的體素裡的顆粒（＝不該存在的體積）有多少？它們是不是懸空的那批？
      事後把它們拿掉，val（val⊂train）與塊內官方 held-out 怎麼變？—— 和**同數量的隨機移除**比（控制組）。
這是「鑿子」接進訓練之前的判決：val 不掉、held-out 變好，才值得做成訓練期機制。

做法：
  光線   本塊分區清單裡每台相機 -> 它看得到的 SfM 點（point3D_ids），只用被 >= --min-track 張影像看到的點；
         每條光線停在點前 max(--margin-abs, --margin-rel x 距離) 處（點附近不算空）
  體素   邊界取顆粒中心的 0.1~99.9 百分位；體素邊長自動放大到總數 <= --max-voxels
         沿光線每 1 個體素長取樣一次，累加「空的票數」（約等於穿過的光線數）
  保護   有 SfM 點（track >= 2）的體素及其 26 鄰居不算空（擋掉離群點射出的假光線在真表面上開洞）
  懸空   顆粒高度 - 同一 xy 欄位（4 個體素寬）裡 SfM 點的最高 z
用法：python tools/freespace_audit.py --ckpt <ckpt> --block 6 [--k 1 3 10] [--eval-k 3]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = "data/matrix_city/aerial/train/block_all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--margin-abs", type=float, default=0.05)
    ap.add_argument("--margin-rel", type=float, default=0.02)
    ap.add_argument("--max-voxels", type=float, default=1.2e8)
    ap.add_argument("--k", type=float, nargs="+", default=[1, 3, 10])
    ap.add_argument("--eval-k", type=float, nargs="+", default=[3, 10])
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    a = ap.parse_args()
    dev = torch.device("cuda")
    from internal.utils.colmap import read_images_binary, read_points3D_binary
    from internal.utils.gaussian_model_loader import GaussianModelLoader

    # ── 顆粒 ──
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    xyz = model.get_xyz.detach().float()
    op = model.get_opacity.detach().reshape(-1).float()
    sc = model.get_scaling.detach().float()
    area = op * sc[:, 0] * sc[:, 1]
    N = xyz.shape[0]
    lo = torch.quantile(xyz[torch.randperm(N, device=dev)[:2_000_000]], torch.tensor([0.001], device=dev), dim=0)[0]
    hi = torch.quantile(xyz[torch.randperm(N, device=dev)[:2_000_000]], torch.tensor([0.999], device=dev), dim=0)[0]
    vol = float(torch.prod(hi - lo))
    vs = max(0.01, (vol / a.max_voxels) ** (1 / 3))
    lo = lo - vs; hi = hi + vs
    dims = torch.ceil((hi - lo) / vs).long()
    NV = int(torch.prod(dims))
    print(f"ckpt {os.path.basename(a.ckpt)}  N={N:,}  體素 {vs:.4f}  格點 {tuple(dims.tolist())} = {NV / 1e6:.1f}M")

    # ── 光線 ──
    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    plist = os.path.join(DATA, "partition", f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08", f"{bx:03d}_{by:03d}.txt")
    want = {l.strip() for l in open(plist) if l.strip()}
    imgs = read_images_binary(os.path.join(DATA, "sparse/0/images.bin"))
    P3 = read_points3D_binary(os.path.join(DATA, "sparse/0/points3D.bin"))
    pid = np.fromiter(P3.keys(), dtype=np.int64)
    pxyz = np.stack([P3[k].xyz for k in pid]).astype(np.float32)
    ptrk = np.array([len(P3[k].image_ids) for k in pid])
    idx_of = {int(k): i for i, k in enumerate(pid)}
    C, Pi = [], []
    ncam = 0
    for im in imgs.values():
        if im.name not in want:
            continue
        ncam += 1
        cen = -im.qvec2rotmat().T @ im.tvec
        ids = [idx_of[int(p)] for p in im.point3D_ids if p >= 0 and int(p) in idx_of]
        ids = [i for i in ids if ptrk[i] >= a.min_track]
        C.append(np.repeat(cen[None].astype(np.float32), len(ids), 0)); Pi.append(np.array(ids, dtype=np.int64))
    C = torch.from_numpy(np.concatenate(C)).to(dev)
    P = torch.from_numpy(pxyz[np.concatenate(Pi)]).to(dev)
    R = C.shape[0]
    print(f"相機 {ncam} 台  光線 {R:,}（track >= {a.min_track}）")

    votes = torch.zeros(NV, dtype=torch.int32, device=dev)
    d = P - C
    L = d.norm(dim=1)
    u = d / L.clamp_min(1e-9)[:, None]
    free = (L - torch.maximum(torch.full_like(L, a.margin_abs), a.margin_rel * L)).clamp_min(0)
    ns = torch.floor(free / vs).long()
    start = 0
    target = 30_000_000
    while start < R:
        cum = torch.cumsum(ns[start:], 0)
        end = start + int(torch.searchsorted(cum, torch.tensor(target, device=dev))) + 1
        end = min(max(end, start + 1), R)
        n = ns[start:end]
        r = torch.repeat_interleave(torch.arange(start, end, device=dev), n)
        if r.numel():
            off = torch.arange(r.numel(), device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
            pts = C[r] + u[r] * ((off.float() + 0.5) * vs)[:, None]
            g = torch.floor((pts - lo) / vs).long()
            ok = ((g >= 0) & (g < dims)).all(1)
            g = g[ok]
            flat = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
            votes.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        start = end

    # ── 保護：有 SfM 點的體素及其鄰居 ──
    sp = torch.from_numpy(pxyz[ptrk >= 2]).to(dev)
    gs = torch.floor((sp - lo) / vs).long()
    ok = ((gs >= 0) & (gs < dims)).all(1)
    occ = torch.zeros(NV, dtype=torch.float16, device=dev)
    occ[(gs[ok, 0] * dims[1] + gs[ok, 1]) * dims[2] + gs[ok, 2]] = 1
    occ = torch.nn.functional.max_pool3d(occ.view(1, 1, *dims.tolist()), 3, 1, 1).view(-1) > 0
    votes[occ] = 0
    print(f"有票的體素 {100 * float((votes > 0).float().mean()):.2f}%  受保護（有 SfM 點及鄰居）{100 * float(occ.float().mean()):.2f}%")

    # ── 顆粒分類 ──
    gp = torch.floor((xyz - lo) / vs).long()
    inside = ((gp >= 0) & (gp < dims)).all(1)
    v = torch.zeros(N, dtype=torch.int32, device=dev)
    gi = gp[inside]
    v[inside] = votes[(gi[:, 0] * dims[1] + gi[:, 1]) * dims[2] + gi[:, 2]]
    # 懸空：高於同欄位 SfM 最高點
    cell = 4 * vs
    cxy = torch.floor((sp[:, :2] - lo[:2]) / cell).long()
    cd = torch.ceil((hi[:2] - lo[:2]) / cell).long() + 1
    okc = ((cxy >= 0) & (cxy < cd)).all(1)
    top = torch.full((int(cd[0] * cd[1]),), -1e9, device=dev)
    top.scatter_reduce_(0, cxy[okc, 0] * cd[1] + cxy[okc, 1], sp[okc, 2], reduce="amax")
    pxy = torch.floor((xyz[:, :2] - lo[:2]) / cell).long().clamp(min=0)
    pxy = torch.minimum(pxy, cd - 1)
    ptop = top[pxy[:, 0] * cd[1] + pxy[:, 1]]
    h = torch.where(ptop > -1e8, xyz[:, 2] - ptop, torch.full_like(ptop, float("nan")))
    print(f"格外顆粒 {100 * float((~inside).float().mean()):.2f}%（不分類）")
    print(f"\n{'k（光線數）':>10} {'顆數%':>7} {'不透明度質量%':>12} {'面積x不透明度%':>13} {'其中 o>0.5':>9} {'高於欄頂 中位':>12} {'（其餘顆粒）':>10}")
    hn = torch.nanmedian(h[inside]).item()
    for k in a.k:
        m = inside & (v >= k)
        if int(m.sum()) == 0:
            print(f"{k:>10g}  無"); continue
        print(f"{k:>10g} {100 * float(m.float().mean()):>6.2f}% {100 * float(op[m].sum() / op.sum()):>11.2f}% "
              f"{100 * float(area[m].sum() / area.sum()):>12.2f}% {100 * float((op[m] > 0.5).float().mean()):>8.1f}% "
              f"{torch.nanmedian(h[m]).item():>12.3f} {hn:>10.3f}")

    # ── 事後移除：val 與塊內官方 held-out，對照同數量隨機移除 ──
    from internal.utils.ssim import ssim as ssim_fn
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from tools.eval_official_test import load_test_cameras, block_bounds
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)
    ck = torch.load(a.ckpt, map_location="cpu"); dmh = ck["datamodule_hyper_parameters"]; del ck
    val = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0).get_outputs().val_set
    sets = {"val": [(val.cameras[i], val.image_paths[i]) for i in range(len(val))]}
    if os.path.exists(os.path.join(a.test_dir, ".built_by_official_steps")):
        names, tcams = load_test_cameras(a.test_dir, 1.2)
        blo, bhi, _ = block_bounds(DATA, a.block, a.block_dim)
        cen = np.array([(-tcams.R[i].numpy().T @ tcams.T[i].numpy()) for i in range(len(names))])
        sel = [i for i in range(len(names)) if np.all(cen[i] >= blo) and np.all(cen[i] <= bhi)]
        sets["held-out（塊內官方 test）"] = [(tcams[i], os.path.join(a.test_dir, "images_1.2", names[i])) for i in sel]
    orig = {k_: t for k_, t in model.properties.items()}
    bg = torch.zeros(3, device=dev)
    gt_cache = {}

    def score(mask):
        model.properties = {k_: t[mask] for k_, t in orig.items()}
        res = {}
        with torch.no_grad():
            for nm, items in sets.items():
                ps, ss, ls = [], [], []
                for cam, path in items:
                    img = renderer(cam.to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
                    key = (path, img.shape)
                    if key not in gt_cache:
                        pil = Image.open(path).convert("RGB")
                        if pil.size != (img.shape[2], img.shape[1]):
                            pil = pil.resize((img.shape[2], img.shape[1]), Image.LANCZOS)
                        gt_cache[key] = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
                    gt = gt_cache[key]
                    ps.append(float(-10 * torch.log10(((img - gt) ** 2).mean().clamp_min(1e-12))))
                    ss.append(float(ssim_fn(img, gt))); ls.append(float(lp(img.unsqueeze(0), gt.unsqueeze(0))))
                res[nm] = (np.mean(ps), np.mean(ss), np.mean(ls), len(items))
        model.properties = orig
        return res

    def show(tag, r, base=None):
        s = "  ".join(f"{nm} {p:.3f}/{sv:.4f}/{l:.4f}" + (f"（{p - base[nm][0]:+.3f}）" if base else "")
                      for nm, (p, sv, l, n) in r.items())
        print(f"  {tag:<28} {s}")

    print(f"\n── 事後移除（PSNR/SSIM/LPIPS；括號＝PSNR 相對不剪）視角數：" + "、".join(f"{nm} {len(it)}" for nm, it in sets.items()))
    base = score(torch.ones(N, dtype=torch.bool, device=dev))
    show("不剪", base)
    g = torch.Generator(device=dev).manual_seed(0)
    for k in a.eval_k:
        m = inside & (v >= k)
        n = int(m.sum())
        if n == 0:
            continue
        show(f"移除空區顆粒 k>={k:g}（{n:,}）", score(~m), base)
        rnd = torch.zeros(N, dtype=torch.bool, device=dev)
        rnd[torch.randperm(N, device=dev, generator=g)[:n]] = True
        show(f"  對照：隨機移除同數量", score(~rnd), base)


if __name__ == "__main__":
    main()
