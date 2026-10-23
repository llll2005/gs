#!/usr/bin/env python
"""合併模型的壞區診斷：壞區是「前面多了東西」「表面顏色錯」還是「破洞」，以及是哪一塊的顆粒（10-23 遮罩全場景輸舊最佳）。

## 為什麼
遮罩臂 4x4 全場景（g44s30km25_best1maskdistrf）官方 741 幀 27.53 vs 舊最佳 full44_best 27.74；
損失集中在 b0/b4/b1 視角的少數幀，目視是大片暗色東西蓋住馬路（水面、玻璃帷幕旁）。要修得先知道它是什麼、屬於哪塊。

## 做法（逐幀）
- 壞區＝本模型逐像素誤差 − 參考模型誤差 > --thr（|render-GT| 三通道平均）
- 參考模型（舊最佳合併 ckpt）先渲染 surf_depth 當「真表面」深度；壞區像素按本模型深度 / 參考深度分三類：
    前方（< 1/--tol：前面多了東西＝漂浮物）、同深（顏色錯）、後方（> --tol：破洞／看穿）
- leave-one-block-out：拿掉某塊後壞區誤差的變化（最負＝那塊擁有這片內容；最正＝拿掉它就變好＝純漂浮物）
- 漂浮顆粒：opacity>0.1、投影中心落在壞區、相機深度 < 參考深度/--tol => 按塊計數，報高度（相對地面）與離本塊盒距離
- 顆粒歸屬照合併順序：merge.log 的 checkpoint 清單（字典序）＋ number_of_gaussians 逐塊顆數
--save_pts 存漂浮顆粒（給 tools/blob_mask_check.py 查它們在本塊訓練視角被遮罩分成哪類）。

用法（lab，GPU）：
  python tools/blob_owner.py --run g44s30km25_best1maskdistrf --ref_run full44_best --worst 14 \
      --save_dir outputs/lab/blob_owner_1023/km25 --save_pts outputs/lab/blob_owner_1023/km25_pts.npz
"""
import argparse
import ast
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from internal.utils.gaussian_model_loader import GaussianModelLoader  # noqa: E402


def merge_segments(run_dir):
    """merge.log 最後一次合併：[(block, start, end)]（合併時的串接順序）。"""
    L = open(os.path.join(run_dir, "merge.log"), encoding="utf-8", errors="ignore").read().splitlines()
    files = counts = None
    for l in L:
        if "| INFO | [" in l and "block_" in l:
            files = ast.literal_eval(l.split("| INFO | ", 1)[1])
        m = re.search(r"number_of_gaussians=sum\((\[.*\])\)", l)
        if m:
            counts = ast.literal_eval(m.group(1))
    assert files and counts and len(files) == len(counts), "merge.log 讀不到清單或顆數"
    seg, s = [], 0
    for f, c in zip(files, counts):
        b = int(re.search(r"block_(\d+)", f).group(1))
        seg.append((b, s, s + c)); s += c
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/lab/<run>（要有 checkpoints/merged.ckpt、merge.log、test_official/）")
    ap.add_argument("--ref_run", default="full44_best", help="參考模型 outputs/lab/<ref_run>（merged.ckpt＋test_official/）")
    ap.add_argument("--frames", nargs="*", default=None, help="幀名（如 0038.png）；不給就用 --worst")
    ap.add_argument("--worst", type=int, default=12, help="依兩邊存檔 PNG 的逐幀 PSNR 差，取本模型輸最多的幾幀")
    ap.add_argument("--thr", type=float, default=0.08, help="壞區：本模型誤差 − 參考誤差 > thr")
    ap.add_argument("--tol", type=float, default=1.1, help="深度比的同深容忍（前方 < 1/tol、後方 > tol）")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--part", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--save_pts", default=None, help="漂浮顆粒（世界座標、opacity、塊、幀）＋每塊隨機對照 => npz")
    ap.add_argument("--rand_pts", type=int, default=20000)
    a = ap.parse_args()
    import eval_official_test as eot

    run_dir, ref_dir = os.path.join("outputs/lab", a.run), os.path.join("outputs/lab", a.ref_run)
    names, cams = eot.load_test_cameras(a.test_dir, 1.2)
    idx = {n: i for i, n in enumerate(names)}

    def png(d, n):
        im = np.asarray(Image.open(os.path.join(d, "test_official", n.rsplit(".", 1)[0] + "_gt_vs_render.png")).convert("RGB"), np.float32) / 255.
        w = im.shape[1] // 2
        return im[:, :w], im[:, w:]

    if not a.frames:
        d = []
        for n in names:
            g, r = png(run_dir, n); _, q = png(ref_dir, n)
            d.append(10 * np.log10(((q - g) ** 2).mean() / max(((r - g) ** 2).mean(), 1e-12)))   # 本模型 − 參考（dB）
        d = np.array(d)
        a.frames = [names[i] for i in np.argsort(d)[:a.worst]]
        print(f"[幀] 本模型輸參考 >0.5 dB 的幀 {int((d < -0.5).sum())}／{len(d)}；取最差 {a.worst}：" +
              " ".join(f"{n}({d[idx[n]]:+.2f})" for n in a.frames), flush=True)

    dev = "cuda"
    bg = torch.zeros(3, device=dev)
    # 參考模型：只取深度與渲染
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        os.path.join(ref_dir, "checkpoints", "merged.ckpt"), dev, eval_mode=True)
    ref = {}
    with torch.no_grad():
        for n in a.frames:
            o = renderer(cams[idx[n]].to_device(dev), model, bg_color=bg)
            ref[n] = (o["render"].clamp(0, 1).cpu(), o["surf_depth"].reshape(o["render"].shape[1:]).cpu())
    print(f"[參考] {a.ref_run} N={model.get_xyz.shape[0]:,}", flush=True)
    del model, renderer; torch.cuda.empty_cache()

    seg = merge_segments(run_dir)
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        os.path.join(run_dir, "checkpoints", "merged.ckpt"), dev, eval_mode=True)
    N = model.get_xyz.shape[0]
    assert seg[-1][2] == N, f"merge.log 顆數 {seg[-1][2]:,} != 模型 {N:,}"
    owner = torch.empty(N, dtype=torch.long, device=dev)
    for b, s, e in seg:
        owner[s:e] = b
    print(f"[模型] {a.run} N={N:,}", flush=True)

    parts = torch.load(a.part)
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].float().to(dev)
    ids = parts["partition_coordinates"]["id"].numpy(); xys = parts["partition_coordinates"]["xy"].float().numpy()
    size = parts["scene_config"]["partition_size"].float().numpy()
    lo_t, hi_t = torch.zeros(16, 2, device=dev), torch.zeros(16, 2, device=dev)
    for b in range(16):
        k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0])
        lo_t[b] = torch.from_numpy(xys[k]); hi_t[b] = torch.from_numpy(xys[k] + size[k])
    xyz = model.get_xyz.detach()
    P = xyz @ Rp.T
    zg = float(torch.quantile(P[::97, 2].float(), 0.05))
    op_all = model.get_opacity.detach().reshape(-1)

    orig = {k: v for k, v in model.properties.items()}

    def render(cam, keep=None):
        model.properties = orig if keep is None else {k: v[keep] for k, v in orig.items()}
        with torch.no_grad():
            o = renderer(cam, model, bg_color=bg)
        model.properties = orig
        return o

    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    pts_keep, rand_keep = [], {}
    tot_cls = np.zeros(3); tot_fl = np.zeros(16)
    for n in a.frames:
        cam = cams[idx[n]].to_device(dev)
        g, _ = png(run_dir, n)
        G = torch.from_numpy(g).to(dev).permute(2, 0, 1)
        Q, Dref = ref[n][0].to(dev), ref[n][1].to(dev)
        o = render(cam); R = o["render"].clamp(0, 1); Dn = o["surf_depth"].reshape(R.shape[1:])
        e_new = (R - G).abs().mean(0); e_ref = (Q - G).abs().mean(0)
        bad = (e_new - e_ref) > a.thr
        nb = int(bad.sum())
        if nb == 0:
            print(f"{n}: 沒有壞區"); continue
        ratio = Dn[bad] / Dref[bad].clamp_min(1e-6)
        cls = np.array([float((ratio < 1 / a.tol).float().mean()), float(((ratio >= 1 / a.tol) & (ratio <= a.tol)).float().mean()),
                        float((ratio > a.tol).float().mean())])
        tot_cls += cls * nb
        gain = np.zeros(16)
        for b in range(16):
            Rb = render(cam, owner != b)["render"].clamp(0, 1)
            gain[b] = float(e_new[bad].mean() - (Rb - G).abs().mean(0)[bad].mean())
        # 漂浮顆粒：投影在壞區、比參考表面前面
        ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ cam.world_to_camera
        zc = ph[:, 2]
        u = (cam.fx * ph[:, 0] / zc.clamp_min(1e-6) + cam.cx).long(); v = (cam.fy * ph[:, 1] / zc.clamp_min(1e-6) + cam.cy).long()
        H, W = bad.shape
        ok = (zc > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (op_all > 0.1)
        fl = torch.zeros_like(ok)
        fl[ok] = bad[v[ok], u[ok]] & (zc[ok] < Dref[v[ok], u[ok]] / a.tol)
        cnt = torch.bincount(owner[fl], minlength=16).cpu().numpy()
        tot_fl += cnt
        print(f"{n}: 壞區 {100 * nb / bad.numel():.1f}% 像素（誤差 新 {float(e_new[bad].mean()):.3f} 參考 {float(e_ref[bad].mean()):.3f}）"
              f"  深度 前方/同深/後方 {100 * cls[0]:.0f}/{100 * cls[1]:.0f}/{100 * cls[2]:.0f}%"
              f"  拿掉後變好 " + " ".join(f"b{t}{gain[t]:+.3f}" for t in np.argsort(-gain)[:2]) +
              "  拿掉後變差 " + " ".join(f"b{t}{gain[t]:+.3f}" for t in np.argsort(gain)[:2]), flush=True)
        if fl.any():
            top = np.argsort(-cnt)[:3]
            s = []
            for t in top:
                if cnt[t] == 0:
                    continue
                m = fl & (owner == t)
                Pm = P[m]
                dx = torch.clamp(torch.maximum(lo_t[t] - Pm[:, :2], Pm[:, :2] - hi_t[t]), min=0).amax(1)
                h = Pm[:, 2] - zg
                s.append(f"b{t} {int(cnt[t]):,} 顆（高度中位 {float(h.median()):.3f}、p10 {float(h.quantile(.1)):.3f}；"
                         f"離盒中位 {float(dx.median()):.3f}；相機距離/參考深度 中位 {float((zc[m] / Dref[v[m], u[m]]).median()):.2f}）")
                if a.save_pts:
                    pts_keep.append((xyz[m].cpu().numpy(), op_all[m].cpu().numpy(), int(t), n))
                    if int(t) not in rand_keep:
                        r = torch.nonzero(owner == t).reshape(-1)
                        r = r[torch.randperm(len(r), device=dev)[:a.rand_pts]]
                        rand_keep[int(t)] = (xyz[r].cpu().numpy(), op_all[r].cpu().numpy())
            print("    漂浮顆粒（opacity>0.1、投影在壞區、在參考表面前方）：" + "；".join(s), flush=True)
        if a.save_dir:
            t0 = int(np.argmax(cnt)) if cnt.any() else int(np.argmax(gain))
            Rb = render(cam, owner != t0)["render"].clamp(0, 1)
            im = torch.cat([torch.cat([G, R], 2), torch.cat([Rb, Q], 2)], 1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((im * 255).astype(np.uint8)).resize((im.shape[1] // 2, im.shape[0] // 2)).save(
                os.path.join(a.save_dir, f"{n[:-4]}_minus_b{t0}.jpg"), quality=88)
    tc = tot_cls / max(tot_cls.sum(), 1)
    print(f"\n[加總] 壞區像素 深度 前方/同深/後方 {100 * tc[0]:.1f}/{100 * tc[1]:.1f}/{100 * tc[2]:.1f}%")
    print("[加總] 漂浮顆粒按塊：" + "  ".join(f"b{b} {int(tot_fl[b]):,}" for b in np.argsort(-tot_fl)[:8] if tot_fl[b] > 0))
    print(f"[地面] 顆粒 z 5 百分位 {zg:.3f}（分區座標）；四格圖＝左上 GT、右上本模型、左下拿掉漂浮最多的塊、右下參考")
    if a.save_pts and pts_keep:
        np.savez(a.save_pts,
                 xyz=np.concatenate([p[0] for p in pts_keep]), op=np.concatenate([p[1] for p in pts_keep]),
                 block=np.concatenate([np.full(len(p[0]), p[2]) for p in pts_keep]),
                 frame=np.concatenate([np.full(len(p[0]), p[3]) for p in pts_keep]),
                 rand_block=np.concatenate([np.full(len(v[0]), b) for b, v in rand_keep.items()]),
                 rand_xyz=np.concatenate([v[0] for v in rand_keep.values()]),
                 rand_op=np.concatenate([v[1] for v in rand_keep.values()]))
        print(f"[存] {a.save_pts}：漂浮顆粒 {sum(len(p[0]) for p in pts_keep):,}（塊 {sorted({p[2] for p in pts_keep})}）")


if __name__ == "__main__":
    main()
