#!/usr/bin/env python
"""合併模型的壞區塊歸屬：哪一塊的顆粒造成某些幀的大片漂浮物（10-23 遮罩全場景輸舊最佳的診斷）。

## 為什麼
遮罩臂 4x4 全場景（g44s30km25_best1maskdistrf）官方 741 幀 27.53 vs 舊最佳 full44_best 27.74；
損失集中在 b0/b4/b1 視角的少數幀，目視是大片暗色漂浮物蓋住馬路（水面、玻璃帷幕旁）。
要修就得先知道：漂浮物是哪一塊的顆粒、它們在哪裡（高度、離盒邊多遠）。

## 做法（逐幀，leave-one-block-out）
- 壞區＝本模型逐像素誤差比參考模型（舊最佳的存檔渲染）大 > --thr 的像素（|render-GT| 三通道平均）
- 對每一塊 b：拿掉 b 的顆粒重渲染 => 壞區誤差下降多少（正＝拿掉 b 會變好 => b 是元兇）
- 顆粒歸屬照合併順序：merge.log 的 checkpoint 清單（字典序）＋ number_of_gaussians 逐塊顆數
- 元兇塊：在壞區像素上，渲染深度比參考模型淺（＝擋在前面）的那批顆粒中心 => 報高度（相對地面）與離本塊盒的距離
輸出：逐幀表＋全部幀加總；--save_dir 存「GT／本模型／拿掉元兇塊／參考」四格圖。

用法（lab，GPU）：
  python tools/blob_owner.py --run g44s30km25_best1maskdistrf --ref_dir outputs/lab/full44_best/test_official --worst 12 --save_dir outputs/lab/blob_owner_1023
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
    ap.add_argument("--run", required=True, help="outputs/lab/<run>（要有 checkpoints/merged.ckpt 與 merge.log）")
    ap.add_argument("--ref_dir", required=True, help="參考模型的 eval_official_test --save_dir（左 GT／右渲染 PNG）")
    ap.add_argument("--frames", nargs="*", default=None, help="幀名（如 0038.png）；不給就用 --worst")
    ap.add_argument("--worst", type=int, default=12, help="依本模型存檔 PNG 對參考逐幀 PSNR 差，取最差幾幀")
    ap.add_argument("--thr", type=float, default=0.08, help="壞區：本模型誤差 − 參考誤差 > thr")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--part", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
    ap.add_argument("--save_dir", default=None)
    a = ap.parse_args()
    import eval_official_test as eot

    run_dir = os.path.join("outputs/lab", a.run)
    own_dir = os.path.join(run_dir, "test_official")
    names, cams = eot.load_test_cameras(a.test_dir, 1.2)
    idx = {n: i for i, n in enumerate(names)}

    def png(d, n):
        im = np.asarray(Image.open(os.path.join(d, n.rsplit(".", 1)[0] + "_gt_vs_render.png")).convert("RGB"), np.float32) / 255.
        w = im.shape[1] // 2
        return im[:, :w], im[:, w:]

    if not a.frames:
        d = []
        for n in names:
            g, r = png(own_dir, n); _, q = png(a.ref_dir, n)
            d.append(10 * np.log10(((q - g) ** 2).mean() / max(((r - g) ** 2).mean(), 1e-12)))   # 本模型 − 參考（dB）
        d = np.array(d)
        a.frames = [names[i] for i in np.argsort(d)[:a.worst]]
        print(f"[幀] 本模型輸參考 >0.5 dB 的幀 {int((d < -0.5).sum())}／{len(d)}；取最差 {a.worst}：" +
              " ".join(f"{n}({d[idx[n]]:+.2f})" for n in a.frames))

    seg = merge_segments(run_dir)
    dev = "cuda"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        os.path.join(run_dir, "checkpoints", "merged.ckpt"), dev, eval_mode=True)
    N = model.get_xyz.shape[0]
    assert seg[-1][2] == N, f"merge.log 顆數 {seg[-1][2]:,} != 模型 {N:,}"
    owner = torch.empty(N, dtype=torch.int16, device=dev)
    for b, s, e in seg:
        owner[s:e] = b
    print(f"[模型] N={N:,}；逐塊顆數 " + " ".join(f"b{b}:{(e - s) / 1e6:.2f}M" for b, s, e in sorted(seg)))

    parts = torch.load(a.part)
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].float().to(dev)
    ids = parts["partition_coordinates"]["id"].numpy(); xys = parts["partition_coordinates"]["xy"].float().numpy()
    size = parts["scene_config"]["partition_size"].float().numpy()
    box = {}
    for b in range(16):
        k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0])
        box[b] = (xys[k], xys[k] + size[k])
    P = model.get_xyz.detach() @ Rp.T
    zg = float(torch.quantile(P[::97, 2].float(), 0.05))   # 粗略地面（顆粒 z 5 百分位；僅供相對高度）

    orig = {k: v for k, v in model.properties.items()}
    bg = torch.zeros(3, device=dev)

    def render(cam, keep=None):
        model.properties = orig if keep is None else {k: v[keep] for k, v in orig.items()}
        with torch.no_grad():
            o = renderer(cam, model, bg_color=bg)
        model.properties = orig
        return o

    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    tot = np.zeros(16); tot_bad = 0
    for n in a.frames:
        cam = cams[idx[n]].to_device(dev)
        g, _ = png(own_dir, n); _, q = png(a.ref_dir, n)
        G = torch.from_numpy(g).to(dev).permute(2, 0, 1); Q = torch.from_numpy(q).to(dev).permute(2, 0, 1)
        o = render(cam); R = o["render"].clamp(0, 1)
        e_new = (R - G).abs().mean(0); e_ref = (Q - G).abs().mean(0)
        bad = (e_new - e_ref) > a.thr
        nb = int(bad.sum())
        if nb == 0:
            print(f"{n}: 沒有壞區"); continue
        gain = np.zeros(16)
        for b in range(16):
            Rb = render(cam, owner != b)["render"].clamp(0, 1)
            gain[b] = float(((Rb - G).abs().mean(0)[bad]).mean() - e_new[bad].mean()) * -1
        tot += gain * nb; tot_bad += nb
        top = np.argsort(-gain)[:3]
        b0 = int(top[0])
        # 元兇塊在壞區的顆粒：投影中心落在壞區、且在參考表面之前
        xyz = model.get_xyz.detach()
        sel = (owner == b0)
        pc = xyz[sel]
        w2c = cam.world_to_camera.T if hasattr(cam, "world_to_camera") else None
        msg = ""
        if w2c is not None:
            ph = torch.cat([pc, torch.ones_like(pc[:, :1])], 1) @ w2c.T
            zc = ph[:, 2]
            u = (cam.fx * ph[:, 0] / zc + cam.cx).long(); v = (cam.fy * ph[:, 1] / zc + cam.cy).long()
            H, W = bad.shape
            ok = (zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            hit = torch.zeros_like(ok); hit[ok] = bad[v[ok], u[ok]]
            op = model.get_opacity.detach()[sel].reshape(-1)
            hit &= op > 0.1
            if hit.any():
                Ph = P[sel][hit]
                lo, hi = box[b0]
                dx = torch.clamp(torch.maximum(torch.tensor(lo, device=dev) - Ph[:, :2], Ph[:, :2] - torch.tensor(hi, device=dev)), min=0).amax(1)
                h = Ph[:, 2] - zg
                msg = (f"  元兇 b{b0} 投影在壞區且 opacity>0.1 的顆粒 {int(hit.sum()):,}：高度(相對地面) 中位 {float(h.median()):.3f} "
                       f"p10 {float(h.quantile(.1)):.3f} p90 {float(h.quantile(.9)):.3f}；離盒邊（xy，盒內=0）中位 {float(dx.median()):.3f}"
                       f"；相機距離中位 {float(zc[hit].median()):.3f}")
        print(f"{n}: 壞區 {100 * nb / bad.numel():.1f}% 像素  壞區誤差 新 {float(e_new[bad].mean()):.3f} 參考 {float(e_ref[bad].mean()):.3f}；"
              "拿掉某塊後壞區誤差下降：" + "  ".join(f"b{t} {gain[t]:+.3f}" for t in top))
        if msg:
            print(msg)
        if a.save_dir:
            Rb = render(cam, owner != b0)["render"].clamp(0, 1)
            top_ = torch.cat([G, R], 2); bot = torch.cat([Rb, Q], 2)
            im = torch.cat([top_, bot], 1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((im * 255).astype(np.uint8)).resize((im.shape[1] // 2, im.shape[0] // 2)).save(
                os.path.join(a.save_dir, f"{n[:-4]}_owner_b{b0}.jpg"), quality=88)
    if tot_bad:
        t = tot / tot_bad
        print("\n[加總] 拿掉該塊後壞區誤差下降（壞區像素加權）：" + "  ".join(f"b{b} {t[b]:+.4f}" for b in np.argsort(-t)[:6]))
    print(f"[地面] 顆粒 z 5 百分位 {zg:.3f}（分區座標）")


if __name__ == "__main__":
    main()
