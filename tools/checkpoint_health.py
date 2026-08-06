"""One command to triage a checkpoint: python tools/checkpoint_health.py <ckpt>

Every check here exists because it caught something real, and several encode a mistake that cost
hours. Each line prints the healthy range and which direction is better, so a run can be killed at
the first checkpoint instead of three hours later.

Two tiers, because a training job is usually holding the card:

  CPU  reads the checkpoint only -- instant, safe to run against a live training job
  GPU  renders views -- needs the card, wait for the run to finish (or use --cpu-only)

Block id and grid come out of the checkpoint's own dataparser config, so the only argument is a
path. Override with --block/--block_dim for merged models, which carry no single block.

WHAT EACH CHECK IS FOR
  N                 the cap actually reached. Population shrinking below its init is the signature
                    of an opacity runaway, not of pruning working.
  opacity           REPORTED, NOT SCORED. Calibrating against eleven historical runs killed the
                    thresholds this tool first shipped with: our best-LPIPS model (b7, 24.47/0.283)
                    has a median of 0.142 with 18.5% under 0.05, and our best-PSNR b12 model
                    (reg=0.002, 23.84) has 0.109 and 31.2%. Both would have been flagged red. A
                    crushed opacity distribution is the normal consequence of opacity_reg > 0, not
                    a fault -- reg=0.007 sits at a median of 0.011 and still reaches 23.31. Only
                    the joint extreme (nothing above 0.9 AND over 85% below 0.05) matches a run
                    that actually failed.
  aspect            REPORTED, NOT SCORED, and the naive direction is wrong here. Needles are
                    supposed to signal an over-large position learning rate, but across those same
                    runs the best model has the HIGHEST aspect ratios among healthy ones (P50 3.51,
                    P99 43.1) while the run that died at 30k has the LOWEST (1.19, 2.5).
  surface distance  THE ONE CPU-SIDE CHECK THAT DISCRIMINATES. Median distance from a primitive to
                    the nearest triangulated point, in units of how far real points sit from each
                    other. Not a floater PERCENTAGE: the distribution is smooth, so any threshold
                    lands mid-continuum and the percentage measures the threshold. Calibrated:
                    real points 1x, the 28.9-PSNR model 6x, our b12 models 26-29x, the official
                    b7 rerun 528x (whose depth slope is 0.158, the worst we have measured).
  depth slope       rendered surface depth over true depth, against the COLMAP triangulated points.
                    1.0 is correct. 0.75 is the "blanket" -- a surface at rooftop height that never
                    resolves the streets. Measured 1.082 for our 5x5 blocks and 0.753-0.805 for the
                    3x3 ones, which is how we learned the blanket came from the layout, not the
                    recipe.
  texture ratio     rendered gradient energy over GT's, on building-heavy views only. b12 is about
                    half flat water, where a near-uniform render still scores 32-40 dB; averaging
                    over all views hid a 4x texture difference behind a flat LPIPS line for two
                    months. This is the only quality metric we have verified can separate a run
                    that died at 30k from one that finished.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = "data/matrix_city/aerial/train/block_all"


def band(value, good, warn, higher_better=True, fmt="{:.3f}"):
    """Render a value with a verdict against a healthy threshold and a warning threshold."""
    if value is None:
        return "  n/a"
    ok = value >= good if higher_better else value <= good
    mid = (value >= warn) if higher_better else (value <= warn)
    mark = "✅" if ok else ("🟡" if mid else "❌")
    return f"{fmt.format(value)} {mark}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--block", type=int, default=None, help="預設從 ckpt 自己讀")
    ap.add_argument("--block_dim", type=int, nargs=2, default=None, help="預設從 ckpt 自己讀")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--cpu-only", action="store_true", help="只跑讀檔的檢查；訓練跑著時用這個")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    g = lambda k: sd[f"gaussian_model.gaussians.{k}"]
    xyz = g("means").float()
    n = xyz.shape[0]
    op = torch.sigmoid(g("opacities").float()).numpy().ravel()
    sc = torch.exp(g("scales").float()).numpy()

    dp = ck.get("datamodule_hyper_parameters", {}).get("parser")
    blk = a.block if a.block is not None else getattr(dp, "block_id", None)
    bdim = a.block_dim or getattr(dp, "block_dim", None) or [5, 5]
    step = a.ckpt.split("step=")[-1].split(".")[0] if "step=" in a.ckpt else "?"

    print(f"檢查點 {a.ckpt}")
    print(f"  step={step}  block={blk if blk is not None else '（合併模型/全場景）'}  grid={bdim[0]}×{bdim[1]}")
    print(f"\n{'─' * 78}\n【CPU】只讀檔，訓練中也能跑\n")

    # ---- N -------------------------------------------------------------------
    print(f"  顆數 N              {n:>12,}      cap 是 VRAM 主槓桿；5×5@1.80M 曾峰值 5.63/6.1G")
    print(f"                                    ⚠ N 若低於 depth-init 的起始量（b12 約 1.21M）")
    print(f"                                      代表族群在萎縮，不是剪枝在生效")

    # ---- opacity -------------------------------------------------------------
    med = float(np.median(op))
    lo05, hi09 = float((op < 0.05).mean()), float((op > 0.9).mean())
    runaway = (hi09 < 0.01) and (lo05 > 0.85)
    print(f"\n  不透明度 中位        {med:>10.3f}        僅供參考，不評分")
    print(f"       o < 0.05 佔比   {lo05:>10.1%}        opacity_reg>0 時本來就會低")
    print(f"       o > 0.9  佔比   {hi09:>10.1%}        最佳 LPIPS 的模型只有 9.9%")
    print(f"                          {'❌ 疑似失控' if runaway else '✅ 未達失控條件'}"
          f"（>0.9 佔比 <1% 且 <0.05 佔比 >85% 才算）")
    print(f"                                    對照：最佳 LPIPS(b7) 0.142/18.5%/9.9%；")
    print(f"                                          reg=0.007 0.011/65.7%/3.7% 仍拿 23.31；")
    print(f"                                          已知壞(condense) 0.002/92.9%/0.0%")

    # ---- aspect --------------------------------------------------------------
    smax, smin = sc.max(1), np.maximum(sc.min(1), 1e-12)
    ar = smax / smin
    p50, p90, p99 = (float(np.percentile(ar, q)) for q in (50, 90, 99))
    print(f"\n  長寬比 P50/P90/P99  {p50:>6.2f} /{p90:>6.1f} /{p99:>6.1f}   僅供參考，不評分")
    print(f"                                    ⚠ 天真的方向（越低越好）在實測上是反的：")
    print(f"                                      最佳 LPIPS(b7) 3.51/43.1，已知壞(condense) 1.19/2.5")
    print(f"                                      只有 P99>70 值得注意（官方 b7 是 71.9、官方微調 87.1）")

    # ---- distance to the real surface ---------------------------------------
    try:
        from floater_label import surface_distance
        d, unit = surface_distance(xyz.numpy(), a.data)
        rel = float(np.median(d)) / unit
        mark = "✅" if rel < 10 else ("🟡" if rel < 20 else "❌")
        print(f"\n  離真實表面 中位     {rel:>10.1f}× {mark}      越低越好；健康 <10×")
        print(f"                                    （單位＝真實點彼此的中位距離 {unit:.4f}）")
        print(f"                                    校準：28.9 那次 6×｜我方 b12 26~29×｜")
        print(f"                                          官方 b7 忠實重跑 528×（其深度 slope 0.158）")
        print(f"                                    ⚠ 不報「floater 比例」：距離分布是連續的，")
        print(f"                                      任何門檻切出來的百分比只反映門檻本身")
        print(f"                                    ⚠ 稀疏點雲在水面／空白牆無點，該處會被高估")
    except Exception as e:
        print(f"\n  離真實表面          算不出來：{e}")

    if a.cpu_only:
        print(f"\n{'─' * 78}\n（--cpu-only：略過需要 GPU 的深度與紋理檢查）")
        return

    # ---- GPU checks ----------------------------------------------------------
    print(f"\n{'─' * 78}\n【GPU】需要顯卡，訓練中請改用 --cpu-only\n")
    from eval_official_test import load_test_cameras
    from internal.utils.colmap import read_images_binary, read_points3D_binary
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn

    P3 = read_points3D_binary(f"{a.data}/sparse/0/points3D.bin")
    xyz_all = np.array([p.xyz for p in P3.values()])
    lo, hi = np.percentile(xyz_all, 1, 0), np.percentile(xyz_all, 99, 0)
    pid = {k: v.xyz for k, v in P3.items()}
    by_name = {im.name: im for im in read_images_binary(f"{a.data}/sparse/0/images.bin").values()}
    names, cams = load_test_cameras(a.data, 1.2)
    files = sorted(f for f in os.listdir(f"{a.data}/images_1.2") if f.lower().endswith(".png"))

    if blk is not None:
        by, bx = blk // bdim[0], blk % bdim[0]
        want = {l.strip() for l in open(os.path.join(
            a.data, "partition", f"partitions-dim_{bdim[0]}_{bdim[1]}_visibility_0.08",
            f"{bx:03d}_{by:03d}.txt")) if l.strip()}
        pool = [i for i, nm in enumerate(names) if nm in want]
        from eval_official_test import block_bounds
        blo, bhi, _ = block_bounds(a.data, blk, bdim)
        lo, hi = np.maximum(lo, blo - 0.5), np.minimum(hi, bhi + 0.5)   # 只用該塊自己的內容評分
    else:
        pool = list(range(len(names)))
    pool = [i for i in pool if 0 <= int(names[i][:-4]) - 1 < len(files)]

    model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, "cuda", eval_mode=True)
    bg = torch.zeros(3, device="cuda")
    W, H = int(cams.width[0]), int(cams.height[0])

    def gt(i):
        from PIL import Image
        p = Image.open(f"{a.data}/images_1.2/{files[int(names[i][:-4]) - 1]}").convert("RGB")
        return torch.from_numpy(np.array(p.resize((W, H), Image.LANCZOS), np.uint8)) \
                    .float().permute(2, 0, 1).cuda() / 255.

    grad = lambda t: float((t[:, 1:, :] - t[:, :-1, :]).abs().mean()
                           + (t[:, :, 1:] - t[:, :, :-1]).abs().mean())
    probe = [pool[j] for j in np.linspace(0, len(pool) - 1, min(40, len(pool))).astype(int)]
    tex_rank = sorted((grad(gt(i)), i) for i in probe)
    building = [i for _, i in tex_rank[-a.views:]]     # 建築組：水面會把一切稀釋掉

    zr_all, zt_all, ratios, psnrs = [], [], [], []
    with torch.no_grad():
        for i in building:
            G = gt(i)
            out = rend(cams[i].to_device("cuda"), model, bg_color=bg)
            R = out["render"].clamp(0, 1)
            ratios.append(grad(R) / max(grad(G), 1e-9))
            psnrs.append(float(-10 * torch.log10(((R - G) ** 2).mean().clamp_min(1e-12))))
            im = by_name.get(names[i])
            if im is None:
                continue
            ids = im.point3D_ids
            X = np.array([pid[j] for j in ids[ids >= 0] if j in pid])
            if len(X) == 0:
                continue
            X = X[np.all((X > lo) & (X < hi), 1)]
            if len(X) < 100:
                continue
            cam = cams[i].to_device("cuda")
            dep, alpha = out["surf_depth"].squeeze(), out["rend_alpha"].squeeze()
            Xc = (cam.R @ torch.from_numpy(X).float().cuda().T).T + cam.T
            z = Xc[:, 2]
            m = z > 0.05
            u = cam.fx * Xc[m, 0] / z[m] + cam.cx
            v = cam.fy * Xc[m, 1] / z[m] + cam.cy
            k = (u >= 0) & (u < dep.shape[1]) & (v >= 0) & (v < dep.shape[0])
            vv, uu = v[k].long(), u[k].long()
            ok = (dep[vv, uu] > 1e-4) & (alpha[vv, uu] >= 0.5)
            zr_all.append(dep[vv, uu][ok].cpu().numpy())
            zt_all.append(z[m][k][ok].cpu().numpy())

    tr, pz = float(np.mean(ratios)), float(np.mean(psnrs))
    print(f"  建築視角 紋理比      {band(tr, 0.30, 0.15, True):>14}      越高越好；我方 5×5 約 0.24~0.39")
    print(f"                                    舊 kernel 只有 0.07~0.10、30k 就死掉的只有 0.011")
    print(f"  建築視角 PSNR       {band(pz, 19.0, 17.0, True, '{:.2f}'):>14}      越高越好；健康 >19 dB")

    if zr_all:
        zr, zt = np.concatenate(zr_all), np.concatenate(zt_all)
        slope = float((zr * zt).sum() / (zt * zt).sum())
        corr = float(np.corrcoef(zr, zt)[0, 1])
        dev = abs(slope - 1.0)
        mark = "✅" if dev < 0.08 else ("🟡" if dev < 0.15 else "❌")
        print(f"\n  深度 slope          {slope:>10.3f} {mark}      越接近 1.0 越好；我方 5×5 = 1.082")
        print(f"  深度 corr           {band(corr, 0.80, 0.50, True):>14}      越高越好；<0.5 時 slope 無意義")
        print(f"                                    0.75 左右＝毯子（表面浮在屋頂高度、街道解不出來）")
        print(f"                                    3×3 佈局實測 0.753~0.805，5×5 是 1.082")
    else:
        print("\n  深度 slope          n/a（該塊沒有足夠的可見稀疏點）")

    print(f"\n{'─' * 78}")
    print("VRAM 不在這裡：它是執行期性質，由 internal/callbacks.py 自動寫進 logs/quad_progress.log")
    print("（每次的 N / it-s / VRAM / PSNR，OOM 另寫 DIED 加例外與死亡位置）")


if __name__ == "__main__":
    main()
