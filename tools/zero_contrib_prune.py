"""「只剪幾乎無貢獻的」安全嗎、省多少？（2026-10-09，使用者的 trim 改進構想的離線閘門）

構想：不再每 500 步剪固定 10%，改成只在幾個時點、用**門檻**剪掉「幾乎沒有貢獻」的顆粒，剪了不補。
本工具在已訓練的 ckpt 上做事後量測（不重訓）：
  v_i = max_{全部訓練視角} 每像素平均 T·α（＝現行週期 trim 的判準，contribution_accumulator reduce=max）
  對每個門檻 tau：刪掉 v_i <= tau 的顆粒（tau=0 ＝ 只刪貢獻恰好為 0 的），量
    刪掉的顆數比例、val 的精確 Load 中位（光柵器 Σtiles）、val PSNR/SSIM/LPIPS、塊內官方 held-out PSNR/SSIM/LPIPS
判讀：某個 tau 下「刪掉的 Load 比例明顯」而「品質幾乎不動（≤ 噪音：val ±0.1、held-out ±0.2）」
      => 只剪無貢獻者是安全且划算的；那個 tau 就是三次門檻的候選。

用法：python tools/zero_contrib_prune.py --ckpt <ckpt> [--tau 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    a = ap.parse_args()

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import eval_official_test as eot

    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0)
    outs = dp.get_outputs()
    train, val = outs.train_set, outs.val_set
    blk = getattr(dmh["parser"], "block_id", None)
    bdim = getattr(dmh["parser"], "block_dim", None) or [5, 5]
    N = model.n_gaussians
    bg = torch.zeros(3, device=dev)
    print(f"ckpt {a.ckpt}\n  N={N:,}  訓練視角 {len(train)}  val 視角 {len(val)}  block {blk} {list(bdim)}")

    # ── 掃描全部訓練視角：v（每像素平均 T·α 的最大值）──
    v = torch.zeros(N, device=dev)
    with torch.no_grad():
        for ci in range(len(train)):
            trans = renderer(train.cameras[ci].to_device(dev), model, bg_color=bg, record_transmittance=True)
            if isinstance(trans, tuple):
                trans = trans[0]
            v = torch.maximum(v, trans.reshape(-1))
    q = torch.quantile(v[torch.randperm(N, device=dev)[:min(N, 2_000_000)]], torch.tensor([.01, .05, .1, .25, .5], device=dev))
    print("  v 分位（1/5/10/25/50%）:", " ".join(f"{x:.2e}" for x in q.tolist()), f"｜v==0 {float((v == 0).float().mean()) * 100:.2f}%")

    # ── held-out 視角（我方塊 AABB 內的官方 test 幀）──
    names, tcams = eot.load_test_cameras(a.test_dir, 1.2)
    centres = np.array([(-tcams.R[i].numpy().T @ tcams.T[i].numpy()) for i in range(len(names))])
    if blk is not None:
        lo, hi, _ = eot.block_bounds(dmh["path"], int(blk), list(bdim), getattr(dmh["parser"], "content_threshold", 0.08))
        hsel = [i for i in range(len(names)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)]
    else:
        hsel = list(range(len(names)))
    print(f"  held-out 視角 {len(hsel)}")

    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)
    orig = {k: t for k, t in model.properties.items()}
    gts = {}

    def gt_of(key, path, like):
        if key not in gts:
            pil = Image.open(path).convert("RGB")
            if pil.size != (like.shape[2], like.shape[1]):
                pil = pil.resize((like.shape[2], like.shape[1]), Image.LANCZOS)
            gts[key] = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
        return gts[key]

    def score(cams_paths):
        ps, ss, ls, loads = [], [], [], []
        with torch.no_grad():
            for key, cam, path in cams_paths:
                o = renderer(cam.to_device(dev), model, bg_color=bg)
                img = o["render"].clamp(0, 1)
                gt = gt_of(key, path, img)
                ps.append(float(-10 * torch.log10(((img - gt) ** 2).mean().clamp_min(1e-12))))
                ss.append(float(ssim_fn(img, gt)))
                ls.append(float(lp(img.unsqueeze(0), gt.unsqueeze(0))))
                if o.get("tiles") is not None:
                    loads.append(float(o["tiles"].sum()))
        return np.mean(ps), np.mean(ss), np.mean(ls), (np.median(loads) if loads else float("nan"))

    vset = [(("v", i), val.cameras[i], val.image_paths[i]) for i in range(len(val))]
    hset = [(("h", i), tcams[i], os.path.join(a.test_dir, "images_1.2", names[i])) for i in hsel]
    print(f"\n{'門檻 tau':>9} {'刪除顆數':>10} {'刪除%':>6} {'val Load中位':>12} {'ΔLoad%':>7} {'val PSNR':>9} {'Δ':>6} {'SSIM':>6} {'LPIPS':>6}"
          f" {'held PSNR':>9} {'Δ':>6} {'SSIM':>6} {'LPIPS':>6}")
    base = None
    for tau in [None] + list(a.tau):
        keep = torch.ones(N, dtype=torch.bool, device=dev) if tau is None else (v > tau)
        model.properties = {k: t[keep] for k, t in orig.items()}
        vp, vs, vl, ld = score(vset)
        hp, hs, hl, _ = score(hset)
        model.properties = orig
        if base is None:
            base = (vp, hp, ld)
        n_del = N - int(keep.sum())
        tag = "不剪" if tau is None else f"{tau:.0e}" if tau > 0 else "==0"
        print(f"{tag:>9} {n_del:>10,} {n_del / N * 100:6.2f} {ld / 1e6:11.3f}M {(ld / base[2] - 1) * 100:+7.2f} {vp:9.3f} {vp - base[0]:+6.3f} {vs:.4f} {vl:.4f}"
              f" {hp:9.3f} {hp - base[1]:+6.3f} {hs:.4f} {hl:.4f}", flush=True)


if __name__ == "__main__":
    main()
