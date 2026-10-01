"""逐 tile 前 K 名的事後剪枝曲線（2026-10-01，使用者提的「每個 tile 只留最好的幾顆」＝背包的 tile 局部版）。

問題：如果每個 (視角, tile) 只保留貢獻前 K 名的顆粒（至少在某處進過前 K 的才留），
      品質掉多少、渲染成本（Load）降多少？和「同顆數下的其他剪法」比誰好？
      —— 這是不訓練、在已訓練模型上做的事後量測（post-hoc），用來判斷值不值得做成訓練期機制。

做法（不改光柵器 kernel 的近似）：
  kernel 只回傳「每顆在一個視角裡的總貢獻」（record_transmittance：平均 T·α × 覆蓋像素數），
  沒有逐 tile 的值 ⇒ 把該顆在該視角的總貢獻**平均分到它覆蓋的 tile**（外接盒 = 投影中心 ± radii，
  與光柵器 getRect 同規則），在每個 tile 內排名；每顆記下它在所有 (視角, tile) 拿到的**最好名次**。
  ⇒ 一趟掃描就得到所有 K：保留集合 = {最好名次 < K}。
  ⚠ 近似：同一顆在不同 tile 的真實貢獻不同（邊緣淡、中心濃），平均分攤會高估邊緣 tile 的分數。
    外接盒用對稱半徑（conic 開時光柵器實際用的是不對稱盒，這裡偏大）。

對照（同顆數 N_K）：opacity 由高到低、隨機，以及三種貢獻判準：
  總貢獻最大   max_v (每像素平均 T·α × 覆蓋像素)  —— ⚠ 09-30 版圖 3 誤標為「現行 trim 判準」，它偏好又大又亮的顆粒
  現行 trim v  max_v (每像素平均 T·α)              —— renderer 週期 trim 的預設判準（contribution_accumulator reduce=max）
  v/c          v ÷ Σ_v 覆蓋像素                    —— trim_by_value_per_cost（α=1）
評分：val 視角（val ⊂ train，只用於同模型不同剪法之間的相對比較）PSNR/SSIM/LPIPS，
      與同批視角的精確 Load 中位（Σtiles，光柵器回傳）。

用法：python tools/tile_topk_prune.py --ckpt <ckpt> [--K 1 2 4 8 16 32 64] [--max-cam 0]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--max-cam", type=int, default=0, help="掃描用的訓練視角上限（0＝全部）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0)
    outs = dp.get_outputs()
    train, val = outs.train_set, outs.val_set
    N = model.n_gaussians
    bg = torch.zeros(3, device=dev)
    print(f"ckpt {os.path.basename(a.ckpt)}  N={N:,}  訓練視角 {len(train)}  val 視角 {len(val)}")

    # ── opacity 分布（使用者問：讓不透明度收斂得更高，能否減少顆數）──
    op = model.get_opacity.detach().reshape(-1).float()
    qs = torch.quantile(op[torch.randperm(N, device=dev)[:min(N, 2_000_000)]],
                        torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev))
    print("opacity 分位（10/25/50/75/90%）:", " ".join(f"{q:.3f}" for q in qs.tolist()),
          f"｜o<0.05 {float((op < 0.05).float().mean()) * 100:.1f}%  o>0.9 {float((op > 0.9).float().mean()) * 100:.1f}%")

    # ── 掃描：每顆的「最好 tile 名次」與「單視角最大貢獻」──
    idxs = list(range(len(train)))
    if a.max_cam and a.max_cam < len(idxs):
        idxs = idxs[::max(1, len(idxs) // a.max_cam)][:a.max_cam]
    best_rank = torch.full((N,), 1 << 30, dtype=torch.int64, device=dev)
    max_contrib = torch.zeros(N, device=dev)
    v_trim = torch.zeros(N, device=dev)          # 現行 trim 的 v：各視角每像素平均貢獻取最大
    cost_sum = torch.zeros(N, device=dev)        # c：各視角覆蓋像素數加總（renderer 的 cost_acc）
    B = 16
    with torch.no_grad():
        for vi, ci in enumerate(idxs):
            cam = train.cameras[ci].to_device(dev)
            W, H = int(cam.width), int(cam.height)
            gx, gy = (W + B - 1) // B, (H + B - 1) // B
            out = renderer(cam, model, bg_color=bg)
            radii = out["radii"].reshape(-1)
            trans, cov = renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True)
            C = (trans * cov.to(trans.dtype)).reshape(-1)                 # 該視角總貢獻 Σ T·α
            max_contrib = torch.maximum(max_contrib, C)
            v_trim = torch.maximum(v_trim, trans.reshape(-1))
            cost_sum += cov.reshape(-1).float()
            ph = torch.cat([model.get_xyz, torch.ones(N, 1, device=dev)], 1) @ cam.full_projection
            w = ph[:, 3].clamp_min(1e-7)
            px = ((ph[:, 0] / w + 1.0) * W - 1.0) * 0.5
            py = ((ph[:, 1] / w + 1.0) * H - 1.0) * 0.5
            vis = (radii > 0) & (C > 0)
            g = torch.nonzero(vis).squeeze(1)
            if g.numel() == 0:
                continue
            r = radii[g].float()
            x0 = ((px[g] - r) / B).floor().clamp(0, gx).long(); x1 = ((px[g] + r + B - 1) / B).floor().clamp(0, gx).long()
            y0 = ((py[g] - r) / B).floor().clamp(0, gy).long(); y1 = ((py[g] + r + B - 1) / B).floor().clamp(0, gy).long()
            nx, ny = (x1 - x0), (y1 - y0)
            nt = nx * ny
            keep = nt > 0
            g, x0, y0, nx, nt = g[keep], x0[keep], y0[keep], nx[keep], nt[keep]
            score = C[g] / nt.float()
            # 展開成 (顆, tile) 對
            rep = torch.repeat_interleave(torch.arange(g.numel(), device=dev), nt)
            start = torch.cumsum(nt, 0) - nt
            local = torch.arange(rep.numel(), device=dev) - start[rep]
            tx = x0[rep] + local % nx[rep]
            ty = y0[rep] + local // nx[rep]
            tile = ty * gx + tx
            s = score[rep]
            o1 = torch.argsort(-s, stable=True)
            o2 = torch.argsort(tile[o1], stable=True)
            order = o1[o2]
            t_sorted = tile[order]
            _, counts = torch.unique_consecutive(t_sorted, return_counts=True)
            seg_start = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
            rank = torch.arange(order.numel(), device=dev) - seg_start
            best_rank.scatter_reduce_(0, g[rep[order]], rank, reduce="amin")
            if (vi + 1) % 50 == 0:
                print(f"  掃描 {vi + 1}/{len(idxs)}（本視角 (顆,tile) 對 {rep.numel():,}）", flush=True)

    # ── 評分工具 ──
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)
    orig = {k: v for k, v in model.properties.items()}

    def load_gt(i, like):
        pil = Image.open(val.image_paths[i]).convert("RGB")
        if pil.size != (like.shape[2], like.shape[1]):
            pil = pil.resize((like.shape[2], like.shape[1]), Image.LANCZOS)
        return torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.

    gts = {}

    def evaluate(mask, tag):
        model.properties = {k: v[mask] for k, v in orig.items()}
        ps, ss, ls, loads = [], [], [], []
        with torch.no_grad():
            for i in range(len(val)):
                cam = val.cameras[i].to_device(dev)
                o = renderer(cam, model, bg_color=bg)
                img = o["render"].clamp(0, 1)
                if i not in gts:
                    gts[i] = load_gt(i, img)
                gt = gts[i]
                ps.append(float(-10 * torch.log10(((img - gt) ** 2).mean().clamp_min(1e-12))))
                ss.append(float(ssim_fn(img, gt)))
                ls.append(float(lp(img.unsqueeze(0), gt.unsqueeze(0))))
                if o.get("tiles") is not None:
                    loads.append(float(o["tiles"].sum()))
        model.properties = orig
        n = int(mask.sum())
        ld = f"{np.median(loads) / 1e6:6.2f}M" if loads else "   n/a"
        print(f"  {tag:<22} N {n / 1e6:6.3f}M ({n / N * 100:5.1f}%)  精確Load中位 {ld}  "
              f"PSNR {np.mean(ps):6.3f}  SSIM {np.mean(ss):.4f}  LPIPS {np.mean(ls):.4f}", flush=True)

    print("\n═══ val 評分（val⊂train；只比同模型的不同剪法）═══")
    evaluate(torch.ones(N, dtype=torch.bool, device=dev), "不剪")
    g = torch.Generator(device=dev).manual_seed(a.seed)
    rnd = torch.rand(N, device=dev, generator=g)
    for K in a.K:
        m = best_rank < K
        n = int(m.sum())
        if n == 0 or n == N:
            print(f"  K={K}: 保留 {n:,}（{'全部' if n == N else '無'}），略過"); continue
        print(f"── K={K}：保留「至少在某個 (視角,tile) 進前 {K} 名」的 {n:,} 顆 ──")
        evaluate(m, f"tile 前{K}名")
        evaluate(op >= torch.topk(op, n).values[-1], "同 N：opacity 最高")
        evaluate(max_contrib >= torch.topk(max_contrib, n).values[-1], "同 N：總貢獻最大")
        evaluate(v_trim >= torch.topk(v_trim, n).values[-1], "同 N：現行 trim v")
        vpc = v_trim / torch.clamp_min(cost_sum, 1.0)
        evaluate(vpc >= torch.topk(vpc, n).values[-1], "同 N：v/c")
        evaluate(rnd >= torch.topk(rnd, n).values[-1], "同 N：隨機")


if __name__ == "__main__":
    main()
