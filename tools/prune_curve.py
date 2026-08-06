"""已訓練模型按貢獻度排序後保留前 N 名（不重新擬合），量建築區紋理比。

⚠ 這測不到「小模型能不能表現場景」——剪枝沒有重新擬合，剩下的高斯 scale/opacity 是在別人
存在的前提下調出來的。唯一有效的宣稱是**部署側**（剪 30% 幾乎免費）。見 紀錄/研究總覽.md §10。
排序＝跨視角 top-K transmittance 平均，與 trim renderer 同一個統計量。
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader

GRAD = lambda t: float((t[:, 1:, :] - t[:, :-1, :]).abs().mean()
                       + (t[:, :, 1:] - t[:, :, :-1]).abs().mean())
PSNR = lambda a, b: float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--rank_views", type=int, default=24, help="用幾個視角算貢獻度排序")
    ap.add_argument("--score_views", type=int, default=8, help="建築組取幾張評分")
    ap.add_argument("--targets", type=int, nargs="+",
                    default=[80_000, 160_000, 320_000, 640_000])
    a = ap.parse_args()

    dev = "cuda"
    names, cams = load_test_cameras(a.data, 1.2)
    files = sorted(f for f in os.listdir(f"{a.data}/images_1.2") if f.lower().endswith(".png"))
    W, H = int(cams.width[0]), int(cams.height[0])
    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        a.data, "partition", f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    idx = [i for i, n in enumerate(names) if n in want and 0 <= int(n[:-4]) - 1 < len(files)]

    def gt(i):
        p = Image.open(f"{a.data}/images_1.2/{files[int(names[i][:-4]) - 1]}").convert("RGB")
        return torch.from_numpy(np.array(p.resize((W, H), Image.LANCZOS), np.uint8)) \
                    .float().permute(2, 0, 1).to(dev) / 255.

    probe = idx[::max(1, len(idx) // 40)]
    tex = sorted((GRAD(gt(i)), i) for i in probe)
    build = [i for _, i in tex[-a.score_views:]]
    gts = {i: gt(i) for i in build}
    rank_views = [idx[j] for j in np.linspace(0, len(idx) - 1, min(a.rank_views, len(idx))).astype(int)]
    print(f"[視角] 排序用 {len(rank_views)} 台，評分用建築組 {len(build)} 張"
          f"（GT 梯度 {tex[-a.score_views][0]:.4f}~{tex[-1][0]:.4f}）")

    model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, dev, eval_mode=True)
    n0 = model.get_xyz.shape[0]
    bg = torch.zeros(3, device=dev)
    print(f"[模型] {a.name or os.path.basename(a.ckpt)}  N={n0:,}")

    # contribution = mean of the top-K transmittances over views (same statistic as the trim)
    K = 5
    top = [torch.zeros(n0, device=dev) for _ in range(K)]
    with torch.no_grad():
        for j, i in enumerate(rank_views):
            o = rend(cams[i].to_device(dev), model, bg_color=bg, record_transmittance=True)
            t = (o if torch.is_tensor(o) else o["transmittance"]).float()
            for k in range(K):                       # keep the K largest per primitive
                bigger = t > top[k]
                if not bigger.any():
                    break
                for m in range(K - 1, k, -1):
                    top[m][bigger] = top[m - 1][bigger]
                top[k][bigger] = t[bigger]
            if (j + 1) % 12 == 0:
                print(f"  ...排序 {j + 1}/{len(rank_views)}")
    contrib = torch.stack(top).mean(0)
    order = torch.argsort(contrib, descending=True)

    saved = {k: v.detach().clone() for k, v in model.gaussians.items()}

    def score():
        P, R = [], []
        with torch.no_grad():
            for i in build:
                o = rend(cams[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
                P.append(PSNR(o, gts[i])); R.append(GRAD(o) / max(GRAD(gts[i]), 1e-9))
        return float(np.mean(P)), float(np.mean(R))

    p0, r0 = score()
    print(f"\n{'保留顆數':>12}{'佔原本':>9}{'建築PSNR':>11}{'建築紋理比':>12}{'紋理比損失':>12}")
    print(f"{n0:>12,}{'100.0%':>9}{p0:>11.2f}{r0:>12.3f}{'—':>12}")
    for t in sorted(a.targets, reverse=True):
        if t >= n0:
            continue
        keep = order[:t]
        for k, v in saved.items():
            model.gaussians[k] = torch.nn.Parameter(v[keep], requires_grad=False)
        p, r = score()
        print(f"{t:>12,}{100 * t / n0:>8.1f}%{p:>11.2f}{r:>12.3f}{100 * (r - r0) / r0:>+11.1f}%")

    print(f"\n  ★ 起始 trim 的存活容量 ≈ 362,000（餵 1.32M 得 360,817、餵 0.99M 得 363,190）")
    print(f"  若品質撐到 ~36 萬才崩 ⇒ 支持「可見表面容量」假說；若一路平滑下滑 ⇒ 不支持")
    print(f"  ⚠ 貢獻度是對【訓練視角】的效用；novel view 未測（塵埃剔除已知與視距相關）")


if __name__ == "__main__":
    main()
