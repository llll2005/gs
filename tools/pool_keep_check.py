"""塊外池要多少顆才夠：離線檢查（2026-10-11 使用者：同 VRAM 的 Ap 池是否太多 => 去重版取代 Ap，先離線驗）

池在 init 後第 2 步建立，成員幾乎全是 dup init 的副本（4x4：b6 3 份、b12 4 份，logs/full44_init.tsv）=> 去重 ≈ 只留 1/K。
本工具拿已訓練好的塊外池 A ckpt（池在載入時凍結），池只留一部分再渲染訓練視角，量品質掉多少：
  區域   塊內內容 ＝可訓練集合單獨渲染 alpha>0.5 的像素；塊外背景 ＝其餘像素（池負責的地方）
  方案   random_f ＝固定種子隨機留 f（去重的近似）；opacity_f ＝留 opacity 最高的 f
⚠ 不重訓：池顆粒沒機會補位 => 這是「去重後背景變差多少」的上界；實際重訓（A 的池可訓練、有尺寸下限）只會更小。
⚠ 背景變差的代價不在背景本身（合併時會丟），而在它會把誤差推給邊界附近的塊內顆粒 => 判準看「塊外背景」掉多少：
   掉很少 => 去重安全；掉很多 => 去重會讓訓練時塊內顆粒去補背景。
用法：python tools/pool_keep_check.py --ckpt <poolA 60k ckpt> [--K 3] [--ncam 24]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", type=int, required=True, help="該塊的 dup 份數（去重 ≈ 留 1/K）")
    ap.add_argument("--ncam", type=int, default=24)
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    dev = torch.device("cuda")
    model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    pool = getattr(model, "_outside_pool", None)
    assert pool is not None and pool.act is not None, "ckpt 沒有塊外池"
    dmh = ck["datamodule_hyper_parameters"]; del ck
    train = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0).get_outputs().train_set
    idx = np.linspace(0, len(train) - 1, min(a.ncam, len(train))).round().astype(int)
    bg = torch.zeros(3, device=dev)
    full = {k: v.clone() for k, v in pool.act.items()}
    P = full["means"].shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(P, generator=g).to(dev)
    op_order = torch.argsort(full["opacities"].reshape(-1), descending=True, stable=True)
    fr = [1.0, 0.5, 1.0 / a.K, 1.0 / (2 * a.K)]
    variants = [("全部池", None)] + [(f"random_{f:.3f}", perm[:int(round(P * f))]) for f in fr[1:]] + \
               [(f"opacity_{1.0 / a.K:.3f}", op_order[:int(round(P / a.K))])]
    print(f"{a.ckpt}\n可訓練 {model.n_gaussians:,} 顆、池 {P:,} 顆、K={a.K}、訓練相機抽 {len(idx)} 台")

    # 先算每台的 GT 與塊內內容遮罩（可訓練集合單獨渲染）
    cams, gts, masks = [], [], []
    with torch.no_grad():
        model._outside_pool = None
        for i in idx:
            cam = train.cameras[int(i)].to_device(dev)
            o = renderer(cam, model, bg_color=bg)
            m = (o["rend_alpha"].reshape(o["render"].shape[1:]) > 0.5)
            pil = Image.open(train.image_paths[int(i)]).convert("RGB").resize((o["render"].shape[2], o["render"].shape[1]), Image.LANCZOS)
            gts.append(torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.)
            cams.append(cam); masks.append(m)
        model._outside_pool = pool
    frac_in = float(np.mean([float(m.float().mean()) for m in masks]))
    print(f"塊內內容佔訓練畫面平均 {100 * frac_in:.1f}%")

    ps = lambda se, n: -10 * np.log10(max(se / max(n, 1), 1e-12))
    res = []
    with torch.no_grad():
        for name, sel in variants:
            pool.act = full if sel is None else {k: v[sel] for k, v in full.items()}
            se_all = se_in = se_out = 0.0; n_all = n_in = n_out = 0
            for cam, gt, m in zip(cams, gts, masks):
                r = renderer(cam, model, bg_color=bg)["render"].clamp(0, 1)
                se = ((r - gt) ** 2).mean(0)
                se_all += float(se.sum()); n_all += se.numel()
                se_in += float(se[m].sum()); n_in += int(m.sum())
                se_out += float(se[~m].sum()); n_out += int((~m).sum())
            res.append((name, pool.n, ps(se_all, n_all), ps(se_in, n_in), ps(se_out, n_out)))
    pool.act = full
    b = res[0]
    print(f"{'方案':<16} {'池顆數':>10} {'全畫面':>8} {'塊內內容':>9} {'塊外背景':>9} {'Δ全':>7} {'Δ塊內':>7} {'Δ背景':>7}")
    for r in res:
        print(f"{r[0]:<16} {r[1]:>10,} {r[2]:>8.3f} {r[3]:>9.3f} {r[4]:>9.3f} {r[2] - b[2]:>+7.3f} {r[3] - b[3]:>+7.3f} {r[4] - b[4]:>+7.3f}")


if __name__ == "__main__":
    main()
