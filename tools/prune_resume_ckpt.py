"""「剪完再短暫 fine-tune」（2026-10-01）：產生可接續訓練的剪枝 ckpt，以及對一批 ckpt 量 val 與精確 Load。

為什麼用 Lightning 接續（--ckpt_path）而不是 initialize_from：
  initialize_from 只載權重、global_step 從 0 起算（稽核清單第 4 項）=> 學習率、SH 階數、Adam 動量全部重來。
  接續則 step 從 60,000 繼續：學習率排程已在最終值（ExponentialDecay 的 t 夾在 [0,1]）、增生與 trim 在 30k 後
  本來就停了 => 60k->65k 就是乾淨的「低學習率 fine-tune」。剪枝必須讓 ckpt 內**所有第一維 = N 的張量**
  （模型參數、Adam exp_avg/exp_avg_sq、density controller 的逐顆狀態）用同一個遮罩剪，否則接續時形狀對不上。

make：python tools/prune_resume_ckpt.py make --ckpt <base> --out <目錄> --K 32 64
      產出 <目錄>/<臂>/checkpoints/<原檔名>：tk{K}（逐 tile 前 K 名）、op{K}（同 N、opacity 最高）、base（不剪，對照「多 5k 步」）
eval：python tools/prune_resume_ckpt.py eval <ckpt> [<ckpt> ...]
      每個 ckpt：N、val PSNR/SSIM/LPIPS（val⊂train，相對比較用）、精確 Load 中位（Σtiles）
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(ck_path, dev):
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck_path, device=dev, eval_mode=True)
    ck = torch.load(ck_path, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(ck_path)), global_rank=0)
    return model, renderer, ck, dp.get_outputs()


def prune_tree(obj, keep, N):
    """遞迴地把所有第一維 = N 的張量用 keep 遮罩剪掉；回傳 (新物件, 剪了幾個張量)。"""
    if torch.is_tensor(obj):
        if obj.dim() >= 1 and obj.shape[0] == N:
            return obj[keep.to(obj.device)].clone(), 1
        return obj, 0
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            out[k], m = prune_tree(v, keep, N); n += m
        return out, n
    if isinstance(obj, list):
        res = [prune_tree(v, keep, N) for v in obj]
        return [r[0] for r in res], sum(r[1] for r in res)
    if isinstance(obj, tuple):
        res = [prune_tree(v, keep, N) for v in obj]
        return tuple(r[0] for r in res), sum(r[1] for r in res)
    return obj, 0


def cmd_make(a):
    from internal.utils.tile_topk import update_best_rank, NEVER
    dev = torch.device("cuda")
    model, renderer, ck, outs = load(a.ckpt, dev)
    N = model.n_gaussians
    bg = torch.zeros(3, device=dev)
    best = torch.full((N,), NEVER, dtype=torch.int64, device=dev)
    with torch.no_grad():
        for i in range(len(outs.train_set)):
            cam = outs.train_set.cameras[i].to_device(dev)
            trans, cov, radii = renderer(cam, model, bg_color=bg, record_transmittance=True,
                                         record_coverage=True, record_radii=True)
            update_best_rank(best, trans * cov.float(), radii, model.get_xyz, cam)
    op = model.get_opacity.detach().reshape(-1)
    name = os.path.basename(a.ckpt)
    arms = {"base": torch.ones(N, dtype=torch.bool, device=dev)}
    for K in a.K:
        m = best < K
        n = int(m.sum())
        arms[f"tk{K}"] = m
        arms[f"op{K}"] = op >= torch.topk(op, n).values[-1]
    for arm, m in arms.items():
        new, cnt = prune_tree(ck, m.cpu(), N)
        d = os.path.join(a.out, arm, "checkpoints"); os.makedirs(d, exist_ok=True)
        torch.save(new, os.path.join(d, name))
        print(f"  {arm:6s} 保留 {int(m.sum()):,}/{N:,}（{100 * float(m.float().mean()):.1f}%）  剪了 {cnt} 個逐顆張量 -> {d}/{name}")


def cmd_eval(a):
    from internal.utils.ssim import ssim as ssim_fn
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    dev = torch.device("cuda")
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)
    gts = {}
    for ck_path in a.ckpts:
        model, renderer, ck, outs = load(ck_path, dev)
        val = outs.val_set
        bg = torch.zeros(3, device=dev)
        ps, ss, ls, loads = [], [], [], []
        with torch.no_grad():
            for i in range(len(val)):
                o = renderer(val.cameras[i].to_device(dev), model, bg_color=bg)
                img = o["render"].clamp(0, 1)
                if (val.image_paths[i], img.shape) not in gts:
                    pil = Image.open(val.image_paths[i]).convert("RGB")
                    if pil.size != (img.shape[2], img.shape[1]):
                        pil = pil.resize((img.shape[2], img.shape[1]), Image.LANCZOS)
                    gts[(val.image_paths[i], img.shape)] = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
                gt = gts[(val.image_paths[i], img.shape)]
                ps.append(float(-10 * torch.log10(((img - gt) ** 2).mean().clamp_min(1e-12))))
                ss.append(float(ssim_fn(img, gt)))
                ls.append(float(lp(img.unsqueeze(0), gt.unsqueeze(0))))
                if o.get("tiles") is not None:
                    loads.append(float(o["tiles"].sum()))
        ld = f"{np.median(loads) / 1e6:.3f}M" if loads else "n/a"
        print(f"{ck_path}\n   step {ck.get('global_step')}  N {model.n_gaussians:,}  精確Load中位 {ld}  "
              f"PSNR {np.mean(ps):.3f}  SSIM {np.mean(ss):.4f}  LPIPS {np.mean(ls):.4f}", flush=True)
        del model, renderer, ck
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("--ckpt", required=True); m.add_argument("--out", required=True)
    m.add_argument("--K", type=int, nargs="+", default=[32, 64])
    e = sp.add_parser("eval"); e.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    cmd_make(a) if a.cmd == "make" else cmd_eval(a)


if __name__ == "__main__":
    main()
