"""逐配對工作量的上限量測（2026-10-02，使用者同意的兩個離線量測；不訓練）。

需要有 audit_tiles 的光柵器（forward.cu 的 auditCUDA）：用 PYTHONPATH 指到另外裝的那份。

回答四件事（同一批訓練視角、同一個塊、多個訓練時點）：
  1 tile 精確剔除（StopThePop）能省多少：binning 的 (tile, 顆粒) 配對裡，有多少在整個 tile 內沒有任何像素 alpha >= 1/255
    —— 這些配對逐像素本來就會被跳過 => 剔除後渲染逐位元相同，但排序、載入、forward/backward 迴圈都少了
  2 遮擋尾巴（Taming §4.1 的 tile last-contributor 略過）：有多少配對在整個 tile 都提早結束之後才輪到
  3 誤差引導 tile 抽樣的上限：每個 tile 的絕對 L1 誤差低於門檻時，那些 tile 佔了多少「工作量」
    （逐像素實際迴圈次數，forward 與 backward 都近似這個數）—— 和 tile 數比例分開報，因為平坦 tile 便宜
  4 「完美過的 tile 會不會變差」：早期判為低誤差的 (視角, tile)，到後面的時點還低誤差的比例（一般訓練下）

用法：
  PYTHONPATH=<新光柵器> python tools/tile_budget_audit.py --ckpts <ckpt@15k> <ckpt@30k> <ckpt@60k> [--nview 60]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TILE = 16


def tile_reduce(x, H, W, op):
    """x: [H, W] -> [Ht*Wt]，邊緣不滿的 tile 只用存在的像素。"""
    Ht, Wt = math.ceil(H / TILE), math.ceil(W / TILE)
    pad = torch.full((Ht * TILE, Wt * TILE), float("nan"), device=x.device, dtype=torch.float64)
    pad[:H, :W] = x.double()
    t = pad.reshape(Ht, TILE, Wt, TILE).permute(0, 2, 1, 3).reshape(Ht * Wt, TILE * TILE)
    if op == "mean":
        return torch.nanmean(t, dim=1)
    if op == "sum":
        return torch.nansum(t, dim=1)
    if op == "max":
        return torch.nan_to_num(t, nan=-1e30).max(dim=1).values
    if op == "std":
        m = torch.nanmean(t, dim=1, keepdim=True)
        return torch.sqrt(torch.nanmean((t - m) ** 2, dim=1))
    raise ValueError(op)


def audit_ckpt(ck_path, nview, dev):
    import diff_trim_surfel_rasterization as R
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    assert "audit_tiles" in R.GaussianRasterizationSettings._fields, f"光柵器沒有 audit_tiles：{R.__file__}"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck_path, device=dev, eval_mode=True)
    ck = torch.load(ck_path, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(ck_path)),
                                   global_rank=0).get_outputs().train_set
    del ck
    idx = np.linspace(0, len(ts) - 1, nview).round().astype(int).tolist()
    bg = torch.zeros(3, device=dev)
    rec = {"err": [], "std": [], "F": [], "B": [], "blend": []}
    tot = {"pairs": 0, "useful": 0, "reached": 0, "proc": 0.0, "blend": 0.0, "B_check": 0.0}
    with torch.no_grad():
        for i in idx:
            cam = ts.cameras[i].to_device(dev)
            img = renderer(cam, model, bg_color=bg)["render"].clamp(0, 1)
            H, W = int(img.shape[1]), int(img.shape[2])
            pil = Image.open(ts.image_paths[i]).convert("RGB")
            if pil.size != (W, H):
                pil = pil.resize((W, H), Image.LANCZOS)
            gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
            err = (img - gt).abs().mean(0)
            luma = 0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2]
            reached, useful, _radii, others, tiles = renderer(cam, model, bg_color=bg, _audit=True)
            n_proc, rng, n_blend = others[0], others[1], others[2]
            rec["err"].append(tile_reduce(err, H, W, "mean").cpu())
            rec["std"].append(tile_reduce(luma, H, W, "std").cpu())
            rec["F"].append(tile_reduce(n_proc, H, W, "sum").cpu())
            rec["blend"].append(tile_reduce(n_blend, H, W, "sum").cpu())
            Bt = tile_reduce(rng, H, W, "max").cpu()
            rec["B"].append(Bt)
            tot["pairs"] += int(tiles.long().sum())
            tot["useful"] += int(useful.long().sum())
            tot["reached"] += float(reached.double().sum())
            tot["proc"] += float(n_proc.double().sum())
            tot["blend"] += float(n_blend.double().sum())
            tot["B_check"] += float(Bt.clamp_min(0).sum())
    out = {k: torch.stack(v).numpy() for k, v in rec.items()}       # [nview, ntile]
    return out, tot, model.n_gaussians, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--nview", type=int, default=60)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03, 0.05])
    ap.add_argument("--textured-std", type=float, default=0.10, help="GT 亮度標準差 >= 此值的 tile 算『有紋理』（絕對門檻）")
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = []
    for ck in a.ckpts:
        o, t, N, nv = audit_ckpt(ck, a.nview, dev)
        res.append((ck, o, t))
        name = os.path.basename(ck)
        print(f"\n════ {name}  N={N:,}  視角 {nv}")
        print(f"  binning 配對 {t['pairs']:,}（逐 tile 加總 {t['B_check']:,.0f}，兩者應相等）")
        print(f"  ① tile 精確剔除：整個 tile 內沒有任何像素 alpha>=1/255 的配對 {100 * (1 - t['useful'] / max(t['pairs'], 1)):.1f}%"
              f"  => 剔除後配對剩 {100 * t['useful'] / max(t['pairs'], 1):.1f}%（渲染逐位元不變）")
        print(f"  ② 遮擋尾巴：整個 tile 都提早結束後才輪到的配對 {100 * (1 - t['reached'] / max(t['pairs'], 1)):.1f}%"
              f"（forward 已會整塊跳過；backward 目前仍逐輪載入）")
        print(f"  逐像素迴圈 {t['proc']:,.0f} 次，其中真正混合 {t['blend']:,.0f} 次（{100 * t['blend'] / max(t['proc'], 1):.1f}%；"
              f"其餘是 alpha<1/255 或幾何不相交被跳過的迴圈）")
        err, std, F, B = o["err"], o["std"], o["F"], o["B"]
        valid = ~np.isnan(err)
        Ftot, Btot, ntile = F[valid].sum(), B[valid].clip(min=0).sum(), valid.sum()
        tex = valid & (std >= a.textured_std)
        print(f"  ③ 誤差引導上限（絕對 L1，tile {TILE}px；有紋理＝GT 亮度 std>={a.textured_std}，佔 tile {100 * tex.sum() / ntile:.1f}%、"
              f"佔工作量 {100 * F[tex].sum() / Ftot:.1f}%）")
        print(f"     {'門檻':>7} {'tile 比例':>9} {'工作量比例':>10} {'配對比例':>8} {'有紋理 tile 中的比例':>18}")
        for tau in a.tau:
            m = valid & (err < tau)
            print(f"     {tau:>7.3f} {100 * m.sum() / ntile:>8.1f}% {100 * F[m].sum() / Ftot:>9.1f}% "
                  f"{100 * B[m].clip(min=0).sum() / Btot:>7.1f}% {100 * (m & tex).sum() / max(tex.sum(), 1):>17.1f}%")
    if len(res) > 1:
        print("\n════ ④ 一般訓練下，早期低誤差的 (視角, tile) 到後期還低誤差的比例")
        for tau in a.tau:
            row = []
            for k in range(len(res) - 1):
                e0 = res[k][1]["err"]
                m0 = ~np.isnan(e0) & (e0 < tau)
                for k2 in range(k + 1, len(res)):
                    e1 = res[k2][1]["err"]
                    if e1.shape != e0.shape:
                        row.append("形狀不同"); continue
                    still = (m0 & (e1 < tau)).sum() / max(m0.sum(), 1)
                    worse = (m0 & (e1 > 1.5 * e0)).sum() / max(m0.sum(), 1)
                    row.append(f"{k}->{k2}: 仍低 {100 * still:.1f}%／誤差變 1.5 倍以上 {100 * worse:.1f}%")
            print(f"   門檻 {tau:.3f}：" + "  ".join(row))


if __name__ == "__main__":
    main()
