#!/usr/bin/env python
"""O0 oracle：**凍結拓撲**，只讓失敗 tile 的影響集重新優化 —— 現有 basis 能不能被救活？

## 這回答什麼

§11.77 已證**表徵容量足夠**（20 個 2D 高斯 -> corr 0.92，而模型用 8,200 個只有 0.29）；
§11.80 已證**梯度盲區**（每單位殘差的訊號少 4.6 倍）且**先於失敗**（step 1,499 就存在）。
剩下的分岔是：

```
O0 能救回來  => 現有 basis 夠，是**優化到不了** => 該改優化（LR/排程/觸發訊號）
O0 救不回來  => 現有 basis 不夠 => 才需要 O1（加 basis / 改 basis）
```
**必須先測 O0 再測 O1** —— 順序反了因果鏈就不乾淨（外部諮詢第二輪，GPT）。

## 設計（採納外部指出的四個要點）

1. **影響集要按「光線貢獻」取，不能按 primitive 中心落在哪個 tile**
   （footprint 會跨 tile；兩邊都指出這點）。本工具用**投影半徑覆蓋**近似：
   凡投影圓與目標 tile 相交者納入。
2. **對照組**：同流程跑成功 tile。`ΔQ_失敗 >> ΔQ_成功` 才有意義。
3. **鄰域檢查**：同時量目標 tile 外圈，避免只是把誤差搬到旁邊。
4. **措辭限制**：結論只能說「**在此局部預算下無法恢復**」，不能說「無法表示」。

⚠ 本工具**不改變 topology**（不 split/clone/prune），只更新
`means / scales / rotations / opacities / SH`，且**只對影響集**（其餘梯度歸零）。

用法: python tools/oracle_o0.py agd2_b12 sched30_b12 --blk 12 --steps 300
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def tile_corr(gt, rd, ty, tx, T):
    a = gt[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T].ravel()
    b = rd[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T].ravel()
    d = a.std() * b.std()
    return float(((a - a.mean()) * (b - b.mean())).mean() / d) if d > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--model-run", default=None)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-tile", type=int, default=8, help="每組取幾個 tile")
    ap.add_argument("--lambda-dssim", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="★ 第二臂用：改成訓練後期的實際 means_lr，回答『是不是 LR 太低』。"
                         "1e-3 是 oracle 臂（比訓練後期高約兩個數量級）")
    ap.add_argument("--multi-view", type=int, default=0,
                    help="★★ 決定性：用 K 台相機**一起**優化同一批影響集（整張 loss，＝訓練的形狀）。"
                         "仍達 ~0.85 => 多視角一致解存在；卡在低點 => 衝突為真。")
    ap.add_argument("--split", type=int, default=1,
                    help="★★ 因果測試：把影響集的每顆**分裂成 K 顆**再優化（拓撲仍凍結，"
                         "只是起點不同）。§11.98 的假說是『大粒子跨 tile/跨視角共享 => 無法特化』，"
                         "它預測**分裂會降低留出視角的代價**。\n"
                         "  對照（K=1）：Δcorr +0.401，留出視角 **-0.675 dB**\n"
                         "  K=4 若 Δ 上升且留出代價明顯縮小 => 分裂確實解耦 => 支持按足跡增生\n"
                         "  留出代價不變 => 「耦合」只是相關，該區本來就難")
    ap.add_argument("--split-jitter", type=float, default=1.0,
                    help="子代位置抖動的幅度（單位＝子代自身尺度）。**0 = 共位**。\n"
                         "⚠ MCMC Eq.9 的推導假設 N 個子代**共位**（本專案 `long_axis_spread` "
                         "的 docstring 明文記過：抖動會 break Eq.9）。第一次跑用了 1.0，"
                         "結果 `corr 前` 從 0.270 掉到 0.140 —— 分裂在優化前就先弄壞畫面，"
                         "使比較被起點汙染。乾淨的因果測試要用 **0**。")
    ap.add_argument("--with-reg", action="store_true",
                    help="★ 把訓練的兩個 L1 正則項加進 O0 的 loss（權重從 ckpt 的 "
                         "hyper_parameters 讀真值，不是猜的）。O0 原本只優化光度，"
                         "而**訓練的終點是「光度 + 正則」的最優** —— 若 Δ 因此塌掉，"
                         "正則就是元兇，且是「全域統一定價傷到異質區域」的直接證據。")
    ap.add_argument("--freeze", default="",
                    help="★ 逗號分隔的參數名，優化時凍結（例：shs_dc）。"
                         "第二臂顯示 means 用真實 LR 也能恢復、scales/opacity 幾乎沒動 "
                         "=> 做事的是顏色。凍住它就能驗證。")
    ap.add_argument("--verify-cams", type=int, default=6,
                    help="★ 多視角覆核：優化完後在**另外幾台**相機上量整張 PSNR 的變化。"
                         "工具自己標註的最大風險就是單視角過擬合 —— 若 +0.659 是靠犧牲"
                         "其他視角換來的，『basis 足夠』就不成立。只**多量**、不改優化，"
                         "所以與未開此旗標的跑次仍可比。")
    ap.add_argument("--means-lr", type=float, default=None,
                    help="★ 只把 **means** 換成訓練末期的真實值（config: 6.4e-5 -> **6.4e-7**，"
                         "oracle 的 1e-3 是它的 1,563 倍）。其餘參數維持 --lr，"
                         "否則 opacity/SH 會因為錯的理由失敗，混淆歸因。")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    mr = args.model_run or args.runs[0]
    ck = sorted(glob.glob(f"outputs/{mr}/**/*step=60000.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {mr} 的 60k ckpt")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                  output_path=os.path.dirname(os.path.dirname(ck[0])),
                                  global_rank=0)
    vset = dp.get_outputs().val_set
    _mc = ckpt["hyper_parameters"].get("metric") if args.with_reg else None
    if args.with_reg:
        _w_op = float(getattr(_mc, "opacity_reg", 0.0))
        _w_sc = float(getattr(_mc, "scale_reg", 0.0))
        _immune = float(getattr(_mc, "immune_opacity_threshold", 1.0))
        print(f"  [with-reg] opacity_reg={_w_op:g}（immune>{_immune:g}）  scale_reg={_w_sc:g}"
              f"   —— 與 mcmc_citygsv2_metrics.py:57-71 逐字相同")

    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    fs = sorted(f for f in os.listdir(dirs[args.runs[0]]) if f.endswith(".png"))
    masks = {}
    for f in fs:
        outs = {r: per_image(os.path.join(dirs[r], f), args.tile, args.contrast_q)
                for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, *_z, nx, ny = outs[args.runs[0]]
        ms = np.stack([outs[r][1] <= args.r_min for r in args.runs])
        name = f[:-4] if f.endswith(".png.png") else f
        masks[name] = (keep, ms.all(0), ~ms.any(0), nx, ny)

    T = args.tile
    bg = torch.zeros((3,), device=dev)
    results = {"失敗": [], "成功": []}

    for lab, sel_idx in (("失敗", 1), ("成功", 2)):
        done = 0
        for i in range(len(vset)):
            if done >= args.n_tile:
                break
            name, img_path, _m, cam, _e = vset[i]
            name = os.path.basename(str(name))
            if name not in masks:
                continue
            keep, allb, neverb, nx, ny = masks[name]
            sel = keep[allb if sel_idx == 1 else neverb]
            if len(sel) == 0:
                continue

            # 每張圖只取一個 tile，避免同一張的 tile 彼此干擾
            t = int(sel[len(sel) // 2])
            ty, tx = t // nx, t % nx

            # 每個 tile 都從**乾淨的 ckpt** 重載（前一個 tile 的優化不得污染下一個）
            model, renderer, _ = GaussianModelLoader.\
                initialize_model_and_renderer_from_checkpoint_file(
                    ck[0], device=dev, eval_mode=False, pre_activate=False)
            camd = cam.to_device(dev)
            W, H = int(camd.width), int(camd.height)
            gt_np = np.asarray(Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR),
                               np.float32) / 255.
            gt = torch.from_numpy(gt_np).permute(2, 0, 1).to(dev)

            # ---- 影響集：投影圓與目標 tile 相交（不是「中心落在 tile 內」）----
            with torch.no_grad():
                R = camd.R.to(dev).float() if torch.is_tensor(camd.R) else torch.tensor(camd.R, device=dev)
                Tv = camd.T.to(dev).float() if torch.is_tensor(camd.T) else torch.tensor(camd.T, device=dev)
                pc = model.get_xyz @ R.T + Tv
                z = pc[:, 2].clamp_min(0.2)
                fx = float(camd.fx)
                u = fx * pc[:, 0] / z + W / 2
                v = fx * pc[:, 1] / z + H / 2
                rad = 3.0 * fx * model.get_scales().max(dim=1).values / z
                cx, cy = (tx + 0.5) * T, (ty + 0.5) * T
                infl = ((u - cx).abs() < T / 2 + rad) & ((v - cy).abs() < T / 2 + rad) & (pc[:, 2] > 0.2)
            n_infl = int(infl.sum())

            # ★★ 強制分裂（§11.98 的因果測試）。拓撲在優化期間仍凍結，只是**起點**不同。
            # 用 MCMC 的 Eq.9 保 opacity：o_child = 1 - (1-o_parent)^(1/K)，
            # scale /= sqrt(K)（保總面積），位置在親代尺度內抖動。
            # ⚠ 近似：抖動用各向同性（未按 2DGS 的切平面 rotation 取向）=> 這是探針不是實作。
            if args.split > 1:
                K = args.split
                with torch.no_grad():
                    sel = torch.nonzero(infl, as_tuple=True)[0]
                    keepm = ~infl
                    child = {}
                    for _k, _v in model.gaussians.items():
                        child[_k] = _v.data[sel].repeat_interleave(K, dim=0).clone()
                    # ★ 用 MCMC 真正的 Eq.9（`gsplat.relocation.compute_relocation`），
                    #   不用我原本的 `scale/sqrt(K)` 近似 —— 那不是論文的公式。
                    from gsplat.relocation import compute_relocation as _cr
                    # binoms 表：與 `mcmc_density_controller.py:114-121` 同構
                    _NM = max(K + 1, 8)
                    _binoms = torch.zeros((_NM, _NM), dtype=torch.float, device=dev)
                    for _n in range(_NM):
                        for _k in range(_n + 1):
                            _binoms[_n, _k] = math.comb(_n, _k)
                    _o_par = model.get_opacities()[sel, 0]
                    _s_par = model.get_scales()[sel]
                    # ⚠ gsplat 的 compute_relocation 要求 scales 是 [N,3]，而 2DGS 是 [N,2]。
                    #   補一個 0 的第三軸再截回去 —— 與
                    #   `mcmc_2dgs_density_controller.compute_relocation` 逐字相同
                    #   （coeff 只依賴 opacity，所以對真實的兩軸是精確的）。
                    _d = _s_par.shape[-1]
                    if _d < 3:
                        _s_par = torch.cat(
                            [_s_par, torch.zeros((_s_par.shape[0], 3 - _d),
                                                 dtype=_s_par.dtype, device=_s_par.device)], -1)
                    _Nrep = torch.full(_o_par.shape, K, dtype=torch.long, device=dev)
                    _no, _ns = _cr(_o_par, _s_par, _Nrep, _binoms)
                    _ns = _ns[:, :_d]
                    _no = _no.clamp(min=0.005, max=1.0 - torch.finfo(torch.float32).eps)
                    child["opacities"] = torch.log(_no / (1 - _no)).unsqueeze(-1) \
                        .repeat_interleave(K, dim=0)
                    child["scales"] = torch.log(_ns.clamp_min(1e-12)).repeat_interleave(K, dim=0)
                    if args.split_jitter > 0:
                        _sc = torch.exp(child["scales"]).max(dim=1, keepdim=True).values
                        child["means"] = child["means"] + \
                            (torch.rand_like(child["means"]) - 0.5) * _sc * args.split_jitter
                    for _k in list(model.gaussians.keys()):
                        model.gaussians[_k] = torch.nn.Parameter(
                            torch.cat([model.gaussians[_k].data[keepm], child[_k]], dim=0),
                            requires_grad=model.gaussians[_k].requires_grad)
                    n_keep = int(keepm.sum())
                    infl = torch.zeros(n_keep + child["means"].shape[0],
                                       dtype=torch.bool, device=dev)
                    infl[n_keep:] = True
                if len(results["失敗"]) + len(results["成功"]) == 0:
                    print(f"  [split] 影響集 {n_infl} -> {int(infl.sum())} 顆（K={K}），"
                          f"全體 {n_keep + int(infl.sum()):,}")
                n_infl = int(infl.sum())

            _fz = {x.strip() for x in args.freeze.split(",") if x.strip()}
            rest = [model.gaussians[k] for k in ("scales", "rotations", "opacities")
                    if k not in _fz]
            rest += [model.gaussians[k] for k in ("shs_dc",)
                     if k in model.gaussians and k not in _fz]
            mean_p = model.gaussians["means"]
            params = [mean_p] + rest
            for p in params:
                p.requires_grad_(True)
            if _fz and i == 0:
                print(f"  [freeze] 凍結 {sorted(_fz)}；可優化 {len(params)} 個張量")
            _groups = [{"params": rest, "lr": args.lr}] if rest else []
            if "means" not in _fz:
                _groups.insert(0, {"params": [mean_p],
                                   "lr": args.means_lr if args.means_lr else args.lr})
                params = [mean_p] + rest
            else:
                params = list(rest)
            opt = torch.optim.Adam(_groups)

            def render_gray():
                o = renderer(camd, model, bg_color=bg)
                return o["render"]

            # ★ O0 的解長什麼樣？特別是**它有沒有自己把 scale 縮小**。
            # 這直接決定 ac_shrink 是不是打對變數：
            #   縮小 => oracle 認證「變小就是解」=> ac_shrink 用實測懸崖 ~12px 當目標
            #   沒縮 => 解不在 scale 上 => ac_shrink 打錯變數，該收線
            with torch.no_grad():
                # ⚠ 2026-09-05：`z` 是**分裂前**算的，--split 之後長度會對不上
                #   （IndexError: mask [2340567] vs tensor [2340000]）=> 一律重算。
                _pc0 = model.get_xyz @ R.T + Tv
                _sc0 = model.get_scales()[infl].max(dim=1).values.median().item()
                _rad0 = float((3.0 * fx * model.get_scales()[infl].max(dim=1).values
                               / _pc0[infl][:, 2].clamp_min(0.2)).median())
                _op0 = float(model.get_opacities()[infl].median())
            # ★ 優化用的相機索引（多視角臂會用到）—— **必須先算**，覆核相機要避開它。
            # ⚠ 2026-09-05 修正：原本 `_vc` 與 `_mv` 各自從頭取前 K 台，
            #   6 視角臂的 6 台覆核相機有 **5 台正在被優化** => 量到的是訓練表現不是泛化。
            #   （單視角臂不受影響：`_mv` 只有目標相機，覆核全是留出的。）
            _mv_idx = [i]
            if args.multi_view > 0:
                for _j in range(len(vset)):
                    if len(_mv_idx) >= args.multi_view:
                        break
                    if _j != i:
                        _mv_idx.append(_j)
            _mv_set = set(_mv_idx)

            _vc = []
            if args.verify_cams > 0:
                for _j in range(len(vset)):
                    if len(_vc) >= args.verify_cams:
                        break
                    if _j in _mv_set:          # ★ 真正留出：優化用過的一律排除
                        continue
                    _n2, _p2, _m2, _c2, _e2 = vset[_j]
                    _vc.append((_p2, _c2))
            if _vc and len(results["失敗"]) + len(results["成功"]) == 0:
                print(f"  [覆核] 優化相機 {sorted(_mv_set)} ／ 留出相機 "
                      f"{[j for j in range(len(vset)) if j not in _mv_set][:args.verify_cams]}")

            def _psnr_others():
                """其他視角的整張 PSNR 平均。只讀不改，用來抓單視角過擬合。"""
                if not _vc:
                    return float("nan")
                acc = []
                with torch.no_grad():
                    for _p2, _c2 in _vc:
                        _cd = _c2.to_device(dev)
                        _W2, _H2 = int(_cd.width), int(_cd.height)
                        _g2 = torch.from_numpy(
                            np.asarray(Image.open(_p2).convert("RGB").resize((_W2, _H2),
                                       Image.BILINEAR), np.float32) / 255.
                        ).permute(2, 0, 1).to(dev)
                        _r2 = renderer(_cd, model, bg_color=bg)["render"]
                        acc.append(float(-10 * torch.log10(((_r2 - _g2) ** 2).mean() + 1e-12)))
                return float(np.mean(acc))

            _po0 = _psnr_others()
            with torch.no_grad():
                r0 = render_gray().mean(0).cpu().numpy()
            g_np = gt.mean(0).cpu().numpy()
            c0 = tile_corr(g_np, r0, ty, tx, T)

            _mv = []
            if args.multi_view > 0:
                for _j in _mv_idx:                       # ★ 與覆核相機的排除集合同源
                    if _j == i:
                        _mv.append((camd, gt))
                        continue
                    _n3, _p3, _m3, _c3, _e3 = vset[_j]
                    _cd3 = _c3.to_device(dev)
                    _W3, _H3 = int(_cd3.width), int(_cd3.height)
                    _mv.append((_cd3, torch.from_numpy(np.asarray(
                        Image.open(_p3).convert("RGB").resize((_W3, _H3), Image.BILINEAR),
                        np.float32) / 255.).permute(2, 0, 1).to(dev)))

            for _s in range(args.steps):
                if args.multi_view > 0:
                    # 逐視角輪替 + **整張** loss = 訓練的形狀，只是參數凍到影響集
                    _cs, _gs = _mv[_s % len(_mv)]
                    _is = renderer(_cs, model, bg_color=bg)["render"]
                    loss = (1 - args.lambda_dssim) * torch.abs(_is - _gs).mean() \
                        + args.lambda_dssim * (1 - ssim_fn(_is, _gs))
                    if args.with_reg:
                        _o = model.get_opacities().squeeze(-1)
                        _ni = _o.detach() <= _immune
                        loss = loss + _w_op * (_o[_ni].mean() if bool(_ni.any())
                                               else _o.sum() * 0.0) \
                            + _w_sc * model.get_scales().abs().mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    for p in params:
                        if p.grad is not None and p.grad.shape[0] == infl.shape[0]:
                            p.grad[~infl] = 0
                    opt.step()
                    continue
                img = render_gray()
                y0, x0 = ty * T, tx * T
                sub_r, sub_g = img[:, y0:y0 + T, x0:x0 + T], gt[:, y0:y0 + T, x0:x0 + T]
                loss = (1 - args.lambda_dssim) * torch.abs(sub_r - sub_g).mean() \
                    + args.lambda_dssim * (1 - ssim_fn(sub_r, sub_g))
                if args.with_reg:
                    _o = model.get_opacities().squeeze(-1)
                    _ni = _o.detach() <= _immune
                    loss = loss + _w_op * (_o[_ni].mean() if bool(_ni.any())
                                           else _o.sum() * 0.0) \
                        + _w_sc * model.get_scales().abs().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # ★ 凍結：只留影響集的梯度
                for p in params:
                    if p.grad is not None and p.grad.shape[0] == infl.shape[0]:
                        p.grad[~infl] = 0
                opt.step()

            with torch.no_grad():
                r1 = render_gray().mean(0).cpu().numpy()
                _sc1 = model.get_scales()[infl].max(dim=1).values.median().item()
                _pc1 = model.get_xyz[infl] @ R.T + Tv
                _rad1 = float((3.0 * fx * model.get_scales()[infl].max(dim=1).values
                               / _pc1[:, 2].clamp_min(0.2)).median())
                _op1 = float(model.get_opacities()[infl].median())
            c1 = tile_corr(g_np, r1, ty, tx, T)
            _po1 = _psnr_others()
            # 鄰域（外圈一格）—— 檢查是不是只把誤差搬走
            ring = [tile_corr(g_np, r1, ty + dy, tx + dx, T) - tile_corr(g_np, r0, ty + dy, tx + dx, T)
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and 0 <= ty + dy < ny and 0 <= tx + dx < nx]
            results[lab].append((c0, c1, float(np.mean(ring)) if ring else 0.0, n_infl,
                                 _rad0, _rad1, _op0, _op1, _po0, _po1))
            done += 1
            del model, renderer, opt
            torch.cuda.empty_cache()

    print(f"\nO0：凍結拓撲、只優化影響集、{args.steps} 步、只用單一目標相機\n")
    print(f"{'':>6} {'corr 前':>9} {'corr 後':>9} {'Δcorr':>9} {'鄰域 Δ':>9} {'影響集顆數':>11} {'n':>4}")
    for lab in ("失敗", "成功"):
        A = np.array(results[lab])
        if not len(A):
            continue
        print(f"{lab:>6} {A[:,0].mean():>9.3f} {A[:,1].mean():>9.3f} {(A[:,1]-A[:,0]).mean():>+9.3f} "
              f"{A[:,2].mean():>+9.3f} {A[:,3].mean():>11,.0f} {len(A):>4}")

    # ★★ oracle 的解長什麼樣 —— 決定 ac_shrink 打的是不是對的變數
    print(f"\n  ★ oracle 改了什麼（影響集的中位數）   lr={args.lr:g}")
    print(f"{'':>6} {'投影半徑前':>12} {'投影半徑後':>12} {'比值':>8} "
          f"{'opacity 前':>12} {'opacity 後':>12} {'比值':>8}")
    for lab in ("失敗", "成功"):
        A = np.array(results[lab])
        if not len(A):
            continue
        print(f"{lab:>6} {A[:,4].mean():>12.2f} {A[:,5].mean():>12.2f} "
              f"{A[:,5].mean()/max(A[:,4].mean(),1e-9):>8.3f} "
              f"{A[:,6].mean():>12.4f} {A[:,7].mean():>12.4f} "
              f"{A[:,7].mean()/max(A[:,6].mean(),1e-9):>8.3f}")
    # ★★ 多視角覆核 —— 這條決定「basis 足夠」的結論站不站得住
    if args.verify_cams > 0:
        print(f"\n  ★ 多視角覆核（另外 {args.verify_cams} 台相機的整張 PSNR）")
        print(f"{'':>6} {'其他視角前':>12} {'其他視角後':>12} {'ΔPSNR':>10}")
        for lab in ("失敗", "成功"):
            A = np.array(results[lab])
            if not len(A) or np.isnan(A[:, 8]).all():
                continue
            print(f"{lab:>6} {np.nanmean(A[:,8]):>12.3f} {np.nanmean(A[:,9]):>12.3f} "
                  f"{np.nanmean(A[:,9]-A[:,8]):>+10.3f}")
        print("""  判讀：ΔPSNR ~ 0 或為正 => 局部修復**沒有**犧牲其他視角 => 「basis 足夠」成立
        ΔPSNR 明顯為負    => 是**單視角過擬合**，§11.87 的結論要降級為
                             「單視角下可恢復」，不能推論到多視角一致的解存在""")

    print("""  判讀：失敗組半徑比值 << 1 => **oracle 的解就是「變小」** => ac_shrink 打對變數，
        目標用實測懸崖（radius_cliff：corr 在 12.25px 以上崩，10.8px 仍 0.980）
        比值 ~1 => 解不在 scale 上 => **ac_shrink 收線**，改看 opacity/位置那兩欄""")
    print("""
判讀：
  Δ失敗 >> Δ成功            => **現有 basis 救得回來** => 是優化可及性問題
  Δ失敗 ~ Δ成功 ~ 0         => 在此局部預算下救不回來 => 才輪到 O1（加 basis）
  Δ目標大但**鄰域 Δ 為負**  => 只是把誤差搬到旁邊，不算恢復
⚠ 措辭限制：只能說「在此局部預算下無法恢復」，**不能說「無法表示」**
  （§11.77 已證表徵容量足夠 —— 20 個 2D 高斯就能到 0.92）。
⚠ 本版只用**單一目標相機**優化 => 有過擬合單視角的風險，
  Δ 很大時必須再用其他視角覆核（外部指出的失敗模式之一）。""")


if __name__ == "__main__":
    main()
