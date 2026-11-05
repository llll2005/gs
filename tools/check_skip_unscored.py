"""不計分像素（塊外遮罩的 C 區）略過的等價性與速度驗證（2026-11-05，使用者「C 區的部分排一次測試」）。

兩個旗標（renderer，見 sep_depth_trim_2dgs_renderer.py；光柵器 forward.cu pix_skip、backward.cu ZSKIP）：
  bwd_zero_grad_skip    backward 不處理上游梯度全 0 的像素 => 梯度應在噪音底內（只差浮點加總順序）
  skip_unscored_pixels  lean 訓練 forward 略過 C 區 => A／B 像素的渲染應逐位元相同；梯度會變（C 區不再算 dist、
                        交界處 SSIM 視窗看到的 C 值變成背景色），這裡只量變了多少
loss 照 metric 的寫法：B 的目標＝背景色、C 的目標＝render.detach()、送進 loss 的 render 在 C 區也換成 detach。

設定（同一個 ckpt、同一組相機、同一組隨機目標，依序跑）：
  base   lean＋飽和跳過＋先加總（現行）            base2  同上再跑一次 => 噪音底
  zskip  base＋bwd_zero_grad_skip                   fskip  zskip＋skip_unscored_pixels
--dist w：loss 加 w*mean(rend_dist)、走 dist 版 lean（現行配方每 4 步一次的那種步）。
--vc 1（2026-11-05 使用者「V/C 累加直接對 forward 作看看」）：改比訓練中累加 V/C 的兩種寫法（增生期的 forward）：
  vc     累加式 V/C（現行：每個配對 32 條執行緒對同一個 shared 位址 atomic）   vc2  同上再跑一次 => 噪音底
  vcw    vc＋trim_running_warp（warp 內先加總）=> 渲染應逐位元相同、覆蓋數逐位元相同、V 只差加總順序
  vcwall vcw＋兩個 C 區旗標
--pairbit 1（2026-11-05 使用者「都排進去」：配對貢獻 bit 在現行配方上重量；10-16 lab 只快 1.1~1.9% 而關閉）：
  base／base2 同上   pbit  base＋bwd_pair_bit（forward 記配對有沒有被混合、backward 只處理混合過的）
  pball pbit＋兩個 C 區旗標（兩者省的是不同的配對，看能不能疊）
  ⚠ 梯度與 base 不完全相同是已知的：上游光柵器近平面判斷 forward 用交點深度、backward 濾波分支用 Tw.z（10-16 查出）
用法（新光柵器用 PYTHONPATH 指到另外編好的目錄，不碰共用環境）：
  PYTHONPATH=<編好的目錄> python tools/check_skip_unscored.py --ckpt <ckpt> --mask_dir <block_masks44/block_B> [--ncam 8] [--repeat 8] [--dist 400]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_mask(mask_dir, name, h, w):
    from PIL import Image
    p = os.path.join(mask_dir, os.path.basename(str(name)))
    if not os.path.exists(p):
        return None
    t = torch.from_numpy(np.array(Image.open(p)))[None, None].float()
    return torch.nn.functional.interpolate(t, size=(h, w), mode="nearest")[0, 0].to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--ncam", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--dist", type=float, default=0.0)
    ap.add_argument("--absgrad", type=int, default=1, help="1＝增生期（backward 累加 |g|）；0＝收割期")
    ap.add_argument("--vc", type=int, default=0, help="1＝比訓練中累加 V/C 的 warp 先加總（見檔頭）")
    ap.add_argument("--pairbit", type=int, default=0, help="1＝比配對貢獻 bit（見檔頭）")
    a = ap.parse_args()

    import diff_trim_surfel_rasterization as R
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    dev = torch.device("cuda")
    print(f"光柵器：{R.__file__}  支援 pix_skip：{getattr(R, 'PIX_SKIP_SUPPORTED', False)}")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    renderer.lean_train = True
    renderer._lean_announced = renderer._lean_dist_announced = True
    renderer.bwd_sat_skip = renderer.bwd_reduce = True
    renderer.bwd_pair_bit = False
    renderer.trim_running_vc = bool(a.vc)     # --vc 0：只比渲染與梯度；--vc 1：累加值另外攔下來比
    renderer._acc_announced = True
    renderer._acc_stop = False                # ⚠ 11-05：訓練到底的 ckpt 帶著「增生期結束、停止累加」的狀態 => 不重設就一顆都不累加（第一次 --vc 1 因此白跑）
    ACC, CAP = [], [False]      # 只在算梯度那一輪攔（計時那幾輪不 clone，免得多吃顯存、也不算進時間）
    renderer._acc_add = lambda pc, ta, cn: ACC.append((ta.detach().clone(), cn.detach().clone())) if CAP[0] else None
    renderer.sh_adam_in_backward = False      # SH 的 Adam 在 backward 就地改參數 => 各設定之間會互相污染
    renderer.absgrad_gate = True
    renderer._absgrad_announced = True
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"
    ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                                   global_rank=0).get_outputs().train_set
    del ck
    cams, names = ts.cameras, ts.image_names
    idx = np.linspace(0, len(cams) - 1, a.ncam).round().astype(int).tolist()
    params = dict(model.gaussians.items())
    for p in params.values():
        p.requires_grad_(True)
    bg = torch.zeros(3, device=dev)
    print(f"N={model.n_gaussians:,}  相機 {idx}  dist={a.dist}  absgrad={a.absgrad}")

    data = []
    for c in idx:
        cam = cams[c].to_device(dev)
        h, w = int(cam.height), int(cam.width)
        m = _load_mask(a.mask_dir, names[c], h, w)
        if m is None:
            print(f"⚠⚠ 相機 {c} {names[c]} 沒有遮罩，跳過")
            continue
        m = m.to(dev)
        g = torch.Generator(device=dev).manual_seed(1000 + c)
        tgt = torch.rand((3, h, w), device=dev, generator=g)
        data.append((c, cam, m, tgt))
    cfrac = float(np.mean([float((m == 0).float().mean()) for _, _, m, _ in data]))
    print(f"有遮罩的相機 {len(data)} 台，C 像素平均 {cfrac:.3f}")

    def run(cam, m, tgt, skip):
        TF = {"_absgrad": bool(a.absgrad), "_geo": False, "_lean_dist": a.dist > 0}
        if skip:
            TF["_pix_skip"] = (m == 0).to(torch.uint8).contiguous().view(-1)
        out = renderer.training_forward(0, None, cam, model, bg, **TF)
        r = out["render"]
        Bm, Cm = (m == 2)[None], (m == 0)[None]
        t2 = torch.where(Bm, bg.view(3, 1, 1).expand_as(tgt), tgt)
        t2 = torch.where(Cm, r.detach(), t2)
        cf = Cm.to(r.dtype)
        rl = r * (1 - cf) + r.detach() * cf
        loss = 0.8 * (rl - t2).abs().mean() + 0.2 * (1 - ssim(rl, t2))
        if a.dist > 0:
            loss = loss + a.dist * out["rend_dist"].mean()
        return out, loss

    def config(tag, zskip, fskip, warp=False, pbit=False):
        renderer.trim_running_warp = warp
        renderer.bwd_pair_bit = pbit
        renderer.bwd_zero_grad_skip = zskip
        renderer.skip_unscored_pixels = fskip
        renderer._pix_skip_announced = True
        gsum = {k: torch.zeros_like(p, dtype=torch.float64) for k, p in params.items()}
        renders = []
        ACC.clear(); CAP[0] = True
        for c, cam, m, tgt in data:
            for p in params.values():
                p.grad = None
            out, loss = run(cam, m, tgt, fskip)
            loss.backward()
            renders.append(out["render"].detach().cpu())   # 存 CPU：6 GB 卡放不下四組設定的渲染與梯度
            for k, p in params.items():
                if p.grad is not None:
                    gsum[k] += p.grad.double()
        CAP[0] = False
        tf, tb = [], []
        for _, cam, m, tgt in data:
            for r_ in range(a.repeat + 2):
                for p in params.values():
                    p.grad = None
                e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                e0.record()
                out, loss = run(cam, m, tgt, fskip)
                e1.record()
                loss.backward()
                e2.record()
                torch.cuda.synchronize()
                if r_ >= 2:
                    tf.append((cam, e0.elapsed_time(e1))); tb.append((cam, e1.elapsed_time(e2)))
        # 每台相機取中位，再對相機加總 => 一步的期望時間（每步隨機一台）
        def per_cam(lst):
            d = {}
            for cam, t in lst:
                d.setdefault(id(cam), []).append(t)
            return float(np.mean([np.median(v) for v in d.values()]))
        acc = [(t.cpu(), c.cpu()) for t, c in ACC[:len(data)]]
        ACC.clear()
        res = {"tag": tag, "g": {k: v.float().cpu() for k, v in gsum.items()}, "renders": renders, "acc": acc,
               "fwd": per_cam(tf), "bwd": per_cam(tb)}
        print(f"  {tag:6s} forward+loss {res['fwd']:6.2f} ms   backward {res['bwd']:6.2f} ms   合計 {res['fwd'] + res['bwd']:6.2f} ms", flush=True)
        return res

    print("計時（每台相機中位、再對相機平均；ms）：")
    if a.vc:
        base = config("vc", False, False)
        base2 = config("vc2", False, False)
        zs = config("vcw", False, False, warp=True)
        fs = config("vcwall", True, True, warp=True)
    elif a.pairbit:
        base = config("base", False, False)
        base2 = config("base2", False, False)
        zs = config("pbit", False, False, pbit=True)
        fs = config("pball", True, True, pbit=True)
    else:
        base = config("base", False, False)
        base2 = config("base2", False, False)
        zs = config("zskip", True, False)
        fs = config("fskip", True, True)

    def gdiff(x, y):
        out = {}
        for k in x["g"]:
            a_, b_ = x["g"][k], y["g"][k]
            den = float(a_.abs().max()) or 1.0
            out[k] = float((a_ - b_).abs().max()) / den
        return out

    print(f"\n梯度差（各參數 max|Δ| / max|g|；{base['tag']} vs {base2['tag']} 是噪音底）：")
    for lab, y in ((base2['tag'], base2), (zs['tag'], zs), (fs['tag'], fs)):
        d = gdiff(base, y)
        print(f"  {base['tag']} vs {lab:6s} " + "  ".join(f"{k} {v:.2e}" for k, v in d.items()))
    print("\n渲染（A／B 像素；C 區在略過時會變成背景色）：")
    for lab, y in ((zs['tag'], zs), (fs['tag'], fs)):
        mx = 0.0; same = True
        for (c, cam, m, tgt), r0, r1 in zip(data, base["renders"], y["renders"]):
            keep = (m != 0).cpu()[None].expand_as(r0)
            dd = (r0[keep] - r1[keep]).abs().max()
            mx = max(mx, float(dd)); same = same and bool(torch.equal(r0[keep], r1[keep]))
        print(f"  {base['tag']} vs {lab}: A／B 像素 {'逐位元相同' if same else '不同'}（max|Δ| {mx:.2e}）")
    if a.vc:
        print("\n訓練中累加的 V/C（逐顆 Σ T·α 與覆蓋數；每台相機一份）：")
        for lab, y in ((base2['tag'], base2), (zs['tag'], zs), (fs['tag'], fs)):
            if not y['acc'] or not base['acc']:
                print(f"  ⚠⚠ {lab}：沒攔到累加值（train_record 沒開？）"); continue
            ceq = all(torch.equal(p[1], q[1]) for p, q in zip(base['acc'], y['acc']))
            tr = max(float((p[0] - q[0]).abs().max()) / max(float(p[0].abs().max()), 1e-12) for p, q in zip(base['acc'], y['acc']))
            print(f"  {base['tag']} vs {lab:6s} 覆蓋數 {'逐位元相同' if ceq else '不同'}   Σ T·α max|Δ|/max {tr:.2e}")
    t0 = base["fwd"] + base["bwd"]
    print(f"\n相對 {base['tag']} 的每步光柵化＋loss 時間：" + "  ".join(
        f"{y['tag']} {100 * ((y['fwd'] + y['bwd']) / t0 - 1):+.1f}%（fwd {100 * (y['fwd'] / base['fwd'] - 1):+.1f}%、bwd {100 * (y['bwd'] / base['bwd'] - 1):+.1f}%）"
        for y in (base2, zs, fs)))


if __name__ == "__main__":
    main()
