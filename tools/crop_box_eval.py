#!/usr/bin/env python
"""裁掉 SfM 盒外的粒子，畫面掉多少、binning 省多少？（事後裁切，不重新訓練）

## 為什麼問這個

`tools/aabb_coverage.py`（純 CPU）量到 `speed3_b12 @60k`：
```
SfM 點盒 1~99%  [-4.23,-4.62,-1.00] ~ [5.10,3.63,2.21]
  界內  97.89% 顆數 ／ 96.60% 不透明度質量 ／ **僅 48.40% binning 成本代理**
```
⇒ **盒外那 2.11% 的粒子扛了 51.6% 的 binning 成本。** 極少數巨大粒子主宰渲染成本，
與 `elongation_binning_waste` 量到的長寬比極端值（p100 = 6,060,205）指向同一批東西。

而模型自身範圍是 `[-77.3,-68.0,-61.9] ~ [67.7,64.6,45.6]`，遠超場景的 102 x 80
⇒ 那些東西在場景外面。

**但「成本高」不等於「沒用」**：它們帶著 3.4% 的不透明度質量，而
`tools/novel_view_coverage.py` 量到 1500 步時它們讓 `rend_alpha` 從 0.9935
變成只算盒內的 0.5139 —— 也就是它們**確實在遮住整個畫面**。
所以必須直接量「裁掉之後 val 掉多少」。

## 量什麼

同一個 ckpt、同一批 val 相機，比較「全部粒子」與「只留 SfM 盒內」：
```
PSNR / SSIM        畫面代價（噪音底 3sd = 0.24 / 0.0015）
Σtile（binning）    省下的渲染工作量，用光柵器回傳的 radii 算 (2r/16)²
```
判準：
```
PSNR 掉 < 0.24（3sd）而 binning 省 > 30%  => 值得做成訓練期的機制（或至少部署時裁）
PSNR 掉很多                               => 那些場外粒子在承擔真實的外觀，不能直接裁
```
⚠ 這是**事後裁切**，不是訓練期介入。訓練期裁掉它們，MCMC 會把預算用在別處
  ⇒ 端到端的結果可能更好也可能更差（記憶 `absgrad_densify_current_best`：
  「改變集合」有效、「破壞集合」無效）。這支只回答「它們現在值多少」。

用法: python tools/crop_box_eval.py --runs speed3_b12 sfminit2_b12
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["speed3_b12", "sfminit2_b12"])
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--max-cam", type=int, default=36)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    from PIL import Image

    print(f"\n{'跑次':<18}{'':<14}{'顆數':>12}{'PSNR':>9}{'SSIM':>9}{'Σtile（binning）':>20}")
    for run in a.runs:
        ck = sorted(glob.glob(f"outputs/{run}/**/*step={a.step}.ckpt", recursive=True))
        if not ck:
            print(f"{run}: 找不到 ckpt"); continue
        model, renderer, _ = GaussianModelLoader.\
            initialize_model_and_renderer_from_checkpoint_file(ck[0], device=dev,
                                                               eval_mode=True, pre_activate=False)
        c = torch.load(ck[0], map_location="cpu")
        dmh = c["datamodule_hyper_parameters"]
        dp = dmh["parser"].instantiate(path=dmh["path"],
                                       output_path=os.path.dirname(os.path.dirname(ck[0])),
                                       global_rank=0)
        out = dp.get_outputs(); vset = out.val_set
        pcd = torch.as_tensor(np.asarray(out.point_cloud.xyz), dtype=torch.float32)
        lo = torch.quantile(pcd, 0.01, dim=0).to(dev)
        hi = torch.quantile(pcd, 0.99, dim=0).to(dev)
        del c
        props = {k: v for k, v in model.properties.items()}
        keep = ((model.get_xyz.detach() >= lo) & (model.get_xyz.detach() <= hi)).all(dim=1)
        N = keep.numel()
        bg = torch.zeros(3, device=dev)

        def ev(mask=None):
            if mask is not None:
                model.properties = {k: v[mask] for k, v in props.items()}
            ps, ss, tiles = [], [], 0.0
            with torch.no_grad():
                for i in range(min(len(vset), a.max_cam)):
                    _n, ip, _m, cam, _e = vset[i]
                    cam = cam.to_device(dev)
                    r = renderer(cam, model, bg_color=bg)
                    img = r["render"].clamp(0, 1)
                    gt = torch.from_numpy(np.asarray(
                        Image.open(ip).convert("RGB").resize(
                            (int(cam.width), int(cam.height)), Image.BILINEAR),
                        np.float32) / 255.).permute(2, 0, 1).to(dev)
                    ps.append(float(-10 * torch.log10(((img - gt) ** 2).mean())))
                    ss.append(float(ssim_fn(img[None], gt[None])))
                    rad = r.get("radii")
                    if rad is not None:
                        tiles += float(((2.0 * rad.double() / 16).clamp_min(0) ** 2).sum())
                    del r, img, gt
            if mask is not None:
                model.properties = props
            return float(np.mean(ps)), float(np.mean(ss)), tiles

        p0, s0, t0 = ev()
        p1, s1, t1 = ev(keep)
        print(f"{run:<18}{'原始':<14}{N:>12,}{p0:>9.4f}{s0:>9.4f}{t0:>20,.0f}")
        print(f"{'':<18}{'裁到 SfM 盒':<12}{int(keep.sum()):>12,}{p1:>9.4f}{s1:>9.4f}{t1:>20,.0f}")
        print(f"{'':<18}{'差':<14}{int(keep.sum())-N:>12,}{p1-p0:>+9.4f}{s1-s0:>+9.4f}"
              f"{100*(t1/max(t0,1e-9)-1):>19.1f}%")
        del model, renderer, props
        torch.cuda.empty_cache()
    print("""
噪音底：PSNR 3sd = 0.24 ／ SSIM 3sd = 0.0015
判準：PSNR 掉 < 0.24 而 binning 省 > 30% => 值得做成機制；掉很多 => 那些粒子在承擔真實外觀
⚠ 這是**事後裁切**。訓練期裁掉它們，MCMC 會把預算用在別處 => 端到端可能更好也可能更差。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
