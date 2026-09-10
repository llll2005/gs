#!/usr/bin/env python
"""arm B 的覆蓋率 0.9998 是「真表面」還是「場外的巨大幕布」？

## 起因

使用者在 web viewer 看 1500 步四臂：A/C/D 被 floater 佔滿的地方，**B（SfM-init）是空的**；
B 的 floater「幾乎都在場景外（很大的）」。
但用**驗證相機**量渲染 alpha，B 的覆蓋是四臂最好（中位 0.9998、零個破洞 tile）—— 矛盾。

線索：兩臂的粒子空間跨度差 33 倍（A 21.55 vs **B 718.81**）
=> **假說：B 的 alpha 覆蓋大部分來自場外那片巨大 floater，它從任何相機看都是滿版背景。**
   `rend_alpha` 只問「這條光線有沒有撞到東西」，撞到 700 單位外的幕布也算 1.0。

## 量什麼

```
① 場外比例        SfM 點雲的 1~99 百分位盒 = 真場景的範圍；盒外粒子佔多少顆、佔多少「面積」
② 遮掉場外再量    只留盒內粒子重新渲染 => 這才是「真表面的覆蓋率」
③ 覆蓋來自多遠    alpha 加權的 surf_depth 中位 vs SfM 點的深度中位
```
判準：
```
遮掉場外後 B 的覆蓋率**崩掉** => 0.9998 是幕布，使用者看到的破洞是真的
遮掉後仍然高                  => 覆蓋是真表面，破洞來自別的原因（視角/顯示）
```
⚠ 本工具只量覆蓋（有沒有東西），不量正確性；要連 1500 步的光度指標一起看。

用法: python tools/novel_view_coverage.py --arms probe_armA_baseline probe_armB_sfm
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["probe_armA_baseline", "probe_armB_sfm",
                                                  "probe_armC_low_opacity",
                                                  "probe_armD_absgrad_early"])
    ap.add_argument("--step", type=int, default=1500)
    ap.add_argument("--max-cam", type=int, default=12)
    a = ap.parse_args()

    dev = torch.device("cuda")
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    rows = []
    for arm in a.arms:
        ck = sorted(glob.glob(f"outputs/{arm}/**/*step={a.step}.ckpt", recursive=True))
        if not ck:
            print(f"{arm}: 找不到 ckpt"); continue
        model, renderer, _ = GaussianModelLoader.\
            initialize_model_and_renderer_from_checkpoint_file(ck[0], device=dev,
                                                               eval_mode=True, pre_activate=False)
        c = torch.load(ck[0], map_location="cpu")
        dmh = c["datamodule_hyper_parameters"]
        dp = dmh["parser"].instantiate(path=dmh["path"],
                                       output_path=os.path.dirname(os.path.dirname(ck[0])),
                                       global_rank=0)
        out = dp.get_outputs()
        vset = out.val_set
        pcd = out.point_cloud.xyz
        pcd = torch.as_tensor(np.asarray(pcd), dtype=torch.float32)
        lo = torch.quantile(pcd, 0.01, dim=0).to(dev)
        hi = torch.quantile(pcd, 0.99, dim=0).to(dev)

        props = {k: v for k, v in model.properties.items()}
        xyz = model.get_xyz.detach()
        N = xyz.shape[0]
        inbox = ((xyz >= lo) & (xyz <= hi)).all(dim=1)
        bg = torch.zeros((3,), device=dev)

        def cover(mask=None):
            if mask is not None:
                model.properties = {k: v[mask] for k, v in props.items()}
            cov, dep = [], []
            with torch.no_grad():
                for i in range(min(len(vset), a.max_cam)):
                    cam = vset[i][3].to_device(dev)
                    r = renderer(cam, model, bg_color=bg)
                    al = r["rend_alpha"].float()
                    cov.append(float(al.mean()))
                    d = r["surf_depth"].float()
                    w = al.flatten()
                    dv = d.flatten()[w > 0.5]
                    if dv.numel():
                        dep.append(float(dv.median()))
            if mask is not None:
                model.properties = props
            return float(np.mean(cov)), (float(np.median(dep)) if dep else float("nan"))

        cov_all, dep_all = cover()
        cov_in, dep_in = cover(inbox)
        # SfM 點從 val 相機看的深度中位（真表面的距離尺度）
        sd = []
        for i in range(min(len(vset), a.max_cam)):
            cam = vset[i][3]
            R = cam.R.float(); T = cam.T.float()
            z = (pcd @ R.T + T)[:, 2]
            z = z[z > 0]
            if z.numel():
                sd.append(float(z.median()))
        sfm_depth = float(np.median(sd)) if sd else float("nan")

        rows.append((arm, N, int(inbox.sum()), cov_all, cov_in, dep_all, dep_in, sfm_depth))
        print(f"  {arm} 完成", flush=True)
        del model, renderer
        torch.cuda.empty_cache()

    print(f"\n{'arm':>24} {'N':>9} {'盒內':>9} {'盒內%':>7} "
          f"{'覆蓋(全部)':>10} {'覆蓋(只盒內)':>12} {'渲染深度':>9} {'SfM深度':>9}")
    for arm, N, nin, ca, ci, da, di, sfm in rows:
        print(f"{arm:>24} {N:>9,} {nin:>9,} {100*nin/N:>6.2f}% "
              f"{ca:>10.4f} {ci:>12.4f} {da:>9.2f} {sfm:>9.2f}")
    print("""
判讀：
  「只盒內」的覆蓋率**崩掉** => 全部覆蓋是**場外幕布**，使用者看到的破洞是真的
  兩者接近                   => 覆蓋來自真表面，破洞的成因在別處
  渲染深度 >> SfM 深度       => 同一件事的獨立佐證（撞到的東西比真表面遠得多）""")


if __name__ == "__main__":
    main()
