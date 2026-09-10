#!/usr/bin/env python
"""SfM-init 的「破洞」是不是就是持續糊掉的區域？—— 用**渲染出來的 alpha 覆蓋率**直接量。

## 為什麼要重做（前一版 `sfm_blur_overlap.py` 的兩個漏洞）

使用者在 web viewer 看 1500 步的四臂，觀察到「A/C/D 被 floater 佔滿的地方，B 是**空的**」。
我前一版的測試否證了這個假說，但那個測試有兩個問題：
```
1 只看**最高對比的 15%** tile（contrast_q=0.85）=> 若破洞落在低對比區，根本看不到
2 用**原始 SfM 點**投影當代理，不是 arm B **訓練後實際渲染出來的覆蓋**
   （真正零點的 tile 只有 8 個 => 樣本不足以談「破洞」）
```
本版改為：**渲染 arm B 拿 `rend_alpha`**，逐 tile 算平均覆蓋率，且**不預先篩掉任何 tile**。

## ⚠ 第三個漏洞（2026-09-11 補，`novel_view_coverage.py` 抓到）

`rend_alpha` 只問「這條光線有沒有撞到東西」—— **撞到場外 700 單位遠的巨大 floater 也算 1.0**。
arm B 的 floater 幾乎都在場景外且很大，實測：
```
覆蓋(全部粒子) 0.9919  ->  覆蓋(只算 SfM 盒內) 0.4744     渲染深度 5.79 vs 真表面 2.89
```
A/C/D 只掉 1~2%，只有 B 崩掉一半 ⇒ **不濾場外粒子，這個量測對 B 完全無效**。
=> `--inbox` 預設**開啟**：只保留落在 SfM 點雲 1~99 百分位盒內的粒子再渲染。

## 量什麼

```
1 覆蓋率的分布           哪些 tile 真的是「空的」（alpha 接近 0）
2 破洞 tile 的 GT 對比    它們是**高對比**（＝糊掉判準涵蓋的）還是**低對比**（判準排除的）
                         -> 若是低對比，那使用者看到的洞與「40~46% 糊掉」是**兩個不同問題**
3 高對比 tile 內：糊掉 vs 沒糊 的覆蓋率差異
                         -> 覆蓋率明顯較低 => 「那裡沒資訊」成立
```

用法: python tools/coverage_vs_blur.py --arm probe_armB_sfm --step 1500
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="probe_armB_sfm")
    ap.add_argument("--step", type=int, default=1500)
    ap.add_argument("--runs", nargs="+", default=["agd2_b12", "sched30_b12"])
    ap.add_argument("--blk", default="12")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--max-cam", type=int, default=36)
    ap.add_argument("--inbox", type=int, default=1,
                    help="1=只保留 SfM 盒內粒子（預設；不這樣做會被場外幕布騙）")
    a = ap.parse_args()

    ck = sorted(glob.glob(f"outputs/{a.arm}/**/*step={a.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {a.arm} 的 step={a.step} ckpt")
    dev = torch.device("cuda")
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from PIL import Image as _Im
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck[0], device=dev, eval_mode=True, pre_activate=False)
    print(f"{a.arm} @ {a.step}   N = {model.n_gaussians:,}")

    c = torch.load(ck[0], map_location="cpu")
    dmh = c["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    _out = dp.get_outputs()
    vset = _out.val_set

    if a.inbox:
        _p = torch.as_tensor(np.asarray(_out.point_cloud.xyz), dtype=torch.float32)
        _lo = torch.quantile(_p, 0.01, dim=0).to(dev)
        _hi = torch.quantile(_p, 0.99, dim=0).to(dev)
        _m = ((model.get_xyz.detach() >= _lo) & (model.get_xyz.detach() <= _hi)).all(dim=1)
        model.properties = {k: v[_m] for k, v in model.properties.items()}
        print(f"  只保留 SfM 盒內粒子：{int(_m.sum()):,} / {_m.numel():,} "
              f"({100*float(_m.float().mean()):.2f}%)")

    # 糊掉遮罩 + 逐 tile GT 對比（來自 60k 的成熟跑次）
    dirs = {r: final_test_dir(r, a.blk) for r in a.runs}
    fs = sorted(f for f in os.listdir(dirs[a.runs[0]]) if f.endswith(".png"))
    info = {}
    for f in fs:
        o = {r: per_image(os.path.join(dirs[r], f), a.tile, a.contrast_q) for r in a.runs}
        if any(v is None for v in o.values()):
            continue
        keep, *_z, nx, ny = o[a.runs[0]]
        blur = np.stack([o[r][1] <= a.r_min for r in a.runs]).all(0)
        name = f[:-4] if f.endswith(".png.png") else f
        info[name] = (keep, blur, nx, ny)

    bg = torch.zeros((3,), device=dev)
    rows = []   # (覆蓋率, GT 對比, 是否高對比, 是否糊掉)
    used = 0
    with torch.no_grad():
        for i in range(min(len(vset), a.max_cam)):
            name, ip, _m, cam, _e = vset[i]
            name = os.path.basename(str(name))
            if name not in info:
                continue
            keep, blur, nx, ny = info[name]
            cam = cam.to_device(dev)
            W, H = int(cam.width), int(cam.height)
            out = renderer(cam, model, bg_color=bg)
            al = out.get("rend_alpha")
            if al is None:
                raise SystemExit("renderer 沒回傳 rend_alpha")
            al = al[0, :ny * a.tile, :nx * a.tile].float().cpu().numpy()
            cov = tiles(al, a.tile).mean(axis=1)                 # 逐 tile 平均 alpha
            g = np.asarray(_Im.open(ip).convert("L").resize((W, H), _Im.BILINEAR),
                           np.float32) / 255.
            con = tiles(g[:ny * a.tile, :nx * a.tile], a.tile).std(axis=1)   # 逐 tile GT 對比
            hi = np.zeros(len(cov), bool); hi[keep] = True
            bl = np.zeros(len(cov), bool); bl[keep[blur]] = True
            for t in range(len(cov)):
                rows.append((cov[t], con[t], hi[t], bl[t]))
            used += 1

    A = np.array(rows)
    cov, con, hi, bl = A[:, 0], A[:, 1], A[:, 2] > 0.5, A[:, 3] > 0.5
    print(f"使用 {used} 台相機，**全部** {len(A):,} 個 tile（未預先篩選）\n")

    print("1) 覆蓋率分布")
    for q in (1, 5, 10, 25, 50, 75, 90):
        print(f"    p{q:<3} {np.percentile(cov, q):.4f}", end="   " if q != 90 else "\n")
    for th in (0.01, 0.05, 0.10):
        print(f"    alpha < {th:.2f} 的 tile：{100*np.mean(cov < th):>6.2f}%（n={int((cov<th).sum()):,}）")

    print("\n2) 「破洞」tile 是高對比還是低對比？")
    hole = cov < 0.10
    print(f"    破洞 tile 的 GT 對比 中位 {np.median(con[hole]):.4f}"
          f" ／ 非破洞 {np.median(con[~hole]):.4f}"
          f"  => 比值 **{np.median(con[hole])/max(np.median(con[~hole]),1e-9):.3f}**")
    print(f"    破洞 tile 裡有 **{100*np.mean(hi[hole]):.2f}%** 是高對比"
          f"（全體高對比比例 {100*np.mean(hi):.2f}%）")

    print("\n3) 高對比 tile 內：糊掉 vs 沒糊 的覆蓋率")
    hb, hn = cov[hi & bl], cov[hi & ~bl]
    print(f"    糊掉 {len(hb):,} 個  覆蓋率中位 {np.median(hb):.4f}")
    print(f"    沒糊 {len(hn):,} 個  覆蓋率中位 {np.median(hn):.4f}")
    print(f"    => 比值 **{np.median(hb)/max(np.median(hn),1e-9):.3f}**"
          f"   破洞率：糊掉 {100*np.mean(hb<0.10):.2f}% vs 沒糊 {100*np.mean(hn<0.10):.2f}%")

    print("""
判讀：
  (3) 的比值 << 1 且糊掉組破洞率明顯較高 => **「那裡沒資訊」成立**，糊掉不是優化失敗
  (3) 的比值 ~ 1                        => 糊掉區的覆蓋與其他區相同 => 假說再次否證
  (2) 破洞多為**低對比**                 => 使用者看到的洞與「40~46% 糊掉」是**兩個不同問題**
                                          （糊掉判準只涵蓋高對比 tile）""")


if __name__ == "__main__":
    main()
