#!/usr/bin/env python
"""那些「floater 佔滿 / SfM 破洞」的區域，是不是就是持續糊掉的 40~46%？（純 CPU）

## 假說來源（2026-09-10，使用者的 viewer 觀察）

1500 步四臂探針裡：A/C/D（depth-init）被 floater 佔滿的地方，**B（SfM-init）是破洞**。
```
depth-init  無特徵處也放點（偽深度到處都有值）=> 放錯位置 => 變 floater，但**佔住了**那塊
SfM-init    無特徵處**沒有點**                 => 破洞
```
**⇒ 若那些區域又剛好是那 40~46% 持續糊掉的 tile，整條糊掉調查就換了性質：
不是優化失敗，而是「那裡本來就沒有資訊」** —— 這能解釋為何每個機制解釋都倒（§11.S13）。

## 量什麼

把 COLMAP 的 SfM 稀疏點投影到每台 val 相機，算**逐 tile 的 SfM 點密度**，
再與跨配方的糊掉遮罩（`blur_persistence.per_image`，corr <= r_min）交叉：
```
糊掉率 vs SfM 密度分位     單調下降 => 假說成立（沒特徵 => 糊）
密度中位（糊掉 / 沒糊）    << 1     => 同上
兩者無關                            => 假說否證，糊掉與 SfM 覆蓋無關
```
⚠ **這是相關不是因果**：SfM 密度低的地方通常也是低紋理，而低紋理本來就難重建。
  能分開的追加量測：在**同一 GT 紋理強度**的 tile 內再比一次（本工具已附）。

用法: python tools/sfm_blur_overlap.py agd2_b12 sched30_b12 --blk 12
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--max-cam", type=int, default=36)
    a = ap.parse_args()

    ck = sorted(glob.glob(f"outputs/{a.runs[0]}/**/*step={a.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {a.runs[0]} 的 step={a.step} ckpt")
    c = torch.load(ck[0], map_location="cpu")
    dmh = c["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    out = dp.get_outputs()
    pts = getattr(out, "point_cloud", None)
    xyz = np.asarray(pts.xyz if pts is not None and hasattr(pts, "xyz") else [], np.float64)
    if xyz.size == 0:
        raise SystemExit("拿不到 SfM 點雲（dataparser 沒回傳 point_cloud）")
    print(f"SfM 點 {len(xyz):,} 顆   tile {a.tile}x{a.tile}   糊掉判準 = 所有配方 corr <= {a.r_min}")

    vset = out.val_set
    dirs = {r: final_test_dir(r, a.blk) for r in a.runs}
    fs = sorted(f for f in os.listdir(dirs[a.runs[0]]) if f.endswith(".png"))
    masks = {}
    for f in fs:
        o = {r: per_image(os.path.join(dirs[r], f), a.tile, a.contrast_q) for r in a.runs}
        if any(v is None for v in o.values()):
            continue
        keep, *_z, nx, ny = o[a.runs[0]]
        ms = np.stack([o[r][1] <= a.r_min for r in a.runs])
        name = f[:-4] if f.endswith(".png.png") else f
        masks[name] = (keep, ms.all(0), nx, ny, o[a.runs[0]][3])   # [3] = GT 對比 sg2

    rows = []   # (SfM 密度, 是否糊掉, GT 對比)
    used = 0
    from PIL import Image as _Im
    for i in range(min(len(vset), a.max_cam)):
        name, _p, _m, cam, _e = vset[i]
        name = os.path.basename(str(name))
        if name not in masks:
            continue
        keep, blur, nx, ny, sg = masks[name]
        R = np.asarray(cam.R.cpu() if torch.is_tensor(cam.R) else cam.R, np.float64)
        T = np.asarray(cam.T.cpu() if torch.is_tensor(cam.T) else cam.T, np.float64)
        fx = float(cam.fx); W, H = int(cam.width), int(cam.height)
        pc = xyz @ R.T + T
        z = pc[:, 2]; zc = np.clip(z, .2, None)
        u = fx * pc[:, 0] / zc + W / 2
        v = fx * pc[:, 1] / zc + H / 2
        ok = (z > .2) & (u >= 0) & (u < nx * a.tile) & (v >= 0) & (v < ny * a.tile)
        ti = (v[ok] // a.tile).astype(np.int64) * nx + (u[ok] // a.tile).astype(np.int64)
        dens = np.bincount(ti, minlength=nx * ny).astype(np.float64)
        for j, k in enumerate(keep):
            rows.append((dens[k], bool(blur[j]), float(sg[j])))
        used += 1

    A = np.array([(r[0], r[1], r[2]) for r in rows])
    print(f"使用 {used} 台相機，{len(A):,} 個高對比 tile   整體糊掉率 {100*A[:,1].mean():.2f}%\n")

    print(f"{'SfM 點數/tile 分位':>22} {'n':>8} {'糊掉%':>9} {'密度中位':>10}")
    q = np.quantile(A[:, 0], np.linspace(0, 1, 6)); q[-1] += 1e-6
    for i in range(5):
        m = (A[:, 0] >= q[i]) & (A[:, 0] < q[i + 1])
        if m.sum() < 10: continue
        print(f"{q[i]:>9.1f}~{q[i+1]:<11.1f} {m.sum():>8,} {100*A[m,1].mean():>8.2f}% {np.median(A[m,0]):>10.1f}")
    b, nb = A[A[:, 1] == 1], A[A[:, 1] == 0]
    print(f"\n  SfM 密度中位：糊掉 {np.median(b[:,0]):.1f} ／ 沒糊 {np.median(nb[:,0]):.1f}"
          f"  => 比值 **{np.median(b[:,0])/max(np.median(nb[:,0]),1e-9):.3f}**")
    print(f"  零 SfM 點的 tile 糊掉率 {100*A[A[:,0]==0,1].mean():.2f}%"
          f"（n={int((A[:,0]==0).sum()):,}） vs 有點的 {100*A[A[:,0]>0,1].mean():.2f}%")

    # ★ 控制 GT 紋理強度後再比一次（分開「沒特徵」與「低紋理本來就難」）
    print(f"\n  ★ 控制 GT 對比後（同一紋理強度五分位內再比）")
    print(f"{'GT 對比分位':>18} {'低 SfM 密度 糊掉%':>20} {'高 SfM 密度 糊掉%':>20} {'差':>8}")
    qc = np.quantile(A[:, 2], np.linspace(0, 1, 6)); qc[-1] += 1e-6
    for i in range(5):
        m = (A[:, 2] >= qc[i]) & (A[:, 2] < qc[i + 1])
        if m.sum() < 40: continue
        sub = A[m]; med = np.median(sub[:, 0])
        lo, hi = sub[sub[:, 0] <= med], sub[sub[:, 0] > med]
        if len(lo) < 10 or len(hi) < 10: continue
        print(f"{qc[i]:>8.4f}~{qc[i+1]:<9.4f} {100*lo[:,1].mean():>19.2f}% "
              f"{100*hi[:,1].mean():>19.2f}% {100*(lo[:,1].mean()-hi[:,1].mean()):>+7.2f}")
    print("""
判讀：
  糊掉率隨 SfM 密度**單調下降**，且**控制 GT 對比後仍在**
    => 「那裡沒有資訊」成立 => 糊掉不是優化失敗 => §11.S13 的機制調查可以結案
  控制 GT 對比後差異消失
    => 只是「低紋理本來就難」，與 SfM 覆蓋無關 => 假說否證""")


if __name__ == "__main__":
    main()
