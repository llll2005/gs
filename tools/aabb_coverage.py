#!/usr/bin/env python
"""b12 的邊界到底在哪？切掉界外點會砍掉什麼？（純 CPU 讀 ckpt，可選輸出裁切後的 ckpt）

## 三種「邊界」，不要混用

```
① partition AABB   partitions.pt 的格子範圍 —— **只有 xy，z 無界**，而且非均勻分割
                   b12（格座標 [2,2]）只有 2.861 x 2.588，而整個場景是 102 x 80
② SfM 點盒         該塊相機看到的 SfM 點的 1~99 百分位盒 —— 「真表面在哪」的參考
③ 模型自身範圍     ckpt 裡所有粒子的 min/max —— 含場外 floater，會被少數極端值拉爆
```
⚠⚠ **不要用 ① 當裁切標準。** 記憶 `graded_init` 實測：partition AABB
   「只涵蓋 depth-init 足跡的 1/6 ⇒ 錨點只剩 2.5%」，副產物是
   **block 之間內容重疊約 6 倍面積**。也就是說 b12 訓練時**正當地**重建了格子以外的東西
   （因為它的相機看得到），照 ① 切會砍掉合法內容。
   ① 的正確用途是**合併時決定歸屬**（`merge_citygs_ckpts.py` 做的事），不是裁切單塊。

## 量什麼

對每種邊界，報「界內 / 界外」的三個量：
```
顆數          最直觀，但會被次像素塵埃灌水
不透明度質量   Σopacity —— 決定它們在混色裡真的有多少份量
渲染成本代理   Σ(s_max²) —— binning 是**較長軸撐出的正方形外接盒**（forward.cu:281）
```
⚠ 用 `s_max²` 而不是 `s_max*s_min`：tile 數由外接盒決定，見 `elongation_binning_waste`。

## 為什麼這個問題現在重要

`tools/novel_view_coverage.py` 量到 SfM-init 在 60k 仍有 **5.66%** 的粒子在 SfM 盒外
（depth-init 只有 2.11%），而 1500 步時那些場外 floater 讓 `rend_alpha` 從 0.9935
掉到只算盒內的 0.5139 —— 也就是「巨大的場外幕布」。裁切能直接看到真實覆蓋。

用法:
  python tools/aabb_coverage.py --run speed3_b12 --block 12
  python tools/aabb_coverage.py --run speed3_b12 --block 12 --crop sfm --out /tmp/cropped.ckpt
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PART = ("data/matrix_city/aerial/train/block_all/partition/"
        "partitions-dim_5_5_visibility_0.08/partitions.pt")


def partition_aabb(block):
    d = torch.load(PART, map_location="cpu")
    xy = d["partition_coordinates"]["xy"][block].double()
    sz = torch.as_tensor(d["scene_config"]["partition_size"][block]).double()
    sb = d["scene_bounding_box"]["bounding_box"]
    return xy, xy + sz, sb["min"].double(), sb["max"].double(), \
        d["partition_coordinates"]["id"][block].tolist()


def report(name, inside, o, cost, n):
    ins = int(inside.sum())
    print(f"  {name:<22} 界內 {ins:>9,} ({100*ins/n:>6.2f}%)   "
          f"不透明度質量 {100*float(o[inside].sum())/max(float(o.sum()),1e-9):>6.2f}%   "
          f"成本代理 {100*float(cost[inside].sum())/max(float(cost.sum()),1e-9):>6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="speed3_b12")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--crop", choices=["partition", "sfm", "none"], default="none")
    ap.add_argument("--out", default="")
    ap.add_argument("--pad", type=float, default=0.0, help="partition AABB 外擴（世界單位）")
    a = ap.parse_args()

    ck = sorted(glob.glob(f"outputs/{a.run}/**/*step={a.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {a.run} @ {a.step}")
    c = torch.load(ck[0], map_location="cpu")
    sd = c["state_dict"]
    xyz = sd["gaussian_model.gaussians.means"].double()
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"]).squeeze(-1).double()
    s = torch.exp(sd["gaussian_model.gaussians.scales"]).double()
    cost = s.max(1).values ** 2                      # binning ∝ 外接正方形
    N = xyz.shape[0]

    pmin, pmax, smin, smax, gid = partition_aabb(a.block)
    print(f"\n{a.run} @ {a.step}   block {a.block}（格座標 {gid}）   N={N:,}\n")
    print("=== 邊界定義 ===")
    print(f"  ① partition AABB (xy)  {pmin.tolist()} ~ {pmax.tolist()}"
          f"   大小 {(pmax-pmin).tolist()}")
    print(f"     場景整體 (xy)        {smin.tolist()} ~ {smax.tolist()}"
          f"   大小 {(smax-smin).tolist()}")
    print(f"  ③ 模型自身 (xyz)        min {xyz.min(0).values.tolist()}")
    print(f"                         max {xyz.max(0).values.tolist()}")

    # SfM 點盒（② ）—— 用 dataparser 拿該塊的點雲
    sfm_box = None
    try:
        dmh = c["datamodule_hyper_parameters"]
        dp = dmh["parser"].instantiate(path=dmh["path"],
                                       output_path=os.path.dirname(os.path.dirname(ck[0])),
                                       global_rank=0)
        pcd = torch.as_tensor(np.asarray(dp.get_outputs().point_cloud.xyz), dtype=torch.float64)
        lo = torch.quantile(pcd, 0.01, dim=0); hi = torch.quantile(pcd, 0.99, dim=0)
        sfm_box = (lo, hi)
        print(f"  ② SfM 點盒 1~99% (xyz) min {lo.tolist()}")
        print(f"                         max {hi.tolist()}")
    except Exception as e:
        print(f"  ② SfM 點盒：拿不到（{type(e).__name__}）")

    print("\n=== 各邊界的界內佔比（顆數 / 不透明度質量 / binning 成本代理）===")
    pad = a.pad
    in_part = ((xyz[:, 0] >= pmin[0] - pad) & (xyz[:, 0] <= pmax[0] + pad) &
               (xyz[:, 1] >= pmin[1] - pad) & (xyz[:, 1] <= pmax[1] + pad))
    report(f"① partition AABB", in_part, o, cost, N)
    in_scene = ((xyz[:, 0] >= smin[0]) & (xyz[:, 0] <= smax[0]) &
                (xyz[:, 1] >= smin[1]) & (xyz[:, 1] <= smax[1]))
    report("  場景整體 (xy)", in_scene, o, cost, N)
    in_sfm = None
    if sfm_box is not None:
        lo, hi = sfm_box
        in_sfm = ((xyz >= lo) & (xyz <= hi)).all(dim=1)
        report("② SfM 點盒", in_sfm, o, cost, N)

    print("""
判讀：
  ⚠⚠ **不要用 ① 裁切單塊**。記憶 `graded_init` 實測 partition AABB 只涵蓋 depth-init
     足跡的 1/6（block 之間內容重疊約 6 倍面積）—— b12 訓練時**正當地**重建了格子以外的
     東西（它的相機看得到）。① 的用途是**合併時決定歸屬**，不是裁切。
  ② SfM 點盒是「真表面在哪」的合理代理（`novel_view_coverage.py` 用的就是它）。
     界外那批若「顆數少但成本代理高」=> 就是巨大的場外 floater（幕布）。""")

    if a.crop != "none":
        keep = in_part if a.crop == "partition" else in_sfm
        if keep is None:
            raise SystemExit("拿不到 SfM 盒，無法用 --crop sfm")
        out = a.out or f"/tmp/{a.run}_step{a.step}_crop-{a.crop}.ckpt"
        pre = "gaussian_model.gaussians."
        names = [k for k in sd if k.startswith(pre)]
        for k in names:
            sd[k] = sd[k][keep]
        # ⚠ 優化器狀態也要一起裁，否則長度不一致 => 下次載入/續訓會壞
        for st in c.get("optimizer_states", []):
            for gi, g in enumerate(st.get("param_groups", [])):
                stt = st.get("state", {}).get(gi)
                if not stt:
                    continue
                for mk in ("exp_avg", "exp_avg_sq"):
                    if mk in stt and stt[mk].shape[0] == keep.shape[0]:
                        stt[mk] = stt[mk][keep]
        torch.save(c, out)
        print(f"✅ 已裁切（{a.crop}）：{int(keep.sum()):,} / {N:,} 顆 -> {out}")
        print("   ⚠ 這是**部署/檢視用**的產物，不要拿它的分數與未裁切的跑次比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
