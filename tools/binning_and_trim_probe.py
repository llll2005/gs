#!/usr/bin/env python
"""一次 render loop，兩個答案：① binning 的外接盒浪費有多少 ② trim 改成成本感知會不同嗎

## ① 為什麼問 binning 浪費

`forward.cu:281`：
```c
float radius = ceil(truncated_R * max(max(extent.x, extent.y), FilterSize));
tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
```
**tile 數是用「較長軸」撐出來的正方形外接盒** ⇒ 長寬比 k 的粒子有 `1 - 1/k` 的
binning 是空白。而純 CPU 的代理量測（`tools/elongation_ceiling.py`）給出：
```
speed3_b12 中位長寬比 3.17，>10 的佔 15.3% 顆數／15.8% 不透明度質量
Σs_max² vs Σs_max·s_min => **76.5% 是外接盒空白**
剔除 >10 => binning -31.6% 而真覆蓋只 -7.3%
```
⚠ 但 `s_max²` 忽略了 `truncated_R`（隨 opacity 變、上限 3）與 `FilterSize` 下限
  ⇒ 小粒子被下限主宰、代理會**高估**可回收的比例。這裡用光柵器回傳的**精確**量重算：
```
radii                 光柵器實際用的半徑 => tiles ~ (2r/16)^2
num_covered_pixels    實際做了混色的像素數（精確的 c_i）
浪費 = 1 - Σcovered / Σ(tile 面積)
```
這也是 CityGaussian V2 **Elongation Filter** 針對的機制（「防止單顆覆蓋成千上萬像素」），
而我們把 density controller 換成 MCMC 之後就沒有它了（記憶 `paper_audit`：
「怪物是 3DGS 命名的問題，而 MCMC 把解藥刪掉了」）。

## ② 為什麼問 trim 的判準

trim pass **已經在算精確成本 `c_i`**（`num_covered_pixels` 跨視角求和，
`sep_depth_trim_2dgs_renderer.py:368`，註解自承「累積它是零成本」），
但剪枝判準只看 contribution：
```python
tile = torch.quantile(contribution, self.prune_ratio); prune_mask = contribution <= tile
```
改成 `contribution / c_i`（單位成本的貢獻）是**一行、零額外計算**。
而這是成本感知命題的**第三個位置**：取樣端已否證、約束端未測、**剪枝端從沒想過**。
記憶 `notrim_texture_not_structure` 有直接相關的實測：「**trim 剪掉的是最便宜的那批**
（多 11% 顆粒但 VRAM 反而低）」⇒ 成本感知版會優先刪「貴且沒貢獻」的。

判準（沿用 trim 取樣探針的邏輯）：
```
兩種判準的底部遮罩重疊 > 95%   => 改了也是刪同一批 => 這條線關掉
重疊低 且 成本感知版移除的 Σc_i 明顯更多 => 值得做端到端
```

用法: python tools/binning_and_trim_probe.py --run speed3_b12 --step 60000
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.topk_contribution import contribution_accumulator  # noqa: E402

TILE = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="speed3_b12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--max-cam", type=int, default=284)
    ap.add_argument("--prune-ratio", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    ck = sorted(glob.glob(f"outputs/{a.run}/**/*step={a.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {a.run} @ {a.step}")
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, _ = GaussianModelLoader.\
        initialize_model_and_renderer_from_checkpoint_file(ck[0], device=dev,
                                                           eval_mode=True, pre_activate=False)
    c = torch.load(ck[0], map_location="cpu")
    dmh = c["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    sd = c["state_dict"]
    s = torch.exp(sd["gaussian_model.gaussians.scales"]).double()
    elong = (s.max(1).values / s.min(1).values.clamp_min(1e-12)).to(dev)
    del c
    N = model.n_gaussians
    n = min(a.max_cam, len(cams))
    print(f"\n{a.run} @ {a.step}   N={N:,}   相機 {n}/{len(cams)}\n")

    push, gather = contribution_accumulator(a.K)
    cost = torch.zeros(N, device=dev, dtype=torch.float64)   # Σ num_covered_pixels
    tiles = torch.zeros(N, device=dev, dtype=torch.float64)  # Σ tile 面積（精確半徑算）
    bg = torch.zeros(3, device=dev)
    with torch.no_grad():
        for i in range(n):
            cam = cams[i].to_device(dev)
            mean, covered = renderer(cam, model, bg_color=bg,
                                     record_transmittance=True, record_coverage=True)
            push(mean)
            cost += covered.double()
            r = renderer(cam, model, bg_color=bg).get("radii")
            if r is not None:
                # getRect 的 tile 數 ~ (2r/TILE)^2（外接正方形）
                tiles += ((2.0 * r.double() / TILE).clamp_min(0) ** 2)
            del mean, covered
    contribution = gather()

    print("=== ① binning 的外接盒浪費（精確量，非 s_max² 代理）===")
    T, C = float(tiles.sum()), float(cost.sum())
    print(f"  Σ tile 面積（binning 工作量，單位=像素）= {T:,.0f}")
    print(f"  Σ 實際混色像素（有用的工作）            = {C:,.0f}")
    print(f"  => **浪費 {100*(1-C/max(T,1e-9)):.1f}%**")
    print(f"  {'長寬比門檻':>10} {'顆數':>10} {'佔比':>7} {'binning 省':>11} {'真覆蓋損失':>11} {'交換比':>8}")
    for t in (5, 10, 20, 50):
        m = elong > t
        if not bool(m.any()):
            continue
        db = float(tiles[m].sum()) / max(T, 1e-9)
        dc = float(cost[m].sum()) / max(C, 1e-9)
        print(f"  {'>'+str(t):>10} {int(m.sum()):>10,} {100*float(m.double().mean()):>6.2f}% "
              f"{100*db:>10.2f}% {100*dc:>10.2f}% {db/max(dc,1e-12):>7.1f}:1")

    print("\n=== ② trim 判準：contribution vs contribution/c_i ===")
    k = max(1, int(N * a.prune_ratio))
    base = torch.zeros(N, dtype=torch.bool, device=dev)
    base[contribution.topk(k, largest=False).indices] = True
    ca = (contribution.double() / cost.clamp_min(1.0))
    alt = torch.zeros(N, dtype=torch.bool, device=dev)
    alt[ca.topk(k, largest=False).indices] = True
    ov = float((base & alt).sum()) / k
    print(f"  底部 {100*a.prune_ratio:.0f}% = {k:,} 顆   兩種判準的遮罩重疊 = **{100*ov:.2f}%**")
    print(f"  移除的 Σc_i（渲染成本）：contribution {100*float(cost[base].sum())/C:.2f}%"
          f"  vs  成本感知 {100*float(cost[alt].sum())/C:.2f}%")
    print(f"  移除的 Σcontribution（貢獻）：{float(contribution[base].sum()):.3f}"
          f"  vs  {float(contribution[alt].sum()):.3f}")
    print(f"""
判讀：
  ① 浪費 > 50% 且某個門檻的交換比 > 3:1  => Elongation Filter 有實質目標，值得做
     ⚠ 但「移除」不等於「免費」：被移除的粒子帶著不透明度質量，MCMC 會 relocate
       => 建議的接法是融入 MCMC 的死亡判準（搬走而非刪除），不是硬剪
  ② 重疊 > 95%   => 改判準也是刪同一批，這條線關掉
     重疊低 且 成本感知移除的 Σc_i 明顯更多 => 值得端到端（一行改動）""")


if __name__ == "__main__":
    main()
