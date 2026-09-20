#!/usr/bin/env python
"""標定 `cost_budget` 的單位：現在的操作點，Load 是多少？（單次前向，不用訓練）

## 為什麼需要標定

`cost_budget` 把生長條件從 `N <= cap_max` 換成 `max_view Σ_i (2r_i/16)^2 <= cost_budget`。
旗標的 docstring 明文警告：**不可沿用 `tools/cost_budget_probe.py` 的 23.1M**
——那支用解析投影，而生效的是**光柵器 radii**，同定義不同實作路徑。

原本的標定方式是跑一次 60k 訓練、開 `cost_budget_report`。但 Load 只跟**模型與相機**有關，
不需要訓練：拿已訓練好的 ckpt、用**同一個 renderer** 取 radii 算一次就好。
⇒ 15 小時的 lab 訓練換成本機幾分鐘。

## 為什麼要知道這個數

2026-09-13 `tools/rho_value_cost.py` 量到：背包的天花板是**預算鬆緊**的函數
（預算佔總成本 10% => +57%，75% => +1~2%，兩塊一致）。
而被否證的三個成本感知變體跑在 `cap_max 2.6M` ⇒ 落在**寬鬆端**。
⇒ 介入要排在**緊預算**那一側，而「現在落在哪一格」就是這支工具要回答的。

用法: python tools/cost_budget_calibrate.py --ckpt 'outputs/.../*step=15000.ckpt' --max-cam 150
"""
import argparse
import glob
import os
import sys

import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-cam", type=int, default=150)
    ap.add_argument("--tile", type=int, default=16, help="(2r/tile)^2 的 tile 邊長")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    cks = sorted(glob.glob(args.ckpt))
    if len(cks) != 1:
        raise SystemExit(f"--ckpt 要展開成唯一一個檔，目前 {len(cks)} 個")
    ck = cks[0]

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck, device=dev, eval_mode=False, pre_activate=False)
    N = model.n_gaussians
    ckpt = torch.load(ck, map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck)),
                                   global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    n = min(args.max_cam, len(cams))
    print(f"來源 {ck}\n  N = {N:,}   相機 {n}/{len(cams)}   tile = {args.tile}")

    bg = torch.zeros((3,), device=dev)
    loads, loads_floor, loads_exact, loads_ana = [], [], [], []
    # ⓪ 解析路徑（`cost_budget_probe.py` 的公式）：r = 3·fx·s_max/z·stretch，
    #    tiles = (2r/TILE)^2 clamp 到 [1, 整幀 tile 數]，再用視錐粗剔。
    #    它與 ①② 的差別是**半徑的來源**（解析投影 vs 光柵器），不只是有沒有下限。
    _xyz = model.get_xyz.detach()
    _smax = model.get_scales().detach()[:, :2].max(dim=1).values
    with torch.no_grad():
        for i in range(n):
            out = renderer(cams[i].to_device(dev), model, bg_color=bg)
            r = out.get("radii")
            if r is None:
                raise SystemExit("⛔ renderer 沒回傳 radii，拿不到 Load")
            q = ((2.0 * r.float()) / args.tile) ** 2
            loads.append(float(q.sum()))
            # ★ 2026-09-21：同一組相機上比三條路徑。
            #   ① 現行代理 Σ(2r/16)^2 —— **沒有量化下限**，所以半徑 3px 的顆粒算成 0.14 個 tile，
            #      但它只要可見就至少碰 1 個 => 小顆粒被系統性低估。
            #   ② 代理 + 量化下限 Σ max(q, 1)（只對可見粒子）—— `cost_budget_probe.py` 用的正是
            #      這個 clamp(min=1)，這也是它比 calibrate 大的原因。
            #   ③ 精確：光柵器回傳的逐顆 tile 數（已驗 num_rendered == Σtiles）。
            vis = r > 0
            loads_floor.append(float(torch.clamp(q[vis], min=1.0).sum()))
            t = out.get("tiles")
            loads_exact.append(float(t.double().sum()) if t is not None else float("nan"))
            cam = cams[i].to_device(dev)
            W_, H_ = int(cam.width), int(cam.height)
            fx_ = W_ / (2.0 * math.tan(float(cam.fov_x) * 0.5))
            nft = (W_ / args.tile) * (H_ / args.tile)
            vm = cam.world_to_camera
            pc = _xyz @ vm[:3, :3] + vm[3, :3] if vm.shape == (4, 4) else None
            if pc is None:
                loads_ana.append(float("nan"))
            else:
                z = pc[:, 2]
                m2 = z > 0.2
                if not bool(m2.any()):
                    loads_ana.append(0.0)
                else:
                    pcv, zc = pc[m2], z[m2]
                    u = fx_ * pcv[:, 0] / zc + W_ / 2
                    v = fx_ * pcv[:, 1] / zc + H_ / 2
                    st = torch.sqrt(1 + (pcv[:, 0] / zc) ** 2 + (pcv[:, 1] / zc) ** 2)
                    ra = 3.0 * fx_ * _smax[m2] / zc * st
                    inf = (u > -ra) & (u < W_ + ra) & (v > -ra) & (v < H_ + ra)
                    loads_ana.append(float(((2 * ra[inf] / args.tile) ** 2)
                                           .clamp(min=1.0, max=nft).sum()) if bool(inf.any()) else 0.0)
    loads = np.array(loads)
    loads_floor = np.array(loads_floor)
    loads_exact = np.array(loads_exact)
    loads_ana = np.array(loads_ana)
    L = loads.max()
    print(f"\n  Load（每視角 Σ (2r/{args.tile})^2）")
    print(f"    max   **{L:,.0f}**   <- 這就是 cost_budget 的單位與當前操作點")
    print(f"    p95   {np.percentile(loads,95):,.0f}")
    print(f"    中位   {np.median(loads):,.0f}")
    print(f"    min   {loads.min():,.0f}")
    if np.isfinite(loads_exact).all():
        print(f"\n  ★★ 三條量測路徑（同 ckpt、同 {n} 台相機）")
        print(f"    {'':<34}{'max':>16}{'中位':>16}{'vs 精確':>10}")
        for lab, arr in (("⓪ 解析投影（cost_budget_probe 的公式）", loads_ana),
                         ("① 現行代理 Σ(2r/16)^2（無量化下限）", loads),
                         ("② 代理 + 量化下限 Σ max(q,1)", loads_floor),
                         ("③ 精確 Σtiles（光柵器）", loads_exact)):
            rel = np.median(arr) / max(np.median(loads_exact), 1e-9)
            print(f"    {lab:<34}{arr.max():>16,.0f}{np.median(arr):>16,.0f}{rel:>9.3f}x")
        print("    ⇒ ⓪ 與 ① 的差距＝**半徑來源不同**（解析投影 vs 光柵器 radii），不是下限造成的。")
        print("    ⇒ ① 與 ② 的差距＝量化下限那一項。在**訓練後**的模型上它幾乎不生效")
        print("       （顆粒已經夠大），但在**訓練初期**它主導 —— 實測 SfM init 時代理只有精確的 0.280 倍。")
        print("    ⇒ ② 與 ③ 的差距＝**正方形外接盒**的高估，這是訓練後期的主要誤差來源。")
        print("    ★ 權威順序：③ 精確 > ① 光柵器代理（cost_budget 現行單位）> ⓪ 解析（只可當上界）。")
    # 2026-09-14 補（使用者問「有總成本嗎」）：中位數只代表典型視角；
    #   訓練每步隨機渲染一台相機 => 期望每步 binning 成本＝**平均**；全部相機加總＝走一遍所有視角的總成本。
    #   ⚠ 這仍是「終點模型」的成本，不是整趟訓練的積分（族群在訓練中會變）。
    print(f"    平均   {loads.mean():,.0f}   <- 期望每步成本（每步隨機一台相機）")
    print(f"    總和   {loads.sum():,.0f}   （{len(loads)} 台相機加總）")
    print(f"    B/N   {L/N:.2f}  （每顆平均攤到多少 Load）")
    print(f"\n  ★ 緊預算的候選設定（rho_value_cost.py 量到天花板隨預算收緊而跳）")
    for f in (0.75, 0.5, 0.25, 0.10):
        print(f"    {f*100:>5.0f}% B0  =>  --model.density.init_args.cost_budget {L*f:,.0f}")
    print("""
⚠ 保留：
  ① 這是**已訓練好的模型**的 Load；訓練期 Load 隨 N 成長，約束是在成長途中生效的
     => 設 B < B0 會讓族群停在更早的地方，那正是「緊預算」的意思
  ② `cap_max` 仍要保留當安全閥（VRAM = 逐顆儲存只看 N ＋ binning 只看成本）
  ③ 相機只取了一部分；max 是對這些視角取的，全部相機的 max 只會更大""")


if __name__ == "__main__":
    main()
