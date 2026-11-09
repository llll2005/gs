#!/usr/bin/env python
"""輕量 coarse 的前後處理（使用者 11-09「作一個一樣輕量的 coarse（基於我們當前的 dup 點雲）作為前訓練」）。

## 為什麼
- 11-06 CityGS 起點（`實驗分析/35` §6）：起點的形狀與朝向會留下來（燈柱撐到 30k、> 250 m 立面對齊 +16.6 點），
  但同一個 30k 排程時間沒變（38.3 vs 38.1 分）=> 要省時間得靠「起點好 => 增生期可以短」。
- 增生期是貴的那段（lab A＋dist4 每步約 27 ms、之後約 20 ms；記憶 lab_fullscene_time_breakdown_1109）。
- 全域全解析度跑不動（10-31 全域 3.2M：187 ms／步、峰值 19.3 GB、RAM 被殺）=> coarse 用 1/4 訓練解析度＋SH0。

## 兩個子指令
init   各塊 `sfmfill_sweep44/dup1/block_<B>.ply` 只留自己盒內（不外擴；外圈塊本來就延伸到場景邊）拼成全域，
       隨機抽到 --n 顆 => `<data>/coarse_init/dup1_g44_<n/1000>k.ply`（全域 coarse 的起點）
crop   全域 coarse 的 ckpt 裁到各塊盒外擴 --m_c（同 `block_mask44.py --crop_ply` 的外擴，預設 0.15 x 內側塊尺寸），
       再把最大的面元在面內一拆四（2x2、兩軸尺度各減半、opacity／顏色照抄），顆數拆到 <= --frac x 該塊 cap 為止；
       寫成跑次的樣子 `outputs/<out>/blocks/block_<B>/checkpoints/coarse_crop.ckpt`（訓練用 `--model.initialize_from`；
       SH0 -> SH3 由 `_initialize_from_trained_model` 補零）。拆分的理由：coarse 在 1/4 解析度，一顆面元約等於細修的 4 x 4 px，
       不拆的話每塊起點只有約 0.2M 顆，增生期得從頭長回 cap，排程就短不了。

用法（lab，CPU）：
  python tools/coarse_crop.py init --n 1800000
  python tools/coarse_crop.py crop --ckpt <coarse ckpt> --out lab/<名>_crop --blocks 1 3 4
座標同 `tools/block_mask44.py`：partitions.pt 的 rotation_transform 轉換後取 xy，盒＝[xy_b, xy_b + size_b]。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def boxes(data, part):
    parts = torch.load(os.path.join(data, part, "partitions.pt"))
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    ids = parts["partition_coordinates"]["id"].numpy()
    xys = parts["partition_coordinates"]["xy"].double().numpy()
    size = parts["scene_config"]["partition_size"].double().numpy()
    inner = np.median(np.minimum(size[:, 0], size[:, 1]))
    ref = np.median(size[(size[:, 0] < 10 * inner) & (size[:, 1] < 10 * inner)], axis=0)   # 內側塊的典型尺寸
    out = {}
    for b in range(16):
        k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0])
        out[b] = (xys[k], xys[k] + size[k])
    return Rp, out, float(ref.min())


def in_box(xy, lo, hi):
    return (xy[..., 0] >= lo[0]) & (xy[..., 0] <= hi[0]) & (xy[..., 1] >= lo[1]) & (xy[..., 1] <= hi[1])


def cmd_init(a):
    from plyfile import PlyData, PlyElement
    Rp, bx, _ = boxes(a.data, a.part)
    src = os.path.join(a.data, "sfmfill_sweep44", a.dup)
    parts, tot = [], 0
    for b in range(16):
        v = PlyData.read(os.path.join(src, f"block_{b}.ply"))["vertex"].data
        xy = (np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64) @ Rp.T)[:, :2]
        keep = in_box(xy, *bx[b])
        parts.append(v[keep]); tot += len(v)
        print(f"  block {b}：{len(v):,} -> 盒內 {int(keep.sum()):,}", flush=True)
    allv = np.concatenate(parts)
    rng = np.random.default_rng(a.seed)
    if len(allv) > a.n:
        allv = allv[np.sort(rng.choice(len(allv), a.n, replace=False))]
    od = os.path.join(a.data, "coarse_init"); os.makedirs(od, exist_ok=True)
    of = os.path.join(od, f"{a.dup}_g44_{len(allv) // 1000}k.ply")
    PlyData([PlyElement.describe(allv, "vertex")], text=False).write(of)
    print(f"全域起點：各塊 {a.dup} 合計 {tot:,}、自己盒內拼起來 {sum(len(p) for p in parts):,} -> 抽 {len(allv):,} => {of}")


def quat_to_R(q):
    q = q / q.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
                        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
                        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def cmd_crop(a):
    Rp, bx, ref = boxes(a.data, a.part)
    mc = a.m_c * ref
    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["state_dict"]
    pre = "gaussian_model.gaussians."
    N = sd[pre + "means"].shape[0]
    per = [k for k, v in sd.items() if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N]
    xy = (sd[pre + "means"].double().numpy() @ Rp.T)[:, :2]
    caps = json.load(open(a.alloc))["weights"]
    print(f"coarse：N {N:,}；外擴 {mc:.4f}（{a.m_c} x 內側塊 {ref:.3f}）；逐顆張量 {len(per)} 個")
    summary = {}
    for b in a.blocks:
        lo, hi = bx[b]
        keep = torch.from_numpy(in_box(xy, lo - mc, hi + mc))
        sub = {k: sd[k][keep].clone() for k in per}
        n0 = int(keep.sum())
        if n0 == 0:
            print(f"  ⛔ block {b}：coarse 在這塊盒內＋外擴沒有任何顆粒 => 不寫（細修會找不到起點）", flush=True)
            continue
        cap = int(round(a.avg_cap * float(caps[str(b)])))
        target = int(a.frac * cap)
        n_split = max(0, min(n0, (target - n0) // 3)) if a.split else 0
        if n_split > 0:
            s = sub[pre + "scales"].exp()                       # [n, 2]，面元兩軸的標準差
            order = torch.argsort(s.prod(-1), descending=True)[:n_split]   # 最大的先拆
            R = quat_to_R(sub[pre + "rotations"][order].float())
            u, v = R[:, :, 0], R[:, :, 1]                        # 面內兩軸
            su, sv = s[order, 0:1], s[order, 1:2]
            kids = {}
            for k in per:
                kids[k] = sub[k][order].repeat_interleave(4, 0).clone()
            off = torch.cat([(du * 0.5 * su * u + dv * 0.5 * sv * v).unsqueeze(1)
                             for du, dv in ((-1, -1), (-1, 1), (1, -1), (1, 1))], 1).reshape(-1, 3)
            kids[pre + "means"] = kids[pre + "means"] + off.to(kids[pre + "means"].dtype)
            kids[pre + "scales"] = kids[pre + "scales"] - float(np.log(2.0))
            rest = torch.ones(n0, dtype=torch.bool); rest[order] = False
            sub = {k: torch.cat([sub[k][rest], kids[k]]) for k in per}
        n1 = sub[pre + "means"].shape[0]
        out_sd = dict(sd); out_sd.update(sub)
        out = dict(ck); out["state_dict"] = out_sd; out["optimizer_states"] = []; out["global_step"] = 0; out["epoch"] = 0
        od = os.path.join("outputs", a.out, "blocks", f"block_{b}", "checkpoints"); os.makedirs(od, exist_ok=True)
        torch.save(out, os.path.join(od, "coarse_crop.ckpt"))
        summary[b] = dict(in_box=n0, split=n_split, n=n1, cap=cap)
        print(f"  block {b}：盒內＋外擴 {n0:,}、拆 {n_split:,} 顆 x4 => {n1:,}（cap {cap:,}，{100 * n1 / cap:.0f}%）=> {od}/coarse_crop.ckpt", flush=True)
    json.dump(summary, open(os.path.join("outputs", a.out, "crop_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "crop"])
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--dup", default="dup1", help="init：用 sfmfill_sweep44 底下哪一份 dup")
    ap.add_argument("--n", type=int, default=1800000, help="init：全域起點顆數（要 <= 0.9 x coarse cap）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", help="crop：全域 coarse 的 ckpt")
    ap.add_argument("--out", help="crop：輸出跑次名（outputs/ 底下，例 lab/coarse_crop）")
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--m_c", type=float, default=0.15, help="crop：盒外擴（x 內側塊尺寸；同 block_mask44 的 m_c）")
    ap.add_argument("--alloc", default="configs/cap_alloc_sfm_g44.json", help="crop：各塊 cap 權重")
    ap.add_argument("--avg_cap", type=float, default=1e6, help="crop：平均 cap（各塊 cap＝avg x 權重）")
    ap.add_argument("--frac", type=float, default=0.85, help="crop：拆到 <= frac x cap 為止")
    ap.add_argument("--split", type=int, default=1, help="crop：1＝最大的面元一拆四；0＝不拆")
    a = ap.parse_args()
    cmd_init(a) if a.cmd == "init" else cmd_crop(a)


if __name__ == "__main__":
    main()
