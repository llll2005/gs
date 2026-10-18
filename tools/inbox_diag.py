#!/usr/bin/env python
"""塊內模型的三個目視問題，逐一量（使用者 10-18 看 g44s15k_best1sh1_inbox 的觀察）。

1 黑色變透明：訓練背景固定黑 (0,0,0)（GaussianSplatting.background_color 預設，random_background=false）
  => 暗表面可以用「透明＋黑背景」解釋。量：訓練視角中 GT 暗像素（亮度 < --dark）與亮像素各自的渲染 alpha < 0.5 比例。
2 地底高斯：盒內顆粒中心 z（分區座標）低於地面（SfM 5 百分位）--below 以上的比例，按顆數與 opacity x 面積加權。
3 邊緣：同一視角，完整模型 vs 只留盒內顆粒（判定同 merge），在「第一表面落在盒內」的像素上，依離盒邊的距離分帶，
  比較 alpha 的保留率 => 若靠邊帶掉很多，是盒外中心的顆粒本來在蓋盒內（切法本身的性質），不是切錯。

用法：python tools/inbox_diag.py --ckpt <完整單塊 ckpt> [--views 24]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--dark", type=float, default=0.06)
    ap.add_argument("--below", type=float, default=0.01, help="地面以下多少才算地底（場景單位；0.01 ≈ 1 m）")
    ap.add_argument("--no_render", action="store_true", help="只算地底統計（CPU 也行）")
    ap.add_argument("--ref_ckpt", default=None, help="相機與分區盒取自這個 ckpt（評官方模型時給我方同塊 ckpt => 同一批視角、同一個盒）")
    a = ap.parse_args()
    dev = torch.device("cuda")
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from crop_inbox_ckpt import inbox_mask
    from internal.utils.colmap import read_points3D_binary
    ck = torch.load(a.ref_ckpt or a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]; pcfg = dmh["parser"]
    del ck
    if getattr(pcfg, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], pcfg.ply_file)):
        pcfg.points_from = "random"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        if hasattr(renderer, f):
            setattr(renderer, f, False)
    means = model.get_xyz.detach()
    inb = inbox_mask(means.cpu(), pcfg, 0.0).to(dev)
    parts = torch.load(os.path.join(os.path.dirname(pcfg.image_list), "partitions.pt"))
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].float().to(dev)
    k = int(np.where((parts["partition_coordinates"]["id"][:, 0] == pcfg.block_id % 4) & (parts["partition_coordinates"]["id"][:, 1] == pcfg.block_id // 4))[0][0])
    lo = parts["partition_coordinates"]["xy"][k].float().to(dev); hi = lo + parts["scene_config"]["partition_size"][k].float().to(dev)
    pts = read_points3D_binary(os.path.join(dmh["path"], "sparse/0/points3D.bin"))
    P = torch.tensor(np.stack([v.xyz for v in pts.values()]), dtype=torch.float32) @ Rp.cpu().T
    lc, hc = lo.cpu(), hi.cpu()
    inP = (P[:, 0] >= lc[0]) & (P[:, 0] <= hc[0]) & (P[:, 1] >= lc[1]) & (P[:, 1] <= hc[1])
    zs = P[inP, 2]
    zg = float(torch.quantile(zs[torch.randperm(len(zs))[:200000]], 0.05))     # ★ 本塊盒內的 SfM 點（局部地形）

    # ── 2 地底高斯（盒內）
    pr = means @ Rp.T
    o = model.get_opacity.detach().reshape(-1)
    sc = model.get_scaling.detach()
    area = sc[:, 0] * sc[:, 1] if sc.shape[1] >= 2 else sc[:, 0] ** 2
    m_in = inb
    below = m_in & (pr[:, 2] < zg - a.below)
    w = o * area
    print(f"N {means.shape[0]:,}；盒內 {int(m_in.sum()):,}；本塊地面 z={zg:.4f}（盒內 SfM 點 5 百分位，{int(inP.sum()):,} 點）")
    print(f"[2 地底] 盒內顆粒中心低於地面 {a.below}：顆數 {100 * float(below.sum()) / max(int(m_in.sum()), 1):.2f}%，"
          f"opacity x 面積 {100 * float(w[below].sum() / w[m_in].sum()):.2f}%；"
          f"這批的尺度中位 {float(sc[below].max(1).values.median()) if below.any() else 0:.4f} vs 盒內全部 {float(sc[m_in].max(1).values.median()):.4f}")
    for t in (0.01, 0.03, 0.1):
        b2 = m_in & (pr[:, 2] < zg - t)
        print(f"     低於地面 {t}：顆數 {100 * float(b2.sum()) / max(int(m_in.sum()), 1):.2f}%  權重 {100 * float(w[b2].sum() / w[m_in].sum()):.2f}%")

    if a.no_render:
        return
    # ── 1、3 需要渲染
    dp = pcfg.instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ref_ckpt or a.ckpt)), global_rank=0)
    ts = dp.get_outputs().train_set
    idx = np.linspace(0, len(ts.cameras) - 1, a.views).astype(int)
    bg = torch.zeros(3, device=dev)
    props_full = {kk: v for kk, v in model.properties.items()}
    props_in = {kk: v[inb] for kk, v in props_full.items()}
    dark_a = []; bright_a = []; bands = np.zeros((4, 2)); dark_in = []
    Vacc = torch.zeros(means.shape[0], device=dev)
    edges = [0, 0.05, 0.15, 0.4, 1e9]       # 離盒邊距離（x 內側塊短邊；外圈塊的盒子延伸到場景邊，不能用自己的短邊）
    _sz = parts["scene_config"]["partition_size"].float()
    _inner = float(torch.median(torch.minimum(_sz[:, 0], _sz[:, 1])))
    short = float(torch.median(torch.minimum(_sz[:, 0], _sz[:, 1])[(_sz[:, 0] < 10 * _inner) & (_sz[:, 1] < 10 * _inner)]))
    for i in idx:
        cam = ts.cameras[int(i)].to_device(dev)
        gt = torch.from_numpy(np.asarray(Image.open(ts.image_paths[int(i)]).convert("RGB").resize((int(cam.width), int(cam.height))), dtype=np.float32) / 255.).to(dev)
        lum = gt.mean(-1)
        with torch.no_grad():
            model.properties = props_full
            of = renderer(cam, model, bg_color=bg)
            af = of["rend_alpha"].reshape(lum.shape); dep = of["surf_depth"].reshape(lum.shape)
            model.properties = props_in
            ai = renderer(cam, model, bg_color=bg)["rend_alpha"].reshape(lum.shape)
        dk = lum < a.dark; br = lum > 0.25
        if i == idx[0]:
            print(f"  （檢查）alpha 分位 p1/p5/p50 = {[round(float(torch.quantile(af.reshape(-1)[::97], q)), 3) for q in (0.01, 0.05, 0.5)]}；暗像素佔 {100 * float(dk.float().mean()):.1f}%")
        dark_a.append(float((af[dk] < 0.5).float().mean()) if dk.any() else np.nan)
        bright_a.append(float((af[br] < 0.5).float().mean()) if br.any() else np.nan)
        # 第一表面位置（surf_depth 反投影）=> 在盒內者依離邊距離分帶
        H, W = lum.shape
        v, u = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
        x = (u - cam.cx) / cam.fx * dep; y = (v - cam.cy) / cam.fy * dep
        pc = torch.stack([x, y, dep], -1).reshape(-1, 3)
        w2c = cam.world_to_camera.T if cam.world_to_camera.shape == (4, 4) else None
        c2w = torch.linalg.inv(cam.world_to_camera.T)
        pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
        xy = (pw @ Rp.T)[:, :2]
        inside = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]) & (af.reshape(-1) > 0.5)
        dist = torch.minimum(torch.minimum(xy[:, 0] - lo[0], hi[0] - xy[:, 0]), torch.minimum(xy[:, 1] - lo[1], hi[1] - xy[:, 1])) / short
        with torch.no_grad():   # 逐顆實際混合貢獻 V = Σ_像素 T·α（同 trim 的 V）
            model.properties = props_full
            mean_, cov_ = renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True)
            Vacc += mean_.reshape(-1) * cov_.reshape(-1).float()
        dkf = dk.reshape(-1) & inside
        if dkf.any():
            dark_in.append((float(ai.reshape(-1)[dkf].sum()), float(af.reshape(-1)[dkf].sum()), int(dkf.sum()), float((ai.reshape(-1)[dkf] < 0.5).float().sum())))
        for bi in range(4):
            m = inside & (dist >= edges[bi]) & (dist < edges[bi + 1])
            if m.any():
                bands[bi, 0] += float(ai.reshape(-1)[m].sum()); bands[bi, 1] += float(af.reshape(-1)[m].sum())
    model.properties = props_full
    print(f"[1 黑變透明] alpha<0.5 的比例：GT 暗像素（亮度<{a.dark}）{100 * np.nanmean(dark_a):.1f}%  vs 亮像素（>0.25）{100 * np.nanmean(bright_a):.1f}%（{len(idx)} 個訓練視角）")
    tot = float(Vacc[m_in].sum())
    for t in (0.01, 0.03, 0.1):
        b2 = m_in & (pr[:, 2] < zg - t)
        print(f"[2b 地底實際貢獻] 低於地面 {t}：佔盒內顆粒實際混合貢獻 Σ T·α 的 {100 * float(Vacc[b2].sum()) / max(tot, 1e-12):.2f}%（{len(idx)} 個訓練視角）")
    if dark_in:
        d = np.array(dark_in)
        print(f"[1b 塊內暗像素] 第一表面在盒內且 GT 暗的像素 {int(d[:, 2].sum()):,} 個：只留盒內後 alpha 保留 {100 * d[:, 0].sum() / d[:, 1].sum():.1f}%、"
              f"alpha<0.5 佔 {100 * d[:, 3].sum() / d[:, 2].sum():.1f}%")
    print(f"[3 邊緣] 第一表面在盒內的像素，只留盒內顆粒後 alpha 保留率（依離盒邊距離 / 盒短邊 {short:.2f}）：")
    for bi in range(4):
        r = bands[bi, 0] / bands[bi, 1] if bands[bi, 1] > 0 else float("nan")
        print(f"     {edges[bi]:.2f}~{edges[bi + 1] if edges[bi + 1] < 1e8 else '中心'}：{100 * r:.1f}%")


if __name__ == "__main__":
    main()
