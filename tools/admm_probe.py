"""ADMM 統一框架 P1 探針（2026-10-14，`紀錄/實驗分析/14` §P1）：共識約束是不是「活」的、用它要付多少成本。只讀現成 ckpt，不訓練。

A 相鄰區塊的幾何不一致（＝ADMM 共識要壓到 0 的量 ||A_v x_v - A_u x_u||）
  4x4 的 24 對相鄰區塊；每對取兩塊**共同的**訓練視角，兩個單塊模型（各自完整，含塊外顆粒）各渲染一次深度。
  兩者 alpha 都 > 0.5 的像素，依「v 的表面點落在哪裡」分組：v 的盒內（u 在這裡是塊外）、u 的盒內、離共同邊界 < 0.1 格（接縫帶）、其他。
  報：相對深度差 |d_v - d_u| / min(d_v, d_u) 的中位／p90；絕對差超過 0.01（官方網格 voxel）、0.04（官方 sdf_trunc）的像素比例；兩張渲染之間的 PSNR。
B 每視角單目深度尺度的跨區塊一致性（「尺度共識」最便宜的一層：每台相機 2 個數）
  現行：每張影像的 Depth-Anything-V2 逆深度 m 只在資料準備時對 SfM 點擬合一次 (a, b)（utils/get_depth_scales.py：
        t = median, s = mean|x - t|，a = s_sfm / s_mono，b = t_sfm - a * t_mono），訓練中固定。
  這裡：同一張影像出現在 >= 2 塊時，用**各塊的渲染逆深度**照同一公式擬合 (a_v, b_v)，比較：
    各塊之間 a 的變異係數；a_v 相對 SfM 擬合的偏差；各塊的監督目標 a_v m + b_v 之間的相對差；仿射擬合本身的殘差（單目深度有多少是仿射換不過來的）。
C 成本（ADMM 會多出來的部分）
  ① 幾何通道：同一區塊、同一批視角，lean（現行，不算幾何通道）vs 非 lean＋深度共識項的 forward+backward 時間與峰值 VRAM
  ② 每輪 A_v：渲染一個區塊全部訓練視角一次（深度）的時間
  ③ 每輪切換區塊：ckpt 讀與寫的時間
  ④ z／對偶變數的記憶體（地面網格 0.05、0.02 單位；每視角 2 個數）
  ⚠ lab 同卡有容器外的佔用時有時無 => 開頭記錄整卡已用量；時間數字要對照它看。
用法：python tools/admm_probe.py --run outputs/lab/full44_best --out outputs/lab/admm_probe
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
TRAIN_FLAGS = ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train")


def gpu_used_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return -1


def robust_affine(x, y):
    """utils/get_depth_scales.py 同一公式：把 x 換到 y 的尺度。回傳 (a, b)。"""
    tx, ty = x.median(), y.median()
    sx, sy = (x - tx).abs().mean(), (y - ty).abs().mean()
    a = sy / sx.clamp_min(1e-12)
    return float(a), float(ty - a * tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/lab/full44_best")
    ap.add_argument("--out", default="outputs/lab/admm_probe")
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 4])
    ap.add_argument("--views_per_pair", type=int, default=8)
    ap.add_argument("--scale_views", type=int, default=120)
    ap.add_argument("--time_views", type=int, default=30)
    ap.add_argument("--time_block", type=int, default=6)
    ap.add_argument("--blocks", type=int, nargs="*", default=None, help="只用這些區塊（本機功能測試）")
    ap.add_argument("--skip", nargs="*", default=[], help="略過 A／B／C")
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.outside_pool import block_inside_mask
    import seam_failure as sf
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    ext0 = gpu_used_mib()
    print(f"[卡] 開始時整卡已用 {ext0} MiB（容器外佔用會影響時間數字）", flush=True)
    GX, GY = a.grid
    K = GX * GY
    cks = {}
    for b in (a.blocks if a.blocks is not None else range(K)):
        c = sorted(glob.glob(os.path.join(a.run, "blocks", f"block_{b}", "checkpoints", "*step=60000.ckpt")))
        if c:
            cks[b] = c[0]
    assert cks, "找不到任何區塊 ckpt"
    b0 = sorted(cks)[0]
    model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(cks[b0], device=dev, eval_mode=True)
    for f in TRAIN_FLAGS:
        if hasattr(renderer, f):
            setattr(renderer, f, False)
    dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; del ck
    pc.block_id = None
    if getattr(pc, "points_from", None) == "ply":
        pc.points_from = "random"
    ts = pc.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
    names = list(ts.image_names)
    name2i = {n: i for i, n in enumerate(names)}
    pdir = os.path.join(dmh["path"], "partition", f"partitions-dim_{GX}_{GY}_visibility_0.08")
    members = {}
    for b in range(K):
        x, y = b % GX, b // GX
        with open(os.path.join(pdir, f"{x:03d}_{y:03d}.txt")) as f:
            members[b] = [name2i[l.strip()] for l in f if l.strip() in name2i]
    nb = np.zeros(len(names), int)
    for b in members:
        nb[members[b]] += 1
    print(f"[資料] 訓練影像 {len(names)}；出現在幾塊：" + "、".join(f"{k} 塊 {int((nb == k).sum())}" for k in range(1, nb.max() + 1)), flush=True)
    props = {}
    for b, c in cks.items():
        m, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(c, device=torch.device("cpu"), eval_mode=True)
        props[b] = {k: v.detach().cpu() for k, v in m.properties.items()}
        del m
    print(f"[模型] 載入 {len(props)} 塊（放 CPU，用到才搬上 GPU）", flush=True)
    parts = torch.load(os.path.join(pdir, "partitions.pt"))
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].float().to(dev)
    mn, mx, internal, unit = (t.to(dev) for t in sf.seam_frame(parts))
    bg = torch.zeros(3, device=dev)
    pcs = {}

    def pc_of(b):
        if b not in pcs:
            import copy
            q = copy.copy(pc); q.block_id = b; q.block_dim = [GX, GY]
            q.image_list = os.path.join(pdir, f"{b % GX:03d}_{b // GX:03d}.txt")      # block_inside_mask 從這裡找 partitions.pt
            pcs[b] = q
        return pcs[b]

    def put(b):
        model.properties = {k: v.to(dev) for k, v in props[b].items()}

    def render(b, i):
        put(b)
        with torch.no_grad():
            o = renderer(ts.cameras[i].to_device(dev), model, bg_color=bg)
        return o

    def world_pts(i, d):
        cam = ts.cameras[i].to_device(dev)
        H, W = d.shape
        vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        pcm = torch.stack([(uu + 0.5 - cam.cx) / cam.fx * d, (vv + 0.5 - cam.cy) / cam.fy * d, d], -1).reshape(-1, 3)
        return (pcm - ts.cameras.T[i].to(dev).float()) @ ts.cameras.R[i].to(dev).float()

    res = {"gpu_used_at_start_mib": ext0}
    rng = np.random.default_rng(0)

    # ── A 相鄰區塊幾何不一致 ──
    if "A" not in a.skip:
        pairs = [(b, b + 1) for b in cks if b % GX < GX - 1 and b + 1 in cks] + [(b, b + GX) for b in cks if b + GX in cks]
        rows = []
        for v, u in pairs:
            shared = sorted(set(members[v]) & set(members[u]))
            if not shared:
                continue
            sel = [shared[j] for j in np.linspace(0, len(shared) - 1, min(a.views_per_pair, len(shared))).round().astype(int)]
            acc = {g: [] for g in ("v盒內", "u盒內", "接縫帶", "其他")}
            ps = []
            for i in sel:
                ov, ou = render(v, i), render(u, i)
                dv, du = ov["surf_depth"][0], ou["surf_depth"][0]
                H, W = dv.shape
                ok = ((ov["rend_alpha"].reshape(H, W) > 0.5) & (ou["rend_alpha"].reshape(H, W) > 0.5) & (dv > 0) & (du > 0)).reshape(-1)
                xw = world_pts(i, dv)
                inv = block_inside_mask(xw, pc_of(v), 0.0); inu = block_inside_mask(xw, pc_of(u), 0.0)
                dist = sf.seam_distance((xw @ Rp.T)[:, :2], mn, mx, internal, unit)
                rel = ((dv - du).abs() / torch.minimum(dv, du).clamp_min(1e-6)).reshape(-1)
                ab = (dv - du).abs().reshape(-1)
                grp = {"接縫帶": ok & (dist < 0.1), "v盒內": ok & inv & ~(dist < 0.1), "u盒內": ok & inu & ~(dist < 0.1)}
                grp["其他"] = ok & ~grp["接縫帶"] & ~grp["v盒內"] & ~grp["u盒內"]
                for g, m in grp.items():
                    if int(m.sum()) > 0:
                        idx = torch.nonzero(m).squeeze(1)
                        if idx.numel() > 200000:
                            idx = idx[torch.randperm(idx.numel(), device=dev)[:200000]]
                        acc[g].append(torch.stack([rel[idx], ab[idx]], 1).cpu())
                se = ((ov["render"].clamp(0, 1) - ou["render"].clamp(0, 1)) ** 2).mean(0).reshape(-1)[ok]
                if se.numel():
                    ps.append(float(-10 * torch.log10(se.mean().clamp_min(1e-12))))
            for g, lst in acc.items():
                if lst:
                    t = torch.cat(lst)
                    rows.append(dict(pair=f"{v}-{u}", group=g, n=int(t.shape[0]), rel_med=float(t[:, 0].median()),
                                     rel_p90=float(t[torch.randperm(t.shape[0])[:2_000_000], 0].quantile(0.9)),
                                     gt001=float((t[:, 1] > 0.01).float().mean()), gt004=float((t[:, 1] > 0.04).float().mean()),
                                     psnr_between=float(np.mean(ps)) if ps else float("nan")))
            print(f"  [A] {v}-{u}：共同視角 {len(shared)}，用 {len(sel)}；兩模型渲染之間 PSNR {np.mean(ps) if ps else float('nan'):.2f}", flush=True)
        import pandas as pd
        dfA = pd.DataFrame(rows); dfA.to_csv(os.path.join(a.out, "A_pairs.csv"), index=False)
        if len(dfA):
            g = dfA.groupby("group").apply(lambda x: pd.Series({"像素": x.n.sum(), "相對差中位(加權)": np.average(x.rel_med, weights=x.n),
                                                                 ">0.01 比例": np.average(x.gt001, weights=x.n), ">0.04 比例": np.average(x.gt004, weights=x.n)}))
            print("\n═══ A 相鄰區塊深度不一致（兩模型 alpha 都 > 0.5 的像素；分組＝v 的表面點位置）═══\n" + g.to_string(), flush=True)
            res["A"] = g.reset_index().to_dict("records")

    # ── B 每視角單目深度尺度 ──
    if "B" not in a.skip:
        mset = {b: set(members[b]) for b in members}
        multi = [i for i in range(len(names)) if nb[i] >= 2 and ts.extra_data[i] is not None]
        print(f"[B] 有深度圖且出現在 >= 2 塊的影像 {len(multi)} 張", flush=True)
        sel = sorted(rng.choice(multi, min(a.scale_views, len(multi)), replace=False).tolist())
        rowsB = []
        for i in sel:
            dpath, dsc = ts.extra_data[i]
            mono = torch.from_numpy(np.load(dpath).astype(np.float32)).to(dev)
            if mono.dim() == 3:
                mono = mono[..., 0] if mono.shape[-1] <= 4 else mono[0]
            per = []
            for b in [bb for bb in cks if i in mset[bb]]:
                o = render(b, i)
                d = o["surf_depth"][0]; H, W = d.shape
                m = F.interpolate(mono[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                ok = (o["rend_alpha"].reshape(H, W) > 0.5) & (d > 1e-6)
                if int(ok.sum()) < 1000:
                    continue
                y = 1.0 / d[ok]; x = m[ok]
                aa, bb_ = robust_affine(x, y)
                fit_res = float(((aa * x + bb_ - y).abs() / y.abs().clamp_min(1e-9)).median())
                per.append((b, aa, bb_, fit_res, x, y))
            if len(per) < 2:
                continue
            A_ = np.array([p[1] for p in per]); ref = per[0][4]
            tg = torch.stack([p[1] * ref + p[2] for p in per])           # 各塊的監督目標（同一組 mono 像素）
            spread = float(((tg.max(0).values - tg.min(0).values) / tg.median(0).values.abs().clamp_min(1e-9)).median())
            rowsB.append(dict(view=names[i], n_blocks=len(per), a_cv=float(A_.std() / max(abs(A_.mean()), 1e-12)),
                              a_vs_sfm=float(np.median(np.abs(A_ / dsc["scale"] - 1))) if dsc["scale"] else float("nan"),
                              target_spread=spread, fit_res_med=float(np.median([p[3] for p in per]))))
        import pandas as pd
        dfB = pd.DataFrame(rowsB); dfB.to_csv(os.path.join(a.out, "B_scales.csv"), index=False)
        if len(dfB):
            q = lambda c: f"中位 {dfB[c].median():.3f}／p90 {dfB[c].quantile(0.9):.3f}"
            print(f"\n═══ B 每視角深度尺度（{len(dfB)} 張出現在 >= 2 塊的影像）═══\n"
                  f"  各塊擬合的 a 變異係數          {q('a_cv')}\n"
                  f"  各塊的 a 相對現行 SfM 擬合的偏差 {q('a_vs_sfm')}\n"
                  f"  各塊監督目標之間的相對差        {q('target_spread')}\n"
                  f"  仿射擬合殘差（單目換不過來的部分）{q('fit_res_med')}", flush=True)
            res["B"] = {c: [float(dfB[c].median()), float(dfB[c].quantile(0.9))] for c in ("a_cv", "a_vs_sfm", "target_spread", "fit_res_med")}

    # ── C 成本 ──
    if "C" not in a.skip:
        tb = a.time_block if a.time_block in cks else b0
        C = {}
        vs = members[tb][:: max(1, len(members[tb]) // a.time_views)][: a.time_views]
        gts = {}
        for i in vs:
            cam = ts.cameras[i]
            pil = Image.open(ts.image_paths[i]).convert("RGB").resize((int(cam.width), int(cam.height)), Image.LANCZOS)
            gts[i] = torch.from_numpy(np.array(pil, np.uint8)).permute(2, 0, 1).float().to(dev) / 255.
        P = {k: v.to(dev).requires_grad_(True) for k, v in props[tb].items()}
        model.properties = P

        def step_time(lean, with_depth):
            renderer.lean_train = lean
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ts_ = []
            for r in range(2):
                for i in vs:
                    t0 = time.perf_counter()
                    o = renderer(ts.cameras[i].to_device(dev), model, bg_color=bg, **({"_lean": True} if lean else {}))
                    loss = (o["render"] - gts[i]).abs().mean()
                    if with_depth:
                        d = o["surf_depth"][0]
                        loss = loss + 0.1 * (d - d.detach().median()).abs().mean()     # 形狀同共識項：渲染深度對一個目標的 L1
                    loss.backward()
                    for t in P.values():
                        t.grad = None
                    torch.cuda.synchronize()
                    if r == 1:
                        ts_.append(time.perf_counter() - t0)
            return 1000 * float(np.median(ts_)), torch.cuda.max_memory_allocated() / 2**30

        try:
            t_l, m_l = step_time(True, False)
            t_n, m_n = step_time(False, True)
            C["fwd_bwd_lean_ms"], C["fwd_bwd_geom_depth_ms"] = t_l, t_n
            C["peak_lean_GiB"], C["peak_geom_depth_GiB"] = m_l, m_n
            print(f"\n═══ C 成本（block {tb}，N={props[tb]['means'].shape[0]:,}，{len(vs)} 視角）═══\n"
                  f"  ① forward+backward：lean {t_l:.1f} ms → 幾何通道＋深度共識項 {t_n:.1f} ms（{100 * (t_n / t_l - 1):+.1f}%）；"
                  f"峰值 {m_l:.2f} → {m_n:.2f} GiB", flush=True)
        except Exception as e:
            print(f"⚠ ① 量不到（{type(e).__name__}: {e}）——光柵器可能沒有 lean 路徑", flush=True)
        renderer.lean_train = False
        model.properties = {k: v.detach() for k, v in P.items()}; del P
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            for i in members[tb]:
                renderer(ts.cameras[i].to_device(dev), model, bg_color=bg)
        torch.cuda.synchronize(); C["Av_pass_s"] = time.perf_counter() - t0; C["Av_pass_views"] = len(members[tb])
        print(f"  ② 每輪 A_v：渲染 block {tb} 全部 {len(members[tb])} 個訓練視角 {C['Av_pass_s']:.1f} s", flush=True)
        t0 = time.perf_counter(); ck = torch.load(cks[tb], map_location="cpu"); C["ckpt_load_s"] = time.perf_counter() - t0
        tmp = os.path.join(a.out, "_tmp_save.ckpt")
        t0 = time.perf_counter(); torch.save(ck, tmp); C["ckpt_save_s"] = time.perf_counter() - t0
        C["ckpt_GiB"] = os.path.getsize(tmp) / 2**30; os.remove(tmp); del ck
        print(f"  ③ 每輪切換：ckpt {C['ckpt_GiB']:.2f} GiB，讀 {C['ckpt_load_s']:.1f} s、寫 {C['ckpt_save_s']:.1f} s", flush=True)
        # 範圍＝SfM 點（分區座標系）1~99 百分位；不用分區盒（最外圈的格子一路延伸到場景外，會把範圍算成 6 倍）
        from internal.utils.colmap import read_points3D_binary
        P3 = np.array([q.xyz for q in read_points3D_binary(os.path.join(dmh["path"], "sparse", "0", "points3D.bin")).values()], np.float32)
        P3 = P3 @ Rp.cpu().numpy().T
        ext = np.percentile(P3[:, :2], 99, axis=0) - np.percentile(P3[:, :2], 1, axis=0)
        for r_ in (0.05, 0.02):
            cells = int(np.ceil(ext[0] / r_) * np.ceil(ext[1] / r_))
            C[f"z_grid_{r_}_MiB"] = cells * 6 * 4 / 2**20          # 高度、法向 3、權重、尺度
        C["view_scales_KiB"] = len(names) * 2 * 4 / 1024
        print(f"  ④ z（地面網格，每格 6 個 float：高度、法向 3、權重、顆粒尺度）：0.05 格 {C['z_grid_0.05_MiB']:.0f} MiB、0.02 格 {C['z_grid_0.02_MiB']:.0f} MiB"
              f"（範圍 {ext[0]:.1f} x {ext[1]:.1f}）；對偶變數 u_v 與 z 同大小、只存各塊看得到的部分；每視角尺度 {C['view_scales_KiB']:.0f} KiB", flush=True)
        res["C"] = C
        res["gpu_used_at_end_mib"] = gpu_used_mib()
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n摘要：{a.out}/summary.json（整卡已用：開始 {ext0}、結束 {res.get('gpu_used_at_end_mib')} MiB）")


if __name__ == "__main__":
    main()
