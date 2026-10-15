"""離線判別式比賽（2026-10-12，`紀錄/new_archived/第五批_廢案_2026-10-15/13_舊版_trim判準與opacity熵的負交互.md` §4 的驗證第一步；結果在現行 `實驗分析/13` §1）：剪枝判別式該用哪一個？

問題：在**同一個已訓練模型**上，只對「本塊盒內」的顆粒做一次性剪枝（塊外顆粒合併時本來就丟掉，原樣保留），
      各判別式在同一預算下保留哪一批、本塊負責區域的品質掉多少？
      —— 不重訓（post-hoc），用來篩選要做成訓練期機制的判別式；⚠ 訓練期 MCMC 會把名額重新分配，
         「剪完不重訓」與訓練期的排序不一定相同（`06`：v/c 事後最差、訓練期卻有效）。

逐顆統計（掃描本塊全部訓練相機；與訓練期 trim 同一套 record pass）：
  v      各視角「每覆蓋像素平均 T·α」取最大            —— 現行 trim 預設判準
  V      各視角總貢獻 Σ_像素 T·α 加總                    —— 提案 A（總貢獻，不先除以大小）
  Vmax   各視角總貢獻取最大                              —— 最好視角版
  Cpix   各視角覆蓋像素數加總                            —— 訓練期 v/c 的 c
  Tsum   各視角 tile 數加總（光柵器回傳的精確 binning 成本） —— 時間模型的成本單位
  rank   逐 tile 最好名次（internal/utils/tile_topk.py，與訓練期相同）
  Trad   各視角「投影中心 ± radii」外接盒的 tile 數加總 —— trim pass 拿得到的便宜成本（精確 tiles 要多一次正常渲染）
固定成本 β（每顆每視角，以「tile」為單位；擬合自 lab outbox_cost 10 點，2026-10-12）：
  時間  ms = 11.7 + 4.94·N(百萬) + 14.23·Σtiles(百萬/視角)          => β_T = 4.94/14.23 = 0.347
  VRAM  峰值增量 GB = 0.355 + 0.592·N + 0.057·Σtiles                => β_M = 0.592/0.057 = 10.4
  => 每顆時間成本 = β_T·視角數 + Tsum；判別式 V/(β·視角數 + Tsum)。
  對偶 V − λ·成本 > 0 ⟺ V/成本 > λ：在預算切斷下與比值排序**是同一個集合**，不另列。

預算（只算本塊盒內顆粒；保留比例 p）：
  顆數  依分數保留前 p·N_in 顆
  成本  依分數累加「時間成本 β_T + Tsum/視角數」到總成本的 p 為止（Gemini 建議的隱式 λ）
  隨機下界：隨機順序、同樣兩種預算（等成本隨機剪枝）

評分：官方 test 集（held-out）中「本塊負責區域」佔畫面 >= 10% 的幀，
  「本塊負責區域」＝未剪模型的表面深度反推回世界座標後落在本塊分區盒內的像素（幾何定義，不受浮點顆粒影響）。
  報：區域 PSNR、整張 PSNR、LPIPS、精確 Load 中位、時間模型預估 ms／步、VRAM 模型峰值增量、被剪掉的微小顆粒比例。

用法：python tools/discriminant_race.py --ckpt <block ckpt> [--p 0.9 0.7 0.5] [--max_cam 0] [--out <csv>]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

BETA_T, BETA_M = 4.94 / 14.23, 0.592 / 0.057
TINY = 1e-4          # 最大軸 < 1e-4 ＝ 微小顆粒（舊版 13 §2.2 同門檻，`紀錄/new_archived/第五批_廢案_2026-10-15/13_舊版_trim判準與opacity熵的負交互.md`）


def radii_tiles(radii, xyz, cam, TILE=16):
    """trim pass 拿得到的便宜成本：投影中心 ± radii 的外接盒 tile 數（同 tile_topk／光柵器 getRect 的對稱盒）。"""
    W, H = int(cam.width), int(cam.height)
    gx, gy = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    ph = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)], 1) @ cam.full_projection
    w = ph[:, 3].clamp_min(1e-7)
    px = ((ph[:, 0] / w + 1.0) * W - 1.0) * 0.5
    py = ((ph[:, 1] / w + 1.0) * H - 1.0) * 0.5
    r = radii.float()
    nx = (((px + r + TILE - 1) / TILE).floor().clamp(0, gx) - ((px - r) / TILE).floor().clamp(0, gx))
    ny = (((py + r + TILE - 1) / TILE).floor().clamp(0, gy) - ((py - r) / TILE).floor().clamp(0, gy))
    return torch.where(radii > 0, nx * ny, torch.zeros_like(nx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--p", type=float, nargs="+", default=[0.9, 0.7, 0.5])
    ap.add_argument("--max_cam", type=int, default=0, help="掃描用的訓練視角上限（0＝全部）")
    ap.add_argument("--max_views", type=int, default=100000, help="評分幀上限")
    ap.add_argument("--min_region", type=float, default=0.10, help="評分幀：本塊負責區域佔畫面的下限")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="結果 CSV")
    ap.add_argument("--only", nargs="*", default=None, help="只跑名稱含這些字串的候選（補跑用）")
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.outside_pool import block_inside_mask
    from internal.utils.tile_topk import update_best_rank, NEVER
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import eval_official_test as eot
    dev = torch.device("cuda")
    model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=dev, eval_mode=True)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        setattr(renderer, f, False)      # 訓練期加速旗標（渲染逐位元相同）；本機光柵器可能沒重編這些欄位（同 tower_probe）
    dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; del ck
    if getattr(pc, "points_from", None) == "ply":
        pc.points_from = "random"
    ts = pc.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
    N = model.n_gaussians
    B = pc.block_id
    inb = block_inside_mask(model.get_xyz, pc, 0.0)
    n_in = int(inb.sum())
    bg = torch.zeros(3, device=dev)
    print(f"ckpt {a.ckpt}\n  block {B}  N={N:,}  盒內 {n_in:,}（{100 * n_in / N:.1f}%）  訓練視角 {len(ts)}", flush=True)

    # ── 掃描 ──
    idxs = list(range(len(ts)))
    if a.max_cam and a.max_cam < len(idxs):
        idxs = idxs[::max(1, len(idxs) // a.max_cam)][:a.max_cam]
    z = lambda: torch.zeros(N, device=dev, dtype=torch.float64)
    v, V, Vmax, Cpix, Tsum, Trad = z(), z(), z(), z(), z(), z()
    rank = torch.full((N,), NEVER, dtype=torch.int64, device=dev)
    with torch.no_grad():
        for k, ci in enumerate(idxs):
            cam = ts.cameras[ci].to_device(dev)
            o = renderer(cam, model, bg_color=bg)
            assert o.get("tiles") is not None, "光柵器沒有回傳逐顆 tiles（未重編？）"
            Tsum += o["tiles"].reshape(-1).double()
            mean, cov, radii = renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True, record_radii=True)
            mean, cov = mean.reshape(-1).double(), cov.reshape(-1).double()
            tot = mean * cov
            v = torch.maximum(v, mean); V += tot; Vmax = torch.maximum(Vmax, tot); Cpix += cov
            update_best_rank(rank, (mean * cov).float(), radii, model.get_xyz, cam)
            Trad += radii_tiles(radii.reshape(-1), model.get_xyz, cam).double()
            if (k + 1) % 100 == 0:
                print(f"  掃描 {k + 1}/{len(idxs)}", flush=True)
    nv = float(len(idxs))
    cost_T = BETA_T * nv + Tsum                       # 每顆時間成本（tile 當量，全部掃描視角加總）
    cost_M = BETA_M * nv + Tsum
    smax = model.get_scaling.max(-1).values.detach()
    tiny = smax < TINY
    print(f"  盒內：微小顆粒 {100 * float(tiny[inb].float().mean()):.1f}%  零貢獻 {100 * float((V[inb] <= 0).float().mean()):.1f}%  "
          f"從未上榜 {100 * float((rank[inb] >= NEVER).float().mean()):.1f}%", flush=True)

    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rnd = torch.rand(N, device=dev, generator=gen).double()
    vpc = v / Cpix.clamp_min(1.0)
    # β 換成像素單位：盒內「覆蓋像素／精確 tile」的總和比（每個 tile 當量平均對應多少個實際覆蓋像素）
    beta_pix = BETA_T * float(Cpix[inb].sum() / Tsum[inb].sum().clamp_min(1.0))
    rho = lambda x, y: float(np.corrcoef(np.argsort(np.argsort(x.cpu().numpy())), np.argsort(np.argsort(y.cpu().numpy())))[0, 1])
    _ii = torch.nonzero(inb).squeeze(1)
    print(f"  β像素 = {beta_pix:.1f}；盒內 Spearman：radii 盒 tiles vs 精確 tiles {rho(Trad[_ii], Tsum[_ii]):.3f}、"
          f"覆蓋像素 vs 精確 tiles {rho(Cpix[_ii], Tsum[_ii]):.3f}", flush=True)
    cands = [
        ("現行 trim：v（每像素平均貢獻，最好視角）", v),
        ("v/c（訓練期判準）", vpc),
        ("v/c × tile 名次（訓練期合成判準）", vpc / (1.0 + rank.double() / 32.0)),
        ("opacity", model.get_opacity.detach().reshape(-1).double()),
        ("A：總貢獻 V", V),
        ("A：V / 覆蓋像素", V / Cpix.clamp_min(1.0)),
        ("A+B：V / (β時間 + tiles)", V / cost_T),
        ("A+B：V / (βVRAM + tiles)", V / cost_M),
        ("A+B 冪次：V / (β時間 + tiles)^0.5", V / cost_T.sqrt()),
        ("最好視角：Vmax / (β時間 + tiles 平均)", Vmax / (BETA_T + Tsum / nv)),
        ("便宜成本：V / (β時間 + radii 盒 tiles)", V / (BETA_T * nv + Trad)),
        ("便宜成本：V / (β像素 + 覆蓋像素)", V / (beta_pix * nv + Cpix)),
        ("隨機（下界）", rnd),
    ]
    if a.only:
        cands = [c for c in cands if any(k in c[0] for k in a.only)]

    # ── 評分幀與本塊負責區域 ──
    names_t, cams = eot.load_test_cameras(a.test_dir, 1.2)
    # 「本塊負責區域」（幾何定義，2026-10-12）：完整模型渲染的表面深度把每個像素反推回世界座標，
    #   落在本塊分區盒內（merge 歸屬規則）且 alpha > 0.5 的像素。
    #   ⚠ 不用「盒內顆粒單獨渲染 alpha > 0.5」：本機試跑時那樣選到的幀，區域是少數靠近相機的巨大盒內顆粒撐出來的
    #     （隨機剪 30% 區域 PSNR 一位不差、v/c 一剪就掉 0.8）=> 量到的是浮點顆粒，不是本塊內容。
    #   選幀：區域 >= --min_region 的幀。
    orig = {kk: t for kk, t in model.properties.items()}
    region, gts = {}, {}
    with torch.no_grad():
        for i in range(len(names_t)):
            cam = cams[i].to_device(dev)
            oo = renderer(cam, model, bg_color=bg)
            H, W = oo["render"].shape[1:]
            zz = oo["surf_depth"][0].reshape(H, W)
            vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
            pcm = torch.stack([(uu + 0.5 - cam.cx) / cam.fx * zz, (vv + 0.5 - cam.cy) / cam.fy * zz, zz], -1).reshape(-1, 3)
            xw = (pcm - cams.T[i].to(dev).float()) @ cams.R[i].to(dev).float()      # 同 tools/tower_probe.py（已自檢地面 z）
            m = block_inside_mask(xw, pc, 0.0).reshape(H, W) & (zz > 0) & (oo["rend_alpha"].reshape(H, W) > 0.5)
            if float(m.float().mean()) >= a.min_region:
                region[i] = m
    sel = sorted(region)[:a.max_views]
    print(f"  評分：官方 test 741 幀中本塊負責區域 >= {100 * a.min_region:.0f}% 的 {len(sel)} 幀；區域平均佔畫面 "
          f"{100 * np.mean([float(region[i].float().mean()) for i in sel]):.1f}%", flush=True)
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)

    def gt_of(i, like):
        if i not in gts:
            pil = Image.open(os.path.join(a.test_dir, "images_1.2", names_t[i])).convert("RGB")
            if pil.size != (like.shape[2], like.shape[1]):
                pil = pil.resize((like.shape[2], like.shape[1]), Image.LANCZOS)
            gts[i] = torch.from_numpy(np.array(pil, np.uint8)).permute(2, 0, 1).to(dev)   # uint8 快取：b10 有 130 幀，float32 要 2.2 GB（6GB 卡 OOM）
        return gts[i].float() / 255.

    rows = []

    def evaluate(keep, tag, budget, p):
        model.properties = {kk: t[keep] for kk, t in orig.items()}
        rse, rn, ps, ls, loads = 0.0, 0, [], [], []
        with torch.no_grad():
            for i in sel:
                oo = renderer(cams[i].to_device(dev), model, bg_color=bg)
                img = oo["render"].clamp(0, 1); gt = gt_of(i, img)
                se = ((img - gt) ** 2).mean(0)
                rse += float(se[region[i]].sum()); rn += int(region[i].sum())
                ps.append(float(-10 * torch.log10(se.mean().clamp_min(1e-12))))
                ls.append(float(lp(img.unsqueeze(0), gt.unsqueeze(0))))
                loads.append(float(oo["tiles"].sum()))
        model.properties = orig
        n_keep = int(keep.sum()); n_kin = int((keep & inb).sum())
        load = float(np.median(loads)) / 1e6
        cut = inb & ~keep
        r = dict(判別式=tag, 預算=budget, p=p, 盒內保留=n_kin, N=n_keep,
                 區域PSNR=-10 * np.log10(max(rse / max(rn, 1), 1e-12)), 整張PSNR=float(np.mean(ps)), LPIPS=float(np.mean(ls)),
                 Load中位M=load, 預估ms=11.7 + 4.94 * n_keep / 1e6 + 14.23 * load, VRAM增量GB=0.355 + 0.592 * n_keep / 1e6 + 0.057 * load,
                 剪掉中微小顆粒pct=100 * float((cut & tiny).sum()) / max(int(cut.sum()), 1),
                 盒內成本保留pct=100 * float((cost_T / nv)[keep & inb].sum() / (cost_T / nv)[inb].sum()))
        rows.append(r)
        print(f"  {tag:<34} {budget:<3} p={p:.2f}  盒內 {n_kin / 1e6:5.3f}M  區域 {r['區域PSNR']:6.3f}  整張 {r['整張PSNR']:6.3f}  "
              f"LPIPS {r['LPIPS']:.4f}  Load {load:5.2f}M  預估 {r['預估ms']:5.1f} ms  剪掉中微小 {r['剪掉中微小顆粒pct']:5.1f}%  "
              f"成本保留 {r['盒內成本保留pct']:5.1f}%", flush=True)

    print("\n═══ 不剪 ═══")
    evaluate(torch.ones(N, dtype=torch.bool, device=dev), "不剪", "-", 1.0)
    ii = torch.nonzero(inb).squeeze(1)
    unit_cost = (cost_T / nv)[ii]                     # 每顆每視角時間成本（tile 當量）
    for p in a.p:
        print(f"\n═══ 保留比例 p = {p:.2f} ═══")
        for tag, s in cands:
            order = ii[torch.argsort(s[ii], descending=True, stable=True)]
            for budget in ("顆數", "成本"):
                if budget == "顆數":
                    kept = order[:int(round(p * n_in))]
                else:
                    cc = torch.cumsum((cost_T / nv)[order], 0)
                    kept = order[cc <= p * float(unit_cost.sum())]
                keep = ~inb.clone(); keep[kept] = True
                evaluate(keep, tag, budget, p)
    if a.out:
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\nCSV：{a.out}")


if __name__ == "__main__":
    main()
