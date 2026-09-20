#!/usr/bin/env python
"""正方形外接盒 vs 矩形外接盒：離線量「改光柵器能省多少 binning」的天花板（2026-09-20）。

## 背景

`forward.cu:281` 目前是
    radius = ceil(truncated_R * max(max(extent.x, extent.y), FilterSize))
把 `computeAABB` 已經算好的**兩軸**半寬 `extent.x / extent.y` 用 `max()` 壓成一個純量，
再用它圍出**正方形** tile 盒（`getRect`）。長寬比 k 的粒子因此有 `1 - 1/k` 的 tile 是空白的。

提案：改成各軸獨立
    rx = ceil(truncated_R * max(extent.x, FilterSize))
    ry = ceil(truncated_R * max(extent.y, FilterSize))

⚠ 但 `computeAABB` 給的已經是**旋轉後**橢圓的軸對齊外接盒：長軸貼著螢幕軸時省很多，
  長軸 45 度時 `ex ≈ ey`、正方形本來就是緊的、**省 0**。再加上 tile 量化（半寬 < 16px 的
  粒子不管怎麼算都佔 1~2 格），真實收益遠低於「Σs_max² vs Σs_max·s_min = 浪費 76.5%」那個
  解析上界。這支工具就是要量**含旋轉、含量化**的真實比值，決定值不值得動 CUDA。

## 做法

在 Python 重現 `computeTransMat` + `computeAABB` + `getRect`，對每台相機算兩種 tile 數。
**先驗關卡**：自己算的 `radius` 必須與光柵器回傳的 `radii` 對得上（同一批粒子、同一台相機）。
對不上就不要信後面的數字 —— 這支工具用解析投影，而生效的是光柵器，是「同定義不同實作路徑」。

編譯期常數（auxiliary.h，寫死在這裡，改了要同步）：
    EXACT_SUPPORT 1 / EXACT_SUPPORT_RADIUS 0 / EXACT_SUPPORT_GROW 0 / EXACT_SUPPORT_MARGIN_PX 0
    TIGHTBBOX 0  =>  truncated_R 恆為 3.0；opacity <= 1/255 直接丟棄

用法:
    python tools/bbox_rect_ceiling.py --ckpt <path> [--max-cam 60] [--tile 16]
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILTER_SIZE = 0.7071067811865476
TRUNCATED_R = 3.0          # EXACT_SUPPORT=1 且 EXACT_SUPPORT_RADIUS=0 => 固定 3 sigma
OPACITY_DROP = 1.0 / 255.0


def quat_to_rotmat(q):
    """與 CUDA 的 quat_to_rotmat 同慣例（w, x, y, z，已正規化），回傳 (N,3,3) 的**行為列**矩陣。

    glm 是 column-major：R[0]/R[1]/R[2] 是三個**行向量（column）**。
    這裡回傳 R[:, :, j] = 第 j 個 column，與 CUDA 的 R[j] 對應。
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 1, 0] = 2 * (x * y + w * z); R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 0, 1] = 2 * (x * y - w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 0, 2] = 2 * (x * z + w * y); R[:, 1, 2] = 2 * (y * z - w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def compute_extent(xyz, quat, scale2, viewmat_flat, fx, fy, cx, cy, W, H):
    """重現 computeTransMat + computeAABB。回傳 (center_x, center_y, ex, ey, ok)。

    `viewmat_flat` 是光柵器實際收到的那塊記憶體（16 個 float，與 CUDA 同索引）。
    """
    vm = viewmat_flat
    # glm::mat3 W = mat3(vm[0],vm[1],vm[2],  vm[4],vm[5],vm[6],  vm[8],vm[9],vm[10])
    # => 三個 column 分別是 (vm0,vm1,vm2)、(vm4,vm5,vm6)、(vm8,vm9,vm10)
    Wm = torch.tensor([[vm[0], vm[4], vm[8]],
                       [vm[1], vm[5], vm[9]],
                       [vm[2], vm[6], vm[10]]], dtype=xyz.dtype, device=xyz.device)
    cam_pos = torch.tensor([vm[12], vm[13], vm[14]], dtype=xyz.dtype, device=xyz.device)

    p_view = xyz @ Wm.T + cam_pos                      # W * p_world + cam_pos
    Rq = quat_to_rotmat(quat)                          # column j = Rq[:, :, j]
    R0 = Rq[:, :, 0] * scale2[:, 0:1]                  # R[0] * sx
    R1 = Rq[:, :, 1] * scale2[:, 1:2]                  # R[1] * sy
    # M 的三個 column：W*R[0]、W*R[1]、p_view
    M0 = R0 @ Wm.T
    M1 = R1 @ Wm.T
    M2 = p_view

    # P 的 column：(fx,0,0,0)、(0,fy,0,0)、(cx,cy,1,1)、(0,0,0,0)
    # T = transpose(P * mat3x4(vec4(M0,0), vec4(M1,0), vec4(M2,1)))
    # 逐 column 展開：P * (m, s) 其中 s 是第四分量
    def P_mul(m, s):
        return (fx * m[:, 0] + cx * m[:, 2],
                fy * m[:, 1] + cy * m[:, 2],
                m[:, 2],
                m[:, 2] + s)          # 第四列 = (0,0,1,0)·m + 1·s ... 見下方註記
    # ⚠ P 的第 3 列（index 2）= (0,0,1,0)，第 4 列 = (0,0,1,0) 的第四個 column 是 1
    #   glm P = mat4(col0,col1,col2,col3)，row2 = (0,0,1,0)、row3 = (0,0,1,0)
    #   => (P*v)[2] = v.z, (P*v)[3] = v.z ... 但 col2 的第四分量是 1 => (P*v)[3] = v.z*1 + v.w*0
    #   實際 col2 = (cx, cy, 1, 1) => row3 = (0, 0, 1, 0) 取自各 column 的第 4 分量 (0,0,1,0)
    #   => (P*v)[3] = v.z
    c0 = P_mul(M0, 0.0)
    c1 = P_mul(M1, 0.0)
    c2 = P_mul(M2, 1.0)
    # 修正第四分量：row3 = 各 column 的第 4 分量 = (0, 0, 1, 0) => (P*v)[3] = v[2]
    c0 = (c0[0], c0[1], c0[2], M0[:, 2])
    c1 = (c1[0], c1[1], c1[2], M1[:, 2])
    c2 = (c2[0], c2[1], c2[2], M2[:, 2])

    # transpose => mat4x3：T[0]=(c0[0],c1[0],c2[0]), T[1]=(c0[1],c1[1],c2[1]),
    #                      T[2]=(c0[2],c1[2],c2[2]), T[3]=(c0[3],c1[3],c2[3])
    T0 = torch.stack([c0[0], c1[0], c2[0]], -1)
    T1 = torch.stack([c0[1], c1[1], c2[1]], -1)
    T2 = torch.stack([c0[2], c1[2], c2[2]], -1)
    T3 = torch.stack([c0[3], c1[3], c2[3]], -1)
    # computeAABB 把 T[3] 覆寫成 T[2]（見 forward.cu:133-140 的建構）
    T3 = T2

    sgn = torch.tensor([1.0, 1.0, -1.0], dtype=xyz.dtype, device=xyz.device)
    d = (sgn * T3 * T3).sum(-1)
    ok = d != 0
    d = torch.where(ok, d, torch.ones_like(d))
    f = sgn.unsqueeze(0) / d.unsqueeze(-1)

    px = (f * T0 * T3).sum(-1)
    py = (f * T1 * T3).sum(-1)
    pz = (f * T2 * T3).sum(-1)
    p = torch.stack([px, py, pz], -1)

    inb = (px >= -W / 4) & (px <= W * 5 / 4) & (py >= -H / 4) & (py <= H * 5 / 4)
    h0 = p * p - torch.stack([(f * T0 * T0).sum(-1),
                              (f * T1 * T1).sum(-1),
                              (f * T2 * T2).sum(-1)], -1)
    h = torch.sqrt(h0.clamp_min(0.0))
    return px, py, h[:, 0], h[:, 1], ok & inb, T0, T1, T3


def conic_box(T0, T1, T3, R):
    """**精確**圓錐曲線外接盒（2026-09-20 新增）。

    現行 CUDA 走的是「算 r=1 的盒，再乘 truncated_R，中心不動」—— 那是線性化。
    ray-splat 交點 p = k x l 對像素是**仿射**的（因為 Tw x Tw = 0），所以
    `u^2 + v^2 <= R^2` 在螢幕上是**真正的圓錐曲線**；把對偶簽名從 (1,1,-1) 換成
    (R^2, R^2, -1) 就得到該層等高線的精確外接盒，中心也會跟著 R 移動。
    正交極限下會自動退化成 R * extent(1)，所以這是現行式子的嚴格推廣。

    ⚠ d <= 0 代表該層等高線在螢幕上是**無界**的（穿過地平線）：現行 CUDA 只擋 d == 0，
      d < 0 時 sqrt 的引數轉負被 clamp 成 0 => 給出一個極小的盒（靜默漏覆蓋）。
      這裡把 d == 0 標成 ok=False 並另外計數（沿用原始 CUDA 的判準）。
    """
    R2 = R * R
    ones = torch.ones_like(R2)
    sgn = torch.stack([R2, R2, -ones], -1)              # (N,3)
    d = (sgn * T3 * T3).sum(-1)
    # ⚠ d = C*_33 ~= -(深度^2)：**負號是常態**（Tw = (u軸.z, v軸.z, 中心深度)，中心深度遠大於軸的 z）。
    #   中心與半寬都是 C* 的比值 => 對 C* 的整體符號不變 => 只有 d == 0 才是真退化（與原始 CUDA 同）。
    ok = d != 0
    dd = torch.where(ok, d, torch.ones_like(d))
    f = sgn / dd.unsqueeze(-1)
    px = (f * T0 * T3).sum(-1)
    py = (f * T1 * T3).sum(-1)
    hx2 = px * px - (f * T0 * T0).sum(-1)
    hy2 = py * py - (f * T1 * T1).sum(-1)
    return px, py, torch.sqrt(hx2.clamp_min(0.0)), torch.sqrt(hy2.clamp_min(0.0)), ok


def tiles_from_rect(cx_, cy_, rx, ry, tile, gw, gh):
    """重現 getRect + tiles_touched。"""
    def lo(c, r, g):
        return torch.clamp(torch.floor_divide((c - r), tile).long(), min=0).clamp(max=g)
    def hi(c, r, g):
        return torch.clamp(torch.floor_divide((c + r + tile - 1), tile).long(), min=0).clamp(max=g)
    x0, x1 = lo(cx_, rx, gw), hi(cx_, rx, gw)
    y0, y1 = lo(cy_, ry, gh), hi(cy_, ry, gh)
    return ((x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)).double()


def tiles_from_minmax(x0, x1, y0, y1, tile, gw, gh):
    """非對稱盒的 tile 數：getRect 改成 (rect_min, rect_max) 後才表達得出來。"""
    def lo(v, g):
        return torch.clamp(torch.floor_divide(v, tile).long(), min=0).clamp(max=g)
    def hi(v, g):
        return torch.clamp(torch.floor_divide(v + tile - 1, tile).long(), min=0).clamp(max=g)
    a, b = lo(x0, gw), hi(x1, gw)
    c, d = lo(y0, gh), hi(y1, gh)
    return ((b - a).clamp_min(0) * (d - c).clamp_min(0)).double()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-cam", type=int, default=60)
    ap.add_argument("--tile", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cuda")
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        args.ckpt, device=dev, eval_mode=True, pre_activate=False)
    ck = torch.load(args.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(args.ckpt)),
                                   global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    n_cam = min(args.max_cam, len(cams))
    print(f"ckpt {os.path.basename(args.ckpt)}   N = {model.n_gaussians:,}   相機 {n_cam}/{len(cams)}   tile {args.tile}")

    xyz = model.get_xyz.detach().double()
    # ⚠ 這三個是**方法**不是 property（get_xyz 才是 property）
    quat = model.get_rotations().detach().double()
    sc = model.get_scales().detach().double()[:, :2]
    op = model.get_opacities().detach().double().squeeze(-1)
    keep = op > OPACITY_DROP                       # EXACT_SUPPORT：整顆不進 binning
    print(f"opacity > 1/255 的粒子：{int(keep.sum()):,} / {len(keep):,}（{100*float(keep.float().mean()):.1f}%）")

    bg = torch.zeros((3,), device=dev)
    acc = {k: 0.0 for k in ("lin_sq", "lin_rect", "lin_sq_e", "lin_rect_e",
                            "con_sq_f", "con_rect_f", "con_sq_e", "con_rect_e",
                            "con_asym_f", "con_asym_e")}
    stats = {"unbounded": 0, "n": 0, "bigger": 0, "cmp_n": 0, "dmax": 0.0, "dsum": 0.0}
    chk_n = chk_bad = 0
    chk_maxdiff = 0.0

    for i in range(n_cam):
        cam = cams[i].to_device(dev)
        vm = cam.world_to_camera.detach().flatten().tolist()
        W, H = int(cam.width), int(cam.height)
        fx = float(W) / (2.0 * math.tan(float(cam.fov_x) * 0.5))
        fy = float(H) / (2.0 * math.tan(float(cam.fov_y) * 0.5))
        cx_i, cy_i = W / 2.0, H / 2.0
        gw = (W + args.tile - 1) // args.tile
        gh = (H + args.tile - 1) // args.tile

        ux, uy, ex, ey, ok, T0, T1, T3 = compute_extent(xyz, quat, sc, vm, fx, fy, cx_i, cy_i, W, H)
        m = ok & keep
        if int(m.sum()) == 0:
            continue
        exm, eym = ex[m], ey[m]
        cxm, cym = ux[m], uy[m]
        # ★ 逐顆的「精確支撐半徑」：alpha = o*exp(-rho/2) >= 1/255  <=>  rho <= 2*ln(255*o)
        #   => truncated_R = sqrt(2*ln(255*o))，與 3 取 min（EXACT_SUPPORT_GROW=0 的保守版）。
        tr_exact = torch.sqrt((2.0 * torch.log(255.0 * op[m])).clamp_min(1e-6)).clamp(max=3.0)
        tr_flat = torch.full_like(exm, TRUNCATED_R)

        def rad(tr, e):
            """線性化路徑（現行 CUDA）：ceil(R * max(extent_at_1, FilterSize))"""
            return torch.ceil(tr * torch.maximum(e, torch.full_like(e, FILTER_SIZE)))

        def rad_c(e_at_R, tr):
            """精確圓錐路徑：extent 已經是該層的，低通圓盤項仍是 FilterSize * R"""
            return torch.ceil(torch.maximum(e_at_R, FILTER_SIZE * tr))

        # ── 線性化（現行實作）──
        acc["lin_sq"] += float(tiles_from_rect(cxm, cym, rad(tr_flat, torch.maximum(exm, eym)),
                                               rad(tr_flat, torch.maximum(exm, eym)), args.tile, gw, gh).sum())
        acc["lin_rect"] += float(tiles_from_rect(cxm, cym, rad(tr_flat, exm), rad(tr_flat, eym),
                                                 args.tile, gw, gh).sum())
        acc["lin_sq_e"] += float(tiles_from_rect(cxm, cym, rad(tr_exact, torch.maximum(exm, eym)),
                                                 rad(tr_exact, torch.maximum(exm, eym)), args.tile, gw, gh).sum())
        acc["lin_rect_e"] += float(tiles_from_rect(cxm, cym, rad(tr_exact, exm), rad(tr_exact, eym),
                                                   args.tile, gw, gh).sum())

        # ── 精確圓錐盒 ──
        T0m, T1m, T3m = T0[m], T1[m], T3[m]
        for tag, tr in (("f", tr_flat), ("e", tr_exact)):
            ccx, ccy, cex, cey, cok = conic_box(T0m, T1m, T3m, tr)
            stats["unbounded"] += int((~cok).sum()); stats["n"] += int(cok.numel())
            # d <= 0 => 該層在螢幕上無界，精確盒不存在；保守退回線性化盒（與現行同）
            lx = rad(tr, torch.maximum(exm, eym)); ly = lx
            # ⚠ 中心**不能動**：`points_xy_image` 同時是 renderCUDA 低通圓盤 rho2d 的中心，
            #   backward.cu 的 computeAABB 也按 r=1 的中心公式回傳梯度。
            #   => 盒子必須仍以 r=1 中心表示，半徑撐大到蓋住位移後的圓錐盒：|位移| + 該層半寬。
            rx = torch.where(cok, rad_c((ccx - cxm).abs() + cex, tr), rad(tr, exm))
            ry = torch.where(cok, rad_c((ccy - cym).abs() + cey, tr), rad(tr, eym))
            sq = torch.maximum(rx, ry)
            acc["con_sq_" + tag] += float(tiles_from_rect(cxm, cym, sq, sq, args.tile, gw, gh).sum())
            acc["con_rect_" + tag] += float(tiles_from_rect(cxm, cym, rx, ry, args.tile, gw, gh).sum())
            # ★ 非對稱盒：直接用 [pR - hR, pR + hR]，再與低通圓盤（中心仍是不動的 r=1 中心）取聯集。
            #   這是 getRect 改成 (rect_min, rect_max) 之後才表達得出來的形狀，
            #   省掉「|中心位移| 被左右各算一次」那筆。center 本身完全不動 => 渲染與梯度不受影響。
            fs = FILTER_SIZE * tr
            ax0 = torch.where(cok, torch.minimum(ccx - cex, cxm - fs), cxm - rad(tr, exm))
            ax1 = torch.where(cok, torch.maximum(ccx + cex, cxm + fs), cxm + rad(tr, exm))
            ay0 = torch.where(cok, torch.minimum(ccy - cey, cym - fs), cym - rad(tr, eym))
            ay1 = torch.where(cok, torch.maximum(ccy + cey, cym + fs), cym + rad(tr, eym))
            acc["con_asym_" + tag] += float(tiles_from_minmax(torch.floor(ax0), torch.ceil(ax1),
                                                              torch.floor(ay0), torch.ceil(ay1),
                                                              args.tile, gw, gh).sum())
            if tag == "f":
                # 現行盒比精確盒小多少 => 現行基準自己漏掉的覆蓋（單位 px，取長軸）
                dlt = (torch.maximum(rx, ry) - lx)[cok]
                if dlt.numel() > 0:
                    stats["bigger"] += int((dlt > 0).sum()); stats["cmp_n"] += int(dlt.numel())
                    stats["dmax"] = max(stats["dmax"], float(dlt.max()))
                    stats["dsum"] += float(dlt.clamp_min(0).sum())

        # ── 先驗關卡：與光柵器回傳的 radii 比對（只做前幾台，省時間）──
        if i < 3:
            with torch.no_grad():
                rr = renderer(cam, model, bg_color=bg).get("radii")
            if rr is not None:
                rr = rr.double()
                vis = rr > 0
                both = vis & m
                if int(both.sum()) > 0:
                    d = (r_sq[m[m]] if False else None)
                    mine = torch.ceil(TRUNCATED_R * torch.maximum(
                        torch.maximum(ex, ey), torch.full_like(ex, FILTER_SIZE)))
                    diff = (mine[both] - rr[both]).abs()
                    chk_n += int(both.sum())
                    chk_bad += int((diff > 1).sum())
                    chk_maxdiff = max(chk_maxdiff, float(diff.max()))

    print()
    if chk_n == 0:
        print("⛔⛔ 先驗關卡沒跑到（光柵器沒回傳 radii 或沒有共同可見粒子）=> 下面的數字不可信")
    else:
        bad_pct = 100.0 * chk_bad / chk_n
        print(f"★ 先驗關卡：解析 radius vs 光柵器 radii，比對 {chk_n:,} 顆")
        print(f"  差 > 1 px 的比例 {bad_pct:.2f}%   最大差 {chk_maxdiff:.1f} px")
        if bad_pct > 1.0:
            print("  ⛔⛔ **對不上（> 1%）=> 解析投影與光柵器不是同一條路，下面的天花板不可引用**")
        else:
            print("  ✅ 對得上（同定義同結果）")

    base = acc["lin_sq"]
    def row(name, v):
        print(f"  {name:<34} = {v:>16,.0f}   {100*(v/base-1):+7.1f}%")
    print()
    print("Σ tile（8 種組合；基準 = 線性化＋正方形＋3 sigma＝現行 CUDA）")
    print("── 線性化外接盒（extent(1) x R，中心不動）──")
    row("正方形 + 3 sigma  ★現行", acc["lin_sq"])
    row("矩形   + 3 sigma", acc["lin_rect"])
    row("正方形 + 精確支撐半徑", acc["lin_sq_e"])
    row("矩形   + 精確支撐半徑", acc["lin_rect_e"])
    print("── 精確圓錐盒（簽名 (R^2,R^2,-1)，中心隨 R 移動）──")
    row("正方形 + 3 sigma", acc["con_sq_f"])
    row("矩形   + 3 sigma", acc["con_rect_f"])
    row("非對稱盒 + 3 sigma  ★解耦後的真解", acc["con_asym_f"])
    row("正方形 + 精確支撐半徑", acc["con_sq_e"])
    row("矩形   + 精確支撐半徑", acc["con_rect_e"])
    row("非對稱盒 + 精確支撐半徑 ★全疊加", acc["con_asym_e"])

    print()
    print("★ 現行線性化盒到底偏多少（同為 3 sigma，逐顆逐相機比長軸半徑）")
    if stats["cmp_n"]:
        print(f"  精確盒**比現行大**的比例 {100.0*stats['bigger']/stats['cmp_n']:.2f}%"
              f"（= 現行漏覆蓋的顆粒佔比）   最大漏 {stats['dmax']:.0f} px"
              f"   平均漏 {stats['dsum']/stats['cmp_n']:.3f} px")
    if stats["n"]:
        print(f"  d == 0（退化，精確盒不存在，精確盒不存在，退回線性化）：{stats['unbounded']:,}"
              f" / {stats['n']:,}（{100.0*stats['unbounded']/stats['n']:.3f}%）")
    print()
    print("  ⚠ 這是 binning 的上界，不是 forward 時間的上界 —— 實測 Load 降 74~77% 時 forward 只降 17~21%")


if __name__ == "__main__":
    main()
