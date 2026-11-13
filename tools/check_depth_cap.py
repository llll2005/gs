"""★ 2026-11-11 逐貢獻深度上限（光柵器 CAP_OFFSET 通道）的檢查：小型合成場景，直接呼叫光柵器。

  base   用**現在裝著的**光柵器存下基準（完整路徑與 lean 路徑的渲染、前 8 個幾何通道、全部梯度）=> 重編前跑
  cmp    重編後跑：沒設上限圖時，渲染／前 8 個通道／梯度要與基準**逐位元相同**（新通道恆為 0）
  grad   上限通道：沒設上限的像素恆為 0、上限放寬到不可能超出時全 0、設了上限圖後前 8 個通道逐位元不變；
         有限差分把關三項（不透明度、尺度、全體沿 z 平移），位置／旋轉的隨機方向只印（會改排序，既有通道同樣對不上）
  ★ 11-11 規格 2（落後權重 W 與逐顆兩筆帳 Vc／Vb）：
  base／cmp 另外比「設了上限圖」的完整路徑：渲染與前 9 個通道（含 B）逐位元相同、梯度在重跑雜訊內
  acc    設了逐顆帳：渲染／10 個通道逐位元不變、梯度在重跑雜訊內；Σ Vc＝Σ_{有上限} alpha；Σ_{有上限} W＝Σ Vb；
         上限極近 => Vb＝Vc、W＝alpha；極遠 => W、Vb 全 0；上限圖全 0 => 全 0；計時（有上限圖的 forward，開帳 vs 不開）
  ★ 11-13 規格 3 §4-3 只補前面（depth_cap_grad_mode 3）：
  fc     渲染／10 個通道與模式 0 逐位元相同；L＝Σ B·w 的梯度：全在上限前面的顆粒（Vb＝0）與模式 0 相同（前面的本來就只經由 α）、
         全在上限後面的顆粒（Vb＝Vc）全部參數梯度恰為 0；不含 B 的 loss 梯度與模式 0 相同；用完模式回到 0（再跑模式 0 與基準相同）
用法：python tools/check_depth_cap.py base|cmp|grad|acc|fc
"""
import sys, math, os
import torch
from diff_trim_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

DEV = "cuda"
W, H = 320, 208
BASE = "/tmp/claude-1000/check_depth_cap_base.pt" if os.path.isdir("/tmp/claude-1000") else "check_depth_cap_base.pt"


def scene(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.stack([(torch.rand(n, generator=g) - 0.5) * 6, (torch.rand(n, generator=g) - 0.5) * 4, 4 + torch.rand(n, generator=g) * 6], 1)
    scales = torch.rand(n, 2, generator=g) * 0.25 + 0.03
    rots = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    opac = torch.rand(n, 1, generator=g) * 0.9 + 0.05
    cols = torch.rand(n, 3, generator=g)
    return [t.to(DEV).requires_grad_(True) for t in (means, scales, rots, opac, cols)]


def settings(**kw):
    fov = math.radians(60)
    tanx = math.tan(fov / 2); tany = tanx * H / W
    view = torch.eye(4, device=DEV)                       # 相機在原點看 +z（world_to_camera 的轉置慣例下 I 仍是 I）
    zn, zf = 0.01, 100.0
    P = torch.zeros(4, 4, device=DEV)
    P[0, 0] = 1 / tanx; P[1, 1] = 1 / tany; P[2, 2] = zf / (zf - zn); P[2, 3] = -(zf * zn) / (zf - zn); P[3, 2] = 1.0
    proj = (P @ view.T).T                                  # full_projection（列向量慣例）
    return GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanx, tanfovy=tany, bg=torch.zeros(3, device=DEV), scale_modifier=1.0,
        viewmatrix=view, projmatrix=proj, sh_degree=0, campos=torch.zeros(3, device=DEV), prefiltered=False,
        record_transmittance=False, debug=False, **kw)


def render(params, **kw):
    means, scales, rots, opac, cols = params
    m2d = torch.zeros_like(means, requires_grad=True)
    out = GaussianRasterizer(settings(**kw))(means3D=means, means2D=m2d, shs=None, colors_precomp=cols, opacities=opac,
                                             scales=scales, rotations=rots, cov3D_precomp=None)
    return out[0], out[2]                                  # 顏色、幾何通道


def grads_of(loss, params):
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(loss, params, allow_unused=True), params)]


def run_paths():
    res = {}
    p = scene()
    tgt = torch.rand(3, H, W, generator=torch.Generator().manual_seed(1)).to(DEV)
    wgt = torch.rand(8, H, W, generator=torch.Generator().manual_seed(2)).to(DEV)
    col, oth = render(p)                                   # 完整路徑：顏色＋全部幾何通道都進 loss
    loss = (col - tgt).abs().mean() + (oth[:8] * wgt).mean()
    res["full"] = dict(col=col.detach().cpu(), oth=oth[:8].detach().cpu(), g=[g.cpu() for g in grads_of(loss, p)], nch=oth.shape[0])
    p = scene()
    col, oth = render(p, lean_render=True, bwd_sat_skip=True, bwd_reduce=True)   # lean 路徑：只有顏色
    loss = (col - tgt).abs().mean()
    res["lean"] = dict(col=col.detach().cpu(), g=[g.cpu() for g in grads_of(loss, p)])
    # ★ 11-11 設了上限圖的完整路徑（前 9 個通道含 B 進 loss）
    cap = cap_map()
    wgt9 = torch.rand(9, H, W, generator=torch.Generator().manual_seed(4)).to(DEV)
    p = scene()
    col, oth = render(p, depth_cap=cap)
    loss = (col - tgt).abs().mean() + (oth[:9] * wgt9).mean()
    res["cap"] = dict(col=col.detach().cpu(), oth=oth[:9].detach().cpu(), g=[g.cpu() for g in grads_of(loss, p)])
    return res


def cap_map(n=4000, seed=0, scale=0.97):
    """上限＝中位深度 x scale（中位面本身與後面都會超出一點）；左四分之一不設上限。"""
    with torch.no_grad():
        _, o0 = render(scene(n=n, seed=seed))
    med = o0[5]
    cap = torch.where(med > 0, 1.0 / (med * scale), torch.zeros_like(med)).contiguous()
    cap[:, : W // 4] = 0.0
    return cap


def main():
    mode = sys.argv[1]
    if mode == "base":
        torch.save(run_paths(), BASE)
        print(f"基準已存 {BASE}（幾何通道數 {run_paths()['full']['nch']}）")
        return
    if mode == "cmp":
        b = torch.load(BASE); r = run_paths(); ok = True
        print(f"幾何通道數：基準 {b['full']['nch']} -> 現在 {r['full']['nch']}")
        for k in ("full", "lean", "cap"):
            if k not in b:
                print(f"  （基準沒有 {k}：是舊版 check 存的）"); continue
            for name in ("col", "oth"):
                if name in b[k]:
                    same = torch.equal(b[k][name], r[k][name]); ok &= same
                    print(f"  {k:5} {name}: {'逐位元相同' if same else '⛔ 不同 max|d|=%.3e' % float((b[k][name] - r[k][name]).abs().max())}")
            for i, nm in enumerate(("means", "scales", "rots", "opac", "cols")):
                d = float((b[k]["g"][i] - r[k]["g"][i]).abs().max()); s = float(b[k]["g"][i].abs().max())
                same = torch.equal(b[k]["g"][i], r[k]["g"][i])
                # lean 的 backward 先在 block 內加總（atomic 順序不固定）=> 允許 1e-5 相對差；完整路徑也有 atomicAdd 的順序差
                good = same or d <= 2e-5 * max(s, 1e-12); ok &= good
                print(f"  {k:5} grad {nm:6}: {'逐位元相同' if same else '差 %.2e（最大值 %.2e）' % (d, s)}{'' if good else ' ⛔'}")
        p = scene(); col, oth = render(p)
        z = float(oth[8:].abs().max()) if oth.shape[0] > 8 else float("nan")
        print(f"  沒設上限圖時新通道（第 9 個起，共 {oth.shape[0] - 8} 個）最大值 {z}（要是 0）"); ok &= (z == 0.0)
        print("✅ 通過" if ok else "⛔ 沒過"); sys.exit(0 if ok else 1)
    if mode == "grad":
        ok = True
        p = scene(n=1500, seed=3)
        with torch.no_grad():
            _, o0 = render(scene(n=1500, seed=3))
        med = o0[5]                                         # 中位深度當「先驗表面」；上限＝中位深度 x 0.97 => 中位面後面與中位面本身都會超出一點
        cap = torch.where(med > 0, 1.0 / (med * 0.97), torch.zeros_like(med)).contiguous()
        cap[:, : W // 4] = 0.0                              # 左四分之一不設上限
        col, oth = render(p, depth_cap=cap)
        B = oth[8]
        print(f"上限通道：有上限的像素 {float((cap > 0).float().mean()):.3f}、其中 B>0 {float((B[cap > 0] > 0).float().mean()):.3f}、B 平均 {float(B[cap > 0].mean()):.5f}、沒上限處最大 {float(B[cap == 0].abs().max()):.1e}")
        ok &= float(B[cap == 0].abs().max()) == 0.0 and float(B.min()) >= 0.0
        # 上限放寬到不可能超出 => 全 0
        _, o2 = render(scene(n=1500, seed=3), depth_cap=(cap * 1e-3).contiguous())
        print(f"上限放得很寬時 B 最大 {float(o2[8].abs().max()):.1e}（要是 0）"); ok &= float(o2[8].abs().max()) == 0.0
        # 其他通道不受上限圖影響
        same = torch.equal(o0[:8], oth[:8].detach()); ok &= same
        print(f"設了上限圖後前 8 個通道{'逐位元相同' if same else ' ⛔ 變了'}")
        # 有限差分：L = Σ B·w（w 固定的隨機權重）
        wB = torch.rand(H, W, generator=torch.Generator().manual_seed(5)).to(DEV)
        L = (B * wB).sum()
        g = grads_of(L, p)

        def Lof(q):
            with torch.no_grad():
                return float((render(q, depth_cap=cap)[1][8].double() * wB.double()).sum())

        # 把關的三項（對排序不敏感、可以精確比）：不透明度、尺度、全體沿 z 平移（dB/dz 那條路）
        gen = torch.Generator().manual_seed(7)
        dirs = [(3, "不透明度", torch.randn(p[3].shape, generator=gen).to(DEV), 0.10),   # alpha<1/255 與飽和提早結束是不連續的 => 容差放寬
                (1, "尺度", torch.randn(p[1].shape, generator=gen).to(DEV), 0.10)]
        dz = torch.zeros_like(p[0]); dz[:, 2] = 1.0
        dirs.append((0, "全體沿 z 平移", dz, 0.03))
        # 只印不把關：位置／旋轉的隨機方向（會改排序與外接盒；既有的 alpha／期望深度通道在同一個檢查下也差 25～53%）
        dirs += [(0, "位置（只印）", torch.randn(p[0].shape, generator=gen).to(DEV), None), (2, "旋轉（只印）", torch.randn(p[2].shape, generator=gen).to(DEV), None)]
        for idx, nm, d, tol in dirs:
            d = d * (g[idx].abs().sum(-1, keepdim=True) > 0)
            ana = float((g[idx].double() * d.double()).sum())
            eps = 2e-4
            qp = [t.detach().clone() for t in p]; qm = [t.detach().clone() for t in p]
            qp[idx] += eps * d; qm[idx] -= eps * d
            num = (Lof(qp) - Lof(qm)) / (2 * eps)
            rel = abs(ana - num) / max(abs(num), abs(ana), 1e-9)
            good = tol is None or rel < tol; ok &= good
            print(f"  方向導數 {nm:10}: 解析 {ana:+.5e}｜有限差分 {num:+.5e}｜相對差 {rel:.3f}{'' if good else ' ⛔'}")
        print("✅ 通過" if ok else "⛔ 沒過"); sys.exit(0 if ok else 1)


    if mode == "acc":
        acc_checks()
    if mode == "fc":
        fc_checks()


def fc_checks():
    """★ 11-13 規格 3 §4-3 只補前面：depth_cap_grad_mode＝3。上限放在中位深度 x 1.15（前面、後面、兩邊都有的顆粒都夠多）。"""
    ok = True
    n = 4000
    cap = cap_map(scale=1.15)
    wB = torch.rand(H, W, generator=torch.Generator().manual_seed(5)).to(DEV)
    tgt = torch.rand(3, H, W, generator=torch.Generator().manual_seed(1)).to(DEV)
    wgt = torch.rand(8, H, W, generator=torch.Generator().manual_seed(2)).to(DEV)

    def run(gm, with_B=True, acc=None):
        q = scene()
        kw = dict(depth_cap=cap, **({"depth_cap_grad_mode": gm} if gm else {}), **({"depth_cap_acc": acc} if acc is not None else {}))
        col, oth = render(q, **kw)
        loss = (oth[8] * wB).sum() if with_B else (col - tgt).abs().mean() + (oth[:8] * wgt).mean()
        return col.detach(), oth.detach(), grads_of(loss, q)
    vb, vc = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV)
    c0, o0, g0 = run(0, acc=(vb, vc))
    _, _, g0b = run(0)
    c3, o3, g3 = run(3)
    fwd = torch.equal(c0, c3) and torch.equal(o0, o3)
    print(f"1. 模式 3 的渲染／{o0.shape[0]} 個通道與模式 0 逐位元相同 {'✅' if fwd else '⛔'}"); ok &= fwd
    seen = vc > 1e-6
    front = seen & (vb == 0)
    behind = seen & ((vc - vb).abs() <= 1e-6 * vc.clamp_min(1e-12))
    mixed = seen & ~front & ~behind
    print(f"   有上限像素裡出現的顆粒 {int(seen.sum()):,}：全在前面 {int(front.sum()):,}、全在後面 {int(behind.sum()):,}、兩邊都有 {int(mixed.sum()):,}")
    noise = max(float((a - b).abs().max()) for a, b in zip(g0, g0b))
    names = ("means", "scales", "rots", "opac", "cols")
    for i, nm in enumerate(names):
        if nm == "cols":
            continue
        d_front = float((g3[i][front] - g0[i][front]).abs().max()) if front.any() else 0.0
        s_front = float(g0[i][front].abs().max()) if front.any() else 0.0
        z_behind = float(g3[i][behind].abs().max()) if behind.any() else 0.0
        nz0 = float(g0[i][behind].abs().max()) if behind.any() else 0.0
        good = d_front <= max(10 * noise, 1e-6 * max(s_front, 1e-12)) and z_behind == 0.0
        ok &= good
        print(f"2. grad {nm:6}：全在前面 與模式 0 最大差 {d_front:.2e}（最大值 {s_front:.2e}、重跑雜訊 {noise:.2e}）｜全在後面 模式 3 最大 {z_behind:.1e}（模式 0 是 {nz0:.2e}）{'✅' if good else '⛔'}")
    # 3. 不含 B 的 loss：梯度與模式 0 相同（只動 B 的梯度）
    _, _, h0 = run(0, with_B=False)
    _, _, h0b = run(0, with_B=False)
    _, _, h3 = run(3, with_B=False)
    nz = max(float((a - b).abs().max()) for a, b in zip(h0, h0b))
    dd = max(float((a - b).abs().max()) for a, b in zip(h0, h3))
    good = dd <= max(10 * nz, 1e-7); ok &= good
    print(f"3. 不含 B 的 loss：模式 3 對模式 0 梯度最大差 {dd:.2e}（重跑雜訊 {nz:.2e}）{'✅' if good else '⛔'}")
    # 4. 用完回到 0：再跑模式 0，B 的梯度與一開始的模式 0 相同
    _, _, g0c = run(0)
    dd = max(float((a - b).abs().max()) for a, b in zip(g0, g0c))
    good = dd <= max(10 * noise, 1e-7); ok &= good
    print(f"4. 模式 3 跑完後再跑模式 0：B 的梯度對最初的模式 0 最大差 {dd:.2e} {'✅' if good else '⛔'}")
    # 5. 方向（只印）：模式 3 下 L 對「全在前面」顆粒不透明度的方向導數 < 0（變更不透明 => 後面分到的少 => B 變小）
    gf = g3[3][front]
    print(f"5. 全在前面且梯度非 0 的顆粒：dL/d不透明度 < 0 的比例 {float((gf[gf != 0] < 0).float().mean()) if (gf != 0).any() else float('nan'):.3f}（要接近 1）")
    print("✅ 通過" if ok else "⛔ 沒過"); sys.exit(0 if ok else 1)


def acc_checks():
    """★ 11-11 規格 2 §1：落後權重 W（第 10 個通道）與逐顆兩筆帳（depth_cap_acc＝(Vb, Vc)）。"""
    import time
    ok = True
    n = 4000
    cap = cap_map()
    tgt = torch.rand(3, H, W, generator=torch.Generator().manual_seed(1)).to(DEV)
    wgt = torch.rand(10, H, W, generator=torch.Generator().manual_seed(2)).to(DEV)

    def run(acc):
        q = scene()
        col, oth = render(q, depth_cap=cap, **({"depth_cap_acc": acc} if acc is not None else {}))
        loss = (col - tgt).abs().mean() + (oth[:9] * wgt[:9]).mean()
        return col.detach(), oth.detach(), grads_of(loss, q)
    c0, o0, g0 = run(None)
    _, _, g0b = run(None)
    vb, vc = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV)
    c1, o1, g1 = run((vb, vc))
    fwd = torch.equal(c0, c1) and torch.equal(o0, o1)
    gd = lambda x, y: max(float((a_ - b_).abs().max()) for a_, b_ in zip(x, y))
    noise, diff = gd(g0, g0b), gd(g0, g1)
    g_ok = diff <= max(10 * noise, 1e-7)
    print(f"1. 開逐顆帳：渲染／{o0.shape[0]} 個通道逐位元相同 {'✅' if fwd else '⛔'}｜梯度最大差 {diff:.2e}（重跑雜訊 {noise:.2e}）{'✅' if g_ok else '⛔'}")
    ok &= fwd and g_ok
    Wm, A = o0[9], o0[1]
    has = cap > 0
    r1 = abs(float(vc.sum()) - float(A[has].sum())) / max(float(A[has].sum()), 1e-9)
    r2 = abs(float(vb.sum()) - float(Wm[has].sum())) / max(float(Wm[has].sum()), 1e-9)
    print(f"2. Σ Vc {float(vc.sum()):.3f} vs Σ_有上限 alpha {float(A[has].sum()):.3f}（相對差 {r1:.1e}）｜Σ Vb {float(vb.sum()):.3f} vs Σ_有上限 W {float(Wm[has].sum()):.3f}（相對差 {r2:.1e}）"
          f"｜沒上限處 W 最大 {float(Wm[~has].abs().max()):.1e}  {'✅' if (r1 < 1e-4 and r2 < 1e-4 and float(Wm[~has].abs().max()) == 0) else '⛔'}")
    ok &= r1 < 1e-4 and r2 < 1e-4 and float(Wm[~has].abs().max()) == 0
    print(f"   有上限的像素 {float(has.float().mean()):.3f}、其中 W>0 {float((Wm[has] > 0).float().mean()):.3f}；有帳的顆粒 {int((vc > 0).sum()):,}／{n:,}、Vb>0 {int((vb > 0).sum()):,}")
    for nm, c_, want in (("極近（全部在後面）", torch.full((H, W), 2.0, device=DEV), "all"), ("極遠（放很寬）", (cap * 1e-3).contiguous(), "none"),
                         ("上限圖全 0", torch.zeros(H, W, device=DEV), "zero")):
        vb, vc = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV)
        with torch.no_grad():
            _, o = render(scene(), depth_cap=c_, depth_cap_acc=(vb, vc))
        if want == "all":
            good = torch.allclose(vb, vc) and torch.allclose(o[9], o[1])
        elif want == "none":
            good = bool((vb == 0).all()) and float(o[9].abs().max()) == 0.0 and float(vc.sum()) > 0
        else:
            good = bool((vb == 0).all() and (vc == 0).all()) and float(o[9].abs().max()) == 0.0
        print(f"3. {nm}：Σ Vb {float(vb.sum()):.3f}、Σ Vc {float(vc.sum()):.3f}、Σ W {float(o[9].sum()):.3f}  {'✅' if good else '⛔'}"); ok &= good
    # 逐顆比例 vs 中心深度（只印）：平面上限 z＝7
    vb, vc = torch.zeros(n, device=DEV), torch.zeros(n, device=DEV)
    q = scene()
    with torch.no_grad():
        render(q, depth_cap=torch.full((H, W), 1.0 / 7.0, device=DEV), depth_cap_acc=(vb, vc))
    z = q[0][:, 2].detach(); seen = vc > 1e-3; rho = torch.where(seen, vb / vc.clamp_min(1e-12), torch.zeros_like(vc))
    print("4. 平面上限 z＝7 的逐顆 ρ＝Vb/Vc（只印）：" + "；".join(
        f"中心 z∈[{lo},{hi}) {int((seen & (z >= lo) & (z < hi)).sum()):,} 顆 ρ 中位 {float(rho[seen & (z >= lo) & (z < hi)].median()):.2f}"
        for lo, hi in ((4, 6.5), (6.5, 7.5), (7.5, 10))))
    # 計時：較大的場景，有上限圖的 forward（no_grad），開帳 vs 不開
    big = scene(n=200000, seed=9)
    capb = torch.full((H, W), 1.0 / 7.0, device=DEV)
    def tf(acc):
        ts = []
        with torch.no_grad():
            for i in range(30):
                if acc:
                    vb_, vc_ = torch.zeros(200000, device=DEV), torch.zeros(200000, device=DEV)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                render(big, depth_cap=capb, **({"depth_cap_acc": (vb_, vc_)} if acc else {}))
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return sorted(ts[5:])[len(ts[5:]) // 2] * 1e3
    t0_, t1_ = tf(False), tf(True)
    print(f"5. 計時（20 萬顆、{W}x{H}、有上限圖的完整路徑 forward 中位）：不開帳 {t0_:.2f} ms、開帳 {t1_:.2f} ms（{100 * (t1_ / t0_ - 1):+.1f}%；"
          f"訓練只在完整路徑步（每 8 步一次）開 => 平均每步約 {100 * (t1_ / t0_ - 1) / 8:+.2f}% 的 forward）")
    print("✅ 通過" if ok else "⛔ 沒過"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
