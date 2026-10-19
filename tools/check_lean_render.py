"""lean_render（剃除沒人用的光柵器計算）的等價性與速度驗證（2026-10-02）。

改了什麼（cuda_rasterizer/forward.cu 的 MODE、backward.cu 的 GEOM）：
  訓練 forward   不算深度／法線／中位深度／distortion（只留顏色＋alpha）
  訓練 backward  跳過幾何梯度：每個 (像素, 顆粒) 配對 3 個加 0 的法線 atomicAdd 與深度/distortion 算術
  trim pass      record 模式只累積 T*alpha 與覆蓋數，不算顏色與幾何
現行配方 normal/dist/depth 權重全為 0 => 被跳過的部分對 loss 與梯度的貢獻恰為 0。

驗證要回答三件事（同一個 ckpt、同一組相機、同一個目標影像）：
  A 舊 .so 跑兩次                  => 噪音底（backward 的浮點 atomicAdd 順序本來就不固定）
  B 舊 .so vs 新 .so（lean 關）    => 新 .so 的舊路徑沒有被改到（渲染必須逐位元相同，梯度在噪音底內）
  C 新 .so：lean 關 vs 開          => lean 等價（渲染逐位元相同、梯度與 record 輸出在噪音底內）並量速度

用法：
  dump    python tools/check_lean_render.py dump --ckpt <ckpt> --out <檔> --lean 0|1 [--ncam 6] [--repeat 10]
          （新 .so 用 PYTHONPATH 指到另外安裝的目錄，不碰共用環境）
  compare python tools/check_lean_render.py compare <a.pt> <b.pt> [--label 說明]
  profile python tools/check_lean_render.py profile --ckpt <ckpt> --lean 0|1 [--steps 12]
          kernel 層級拆解一個訓練步＋一次 trim record pass（逐配對／逐顆／逐像素／逐參數）；--fused 1 用 fused Adam
          ★ 10-19：--fast 1 --running 1 --absgrad 0|1 => 現行最快設定（lean＋backward 加速＋累加式 trim）
  visible python tools/check_lean_render.py visible --ckpt <ckpt> [--ncam 0] [--fast 1]
          每個訓練視角看得到／有梯度的顆粒比例（全部、盒內）=> 稀疏 Adam 能省多少的上限
  adamcheck python tools/check_lean_render.py adamcheck --ckpt <ckpt>
          fused Adam vs 預設 foreach：同梯度 1 步／50 步的參數差（以更新量為單位）與計時
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_dump(a):
    import diff_trim_surfel_rasterization as R
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    dev = torch.device("cuda")
    print(f"光柵器：{R.__file__}  有 lean_render 欄位：{'lean_render' in R.GaussianRasterizationSettings._fields}")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    renderer.lean_train = bool(a.lean)
    renderer._lean_announced = True
    renderer.record_reduce = bool(getattr(a, "record_reduce", 0))
    renderer.tile_cull = bool(getattr(a, "tile_cull", 0))
    # ★ 2026-10-09 absgrad_gate 驗證：--absgrad 0/1 明確指定 backward 要不要做 absgrad 的 atomicAdd（-1＝不碰，舊行為）
    _ab = int(getattr(a, "absgrad", -1))
    if _ab >= 0:
        renderer.absgrad_gate = True
        renderer._absgrad_announced = True
    TF = {"_absgrad": bool(_ab)} if _ab >= 0 else {}
    # ★ 2026-10-19 dist 版 lean 驗證：--dist w => loss 加 w*mean(rend_dist)；lean 開時明確走 lean_dist（MODE 5＋DST）
    DW = float(getattr(a, "dist", 0.0) or 0.0)
    if DW > 0:
        TF["_lean_dist"] = bool(a.lean)
    renderer.bwd_sat_skip = bool(getattr(a, "bwd_sat", 0))   # ★ 2026-10-09 backward 最佳化
    renderer.bwd_reduce = bool(getattr(a, "bwd_red", 0))
    renderer.bwd_pair_bit = bool(getattr(a, "pair_bit", 0))   # ★ 2026-10-16 配對貢獻 bit
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"      # 只要相機；init PLY 不在本機時不讀
    cams = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                                     global_rank=0).get_outputs().train_set.cameras
    del ck
    idx = np.linspace(0, len(cams) - 1, a.ncam).round().astype(int).tolist()
    params = dict(model.gaussians.items())
    for p in params.values():
        p.requires_grad_(True)
    bg = torch.zeros(3, device=dev)
    N = model.n_gaussians
    print(f"N={N:,}  相機 {idx}  lean={a.lean}  exact_conic_aabb={getattr(renderer, 'exact_conic_aabb', False)}")

    def step(cam, tgt):
        for p in params.values():
            p.grad = None
        out = renderer.training_forward(0, None, cam, model, bg, **TF)
        img = out["render"]
        loss = 0.8 * (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        if DW > 0:
            loss = loss + DW * out["rend_dist"].mean()
        loss.backward()
        return out

    res = {"renders": [], "radii": [], "trans": [], "cover": [], "tiles": [], "dists": [], "N": N,
           "lean": f"{a.lean}{'+rr' if renderer.record_reduce else ''}{'+tc' if renderer.tile_cull else ''}{('+ab' + str(_ab)) if _ab >= 0 else ''}{'+sat' if renderer.bwd_sat_skip else ''}{'+red' if renderer.bwd_reduce else ''}{'+pb' if renderer.bwd_pair_bit else ''}{('+dist' + str(DW)) if DW > 0 else ''}"}
    gsum = {k: torch.zeros_like(p, dtype=torch.float64) for k, p in params.items()}
    vsum = None
    for c in idx:
        cam = cams[c].to_device(dev)
        g = torch.Generator(device=dev).manual_seed(1000 + c)
        tgt = torch.rand((3, int(cam.height), int(cam.width)), device=dev, generator=g)
        out = step(cam, tgt)
        res["renders"].append(out["render"].detach().cpu())
        if out.get("rend_dist") is not None:
            res["dists"].append(out["rend_dist"].detach().cpu())
        res["radii"].append(out["radii"].detach().cpu())
        if out.get("tiles") is not None:
            res["tiles"].append(out["tiles"].detach().cpu())
        for k, p in params.items():
            if p.grad is not None:
                gsum[k] += p.grad.double()
        vg = out["viewspace_points"].grad.detach().double()
        vsum = vg if vsum is None else vsum + vg
        with torch.no_grad():
            t, cv = renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True)
        res["trans"].append(t.detach().cpu()); res["cover"].append(cv.detach().cpu())
    res["grads"] = {k: v.float().cpu() for k, v in gsum.items()}
    res["vgrad"] = vsum.float().cpu()

    # ── 計時（相機 idx[0]；CUDA event；中位數）──
    cam = cams[idx[0]].to_device(dev)
    tgt = torch.rand((3, int(cam.height), int(cam.width)), device=dev, generator=torch.Generator(device=dev).manual_seed(7))
    tf, tb, tr = [], [], []
    for r in range(a.repeat + 2):
        for p in params.values():
            p.grad = None
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        out = renderer.training_forward(0, None, cam, model, bg, **TF)
        img = out["render"]
        loss = 0.8 * (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        if DW > 0:
            loss = loss + DW * out["rend_dist"].mean()
        e1.record()
        loss.backward()
        e2.record()
        torch.cuda.synchronize()
        with torch.no_grad():
            e3, e4 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e3.record()
            renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True)
            e4.record()
            torch.cuda.synchronize()
        if r >= 2:                                   # 前兩次暖機
            tf.append(e0.elapsed_time(e1)); tb.append(e1.elapsed_time(e2)); tr.append(e3.elapsed_time(e4))
    res["time"] = {"forward+loss": float(np.median(tf)), "backward": float(np.median(tb)),
                   "record(trim 一台相機)": float(np.median(tr))}
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step(cam, tgt)
    res["peak_fb_MiB"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    print("計時（ms，中位）：" + "  ".join(f"{k} {v:.2f}" for k, v in res["time"].items())
          + f"  ｜forward+backward 峰值增量 {res['peak_fb_MiB']:.0f} MiB")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(res, a.out)
    print(f"-> {a.out}")


def _apply_speed_flags(renderer, a):
    """★ 2026-10-19 現行最快設定（使用者 10-19「第 1 步」）。-1＝沿用 ckpt 裡 renderer 的設定。
    --fast 1     backward 飽和跳過＋先加總、record 先加總（best1 的兩個加速旗標＋record_reduce）
    --running 1  累加式 trim（lean 訓練 forward 順便累加 V/C；不開時 E 欄＝週期 trim 每台相機的 record 成本）
    --absgrad 1  增生期（backward 多累加 |dL/ds|）；0＝收割期；-1＝沿用 training_forward 的判斷（module=None => 關）"""
    fa = int(getattr(a, "fast", -1))
    if fa >= 0:
        renderer.bwd_sat_skip = renderer.bwd_reduce = renderer.record_reduce = bool(fa)
    ru = int(getattr(a, "running", -1))
    if ru >= 0:
        renderer.trim_running_vc = bool(ru)
    renderer._acc_announced = True
    TF = {}
    ab = int(getattr(a, "absgrad", -1))
    if ab >= 0:
        renderer.absgrad_gate = True
        renderer._absgrad_announced = True
        TF["_absgrad"] = bool(ab)
    return TF


def _flag_str(renderer, TF):
    keys = ("lean_train", "bwd_sat_skip", "bwd_reduce", "record_reduce", "trim_running_vc", "exact_conic_aabb", "absgrad_gate")
    return "  ".join(f"{k}={int(bool(getattr(renderer, k, False)))}" for k in keys) + f"  absgrad 累加={int(bool(TF.get('_absgrad', False)))}"


def cmd_visible(a):
    """★ 2026-10-19 每個訓練視角看得到多少比例的顆粒 => 稀疏 Adam（只更新本步有梯度的顆粒）最多能省多少。
    vis   radii>0：通過視錐與尺寸剔除、進了 binning（gsplat／Taming 的 visible Adam 用的就是這個遮罩）
    grad  光度 loss 對該顆有非零梯度：真的混合進至少一個像素（lean＋飽和跳過下＝這步實際有梯度的那批）
    盒內  中心在本塊分區盒內（不外擴；合併時保留的那批）
    ⚠ 正則項（opacity 二元熵、scale）對**全部**顆粒都有梯度 => 稀疏 Adam 等於改了它們的語意（只在可見時生效），不是逐位元等價。"""
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from crop_inbox_ckpt import inbox_mask
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    renderer.lean_train = bool(a.lean)
    renderer._lean_announced = True
    a.running = 0                       # 只量可見性，不累加 trim 狀態
    TF = _apply_speed_flags(renderer, a)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"
    cams = _pp.instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                           global_rank=0).get_outputs().train_set.cameras
    del ck
    params = dict(model.gaussians.items())
    for p in params.values():
        p.requires_grad_(True)
    N = model.n_gaussians
    inb = inbox_mask(model.get_xyz.detach().cpu(), _pp, 0.0).to(dev)
    n_in = int(inb.sum())
    idx = list(range(len(cams))) if a.ncam <= 0 else np.linspace(0, len(cams) - 1, a.ncam).round().astype(int).tolist()
    bg = torch.zeros(3, device=dev)
    ever_vis = torch.zeros(N, dtype=torch.bool, device=dev)
    ever_grad = torch.zeros(N, dtype=torch.bool, device=dev)
    rows = []
    for c in idx:
        cam = cams[c].to_device(dev)
        for p in params.values():
            p.grad = None
        out = renderer.training_forward(0, None, cam, model, bg, **TF)
        img = out["render"]
        tgt = torch.rand(img.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(1000 + c))
        loss = 0.8 * (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        loss.backward()
        vis = out["radii"] > 0
        g = torch.zeros(N, dtype=torch.bool, device=dev)
        for p in params.values():
            if p.grad is not None:
                g |= (p.grad.reshape(N, -1) != 0).any(1)
        ever_vis |= vis
        ever_grad |= g
        rows.append([float(vis.float().mean()), float(g.float().mean()),
                     float((vis & inb).sum()) / max(n_in, 1), float((g & inb).sum()) / max(n_in, 1),
                     float(inb[vis].float().mean()) if bool(vis.any()) else 0.0])
    R = np.array(rows)
    print(f"N={N:,}  盒內 {n_in:,}（{n_in / N:.1%}）  視角 {len(idx)}／{len(cams)}  ckpt={os.path.basename(a.ckpt)}")
    print("   生效旗標：" + _flag_str(renderer, TF))
    print("── 每視角比例（平均／中位／p10／p90）")
    for j, nm in enumerate(["可見 radii>0 ／全部", "有梯度 ／全部", "可見 ／盒內顆粒（只算盒內）", "有梯度 ／盒內顆粒（只算盒內）", "可見顆粒中盒內的佔比"]):
        v = R[:, j]
        print(f"   {nm:<26} {v.mean():7.1%} {np.median(v):7.1%} {np.percentile(v, 10):7.1%} {np.percentile(v, 90):7.1%}")
    print(f"── 這 {len(idx)} 個視角都沒看到：{1 - float(ever_vis.float().mean()):.2%}；都沒梯度：{1 - float(ever_grad.float().mean()):.2%}"
          f"（盒內 {1 - float(ever_grad[inb].float().mean()) if n_in else 0:.2%}）")
    print(f"── 稀疏 Adam 的上限：每步平均只有 {R[:, 1].mean():.1%} 的顆粒有光度梯度 => Adam 工作量最多省 {1 - R[:, 1].mean():.1%}"
          f"（以 radii>0 當遮罩則省 {1 - R[:, 0].mean():.1%}）")


def cmd_profile(a):
    """kernel 層級拆解一個訓練步（forward／loss／backward／Adam）＋一次 trim 的 record pass。
    回答「時間是不是花在梯度」以及梯度裡是「逐 (像素,顆粒) 配對」還是「逐顆」或「逐像素」的部分。"""
    import re
    from torch.profiler import profile, ProfilerActivity, record_function
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    renderer.lean_train = bool(a.lean)
    renderer._lean_announced = True
    TF = _apply_speed_flags(renderer, a)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    _pp = dmh["parser"]
    if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
        _pp.points_from = "random"      # 只要相機；init PLY 不在本機時不讀
    cams = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                                     global_rank=0).get_outputs().train_set.cameras
    del ck
    params = dict(model.gaussians.items())
    for p in params.values():
        p.requires_grad_(True)
    # 與訓練同構：每個參數一個 param group（訓練是 means 一個 optimizer＋其餘一個 optimizer，共 6 組）
    opt = torch.optim.Adam([{"params": [p], "lr": 1e-6} for p in params.values()], eps=1e-15,
                           **({"fused": True} if getattr(a, "fused", 0) else {}))
    bg = torch.zeros(3, device=dev)
    idx = np.linspace(0, len(cams) - 1, max(a.steps, 1)).round().astype(int).tolist()
    tg = {}

    def tgt_for(cam, c):
        if c not in tg:
            tg[c] = torch.rand((3, int(cam.height), int(cam.width)), device=dev,
                               generator=torch.Generator(device=dev).manual_seed(1000 + c))
        return tg[c]

    def one(c, prof_ranges):
        cam = cams[c].to_device(dev)
        tgt = tgt_for(cam, c)
        opt.zero_grad(set_to_none=True)
        with record_function("A_forward"):
            out = renderer.training_forward(0, None, cam, model, bg, **TF)
        with record_function("B_loss"):
            img = out["render"]
            loss = 0.8 * (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        with record_function("C_backward"):
            loss.backward()
        with record_function("D_adam"):
            opt.step()
        with record_function("E_trim_record_1cam"), torch.no_grad():
            renderer(cam, model, bg_color=bg, record_transmittance=True, record_coverage=True)

    for c in idx[:3]:
        one(c, False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for c in idx:
            one(c, True)
        torch.cuda.synchronize()
    S = len(idx)
    ka = prof.key_averages()
    print(f"N={model.n_gaussians:,}  lean={a.lean}  fused_adam={getattr(a, 'fused', 0)}  步數={S}（每步換一台相機）  ckpt={os.path.basename(a.ckpt)}")
    print("   生效旗標：" + _flag_str(renderer, TF))
    print("── 階段（kernel 時間，ms/步；以 record_function 歸屬）")
    for nm in ["A_forward", "B_loss", "C_backward", "D_adam", "E_trim_record_1cam"]:
        e = [k for k in ka if k.key == nm]
        if e:
            print(f"   {nm:<22} {e[0].cuda_time_total / 1e3 / S:8.2f}")

    def bucket(n):
        if "renderCUDA" in n:
            return "光柵 backward（逐配對）" if "float3*" in n else "光柵 forward／record（逐配對）"
        if "preprocessCUDA" in n:
            return "前處理 backward（逐顆）" if "float3 const*" in n else "前處理 forward（逐顆：投影＋SH）"
        if "computeAABB" in n:
            return "前處理 backward（逐顆）"
        if re.search(r"duplicateWithKeys|identifyTileRanges|RadixSort|radix|cub::|DeviceScan|Scan", n):
            return "binning 排序（逐配對）"
        if re.search(r"conv|cudnn|gemm|Gemm|winograd|implicit", n):
            return "SSIM 卷積（逐像素）"
        if re.search(r"multi_tensor|[Aa]dam", n):
            return "Adam（逐參數）"
        return "其他 PyTorch 小運算（逐顆／逐像素）"
    ker = [k for k in ka if k.self_cuda_time_total > 0 and k.key not in
           ("A_forward", "B_loss", "C_backward", "D_adam", "E_trim_record_1cam")]
    tot = sum(k.self_cuda_time_total for k in ker)
    b = {}
    for k in ker:
        b[bucket(k.key)] = b.get(bucket(k.key), 0) + k.self_cuda_time_total
    print(f"── kernel 分類（ms/步，含 1 台相機的 record；全部 kernel 合計 {tot / 1e3 / S:.2f} ms/步）")
    for nm, v in sorted(b.items(), key=lambda x: -x[1]):
        print(f"   {nm:<30} {v / 1e3 / S:8.2f}  {100 * v / tot:5.1f}%")
    print("── 前 20 個 kernel")
    for k in sorted(ker, key=lambda k: -k.self_cuda_time_total)[:20]:
        print(f"   {k.self_cuda_time_total / 1e3 / S:8.2f} ms/步  x{k.count / S:5.1f}  {k.key[:110]}")


def cmd_adamcheck(a):
    """fused Adam vs 預設（foreach）：同一組梯度、同一起點，走 1 步與 50 步後比參數差，並計時。
    差異以「該參數這段期間的更新量」為單位（相對差 ~1e-6 = 浮點捨入等級，與 atomic 順序不固定同級）。"""
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    ck = torch.load(a.ckpt, map_location="cpu")
    lrs = {g["name"]: g["lr"] for o in ck.get("optimizer_states", []) for g in o["param_groups"] if "name" in g}
    dmh = ck["datamodule_hyper_parameters"]
    cam = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)),
                                    global_rank=0).get_outputs().train_set.cameras[0].to_device(dev)
    del ck
    params = dict(model.gaussians.items())
    for p in params.values():
        p.requires_grad_(True); p.grad = None
    bg = torch.zeros(3, device=dev)
    out = renderer.training_forward(0, None, cam, model, bg)
    tgt = torch.rand(out["render"].shape, device=dev, generator=torch.Generator(device=dev).manual_seed(7))
    (0.8 * (out["render"] - tgt).abs().mean() + 0.2 * (1 - ssim(out["render"], tgt))).backward()
    grads = {k: p.grad.detach().clone() for k, p in params.items() if p.grad is not None}
    p0 = {k: params[k].detach().clone() for k in grads}
    print(f"N={model.n_gaussians:,}  lr（取自 ckpt）：" + "  ".join(f"{k}={lrs.get(k, 1e-3):.2e}" for k in grads))
    del out, model, renderer
    torch.cuda.empty_cache()

    def run(fused, steps):
        ps = {k: torch.nn.Parameter(v.clone()) for k, v in p0.items()}
        opt = torch.optim.Adam([{"params": [ps[k]], "lr": lrs.get(k, 1e-3)} for k in ps], eps=1e-15,
                               **({"fused": True} if fused else {}))
        ts = []
        for i in range(steps):
            for k in ps:
                ps[k].grad = grads[k]
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); opt.step(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return {k: v.detach() for k, v in ps.items()}, float(np.median(ts[2:] if len(ts) > 4 else ts))

    for steps in (1, 50):
        A, ta = run(False, steps)
        A2, _ = run(False, steps)
        B, tb = run(True, steps)
        print(f"── {steps} 步（同一組梯度重複套用）：optimizer.step 中位 foreach {ta:.2f} ms -> fused {tb:.2f} ms（{100 * (tb / ta - 1):+.1f}%）")
        for k in A:
            upd = (A[k] - p0[k]).abs().max().item()
            d = (A[k] - B[k]).abs().max().item()
            d2 = (A[k] - A2[k]).abs().max().item()
            print(f"   {k:<12} 更新量最大 {upd:.3e}  fused 與 foreach 最大差 {d:.3e}（= 更新量的 {d / max(upd, 1e-30):.1e}）"
                  f"  foreach 重跑差 {d2:.1e}")
        del A, A2, B


def _cmp(x, y):
    x, y = x.float(), y.float()
    d = (x - y).abs()
    scale = float(torch.maximum(x.abs().max(), y.abs().max()))
    return float(d.max()), float(d.max()) / max(scale, 1e-30), int((x != y).sum()), x.numel()


def cmd_compare(a):
    A, B = torch.load(a.a), torch.load(a.b)
    print(f"══ {a.label or (a.a + ' vs ' + a.b)}  （lean {A['lean']} vs {B['lean']}，N {A['N']:,} / {B['N']:,}）")
    ok = True
    ren = [_cmp(x, y) for x, y in zip(A["renders"], B["renders"])]
    nd = sum(r[2] for r in ren)
    print(f"  渲染 {len(ren)} 張：不同像素值 {nd}（最大差 {max(r[0] for r in ren):.3e}）"
          + ("  ✅ 逐位元相同" if nd == 0 else "  ⛔ 不相同"))
    ok &= nd == 0
    if A.get("dists") and B.get("dists"):
        dd = [_cmp(x, y) for x, y in zip(A["dists"], B["dists"])]
        print(f"  distortion 圖 {len(dd)} 張：不同像素 {sum(r[2] for r in dd)}（最大差 {max(r[0] for r in dd):.3e}、相對 {max(r[1] for r in dd):.2e}）")
    rd = sum(int((x != y).sum()) for x, y in zip(A["radii"], B["radii"]))
    print(f"  radii：不同 {rd}" + ("  ✅" if rd == 0 else "  ⛔"))
    ok &= rd == 0
    cv = sum(int((x != y).sum()) for x, y in zip(A["cover"], B["cover"]))
    print(f"  record 覆蓋數（整數 atomic，應逐位元相同）：不同 {cv}" + ("  ✅" if cv == 0 else "  ⛔"))
    ok &= cv == 0
    tr = [_cmp(x, y) for x, y in zip(A["trans"], B["trans"])]
    print(f"  record T*alpha 平均（浮點 atomic）：最大相對差 {max(r[1] for r in tr):.2e}、不同 {sum(r[2] for r in tr)} 顆")
    # trim 判準：v = 各視角每像素平均貢獻取最大 => 底部 10% 遮罩的重疊（record_reduce 只差加總順序，邊界上可能換掉幾顆）
    va = torch.stack([t.float() for t in A["trans"]]).max(0).values
    vb = torch.stack([t.float() for t in B["trans"]]).max(0).values
    ma, mb = va <= torch.quantile(va, 0.1), vb <= torch.quantile(vb, 0.1)
    print(f"  trim 判準（{len(A['trans'])} 視角的 max v）底部 10% 遮罩重疊 {100 * float((ma & mb).sum()) / max(float(ma.sum()), 1):.4f}%"
          f"（{int((ma ^ mb).sum())} 顆不同）")
    if A.get("tiles") and B.get("tiles"):
        ta = sum(int(t.long().sum()) for t in A["tiles"]); tb = sum(int(t.long().sum()) for t in B["tiles"])
        print(f"  binning 配對（{len(A['tiles'])} 視角合計）{ta:,} -> {tb:,}（{100 * (tb / max(ta, 1) - 1):+.1f}%）")
    print("  梯度（6 台相機加總；浮點 atomic => 與噪音底比）：")
    for k in A["grads"]:
        m, rel, n, tot = _cmp(A["grads"][k], B["grads"][k])
        print(f"    {k:<14} 最大絕對差 {m:.3e}  相對（÷最大幅度）{rel:.2e}  不同 {n:,}/{tot:,}")
    # ★ 2026-10-09：只看「量級夠大」的元素（|A| > 1% 最大值）各自的相對誤差。漏加／重複加貢獻的 bug 會出現在大元素上；
    #   加總順序造成的捨入差集中在互相抵消、接近 0 的小元素。印 max 與 99.9 分位，供判定用（行首 "SIG"）。
    for k in A["grads"]:
        a_, b_ = A["grads"][k].double().flatten(), B["grads"][k].double().flatten()
        mk = a_.abs() > 0.01 * a_.abs().max()
        if int(mk.sum()) > 0:
            r = ((a_[mk] - b_[mk]).abs() / a_[mk].abs())
            q = float(torch.quantile(r[torch.randperm(r.numel())[:2_000_000]], 0.999))
            print(f"    SIG {k:<14} 大元素 {int(mk.sum()):,} 個：逐元素相對誤差 max {float(r.max()):.2e}  p99.9 {q:.2e}")
    m, rel, n, tot = _cmp(A["vgrad"], B["vgrad"])
    print(f"    {'viewspace(+absgrad)':<14} 最大絕對差 {m:.3e}  相對 {rel:.2e}  不同 {n:,}/{tot:,}")
    if A["vgrad"].shape[-1] >= 3:   # ★ 2026-10-09：xy（真的梯度）與 z（absgrad 累加器）分開報
        m, rel, n, tot = _cmp(A["vgrad"][:, :2], B["vgrad"][:, :2])
        print(f"    {'viewspace xy':<14} 最大絕對差 {m:.3e}  相對 {rel:.2e}  不同 {n:,}/{tot:,}")
        za, zb = A["vgrad"][:, 2], B["vgrad"][:, 2]
        print(f"    {'absgrad z':<14} A 非零 {int((za != 0).sum()):,}  B 非零 {int((zb != 0).sum()):,}  （關掉的一邊應該是 0）")
    print("  計時（ms）：" + "  ".join(f"{k} {A['time'][k]:.2f} -> {B['time'][k]:.2f}（{100 * (B['time'][k] / A['time'][k] - 1):+.1f}%）"
                                    for k in A["time"]))
    print(f"  forward+backward 峰值增量 MiB：{A['peak_fb_MiB']:.0f} -> {B['peak_fb_MiB']:.0f}")
    print("  " + ("✅ 渲染／radii／覆蓋數逐位元相同（梯度要對照噪音底那一組）" if ok else "⛔ 有不該不同的東西不同"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump"); d.add_argument("--ckpt", required=True); d.add_argument("--out", required=True)
    d.add_argument("--lean", type=int, default=0); d.add_argument("--ncam", type=int, default=6)
    d.add_argument("--repeat", type=int, default=10)
    d.add_argument("--record-reduce", dest="record_reduce", type=int, default=0)
    d.add_argument("--tile-cull", dest="tile_cull", type=int, default=0)
    d.add_argument("--bwd-sat", dest="bwd_sat", type=int, default=0); d.add_argument("--bwd-red", dest="bwd_red", type=int, default=0)
    d.add_argument("--pair-bit", dest="pair_bit", type=int, default=0)
    d.add_argument("--dist", type=float, default=0.0, help="loss 加 dist*mean(rend_dist)；lean=1 時走 dist 版 lean（2026-10-19）")
    d.add_argument("--absgrad", type=int, default=-1, help="1/0＝absgrad_gate 開且明確要／不要 absgrad 累加；-1＝不碰")
    c = sp.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.add_argument("--label", default="")
    pr = sp.add_parser("profile"); pr.add_argument("--ckpt", required=True); pr.add_argument("--lean", type=int, default=0)
    pr.add_argument("--steps", type=int, default=12); pr.add_argument("--fused", type=int, default=0)
    vi = sp.add_parser("visible"); vi.add_argument("--ckpt", required=True); vi.add_argument("--lean", type=int, default=1)
    vi.add_argument("--ncam", type=int, default=0, help="0＝全部訓練視角")
    for q in (pr, vi):
        q.add_argument("--fast", type=int, default=-1, help="1＝backward 飽和跳過＋先加總、record 先加總；0＝關；-1＝沿用 ckpt")
        q.add_argument("--absgrad", type=int, default=-1, help="1＝增生期（backward 累加 |g|）；0＝收割期；-1＝沿用")
    pr.add_argument("--running", type=int, default=-1, help="1＝累加式 trim（訓練 forward 累加 V/C）；0＝關；-1＝沿用 ckpt")
    ac = sp.add_parser("adamcheck"); ac.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    {"dump": cmd_dump, "compare": cmd_compare, "profile": cmd_profile, "adamcheck": cmd_adamcheck, "visible": cmd_visible}[a.cmd](a)


if __name__ == "__main__":
    main()
