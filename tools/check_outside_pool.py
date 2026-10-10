"""塊外池（internal/utils/outside_pool.py）的行為驗證 —— 上 lab 前在本機跑（2026-10-10 使用者授權）。

用一個現成的單塊 ckpt（不訓練），檢查：
  1 等價：可訓練集合（盒內）＋池（盒外、SH0）渲染 == 同一批顆粒放在單一集合、盒外那批 shs_rest 清零、同樣順序（逐像素）
  2 逐顆輸出：radii／tiles／record 都只有 N_in 筆，且與參考模型的前 N_in 筆相同
  3 梯度：screenspace 梯度只有 N_in 列；可訓練集合的梯度與參考模型前 N_in 列一致；池可訓練時池參數有梯度
  4 池的優化器：step 後池參數改變；凍結後渲染不變、不再有參數、52 B／顆
  5 存檔：state() -> from_state() 渲染不變
用法：python tools/check_outside_pool.py --ckpt outputs/lab/speed3/blocks/block_6/checkpoints/epoch=110-step=60000.ckpt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--ncam", type=int, default=4)
    ap.add_argument("--lean", action="store_true", help="走訓練用的 lean 路徑（lab 的光柵器才有；lean＋record_reduce＋backward 飽和跳過／先加總）")
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.outside_pool import OutsidePool, block_outside_mask, KEYS
    dev = torch.device("cuda")
    model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; del ck
    train = pc.instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0).get_outputs().train_set
    cams = [train.cameras[i].to_device(dev) for i in torch.linspace(0, len(train) - 1, a.ncam).round().long().tolist()]
    bg = torch.zeros(3, device=dev)
    LK = {}
    if a.lean:
        renderer.lean_train = True; renderer.record_reduce = True; renderer.bwd_sat_skip = True; renderer.bwd_reduce = True
        LK = {"_lean": True}
        print("lean 路徑：lean_train＋record_reduce＋bwd_sat_skip＋bwd_reduce（render 與 radii／tiles 比較；lean 不輸出幾何通道）")
    orig = {k: v.detach().clone() for k, v in model.properties.items()}
    out = block_outside_mask(orig["means"], pc, a.margin)
    n_in, n_out = int((~out).sum()), int(out.sum())
    print(f"N={orig['means'].shape[0]:,}  盒＋{a.margin} 內 {n_in:,}／外 {n_out:,}")
    assert n_in > 0 and n_out > 0
    ok = True

    def P(d):
        return {k: torch.nn.Parameter(v.clone()) for k, v in d.items()}

    ref = {k: torch.cat([v[~out], v[out]], 0) for k, v in orig.items()}
    ref["shs_rest"][n_in:] = 0
    inn = {k: v[~out] for k, v in orig.items()}
    raw = {k: orig[k][out] for k in KEYS}
    names = list(orig); del orig      # 6 GB 卡：同時留太多份會 OOM（2026-10-10 第一次跑）

    def render_set(props, pool, **kw):
        model.properties = P(props)
        model._outside_pool = pool
        return renderer(cam, model, bg_color=bg, **{**LK, **kw})

    # 1＋2 等價與逐顆輸出
    pool = OutsidePool(raw, trainable=False); pool.freeze()
    for ci, cam in enumerate(cams):
        with torch.no_grad():
            r = render_set(ref, None); q = render_set(inn, pool)
            d = float((r["render"] - q["render"]).abs().max())
            same_r = torch.equal(r["radii"][:n_in], q["radii"])
            same_t = ("tiles" not in r) or torch.equal(r["tiles"][:n_in], q["tiles"])
            rt = render_set(ref, None, record_transmittance=True, record_coverage=True)
            qt = render_set(inn, pool, record_transmittance=True, record_coverage=True)
            dt = float((rt[0][:n_in] - qt[0]).abs().max()); dc = float((rt[1][:n_in].float() - qt[1].float()).abs().max())
            rt2 = render_set(ref, None, record_transmittance=True, record_coverage=True)   # 噪音底：record 是浮點 atomicAdd，順序不固定
            noise = float((rt[0] - rt2[0]).abs().max())
        good = d == 0 and same_r and same_t and q["radii"].shape[0] == n_in and qt[0].shape[0] == n_in and dt <= 10 * max(noise, 1e-7) and dc == 0
        ok &= good
        print(f"  相機 {ci}：渲染最大差 {d:.3e}  radii 同 {same_r}  tiles 同 {same_t}  radii 長度 {q['radii'].shape[0]:,}  "
              f"record 差 {dt:.2e}（同模型重跑噪音 {noise:.2e}）／覆蓋數差 {dc:.0f}  {'✅' if good else '⛔'}")

    # 3 梯度
    cam = cams[0]
    gt = torch.rand_like(r["render"])
    model.properties = P(ref); model._outside_pool = None
    o = renderer(cam, model, bg_color=bg, **LK); torch.abs(o["render"] - gt).mean().backward()
    g_ref = {k: model.gaussians[k].grad[:n_in].clone() for k in names}; vs_ref = o["viewspace_points"].grad[:n_in].clone()
    pool = OutsidePool(raw, trainable=True)
    model.properties = P(inn); model._outside_pool = pool
    o = renderer(cam, model, bg_color=bg, **LK); torch.abs(o["render"] - gt).mean().backward()
    vs = o["viewspace_points"].grad
    good = vs.shape[0] == n_in
    for k in names:
        gk = model.gaussians[k].grad
        rel = float((gk - g_ref[k]).abs().max() / g_ref[k].abs().max().clamp_min(1e-20))
        good &= rel < 1e-3
        print(f"  梯度 {k:<10} 相對最大差 {rel:.2e}")
    rel_vs = float((vs[:, :2] - vs_ref[:, :2]).abs().max() / vs_ref[:, :2].abs().max().clamp_min(1e-20))
    pg = {k: float(pool.raw[k].grad.abs().sum()) for k in KEYS}
    good &= rel_vs < 1e-3 and all(v > 0 for v in pg.values())
    print(f"  screenspace 梯度列數 {vs.shape[0]:,}（N_in {n_in:,}）相對差 {rel_vs:.2e}；池參數梯度和 {pg}  {'✅' if good else '⛔'}")
    ok &= good

    del ref, g_ref, vs_ref, o
    model.properties = P(inn); torch.cuda.empty_cache()
    # 4 池優化器與凍結
    m0 = pool.raw["means"].detach().clone()
    pool.step([])
    moved = float((pool.raw["means"] - m0).abs().max())
    with torch.no_grad():
        im_a = render_set(inn, pool)["render"]
        pool.freeze()
        im_b = render_set(inn, pool)["render"]
    good = moved > 0 and float((im_a - im_b).abs().max()) == 0 and pool.raw is None and pool.optimizer is None and pool.bytes_per_point() == 52
    print(f"  step 後池位置最大位移 {moved:.2e}；凍結前後渲染差 {float((im_a - im_b).abs().max()):.2e}；凍結後 {pool.bytes_per_point()} B／顆  {'✅' if good else '⛔'}")
    ok &= good
    # 可訓練池的 add（清掃）保留既有 Adam 動量、新的接 0
    p2 = OutsidePool(raw, trainable=True)
    model.properties = P(inn); model._outside_pool = p2
    o = renderer(cam, model, bg_color=bg, **LK); torch.abs(o["render"] - gt).mean().backward(); p2.step([])
    half = {k: v[:1000] for k, v in raw.items()}
    p2.add(half)
    st = p2.optimizer.state[p2.optimizer.param_groups[0]["params"][0]]
    good = p2.n == n_out + 1000 and st["exp_avg"].shape[0] == p2.n and float(st["exp_avg"][n_out:].abs().sum()) == 0 and float(st["exp_avg"][:n_out].abs().sum()) > 0
    print(f"  add 後池 {p2.n:,}、Adam 動量長度 {st['exp_avg'].shape[0]:,}、新的一段為 0  {'✅' if good else '⛔'}")
    ok &= good

    # 5 存檔
    with torch.no_grad():
        s = pool.state(); p3 = OutsidePool.from_state(s, device=dev)
        im_c = render_set(inn, p3)["render"]
    good = float((im_b - im_c).abs().max()) == 0
    print(f"  存檔後重建渲染差 {float((im_b - im_c).abs().max()):.2e}  {'✅' if good else '⛔'}")
    ok &= good
    print("全部通過 ✅" if ok else "⛔ 有項目沒過")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
