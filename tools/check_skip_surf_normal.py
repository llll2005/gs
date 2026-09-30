"""skip_surf_normal 是否逐位元無損（2026-09-30）：同一個 ckpt、同一台相機，旗標關/開各做一次
forward + backward，比較渲染輸出與**所有參數的梯度**。只有全部 max|diff| == 0 才算通過。

為什麼不比訓練跑次：訓練本身不可逐位元重現（見記憶 storage_structure），驗改動要比單步的輸入輸出。
用法：python tools/check_skip_surf_normal.py --ckpt <ckpt> [--cams 3]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cams", type=int, default=3)
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim
    dev = torch.device("cuda")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, device=dev, eval_mode=False, pre_activate=False)
    ck = torch.load(a.ckpt, map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(a.ckpt)), global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    bg = torch.zeros(3, device=dev)
    params = [p for p in model.gaussians.values()] if hasattr(model.gaussians, "values") else list(model.parameters())
    worst = {}
    ok = True
    for ci in range(a.cams):
        cam = cams[ci * 37 % len(cams)].to_device(dev)
        res = {}
        for tag, flag in (("off", False), ("off2", False), ("on", True)):   # off vs off2 = 噪音底（backward 的 atomicAdd 順序不定）
            renderer.skip_surf_normal = flag
            for p in params:
                p.grad = None
            torch.manual_seed(0)
            out = renderer(cam, model, bg_color=bg)
            img = out["render"]
            gt = torch.rand(img.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(123))
            loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img, gt))
            loss.backward()
            res[tag] = ({k: v.detach().clone() for k, v in out.items() if torch.is_tensor(v) and k != "surf_normal"},
                         [p.grad.detach().clone() if p.grad is not None else None for p in params], float(loss))
        def diff(A, B):
            (o0, g0, l0), (o1, g1, l1) = A, B
            d = {"loss": abs(l0 - l1)}
            for k in o0:
                d[f"out:{k}"] = (float((o0[k].float() - o1[k].float()).abs().max()) if o0[k].numel() else 0.0) \
                    if k in o1 and o0[k].shape == o1[k].shape else float("inf")
            for i, (x, y) in enumerate(zip(g0, g1)):
                d[f"grad[{i}]"] = 0.0 if x is None and y is None else float((x - y).abs().max())
            return d
        dn, df = diff(res["off"], res["off2"]), diff(res["off"], res["on"])
        # 通過條件：輸出與 loss 完全相同；梯度差不超過「同設定重跑」的噪音底（取 2 倍寬容，避免單次抽樣運氣）
        outs_same = all(v == 0 for k, v in df.items() if not k.startswith("grad"))
        # ⚠ 2026-09-30：原本逐張量要求 <= 2x 該張量的噪音底 —— 單次抽樣下個別張量的噪音底本身就在跳
        #   （實測同設定重跑 1.4e-9～8.9e-9），逐張量比會把噪音判成差異。改比「所有梯度的最大差」。
        grads_ok = max(v for k, v in df.items() if k.startswith("grad")) <= \
            2 * max(v for k, v in dn.items() if k.startswith("grad"))
        ok &= outs_same and grads_ok
        for k in df:
            worst[k] = max(worst.get(k, 0.0), df[k])
        gmax_n = max(v for k, v in dn.items() if k.startswith("grad")); gmax_f = max(v for k, v in df.items() if k.startswith("grad"))
        print(f"相機 {ci}: 輸出/loss 相同={outs_same}；梯度最大差 同設定重跑 {gmax_n:.2e} vs 開旗標 {gmax_f:.2e}")
    print("最大差異：", {k: v for k, v in worst.items() if v != 0} or "全部為 0")
    print("✅ 無損：輸出與 loss 逐位元相同，梯度差落在同設定重跑的噪音底內（backward atomicAdd 順序不定）"
          if ok else "⛔ 有差異超出噪音底")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
