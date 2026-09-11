#!/usr/bin/env python
"""K-strip 到底有沒有損？—— 逐項量，不看 docstring 的宣稱。（純 CPU）

## 起因（2026-09-11）

舊版手冊寫「K 是 loss 不變量，故可動態調而不影響品質」。而實作
`internal/gaussian_splatting.py:_strip_forward_backward` 的 docstring 自承
*"Image-space SSIM is computed per strip (**boundary-window approximation**)"*。
使用者問：那我們現在的 K-strip 是有損的嗎？

## 三個分開的問題

```
① 現行主線有沒有走條帶   `if n_strips > 1:` 才進；dynamic_strips 預設 False、train_strips 預設 1
                         => **K=1 走完全沒改過的原路徑**（本工具第 0 節印出實際設定）
② K>1 時 SSIM 損多少     本工具直接量：整張 SSIM  vs  sum_k (h_k/H)*SSIM(第 k 條)
③ K>1 時還有什麼被改     absgrad 的 |g| 讀 `outputs["viewspace_points"].grad[:,2]`，
                         而條帶路徑回傳 `dict(last_outputs)` => **只有最後一條帶**
```
③ 比 ② 嚴重得多：`absgrad_densify` 是現行最佳配方的核心機制，而 K=6 時它只看到 1/6 的畫面。
程式裡的 assert 只擋 `READS_VIEWSPACE_GRAD = True` 的控制器，MCMC 是 False ⇒ **擋不到**。
（那段註解寫「verified 2026-08-06」，而 `absgrad_densify` 是 2026-08-25 才加的 ⇒ 註解過期了。）

用法: python tools/strip_loss_audit.py --run speed3_b12 --blk 12
"""
import argparse, glob, os, re, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.ssim import ssim  # noqa: E402


def strip_bounds(H, K):
    """與 internal/utils/strip_cameras.py 相同的切法（等分，餘數給前面幾條）。"""
    from internal.utils.strip_cameras import strip_bounds as sb
    return list(sb(H, K))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="speed3_b12")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--max-img", type=int, default=12)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    a = ap.parse_args()

    print("=== 0. 現行設定：條帶到底有沒有在跑 ===")
    import internal.gaussian_splatting as gs
    import inspect
    sig = inspect.signature(gs.GaussianSplatting.__init__)
    for k in ("train_strips", "dynamic_strips"):
        print(f"    {k:<18} 預設 {sig.parameters[k].default}")
    cfgs = sorted(glob.glob(f"outputs/{a.run}/blocks/block_{a.blk}/lightning_logs/*/config.yaml"))
    if cfgs:
        txt = open(cfgs[-1], encoding="utf-8").read()
        for k in ("dynamic_strips", "train_strips"):
            m = re.search(rf"^\s*{k}:\s*(\S+)", txt, re.M)
            print(f"    {a.run} 的 {k:<14} = {m.group(1) if m else '（未出現）'}")
    print("    ⇒ `if n_strips > 1:` 才走條帶路徑；K=1 是原路徑，逐位元相同。")

    print("\n=== 2. K>1 時 SSIM 的邊界誤差（直接量，用真實的 GT|渲染 並排圖）===")
    ds = sorted(glob.glob(f"outputs/{a.run}/blocks/block_{a.blk}/test/*/"),
                key=lambda d: int(re.search(r"step=(\d+)", d).group(1)) if re.search(r"step=(\d+)", d) else -1)
    if not ds:
        print("    找不到 test 圖 —— 先跑 `main.py test --save_val`"); return 1
    from PIL import Image
    fs = sorted(f for f in os.listdir(ds[-1]) if f.endswith(".png"))[:a.max_img]
    print(f"    來源 {ds[-1]}  {len(fs)} 張")
    rows = {k: [] for k in a.ks}
    full_all = []
    for f in fs:
        im = np.asarray(Image.open(os.path.join(ds[-1], f)).convert("RGB"), np.float32) / 255.
        H, W2, _ = im.shape
        gt = torch.from_numpy(im[:, :W2 // 2].transpose(2, 0, 1))[None]
        rd = torch.from_numpy(im[:, W2 // 2:].transpose(2, 0, 1))[None]
        full = float(ssim(rd, gt))
        full_all.append(full)
        for K in a.ks:
            acc = 0.0
            for v0, v1 in strip_bounds(H, K):
                w = (v1 - v0) / H
                acc += w * float(ssim(rd[:, :, v0:v1], gt[:, :, v0:v1]))
            rows[K].append(acc - full)
    print(f"\n    整張 SSIM 平均 {np.mean(full_all):.4f}")
    print(f"    {'K':>3} {'Δ(條帶和 - 整張) 平均':>22} {'|Δ| 中位':>12} {'|Δ| 最大':>12}   對照噪音底")
    for K in a.ks:
        d = np.array(rows[K])
        print(f"    {K:>3} {np.mean(d):>22.6f} {np.median(np.abs(d)):>12.6f} "
              f"{np.max(np.abs(d)):>12.6f}   SSIM 1sd = 0.0005")
    print("""
    判讀：|Δ| 遠小於 SSIM 的噪音底 0.0005 => 邊界誤差在量測解析度以下，這一項可忽略
          |Δ| 與噪音底同級或更大         => 這一項本身就足以讓跨 K 的比較失效
    ⚠ 這裡量的是 **loss 值**的差；真正影響訓練的是**梯度**的差，量值小不保證梯度小。
      但量值若已經比噪音底大，就不必再談梯度了。""")

    print("\n=== 3. K>1 時 absgrad 只看得到最後一條帶（讀碼，不是量測）===")
    print("""    internal/gaussian_splatting.py:_strip_forward_backward
        outputs = dict(last_outputs)            <- 最後一條帶
        只有 `radii`（取 max）與 `visibility_filter`（取 or）被換掉
        assert not READS_VIEWSPACE_GRAD         <- MCMC 是 False，擋不到
    internal/density_controllers/mcmc_2dgs_density_controller.py:_absgrad_report
        vp = outputs.get("viewspace_points"); a = vp.grad[:, 2].abs()
        self._absgrad_accum = buf + a           <- 只累到最後一條帶的 |g|

    ⇒ K=6 時，`absgrad_densify` 的取樣權重 `probs = o * (1 + w*|g|/mean|g|)`
      只由畫面**最下面 1/6** 的殘差決定；沒出現在該帶的粒子 |g|=0 ⇒ 退化成 `probs = o`
      （等同 absgrad 沒開）。而 absgrad 正是現行最佳配方的核心機制。
    ⚠ 那段註解寫「MCMC never reads them ... verified 2026-08-06」，
      而 `absgrad_densify` 是 2026-08-25 才加的 ⇒ **註解過期了，assert 的守衛沒跟上**。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
