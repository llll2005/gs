#!/usr/bin/env python
"""部署用：把 ckpt 的顆粒參數存成半精度（使用者 10-24：「改儲存形式減少存儲負擔」；紀錄/加速論文對照.md C8）。
- `gaussian_model.gaussians.*` 中，means 以外（shs_dc／shs_rest／opacities／scales／rotations）轉 float16；means 維持 float32
  （座標到幾十個場景單位，float16 在 50 附近的間距 0.03 太粗）。
- 丟掉 optimizer_states（Adam 動量，佔 ckpt 的 2/3；部署用不到；要接著訓練請用原檔）。
- 其餘（hyper_parameters、renderer、density controller 狀態等）原樣保留。
載入不用改：`GaussianModelLoader` 先建 float32 參數再 `load_state_dict`（copy_ 會轉型）=> 渲染時就是 float32。
本機實測（b6 60k SH3，16 張訓練視角）：means 以外全部 float16 => PSNR −0.0008 dB、每顆 232 -> 122 B。
用法：python tools/half_ckpt.py <in.ckpt> [<out.ckpt>]   （預設輸出 <in>_fp16.ckpt）
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # ckpt 裡 pickle 了 repo 的類別

KEEP32 = ("gaussian_model.gaussians.means",)


def main(src, dst=None):
    dst = dst or os.path.splitext(src)[0] + "_fp16.ckpt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    n = sd["gaussian_model.gaussians.means"].shape[0]
    b32 = b16 = 0
    for k in list(sd.keys()):
        if not k.startswith("gaussian_model.gaussians.") or not torch.is_tensor(sd[k]):
            continue
        t = sd[k]
        b32 += t.numel() * 4
        if k in KEEP32 or not t.is_floating_point():
            b16 += t.numel() * t.element_size()
            continue
        if not torch.isfinite(t).all():   # 例：10-24 修正前的 relocation NaN scale（遮罩臂全場景 b1 15,234 顆）
            nf = int((~torch.isfinite(t.reshape(t.shape[0], -1))).any(1).sum())
            raise SystemExit(f"⛔ {k} 原檔（float32）就有 {nf:,} 顆非有限值，不是轉型溢位 —— 不輸出（tools/ckpt_nonfinite.py 查是哪幾塊）")
        h = t.to(torch.float16)
        if not torch.isfinite(h).all():
            raise SystemExit(f"⛔ {k} 轉 float16 後有非有限值（超出 ±65504？）—— 不輸出")
        sd[k] = h
        b16 += h.numel() * 2
    dropped = "optimizer_states" in ck
    ck.pop("optimizer_states", None)
    torch.save(ck, dst)
    print(f"N {n:,}：顆粒參數 {b32 / 2**20:,.1f} -> {b16 / 2**20:,.1f} MiB（每顆 {b32 / n:.0f} -> {b16 / n:.0f} B）"
          f"{'；已丟掉 optimizer_states' if dropped else ''}")
    print(f"檔案 {os.path.getsize(src) / 2**30:.2f} -> {os.path.getsize(dst) / 2**30:.2f} GiB  -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
