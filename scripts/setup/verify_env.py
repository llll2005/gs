#!/usr/bin/env python
"""環境驗證：裝好了不等於裝對了。逐項檢查「它真的在做宣稱的事嗎」。

為什麼需要：本專案最貴的教訓是**七個「看起來正常但沒作用」的機制，全都不報錯**。
環境層同一個形狀：
```
requirements.txt 寫 lightning 2.3 / bitsandbytes 0.45，實際能跑的是 2.0.9 / 0.41.3
requirements/CityGS.txt 用 git+...@9eefc03 裝光柵器 -> **沒有我方的 ABSGRAD / EXACT_SUPPORT**
distutils 只看 .cu 的 mtime -> 改了 .h 不重編，「沒有可量到的差異」其實是沒編到
```
所以這支不只印版本，還做**功能驗證**：ABSGRAD 把 `|g|` 累加到 `dL_dmean2D.z`，
上游根本不寫那個欄位 => backward 後 `means2D.grad[:, 2]` 非零，就證明它真的編進去了。

用法: python scripts/setup/verify_env.py [--no-gpu]
回傳碼 0=全過；1=有 FAIL。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RAST_SRC = os.path.join(ROOT, "submodules", "diff-surfel-rasterization-trim-pp")

OK, BAD = [], []


def chk(name, cond, got, want=""):
    (OK if cond else BAD).append(name)
    print(f"  {'✅' if cond else '❌'} {name:<42} {got}" + (f"   （應為 {want}）" if not cond and want else ""))
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="跳過需要顯卡的功能驗證")
    a = ap.parse_args()

    print("\n=== 1. 版本（以「這個環境實際跑出過結果」為準，不是 requirements.txt）===")
    print(f"  python  {sys.version.split()[0]}")
    chk("python 3.9.x", sys.version_info[:2] == (3, 9), sys.version.split()[0], "3.9")
    try:
        import torch
        chk("torch 2.0.1+cu118", torch.__version__.startswith("2.0.1"), torch.__version__, "2.0.1+cu118")
        chk("torch 認得 CUDA 11.8", torch.version.cuda == "11.8", str(torch.version.cuda), "11.8")
    except Exception as e:
        chk("torch 可匯入", False, repr(e)); return 1
    for mod, want in (("lightning", "2.0."), ("pytorch_lightning", "2.0."), ("numpy", "1.26.")):
        try:
            m = __import__(mod)
            chk(f"{mod} {want}x", m.__version__.startswith(want), m.__version__, want + "x")
        except Exception as e:
            chk(f"{mod} 可匯入", False, repr(e))
    # ⚠ 0.41.3 **沒有** `__version__` 屬性（我第一版就是這樣誤判成「未安裝」）=> 走 metadata。
    # 版本要壓在 0.41.x：0.45 的 state key 是 state1/state2/absmax*（blockwise），
    # 而 density controller 三處寫死 exp_avg/exp_avg_sq => 0.45 會**靜默**對不上。
    try:
        import importlib.metadata as _md
        import bitsandbytes  # noqa: F401  （確認真的能載入，不只是有 metadata）
        v = _md.version("bitsandbytes")
        chk("bitsandbytes 0.41.x", v.startswith("0.41."), v, "0.41.3.post2")
    except ImportError:
        print("  ⚪ bitsandbytes 未安裝（只有 8-bit 優化器實驗需要）")
    except Exception as e:
        chk("bitsandbytes 可載入", False, repr(e))

    print("\n=== 2. Trim 光柵器：原始碼旗標 ===")
    aux = os.path.join(RAST_SRC, "cuda_rasterizer", "auxiliary.h")
    if not os.path.isfile(aux):
        chk("找得到 cuda_rasterizer/auxiliary.h", False, aux,
            "clone 後要有原始碼；若是空目錄代表 submodule 沒帶過來")
    else:
        src = open(aux, encoding="utf-8", errors="ignore").read()
        for flag in ("ABSGRAD", "EXACT_SUPPORT"):
            on = f"#define {flag} 1" in src
            chk(f"原始碼 {flag} = 1", on, "1" if on else "0 或未定義", "1")

    print("\n=== 3. Trim 光柵器：裝的是哪一份 ===")
    try:
        import diff_trim_surfel_rasterization as dtr
        so = [os.path.join(os.path.dirname(dtr.__file__), f)
              for f in os.listdir(os.path.dirname(dtr.__file__)) if f.endswith(".so")]
        so += [os.path.join(os.path.dirname(os.path.dirname(dtr.__file__)), f)
               for f in os.listdir(os.path.dirname(os.path.dirname(dtr.__file__)))
               if f.startswith("diff_trim_surfel_rasterization") and f.endswith(".so")]
        chk("diff_trim_surfel_rasterization 可匯入", True, dtr.__file__)
        if so and os.path.isfile(aux):
            # distutils 只看 .cu 的 mtime => 改了 .h 卻不重編會靜默沿用舊 kernel
            newer = os.path.getmtime(so[0]) > os.path.getmtime(aux)
            chk(".so 比 auxiliary.h 新（有重編過）", newer,
                f"so={os.path.getmtime(so[0]):.0f} h={os.path.getmtime(aux):.0f}",
                "重編：見 scripts/setup/bootstrap.sh 的 build 步驟")
    except Exception as e:
        chk("diff_trim_surfel_rasterization 可匯入", False, repr(e))

    print("\n=== 4. 功能驗證：ABSGRAD 真的在寫 dL_dmean2D.z 嗎 ===")
    if a.no_gpu or not torch.cuda.is_available():
        print("  ⚪ 跳過（--no-gpu 或沒有可用的 CUDA 裝置）")
    else:
        try:
            ok, msg = absgrad_live()
            chk("backward 後 means2D.grad[:,2] 非零", ok, msg,
                "非零。若為零＝光柵器不是我方版本（ABSGRAD 沒編進去）")
        except Exception as e:
            chk("ABSGRAD 功能驗證可執行", False, f"{type(e).__name__}: {e}")

    print("\n=== 5. 資料（不在 git 裡，要自己放）===")
    d = os.path.join(ROOT, "data", "matrix_city", "aerial", "train", "block_all")
    chk("data/matrix_city/.../block_all 存在", os.path.isdir(d), d if os.path.isdir(d) else "缺",
        "見 bootstrap.sh 最後一段")
    for sub, why in (("sparse/0", "COLMAP 稀疏重建，SfM-init 與姿態都吃它"),
                     ("estimated_depths", "Depth Anything V2 的輸出（最貴的一步，5621 張）"),
                     ("depth_init", "depth-init PLY（現行主線的起點）")):
        p = os.path.join(d, sub)
        print(f"  {'✅' if os.path.isdir(p) else '⚪'} {sub:<42} {'有' if os.path.isdir(p) else '缺 — ' + why}")

    print(f"\n=== 結果：{len(OK)} 過 / {len(BAD)} 失敗 ===")
    for b in BAD:
        print(f"  ❌ {b}")
    return 1 if BAD else 0


def absgrad_live():
    """渲染幾顆合成高斯並 backward，看 dL_dmean2D.z 有沒有被寫。

    ABSGRAD 的實作是 `atomicAdd(&dL_dmean2D[id].z, |dL_ds.x| + |dL_ds.y|)`
    （`cuda_rasterizer/backward.cu`）。**上游完全不碰 .z** ⇒ 這個欄位非零是
    「裝的是我方版本」的充分證據，比對 .so 的 mtime 強。
    """
    import torch
    from diff_trim_surfel_rasterization import (GaussianRasterizationSettings,
                                                GaussianRasterizer)
    from internal.cameras.cameras import Cameras

    dev = torch.device("cuda")
    H = W = 64
    cams = Cameras(
        R=torch.eye(3)[None], T=torch.zeros(1, 3),
        fx=torch.tensor([64.0]), fy=torch.tensor([64.0]),
        cx=torch.tensor([W / 2]), cy=torch.tensor([H / 2]),
        width=torch.tensor([W]), height=torch.tensor([H]),
        appearance_id=torch.zeros(1, dtype=torch.long),
        normalized_appearance_id=torch.zeros(1),
        distortion_params=None, camera_type=torch.zeros(1, dtype=torch.int),
    )
    cam = cams[0].to_device(dev)

    n = 16
    g = torch.Generator(device="cpu").manual_seed(0)
    means = (torch.rand(n, 3, generator=g) - 0.5).to(dev)
    means[:, 2] = 2.0                                   # 全部放在相機前方
    means.requires_grad_(True)
    means2D = torch.zeros_like(means, requires_grad=True)
    means2D.retain_grad()
    scales = torch.full((n, 2), 0.10, device=dev, requires_grad=True)
    rots = torch.zeros(n, 4, device=dev); rots[:, 0] = 1.0; rots.requires_grad_(True)
    op = torch.full((n, 1), 0.9, device=dev, requires_grad=True)
    rgb = torch.rand(n, 3, device=dev).requires_grad_(True)

    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=float(torch.tan(cam.fov_x / 2)), tanfovy=float(torch.tan(cam.fov_y / 2)),
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=cam.world_to_camera, projmatrix=cam.full_projection,
        sh_degree=0, campos=cam.camera_center, prefiltered=False,
        record_transmittance=False, debug=False,
    )
    out = GaussianRasterizer(raster_settings=settings)(
        means3D=means, means2D=means2D, shs=None, colors_precomp=rgb,
        opacities=op, scales=scales, rotations=rots, cov3D_precomp=None)
    color = out[0]
    if float(color.abs().sum()) == 0.0:
        return False, "渲染結果全黑 —— 合成場景沒進到畫面，這個檢查無效（不是光柵器的問題）"
    color.sum().backward()
    z = means2D.grad[:, 2]
    nz = int((z != 0).sum())
    return nz > 0, f"{nz}/{n} 顆的 .z 非零，最大 {float(z.abs().max()):.4e}"


if __name__ == "__main__":
    sys.exit(main())
