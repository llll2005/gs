"""評分工具落差的二分（2026-10-09）：用**官方程式碼＋官方環境**直接渲染指定的官方 test 視角。

背景：同一個官方 release 合併模型，官方 `main.py test` 量 25.79、我方 `tools/eval_official_test.py` 量 27.26。
官方存檔的 741 張渲染裡有 8~13% 明顯變暗或全黑（例：0184 全黑、0586 只剩約 1/3 亮度），排除後官方逐張平均 27.18。
我方工具（我方程式碼＋gspl 環境）渲染同一模型時沒有這種幀（最差 21.17）。

本腳本在官方 repo 裡跑（`--repo` 指向 cityGS_origin，用 gspl_official 環境），相機建法與我方工具逐行相同
（官方與我方的 `Cameras` 類別欄位一致），只換掉「程式碼＋光柵器」：
  變暗重現   => 問題在官方 renderer／光柵器（環境層），與 test 迴圈無關
  不變暗     => 問題在官方 test 迴圈（dataloader 相機、config 換 renderer 等）

用法（lab）：cd ../cityGS_origin && conda run -n gspl_official --no-capture-output python <本檔> --repo . --ckpt <合併 ckpt>
              --test_dir <block_all_test_official2> --names 0184.png 0586.png ... --out <tsv>
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--image_subdir", default="images_1.2")
    ap.add_argument("--down_sample", type=float, default=1.2)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.repo))
    import torch
    from PIL import Image
    from internal.cameras.cameras import Cameras
    from internal.utils.colmap import read_cameras_binary, read_images_binary
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    import internal
    print("[程式碼]", os.path.dirname(internal.__file__))
    try:
        import diff_surfel_rasterization as dsr
        print("[光柵器 diff_surfel_rasterization]", dsr.__file__)
    except Exception as e:
        print("[光柵器 diff_surfel_rasterization] 匯入失敗", e)
    try:
        import diff_trim_surfel_rasterization as dtsr
        print("[光柵器 diff_trim_surfel_rasterization]", dtsr.__file__)
    except Exception as e:
        print("[光柵器 diff_trim_surfel_rasterization] 匯入失敗", e)

    # 相機：與 tools/eval_official_test.py 的 load_test_cameras 逐行相同
    sp = os.path.join(a.test_dir, "sparse", "0")
    imgs = read_images_binary(os.path.join(sp, "images.bin"))
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    by_name = {imgs[k].name: imgs[k] for k in imgs}
    dev = "cuda"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, dev, eval_mode=True)
    print(f"[模型] N={model.get_xyz.shape[0]:,}  renderer={type(renderer).__name__}")
    bg = torch.zeros(3, device=dev)
    rows = []
    for nm in a.names:
        e = by_name[nm]
        i = cams[e.camera_id]
        p = i.params
        f_x, f_y, c_x, c_y = (p[0], p[0], p[1], p[2]) if i.model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
        W, H = float(i.width), float(i.height)
        dw, dh = round(W / a.down_sample), round(H / a.down_sample)
        sx, sy = dw / W, dh / H
        t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
        cam = Cameras(R=t(e.qvec2rotmat()), T=t(np.array(e.tvec)), fx=t(f_x * sx), fy=t(f_y * sy),
                      cx=t(c_x * sx), cy=t(c_y * sy), width=torch.tensor([dw], dtype=torch.int16),
                      height=torch.tensor([dh], dtype=torch.int16), appearance_id=torch.zeros(1, dtype=torch.int),
                      normalized_appearance_id=torch.zeros(1), distortion_params=None,
                      camera_type=torch.zeros(1, dtype=torch.int8))[0]
        with torch.no_grad():
            out = renderer(cam.to_device(dev), model, bg_color=bg)
            img = out["render"].clamp(0, 1)
            vis = out.get("visibility_filter")
            nvis = int(vis.sum()) if torch.is_tensor(vis) else -1
            alpha = out.get("rend_alpha")
            amean = float(alpha.mean()) if torch.is_tensor(alpha) else -1.0
        pil = Image.open(os.path.join(a.test_dir, a.image_subdir, nm)).convert("RGB")
        gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
        psnr = float(-10 * torch.log10(((img - gt) ** 2).mean().clamp_min(1e-12)))
        br = float(img.mean() / gt.mean().clamp_min(1e-6))
        rows.append((nm, psnr, br, nvis, amean))
        print(f"  {nm}  PSNR {psnr:6.2f}  亮度比 {br:5.3f}  可見顆數 {nvis:,}  alpha 平均 {amean:.3f}")
    with open(a.out, "w") as f:
        f.write("name\tpsnr\tbright_ratio\tn_visible\talpha_mean\n")
        for r in rows:
            f.write("%s\t%.4f\t%.4f\t%d\t%.4f\n" % r)
    print("[輸出]", a.out)


if __name__ == "__main__":
    main()
