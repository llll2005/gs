"""Score a trained block model on the OFFICIAL MatrixCity held-out test set.

⛔⛔ 2026-09-29 更正（先讀這段）：下面 docstring 說的「照 sparse 名稱 +/-1 偏移校準」**不成立**。
  舊的 `block_all_test/input/` 是我方自己從 10 個 tar 串接、依一份來源不明的清單編號的：
  741 張只有 657 張是官方 test 影格，且對 sparse 相機是**分段**錯開（+1/+9/+26…），
  不是一個常數偏移 ⇒ 過去用它量的 held-out（北極星 16.63、使用者全域版 25.62）全部待重驗。
  正解（不經模型）：sparse 相機 N 的姿態 = transforms_test.json 第 N 幀（殘差 1e-6，741 對 741）
  ＝ 官方 tools/transform_json2txt_mc_aerial.py 產生 input/{N:04d}.png 的方式。
  ⇒ 現在**只接受**照官方步驟重建的 test 目錄（`scripts/lab/task_citygs_origin.sh prep_test_official`
     產生 block_all_test_official2，含 .built_by_official_steps 標記，逐張 md5 驗過），並**照同名配對**；
     偏移校準已移除。另外補上紋理比（逐張平均與能量加權，定義同 citygsv2_metrics）。

Why this exists: every number in 紀錄/實驗總表.csv comes from `split_mode: reconstruction`, where
the validation images are a subset of the TRAINING images (every 8th image, also trained on). Those
numbers are internally comparable but they are not held-out, so they cannot be quoted against
published results and they systematically flatter us.

The official test set has been sitting fully prepared in the repo since April and was skipped in
July on the belief that it was "~150x Sim3 misaligned" with our frame. That belief was wrong. The
transform is exactly

    sparse/0 pose  =  transforms.json pose  x  0.01        (R = I, t = 0)

measured over all 5621 training cameras with a residual of 2e-7 (see tools/audit_pose_consistency.py).
`data/matrix_city/aerial/test/block_all_test/sparse/0` is ALREADY in our frame -- its camera centres
span x[-9.50, 7.20], nested inside the training envelope x[-10.00, 8.70] -- so no transform is
applied here at all. The alignment work was already done; only the belief was stale.

WHAT THIS MEASURES, precisely: a single-block model rendered at test views whose camera centre falls
inside the axis-aligned bounds of that block's TRAINING cameras. This is a lower bound on what the
full method achieves, because a block model only holds its own slice of the city while a test view
centred in the block still sees content past the block edges, and that content is simply missing.
The official protocol scores the MERGED 25-block model. Read these numbers as
  · valid for comparing our own runs against each other on a genuinely held-out set
  · NOT directly comparable to published per-scene numbers until the blocks are merged

Held-out is genuine: none of the 741 test poses appears in our training set (nearest training pose
is ~18.3 scene units away; 0.00% within 1e-3).

The training set is also the right one. It is 5621 frames = the union of MatrixCity's per-block
filtered `transforms.json` over all ten aerial blocks, which matches the paper's "over 4,000
training images". A local `pose/block_all/transforms_train.json` holding 3932 frames looks like an
official split but is not one -- the dataset's own directory listing (data/matrix_city/
small_city_tree.txt) shows `aerial/pose/block_all/` shipping ONLY `transforms_test.json`. That 3932
file was concatenated locally at a moment when block_9's `transforms.json` was missing from disk,
and 3932 is exactly the sum of the other nine blocks. Do not treat it as a reference split.

The per-source-block breakdown below is kept as a diagnostic: test views come from ten different
regions of the city and their difficulty varies a lot, so an average over a block's views can move
simply because the mix shifted.

The camera->image correspondence is CALIBRATED, never assumed. MatrixCity names its files from 1
while the COLMAP converter indexes cameras from 0, so `sparse/0` holds 0000.png..0740.png while
images_1.2 holds 0001.png..0741.png. Pairing them by name silently shifts every view by one frame
and cost 8 dB the first time this ran -- a number that looked like a dramatic held-out finding.
So the tool renders a few views against each candidate offset and picks the one that actually wins;
a healthy run reports a clear margin for a single offset. If the margins are all within noise, the
pairing is not established and the numbers should not be trusted.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.cameras.cameras import Cameras
from internal.utils.colmap import read_cameras_binary, read_images_binary
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.ssim import ssim as ssim_fn


def load_test_cameras(test_dir, down_sample):
    """Build Cameras from block_all_test/sparse/0, matching the dataparser's conventions."""
    sp = os.path.join(test_dir, "sparse", "0")
    imgs = read_images_binary(os.path.join(sp, "images.bin"))
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    R, T, fx, fy, cx, cy, W, H, names = [], [], [], [], [], [], [], [], []
    for key in sorted(imgs, key=lambda k: imgs[k].name):
        e = imgs[key]
        i = cams[e.camera_id]
        assert i.model in ("PINHOLE", "SIMPLE_PINHOLE"), i.model
        p = i.params
        f_x, f_y, c_x, c_y = (p[0], p[0], p[1], p[2]) if i.model == "SIMPLE_PINHOLE" \
            else (p[0], p[1], p[2], p[3])
        R.append(e.qvec2rotmat()); T.append(np.array(e.tvec))
        fx.append(f_x); fy.append(f_y); cx.append(c_x); cy.append(c_y)
        W.append(i.width); H.append(i.height); names.append(e.name)

    t = lambda x, d=torch.float32: torch.tensor(np.array(x), dtype=d)
    fx, fy, cx, cy = t(fx), t(fy), t(cx), t(cy)
    W, H = t(W), t(H)
    if down_sample != 1:
        # "round" is the dataparser default (down_sample_rounding_mode), and the intrinsics must be
        # scaled by the ACTUAL rounded ratio, not by 1/factor, or the principal point drifts.
        dw, dh = torch.round(W / down_sample), torch.round(H / down_sample)
        fx, cx = fx * (dw / W), cx * (dw / W)
        fy, cy = fy * (dh / H), cy * (dh / H)
        W, H = dw, dh
    n = len(names)
    return names, Cameras(
        R=t(R), T=t(T), fx=fx, fy=fy, cx=cx, cy=cy,
        width=W.to(torch.int16), height=H.to(torch.int16),
        appearance_id=torch.zeros(n, dtype=torch.int),
        normalized_appearance_id=torch.zeros(n),
        distortion_params=None, camera_type=torch.zeros(n, dtype=torch.int8),
    )


def source_blocks(test_dir,
                  pose_json="data/matrix_city/aerial/pose/block_all/transforms_test.json"):
    """Map each test image name back to the MatrixCity block it came from.

    `tools/transform_json2txt_mc_aerial.py` names images by their INDEX in transforms_test.json
    (`{idx:04d}.png`), so the json's own `file_path` ("block_9_test/ERR_12.png") recovers the source.
    """
    import json
    frames = json.load(open(pose_json))["frames"]
    return {f"{i:04d}.png": fr["file_path"].split("/")[0] for i, fr in enumerate(frames)}


def block_bounds(train_dir, block_id, block_dim, content_threshold=0.08):
    """AABB of the block's TRAINING camera centres (the region the block model is responsible for)."""
    by, bx = block_id // block_dim[0], block_id % block_dim[0]
    plist = os.path.join(train_dir, "partition",
                         f"partitions-dim_{block_dim[0]}_{block_dim[1]}_visibility_{content_threshold}",
                         f"{bx:03d}_{by:03d}.txt")
    wanted = {ln.strip() for ln in open(plist) if ln.strip()}
    imgs = read_images_binary(os.path.join(train_dir, "sparse", "0", "images.bin"))
    C = np.array([-im.qvec2rotmat().T @ im.tvec for im in imgs.values() if im.name in wanted])
    if len(C) == 0:
        raise SystemExit(f"分區 {plist} 與 sparse 的檔名對不起來")
    return C.min(0), C.max(0), len(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--block", type=int, default=None, help="omit to score all 741 views")
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--train_dir", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--image_subdir", default="images_1.2")
    ap.add_argument("--down_sample", type=float, default=1.2)
    ap.add_argument("--margin", type=float, default=0.0, help="expand the block AABB by this much")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--per_image_csv", default=None, help="逐張 name,psnr,ssim,lpips,bright_ratio（2026-10-09：對官方工具逐幀比對）")
    a = ap.parse_args()

    dev = "cuda"
    if not os.path.exists(os.path.join(a.test_dir, ".built_by_official_steps")):
        raise SystemExit(f"⛔ {a.test_dir} 不是照官方步驟重建的 test 集（缺 .built_by_official_steps）。\n"
                         "  舊的 block_all_test/input 對應是錯的（見檔頭）；先跑 task_citygs_origin.sh prep_test_official")
    names, cameras = load_test_cameras(a.test_dir, a.down_sample)
    centres = np.array([(-cameras.R[i].numpy().T @ cameras.T[i].numpy()) for i in range(len(names))])

    if a.block is None:
        sel = list(range(len(names)))
        print(f"[選視角] 全部 {len(sel)} 幀")
    else:
        lo, hi, ntrain = block_bounds(a.train_dir, a.block, a.block_dim)
        lo, hi = lo - a.margin, hi + a.margin
        sel = [i for i in range(len(names)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)]
        print(f"[選視角] block {a.block}: 訓練相機 {ntrain} 台 → AABB "
              f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}]  "
              f"落在其中的測試視角 {len(sel)}/{len(names)} 幀")
        if not sel:
            raise SystemExit("此塊沒有任何官方測試視角，改用 --margin 放寬或換塊")

    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, dev, eval_mode=True)
    print(f"[模型] N={model.get_xyz.shape[0]:,}")

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)

    img_dir = os.path.join(a.test_dir, a.image_subdir)
    miss = [n for n in names if not os.path.exists(os.path.join(img_dir, n))]
    if miss:
        raise SystemExit(f"⛔ {len(miss)} 台相機在 {img_dir} 找不到同名影像（例：{miss[:3]}）——官方 test 集應該逐一同名")
    bg = torch.zeros(3, device=dev)

    def render(i):
        with torch.no_grad():
            return renderer(cameras[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)

    def load_gt(name, like):
        pil = Image.open(os.path.join(img_dir, name)).convert("RGB")
        cw, ch = int(like.shape[2]), int(like.shape[1])
        if pil.size != (cw, ch):
            pil = pil.resize((cw, ch), Image.LANCZOS)
        return torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.

    from internal.metrics.citygsv2_metrics import _grad_energy

    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    ps, ss, ls, tr, tg, ms, br = [], [], [], [], [], [], []
    import time
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for k, i in enumerate(sel):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = render(i)
            torch.cuda.synchronize(); ms.append((time.perf_counter() - t0) * 1e3)
            gt = load_gt(names[i], out)          # 官方 test 集：相機與影像同名（見檔頭）
            tr.append(float(_grad_energy(out))); tg.append(float(_grad_energy(gt)))

            ps.append(float(-10 * torch.log10(((out - gt) ** 2).mean().clamp_min(1e-12))))
            ss.append(float(ssim_fn(out, gt)))
            ls.append(float(lpips_fn(out.unsqueeze(0), gt.unsqueeze(0))))
            br.append(float(out.mean() / gt.mean().clamp_min(1e-6)))
            if a.save_dir:
                both = torch.cat([gt, out], dim=2).permute(1, 2, 0).cpu().numpy()
                Image.fromarray((both * 255).astype(np.uint8)).save(
                    os.path.join(a.save_dir, f"{names[i].rsplit('.',1)[0]}_gt_vs_render.png"))
            if (k + 1) % 20 == 0:
                print(f"  ...{k+1}/{len(sel)}  PSNR={np.mean(ps):.2f}")

    print(f"\n=== 官方 held-out test（{len(sel)} 幀）===")
    print(f"PSNR  {np.mean(ps):6.3f}   (min {np.min(ps):.2f} / max {np.max(ps):.2f})")
    print(f"SSIM  {np.mean(ss):6.4f}")
    print(f"LPIPS {np.mean(ls):6.4f}")
    w = ms[5:] if len(ms) > 10 else ms                      # 前 5 幀暖身不計
    print(f"渲染  {np.mean(w):6.2f} ms/幀（中位 {np.median(w):.2f}）=> {1e3 / np.mean(w):.1f} FPS"
          f"   峰值 VRAM 配置 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB（含模型；⚠ 計時要獨佔整卡才可比）")
    print(f"紋理比 逐張平均 {np.mean(np.array(tr) / np.maximum(tg, 1e-8)):6.4f}   能量加權 {np.sum(tr) / max(np.sum(tg), 1e-8):6.4f}"
          "（<1 偏糊、>1 多出不存在的細節；要與 LPIPS 一起看）")

    # Break down by which MatrixCity block each test view came from -- difficulty varies a lot
    # across the city, so a shift in the mix can move the average on its own.
    src = source_blocks(a.test_dir)
    groups = {}
    for j, i in enumerate(sel):
        groups.setdefault(src.get(names[i], "?"), []).append(j)
    if len(groups) > 1:
        print(f"\n{'來源區塊':>14} {'幀':>5} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
        for g in sorted(groups, key=lambda x: -len(groups[x])):
            ix = groups[g]
            print(f"{g:>14} {len(ix):>5} {np.mean([ps[j] for j in ix]):>7.3f} "
                  f"{np.mean([ss[j] for j in ix]):>7.4f} {np.mean([ls[j] for j in ix]):>7.4f}")
    if a.per_image_csv:
        with open(a.per_image_csv, "w") as f:
            f.write("name,psnr,ssim,lpips,bright_ratio\n")
            for j, i in enumerate(sel):
                f.write(f"{names[i]},{ps[j]:.5f},{ss[j]:.5f},{ls[j]:.5f},{br[j]:.4f}\n")
        print(f"[逐張] {a.per_image_csv}")
    if a.save_dir:
        print(f"[圖] {a.save_dir}（左=GT 右=渲染）")


if __name__ == "__main__":
    main()
