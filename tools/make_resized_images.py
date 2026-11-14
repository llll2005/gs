"""★ 2026-11-14 demo 模式 C：預先把訓練影像縮成相機大小（使用者「做一個 demo 模式，開這個選項就套用你說的這些」）。

為什麼：每塊啟動時快取 811 張影像要 102 s（lab 冷 HDD）；每張讀 1920x1080 RGBA PNG（3.7 MB）＋LANCZOS 縮 1.2 倍（37 ms）。
        先縮好存成相機大小的無損 PNG => 讀檔量少、不用縮。
逐位元相同：internal/dataset.py get_image 只在 `pil_image.size != (cam_w, cam_h)` 時才縮 => 讀預縮檔就跳過縮放；
  這裡用同一個 PIL LANCZOS、同樣先縮 RGBA 再看 alpha；alpha 全 255 時存 RGB（dataset 本來就丟掉全 255 的 alpha，結果相同），
  否則存 RGBA（dataset 走 float 路徑，與現縮相同）。每張存完都重讀比對，不同就報錯、不留檔。
大小：相機大小用 dataparser 同一個算法（torch.round(寬或高 / down_sample_factor)）。
用法（lab，cwd＝repo 根目錄，CPU）：
  conda run -n gspl --no-capture-output python -u tools/make_resized_images.py data/matrix_city/aerial/train/block_all --nproc 12
  => <dataset>/input_r1.2/<影像名>＋全部到齊時寫 .complete；demo 模式（CITYGS_DEMO=1）看到 .complete 才用 --data.parser.image_dir input_r1.2
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.colmap import read_cameras_binary  # noqa: E402
from internal.utils.colmap_fast import read_images_binary_light  # noqa: E402

A = None


def _one(job):
    src, dst, w, h = job
    if os.path.exists(dst):
        return "skip", 0
    im = Image.open(src)
    if im.size != (w, h):
        im = im.resize((w, h), Image.LANCZOS)
    x = np.array(im, dtype=np.uint8)
    if x.ndim == 3 and x.shape[2] == 4 and bool((x[:, :, 3] == 255).all()):
        x = np.ascontiguousarray(x[:, :, :3])
    tmp = dst + f".tmp{os.getpid()}.png"
    Image.fromarray(x).save(tmp, format="PNG")
    y = np.array(Image.open(tmp), dtype=np.uint8)
    if y.shape != x.shape or not np.array_equal(x, y):
        os.remove(tmp)
        return "mismatch", src
    os.replace(tmp, dst)
    return "ok", os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--image_dir", default="input")
    ap.add_argument("--factor", type=float, default=1.2)
    ap.add_argument("--rounding", default="round")
    ap.add_argument("--out", default=None, help="預設 <image_dir>_r<factor>")
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sp = os.path.join(a.dataset, "sparse", "0")
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    ims = read_images_binary_light(os.path.join(sp, "images.bin"))
    out = os.path.join(a.dataset, a.out or f"{a.image_dir}_r{a.factor:g}")
    os.makedirs(out, exist_ok=True)
    rf = getattr(torch, a.rounding)
    jobs = []
    for im in sorted(ims.values(), key=lambda v: v.name):
        c = cams[im.camera_id]
        w = int(rf(torch.tensor(float(c.width)) / a.factor).item())
        h = int(rf(torch.tensor(float(c.height)) / a.factor).item())
        jobs.append((os.path.join(a.dataset, a.image_dir, im.name), os.path.join(out, im.name), w, h))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"[resize] {len(jobs):,} 張 -> {out}（相機大小例 {jobs[0][2]}x{jobs[0][3]}；{a.nproc} 個行程）", flush=True)
    t0 = time.time(); n_ok = n_skip = 0; tot = 0; bad = []
    with Pool(a.nproc) as p:
        for k, (st, v) in enumerate(p.imap_unordered(_one, jobs, chunksize=4), 1):
            if st == "ok":
                n_ok += 1; tot += v
            elif st == "skip":
                n_skip += 1
            else:
                bad.append(v)
            if k % 500 == 0:
                print(f"[resize] {k:,}/{len(jobs):,}（{time.time() - t0:.0f} s）", flush=True)
    print(f"[resize] 完成 新寫 {n_ok:,}（平均 {tot / max(n_ok, 1) / 2**20:.2f} MB）、已存在 {n_skip:,}、不一致 {len(bad)}｜{time.time() - t0:.0f} s", flush=True)
    if bad:
        print("⛔ 不一致：", bad[:5], flush=True)
        sys.exit(1)
    if not a.limit and all(os.path.exists(j[1]) for j in jobs):   # 全部到齊才寫完成標記（demo 模式只認有標記的目錄）
        open(os.path.join(out, ".complete"), "w").write(f"{len(jobs)}\n")
        print(f"[resize] ✅ 寫完成標記 {os.path.join(out, '.complete')}", flush=True)


if __name__ == "__main__":
    main()
