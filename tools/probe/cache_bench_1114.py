"""11-14 啟動時間拆解：快取 811 張影像要 102 s（8 執行緒）—— 量單張各部分的成本（CPU；lab；只讀）。
用法：conda run -n gspl --no-capture-output python -u tools/probe/cache_bench_1114.py [--n 24]"""
import argparse, glob, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"
names = open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/001_000.txt").read().split()[:a.n]
def t_img(n):
    t0 = time.perf_counter(); im = Image.open(f"{D}/input/{n}"); im.load(); t1 = time.perf_counter()
    w, h = im.size; r = im.resize((int(round(w / 1.2)), int(round(h / 1.2))), Image.LANCZOS); t2 = time.perf_counter()
    x = np.array(r, dtype=np.uint8); ok = (x[:, :, 3] == 255).all() if x.shape[2] == 4 else True; x = np.ascontiguousarray(x[:, :, :3]); t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2, (w, h, im.mode)
def t_extra(n):
    out = {}
    for d in ["estimated_depths_tri", "estimated_depths_tri_rel"]:
        f = f"{D}/{d}/{n}.npy"; t0 = time.perf_counter(); z = np.load(f) if os.path.exists(f) else None; out[d] = time.perf_counter() - t0
    for d in ["block_masks44_tri/block_1", "cap_masks_flat"]:
        fs = glob.glob(f"{D}/{d}/{os.path.splitext(n)[0]}*") + glob.glob(f"{D}/{d}/{n}*")
        t0 = time.perf_counter()
        for f in fs[:2]:
            (np.load(f) if f.endswith(".npy") else np.array(Image.open(f)))
        out[d] = time.perf_counter() - t0
    return out
r = [t_img(n) for n in names]
print(f"影像 {r[0][3]}｜單執行緒每張：讀檔＋解碼 {np.mean([x[0] for x in r]) * 1e3:.0f} ms、LANCZOS 縮 1.2 倍 {np.mean([x[1] for x in r]) * 1e3:.0f} ms、轉陣列＋alpha 檢查 {np.mean([x[2] for x in r]) * 1e3:.0f} ms")
e = [t_extra(n) for n in names]
for k in e[0]:
    print(f"額外資料 {k}：每張 {np.mean([x[k] for x in e]) * 1e3:.1f} ms")
for nt in (1, 8, 16):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(nt) as ex: list(ex.map(t_img, names))
    dt = time.perf_counter() - t0
    print(f"{nt} 執行緒：{len(names)} 張影像 {dt:.1f} s => 每張 {dt / len(names) * 1e3:.0f} ms（換算 811 張 {dt / len(names) * 811:.0f} s）")
print("cpu", os.cpu_count(), "｜sched_getaffinity", len(os.sched_getaffinity(0)))
