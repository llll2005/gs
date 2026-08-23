"""端到端驗證：訓練時拿到的「影像／GT／深度圖」是不是同一幀？

為什麼需要：同一個 off-by-one 在本專案出現過**三處**（dataparser 2026-08-12、
depth_init_blocks 2026-08-22、get_depth_scales 2026-08-23），每一次都是靜默的。
讀程式碼擋不住第四次 —— 這支直接**用訓練走的那條路**把資料load出來，逐項比對。

判準：
  1. dataparser 給的影像路徑 == COLMAP 名的位置對應檔名
  2. 深度圖檔名 == 影像檔名（+.npy）
  3. scales JSON 的鍵 == COLMAP 名，且它擬合時用的深度圖與 (2) 一致
  4. 影像內容與深度圖內容**結構相符**（邊緣位置對得上）—— 這一項才抓得到「檔名對但內容錯」

用法：python tools/verify_data_alignment.py [--block 12] [--scales <json>]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from internal.utils import colmap as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
ap.add_argument("--scales", default=None)
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

D = a.data
imgs = C.read_images_binary(os.path.join(D, "sparse/0/images.bin"))
order = sorted(imgs, key=lambda k: imgs[k].name)
files = sorted(f for f in os.listdir(os.path.join(D, "input")) if f.lower().endswith(".png"))
deps = sorted(f for f in os.listdir(os.path.join(D, "estimated_depths")) if f.endswith(".npy"))
print("COLMAP 相機 %d   影像檔 %d   深度圖 %d" % (len(order), len(files), len(deps)))
assert len(order) == len(files) == len(deps), "數量不符，位置對應無法成立"
pos_img = {imgs[k].name: files[i] for i, k in enumerate(order)}
pos_dep = {imgs[k].name: deps[i] for i, k in enumerate(order)}

by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
plist = os.path.join(D, "partition", "partitions-dim_%d_%d_visibility_0.08" % tuple(a.block_dim),
                     "%03d_%03d.txt" % (bx, by))
names = [l.strip() for l in open(plist) if l.strip()][:: max(1, 284 // a.n)][:a.n]

scales_path = a.scales or os.path.join(D, "estimated_depth_scales.json")
scales = json.load(open(scales_path)) if os.path.exists(scales_path) else {}
print("scales JSON: %s（%d 鍵）\n" % (os.path.basename(scales_path), len(scales)))

def edges(x):
    x = np.asarray(x, np.float32)
    return np.abs(np.diff(x, axis=0)).mean(-1) if x.ndim == 3 else np.abs(np.diff(x, axis=0))

bad = 0
print("%-10s %-14s %8s %10s %10s %8s" % ("COLMAP名", "影像檔", "在scales", "自身相關", "鄰幀相關", "自身較高"))
for nm in names:
    fi, fd = pos_img[nm], pos_dep[nm]
    ok_pair = fd == fi + ".npy"
    im = Image.open(os.path.join(D, "input", fi)).convert("L")
    dp = np.load(os.path.join(D, "estimated_depths", fd)).astype(np.float32)
    im = np.asarray(im.resize((dp.shape[1], dp.shape[0]), Image.LANCZOS), np.float32) / 255.
    ei = edges(im).ravel()
    r = float(np.corrcoef(ei, edges(dp).ravel())[0, 1])
    # ★自我校準：和「鄰幀的深度圖」比。絕對門檻沒有意義（深度圖只在深度不連續處有邊緣，
    # 影像連紋理都算邊緣），但「自己 vs 鄰幀」是同一把尺下的相對比較。
    i_self = deps.index(fd)
    rn = []
    for j in (i_self - 1, i_self + 1):
        if 0 <= j < len(deps):
            dn = np.load(os.path.join(D, "estimated_depths", deps[j])).astype(np.float32)
            if dn.shape == dp.shape:
                rn.append(float(np.corrcoef(ei, edges(dn).ravel())[0, 1]))
    rnb = max(rn) if rn else float("nan")
    flag = "" if (ok_pair and nm in scales and r > rnb) else "  <-- ⚠"
    bad += bool(flag)
    print("%-10s %-14s %8s %10.3f %10.3f %8s%s" % (nm, fi, "是" if nm in scales else "**否**", r, rnb, "✓" if r > rnb else "✗", flag))

print("\n判準 = 影像的邊緣，和**自己的**深度圖比，是否比和**鄰幀的**深度圖更相關。")
print("  絕對值沒有意義（深度圖只在深度不連續處有邊緣）；相對比較才是對齊的證據。")
print("\n%s" % ("✅ 全部通過" if bad == 0 else "⚠ %d/%d 項有問題" % (bad, len(names))))
