"""11-16 使用者看完新底 5 全場景：「港口建築因為靠近邊界被消除一部分」=> 碼頭兩端的像素在訓練時被標成什麼。只讀、CPU；lab 上跑。
輸入 tools/probe/pier_tips_1116.npy（本機從 GT 點雲找碼頭、取兩端各 150 個甲板點，世界座標）。
每座碼頭、每一端：投影到所屬塊的每張訓練影像（在畫面內、在相機前方），讀
  新遮罩 block_masks44_tri（校正深度判；新底在用）與舊遮罩 block_masks44（單目深度判）：A 塊內計分／B 塊外目標背景／C 不計分
  校正深度的可靠度級別（斜率級 0～3、4＝該影像 SfM 點的凸包外）與校正深度對實際距離的相對差
用法（lab，cwd＝repo 根）：python tools/probe/pier_mask_1116.py
"""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from internal.utils.colmap import read_cameras_binary, read_images_binary, qvec2rotmat
D = "data/matrix_city/aerial/train/block_all"
tips = np.load("tools/probe/pier_tips_1116.npy", allow_pickle=True).item()
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin"); imgs = read_images_binary(f"{D}/sparse/0/images.bin")
by_name = {v.name: v for v in imgs.values()}
_cache = {}


def label_at(path, u, v, W0, H0):
    if not os.path.exists(path):
        return None
    if path not in _cache:
        _cache.clear(); _cache[path] = np.array(Image.open(path))
    a = _cache[path]; h, w = a.shape[:2]
    return a[np.clip((v * h / H0).astype(int), 0, h - 1), np.clip((u * w / W0).astype(int), 0, w - 1)]


print(f"{'碼頭':16}{'端':>3}{'GT覆蓋':>7}{'像素數':>8}{'新遮罩 A/B/C %':>18}{'舊遮罩 A/B/C %':>18}{'校正深度無效%':>13}{'凸包外%':>9}{'可靠%':>7}{'深度差中位':>11}")
for name, t in tips.items():
    B = t["blk"]
    lst = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/%03d_%03d.txt" % (B % 4, B // 4)) if l.strip()]
    for end in ("A", "B"):
        X = t["end" + end]
        newl, oldl, rel, ed = [], [], [], []
        for nm in lst:
            im = by_name.get(nm)
            if im is None:
                continue
            cm = cams[im.camera_id]; R = qvec2rotmat(im.qvec); tt = np.asarray(im.tvec)
            Xc = X @ R.T + tt; z = Xc[:, 2]
            fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
            cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
            u = fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy
            ok = (z > 0) & (u >= 0) & (u < cm.width) & (v >= 0) & (v < cm.height)
            if ok.sum() == 0:
                continue
            u, v, z = u[ok], v[ok], z[ok]
            ln = label_at(f"{D}/block_masks44_tri/block_{B}/{nm}", u, v, cm.width, cm.height)
            lo = label_at(f"{D}/block_masks44/block_{B}/{nm}", u, v, cm.width, cm.height)
            if ln is not None:
                newl.append(ln)
            if lo is not None:
                oldl.append(lo)
            fr = f"{D}/estimated_depths_tri_rel/{nm}.npy"; ftd = f"{D}/estimated_depths_tri/{nm}.npy"
            if os.path.exists(fr) and os.path.exists(ftd):
                r = np.load(fr); td = np.load(ftd).squeeze(); h, w = r.shape[:2]
                iu = np.clip((u * w / cm.width).astype(int), 0, w - 1); iv = np.clip((v * h / cm.height).astype(int), 0, h - 1)
                rel.append(r[iv, iu]); inv = td[iv, iu]
                ed.append(np.where(inv > 1e-6, (1.0 / np.maximum(inv, 1e-9)) / z - 1.0, np.nan))
        f3 = lambda L: (lambda a: f"{100 * np.mean(a == 1):.0f}/{100 * np.mean(a == 2):.0f}/{100 * np.mean(a == 0):.0f}")(np.concatenate(L)) if L else "—"
        rr = np.concatenate(rel) if rel else np.zeros(0, int); ee = np.concatenate(ed) if ed else np.zeros(0)
        npx = sum(len(x) for x in newl)
        inval = 100 * np.mean(~np.isfinite(ee)) if len(ee) else float("nan")
        hull = 100 * np.mean((rr & 7) == 4) if len(rr) else float("nan")
        good = 100 * np.mean(((rr & 7) <= 1) & ((rr >> 3) <= 3)) if len(rr) else float("nan")
        med = 100 * np.nanmedian(ee) if np.isfinite(ee).any() else float("nan")
        print(f"{name:16}{end:>3}{100 * t['cov' + end]:>6.0f}%{npx:>8,}{f3(newl):>18}{f3(oldl):>18}{inval:>12.0f}%{hull:>8.0f}%{good:>6.0f}%{med:>10.1f}%", flush=True)
print("DONE pier_mask")
