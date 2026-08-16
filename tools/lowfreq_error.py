"""低頻誤差 = 大團塊假影，直接量渲染結果。純 CPU，讀已存 test 圖。

為什麼要換掉 geometry_health 的 floater%：它數「距相機<0.5 且 opacity>0.1 的粒子」，
但那和螢幕上看到的大團塊對不上 —— dssim08 的 floater% 比 noprior 少 2.7 倍，
低頻誤差卻差 34.7% 且 35/36 視角都差。使用者看到的是後者。
"""
import glob, os, sys, numpy as np
from PIL import Image, ImageFilter
def half(p):
    im = Image.open(p); w = im.width // 2
    return im.crop((0,0,w,im.height)), im.crop((w,0,im.width,im.height))
def lo(im, s=12):
    return np.asarray(im.filter(ImageFilter.GaussianBlur(s)), np.float32)/255.
rows=[]
for run in sys.argv[1:]:
    fs = glob.glob("outputs/%s/blocks/block_12/test/*/*.png" % run)
    if not fs: print(run, "無 test 圖"); continue
    e_lo, e_hi = [], []
    for p in sorted(fs):
        gt, r = half(p)
        g, x = lo(gt), lo(r)
        e_lo.append(np.abs(x-g).mean())
        gh = np.asarray(gt,np.float32)/255. - g; xh = np.asarray(r,np.float32)/255. - x
        e_hi.append(np.abs(xh-gh).mean())
    rows.append((run, float(np.mean(e_lo)), float(np.mean(e_hi))))
rows.sort(key=lambda r: r[1])
print("%-24s %11s %11s" % ("run", "低頻誤差", "高頻誤差"))
for n,a,b in rows: print("%-24s %11.5f %11.5f" % (n.replace('_b12',''), a, b))
print("\n  低頻誤差 = 大團塊/鬼影/整片色偏（越小越好）；高頻 = 細節與銳利度")
