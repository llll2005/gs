"""低頻誤差住在水面還是建築？§12.13 的排名若是水面主導，那它排的不是結構品質。純 CPU。"""
import glob, sys, numpy as np
from PIL import Image, ImageFilter
def half(p):
    im = Image.open(p); w = im.width // 2
    return im.crop((0,0,w,im.height)), im.crop((w,0,im.width,im.height))
def lo(im, s=12): return np.asarray(im.filter(ImageFilter.GaussianBlur(s)), np.float32)/255.
print("%-22s %10s %10s %10s   %s" % ("run","水面低頻","建築低頻","全體","建築佔比"))
for spec in sys.argv[1:]:
    run, _, b = spec.partition(":")
    fs = sorted(glob.glob("outputs/%s/blocks/block_%s/test/*/*.png" % (run, b or "12")))
    if not fs: print(spec, "無 test 圖"); continue
    rec = []
    for p in fs:
        gt, r = half(p)
        g = np.asarray(gt.convert("L"), np.float32)/255.
        tex = float(np.abs(np.diff(g, axis=0)).mean())
        rec.append((tex, float(np.abs(lo(r)-lo(gt)).mean())))
    rec.sort()
    k = max(1, len(rec)//3)
    w_ = np.mean([e for _, e in rec[:k]]); bl = np.mean([e for _, e in rec[-k:]])
    al = np.mean([e for _, e in rec])
    print("%-22s %10.5f %10.5f %10.5f   %5.1f%%" % (spec.replace('_b12',''), w_, bl, al, 100*bl/(w_+bl)))
print("\n  各取 GT 梯度最低/最高的 1/3。建築佔比 >50% => 低頻誤差主要在建築（結構問題）")
print("  <50% => 主要在水面 => §12.13 的排名其實在排「對水面的處理」")
