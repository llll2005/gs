"""我們一直在優化一個雙峰分布的平均。把尾巴修好值多少 dB？"""
import glob
import re, os, numpy as np
from PIL import Image
def _final_test_dir(run, blk):
    """只取最高 step 的 test 目錄（earlyckpt 診斷會在 test/ 留下多個 checkpoint 的圖）。"""
    ds = glob.glob("outputs/%s/blocks/block_%s/test/*/" % (run, blk))
    if not ds:
        return None
    return max(ds, key=lambda d: int(re.search(r"step=(\d+)", d).group(1)) if re.search(r"step=(\d+)", d) else -1)


def half(p):
    im=Image.open(p); w=im.width//2
    return im.crop((0,0,w,im.height)), im.crop((w,0,im.width,im.height))
import sys as _s
SPECS=[(x.split(":")[0], int(x.split(":")[1]) if ":" in x else 12) for x in _s.argv[1:]] \
      or [("sched30_b12",12),("best_b7",7),("noprior_b12",12)]
for run,blk in SPECS:
    ms=[]
    _d=_final_test_dir(run,blk)
    for p in sorted(glob.glob(_d+"*.png") if _d else []):
        gt,r=half(p); g=np.asarray(gt,np.float32)/255.; x=np.asarray(r,np.float32)/255.
        ms.append(float(((g-x)**2).mean()))
    ms=np.array(sorted(ms)); ps=10*np.log10(1/ms)
    mean=ps.mean()
    def fix(frac):
        k=int(len(ms)*frac); m2=ms.copy(); m2[-k:]=np.median(ms)   # 最差的一段拉到中位
        return (10*np.log10(1/m2)).mean()
    print("%-14s 平均 %.3f   最好 %.2f   中位 %.2f   最差 %.2f   全距 %.0fx(MSE)"
          %(run,mean,ps.max(),np.median(ps),ps.min(),ms.max()/ms.min()))
    print("               修好最差 10%% -> %.3f (%+.3f)   25%% -> %.3f (%+.3f)   全拉到中位 -> %.3f (%+.3f)"
          %(fix(.10),fix(.10)-mean,fix(.25),fix(.25)-mean,fix(1.0),fix(1.0)-mean))
print("\n  對照：本 session 所有機制的最大單筆增益 = sched30 的 +0.327 dB")
