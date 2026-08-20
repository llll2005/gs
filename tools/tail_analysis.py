"""我們一直在優化一個雙峰分布的平均。把尾巴修好值多少 dB？"""
import glob, os, numpy as np
from PIL import Image
def half(p):
    im=Image.open(p); w=im.width//2
    return im.crop((0,0,w,im.height)), im.crop((w,0,im.width,im.height))
for run,blk in [("sched30_b12",12),("best_b7",7),("noprior_b12",12)]:
    ms=[]
    for p in sorted(glob.glob("outputs/%s/blocks/block_%d/test/*/*.png"%(run,blk))):
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
