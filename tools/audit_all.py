"""修正後所有跑完的跑次，全指標一表。含尾巴（最差十分位）與結構（建築低頻）。純 CPU。"""
import glob, os, re, sys, numpy as np
from PIL import Image, ImageFilter
FIX_STEPS={}
def half(p):
    im=Image.open(p); w=im.width//2
    return im.crop((0,0,w,im.height)), im.crop((w,0,im.width,im.height))
def lo(im,s=12): return np.asarray(im.filter(ImageFilter.GaussianBlur(s)),np.float32)/255.
pat=re.compile(r"step=(\d+)")
rows=[]
for rd in sorted(glob.glob("outputs/*")):
    run=os.path.basename(rd)
    for bd in sorted(glob.glob(rd+"/blocks/block_*")):
        blk=int(re.search(r"block_(\d+)",bd).group(1))
        st=[int(pat.search(p).group(1)) for p in glob.glob(bd+"/**/*.ckpt",recursive=True) if pat.search(p)]
        st+=[int(pat.search(p).group(1)) for p in glob.glob(bd+"/**/*.ply",recursive=True) if pat.search(p)]
        if not st: continue
        mx=None
        for c in glob.glob(rd+"/**/config.yaml",recursive=True):
            m=re.search(r"^\s*max_steps:\s*([0-9_]+)",open(c).read(),re.M)
            if m: mx=int(m.group(1).replace("_","")); break
        if mx is None or max(st)<mx-1: continue                 # 只留跑完的
        fs=glob.glob(bd+"/**/events*",recursive=True)
        if not fs: continue
        t0=min(os.path.getmtime(x) for x in glob.glob(bd+"/**/*",recursive=True))
        if t0 < 1786000000: continue                            # 2026-08-12 10:08 之後
        # val 指標一律讀 tensorboard，不讀 results.txt —— 後者會被 `main.py test` 覆蓋成
        # test/* 指標（2026-08-21 sched30_b12 就這樣被舊版 earlyckpt.sh 洗掉），tensorboard 不會。
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ev=sorted(glob.glob(bd+"/lightning_logs/version_*/events*"))
        if not ev: continue
        ea=EventAccumulator(ev[-1], size_guidance={'scalars':0}); ea.Reload()
        tg=set(ea.Tags()['scalars']); d={}
        for k in ["psnr","ssim","lpips","texratio"]:
            if "val/"+k in tg: d[k]="%.6f"%ea.Scalars("val/"+k)[-1].value
        imgs=sorted(glob.glob(bd+"/test/*/*.png"))
        tail=lof=lob=float("nan")
        if len(imgs)>=20:
            ms=[];lfs=[]
            for p in imgs:
                gt,rr=half(p); g=np.asarray(gt,np.float32)/255.; x=np.asarray(rr,np.float32)/255.
                ms.append(float(((g-x)**2).mean()))
                gl=np.asarray(gt.convert("L"),np.float32)/255.
                lfs.append((float(np.abs(np.diff(gl,axis=0)).mean()), float(np.abs(lo(rr)-lo(gt)).mean())))
            ms=np.array(sorted(ms)); k=max(1,len(ms)//10)
            tail=float((10*np.log10(1/ms[-k:])).mean())
            lfs.sort(); kk=max(1,len(lfs)//3)
            lob=float(np.mean([e for _,e in lfs[-kk:]]))
        rows.append(dict(run=run,blk=blk,psnr=float(d.get("psnr",0)),ssim=float(d.get("ssim",0)),
                         lpips=float(d.get("lpips",0)),tex=float(d.get("texratio",0)),tail=tail,lob=lob,n=len(imgs)))
rows.sort(key=lambda r:-r["psnr"])
print("%-22s %3s %8s %7s %7s %7s %9s %9s"%("run","blk","PSNR","SSIM","LPIPS","紋理比","最差10%","建築低頻"))
for r in rows:
    print("%-22s %3d %8.3f %7.4f %7.4f %7.4f %9.2f %9.5f"%(r["run"],r["blk"],r["psnr"],r["ssim"],r["lpips"],r["tex"],r["tail"],r["lob"]))
