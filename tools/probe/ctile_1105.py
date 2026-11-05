import os, sys, numpy as np
from PIL import Image
D = sys.argv[1]; W, H = 1600, 900
fs = sorted(f for f in os.listdir(D) if f.endswith('.png'))
rng = np.random.default_rng(0); fs = [fs[i] for i in rng.choice(len(fs), min(150, len(fs)), replace=False)]
acc = dict(pc=0., pb=0., tc=0., wc=0., tcb=0.)
for f in fs:
    m = np.array(Image.open(os.path.join(D, f)).resize((W, H), Image.NEAREST))
    c = (m == 0); b = (m == 2)
    acc['pc'] += c.mean(); acc['pb'] += b.mean()
    Hp = (H + 15) // 16 * 16
    cp = np.ones((Hp, W), bool); cp[:H] = c
    t = cp.reshape(Hp // 16, 16, W // 16, 16)
    acc['tc'] += t.all(axis=(1, 3)).mean()                      # 整個 16x16 tile 都是 C
    w = cp.reshape(Hp // 2, 2, W // 16, 16)                      # warp = 2 列 x 16 像素
    acc['wc'] += w.all(axis=(1, 3)).mean()
    cb = np.ones((Hp, W), bool); cb[:H] = c | b
    acc['tcb'] += cb.reshape(Hp // 16, 16, W // 16, 16).all(axis=(1, 3)).mean()
n = len(fs)
print(f"{D}: {n} 張  C 像素 {acc['pc']/n:.3f}  B 像素 {acc['pb']/n:.3f}  整塊 C 的 tile {acc['tc']/n:.3f}  整個 warp 是 C {acc['wc']/n:.3f}  整塊 B 或 C 的 tile {acc['tcb']/n:.3f}")
