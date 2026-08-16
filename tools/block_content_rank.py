"""逐塊 GT 內容密度（梯度能量）排名，純 CPU。挑「拿哪一塊做實驗」時用。

為什麼需要：全專案的配方都在 **b12** 上調，而 b12 的內容密度只排 **13/25**（0.01550），
半面是水。b7 最重（0.02242，1.45x）。單塊調出來的結論會不會轉移，先看你站在分布的哪裡。

⚠ **取樣數是關鍵，不要偷懶。** 2026-08-16 我先用 5~6 張跑，b12 得到 0.01547 與 0.02188
兩個相差 42% 的值，還把最重的 b7 擠出前六 —— 差點據此宣稱整張排名是雜訊。40 張時
塊間 std 0.00308 vs 均值誤差 0.00103 = **訊噪比 3.0**，排名才穩定（且回頭確認 6 張版的
b12 其實是對的，錯的是另一支 5 張系統取樣的腳本）。低於 ~30 張不要引用。

⚠ 塊內 std（0.003~0.010）本身就和塊間 std（0.00308）同量級 ⇒ **一塊之內的視角差異，
和塊與塊之間的差異一樣大**。所以「這塊比較難」永遠只是平均意義，逐視角不成立。
"""
import os, glob, numpy as np
from PIL import Image
D = "data/matrix_city/aerial/train/block_all"
P = D + "/partition/partitions-dim_5_5_visibility_0.08"
files = sorted(f for f in os.listdir(D + "/input") if f.endswith(".png"))
def grad(i):
    im = Image.open(D + "/input/" + files[i]).convert("L")
    im.draft("L", (im.width // 4, im.height // 4))
    a = np.asarray(im, np.float32) / 255.
    return float(np.abs(np.diff(a, axis=0)).mean())
rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob(P + "/*.txt")):
    b = os.path.basename(f)[:-4]; bid = int(b[4:])*5 + int(b[:3])
    names = [l.strip() for l in open(f) if l.strip()]
    idx = [int(n[:-4]) - 1 for n in names]
    idx = [i for i in idx if 0 <= i < len(files)]
    pick = rng.choice(len(idx), min(40, len(idx)), replace=False)
    g = np.array([grad(idx[k]) for k in pick])
    rows.append((bid, g.mean(), g.std(), g.std()/np.sqrt(len(g)), len(idx)))
rows.sort(key=lambda r: -r[1])
print("%4s %9s %9s %9s %6s" % ("id", "GT梯度", "塊內std", "均值誤差", "視角"))
for bid, m, s, se, n in rows:
    tag = "  <- b12（全部實驗）" if bid == 12 else ("  <- 最重" if bid == rows[0][0] else "")
    print("%4d %9.5f %9.5f %9.5f %6d%s" % (bid, m, s, se, n, tag))
M = np.array([r[1] for r in rows]); SE = np.array([r[3] for r in rows])
print("\n塊間 std %.5f   平均的均值誤差 %.5f   => 訊噪比 %.1f 倍" % (M.std(), SE.mean(), M.std()/SE.mean()))
print("b12 排名 %d / 25" % (1 + sum(1 for r in rows if r[1] > dict((r[0], r[1]) for r in rows)[12])))
