"""水面 vs 建築分組計分，純 CPU：讀已存的 test 並排圖（GT|渲染），不重新渲染、不佔 GPU。

和 `tools/rescore_by_content.py` 做同一件事，差別在**那支要 CUDA**（它從 ckpt 重新渲染）。
這支只讀 `outputs/<run>/blocks/block_<b>/test/*/‌*.png`，所以訓練佔滿 6GB 時照樣能跑分析
——本專案只有一張卡，一個實驗鎖 10 小時，這個差別決定了分析做不做得成。前提是那個 run
已經跑過 `scripts/task_test.sh`。

為什麼一定要分區域：b12 約半面是水，水面渲染是接近均勻色塊卻照樣拿 30 dB，平均起來會把建築
區的差異蓋掉。2026-08-16 用它擋掉一次實作：三組對照裡「depth loss 在水面有益」只有一組成立，
另兩組符號相反 ⇒「紋理閘門式深度監督」的前提不成立（見 `紀錄/研究總覽.md` §12.6）。

⚠ 分組用 GT 梯度排序後取兩端，不需要標註；但 36 張測試視角裡「建築組」GT 梯度 0.0242~0.0302、
  「水面組」0.0025~0.0204 —— 是連續光譜的兩端，不是二分類，別當成乾淨的語意分割。
"""
import glob
import re, os, sys
import numpy as np
from PIL import Image

RUNS = sys.argv[1:] or ["noprior_b12", "npd_b12", "dssim05_b12", "dssim08_b12"]

def _final_test_dir(run, blk):
    """只取最高 step 的 test 目錄（earlyckpt 診斷會在 test/ 留下多個 checkpoint 的圖）。"""
    ds = glob.glob("outputs/%s/blocks/block_%s/test/*/" % (run, blk))
    if not ds:
        return None
    return max(ds, key=lambda d: int(re.search(r"step=(\d+)", d).group(1)) if re.search(r"step=(\d+)", d) else -1)


def halves(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    w = a.shape[1] // 2
    return a[:, :w], a[:, w:]

def grad(x):                       # 與 measure_floater_texture.py 同定義
    return np.abs(np.diff(x, axis=0)).mean()

def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 10 * np.log10(1.0 / max(m, 1e-12))

# 先確定哪一半是 GT：兩個 run 的同一張圖，GT 那半應該完全相同
d0 = sorted(glob.glob(_final_test_dir(RUNS[0], "12")+"*.png"))
d1 = sorted(glob.glob(_final_test_dir(RUNS[1], "12")+"*.png"))
L0, R0 = halves(d0[0]); L1, R1 = halves(d1[0])
gt_is_left = np.abs(L0 - L1).mean() < np.abs(R0 - R1).mean()
print(f"[版面] GT 在{'左' if gt_is_left else '右'}半  "
      f"(左半差 {np.abs(L0-L1).mean():.5f} / 右半差 {np.abs(R0-R1).mean():.5f})")

# 用 GT 梯度把視角排序，兩端各取 12 張
files0 = [os.path.basename(f) for f in d0]
gts = {}
for f in d0:
    g, _ = halves(f) if gt_is_left else halves(f)[::-1]
    gts[os.path.basename(f)] = g
rank = sorted(files0, key=lambda n: grad(gts[n]))
K = 12
groups = {"水面": rank[:K], "建築": rank[-K:]}
print(f"[分組] 各 {K} 張   水面 GT梯度 {grad(gts[rank[0]]):.4f}~{grad(gts[rank[K-1]]):.4f}   "
      f"建築 {grad(gts[rank[-K]]):.4f}~{grad(gts[rank[-1]]):.4f}\n")

print(f"{'run':<15}{'全體PSNR':>10}{'水面PSNR':>10}{'建築PSNR':>10}{'建築紋理比':>12}{'水面紋理比':>12}")
res = {}
for run in RUNS:
    fs = sorted((lambda _d: glob.glob(_d+"*.png") if _d else [])(_final_test_dir(run, "12")))
    by = {os.path.basename(f): f for f in fs}
    row = {}
    for gname, sel in list(groups.items()) + [("全體", rank)]:
        ps, trs = [], []
        for n in sel:
            if n not in by:
                continue
            a, b = halves(by[n])
            g, r = (a, b) if gt_is_left else (b, a)
            ps.append(psnr(g, r)); trs.append(grad(r) / max(grad(g), 1e-9))
        row[gname] = (float(np.mean(ps)), float(np.mean(trs)))
    res[run] = row
    print(f"{run:<15}{row['全體'][0]:>10.3f}{row['水面'][0]:>10.3f}{row['建築'][0]:>10.3f}"
          f"{row['建築'][1]:>12.4f}{row['水面'][1]:>12.4f}")

base = RUNS[0]
print(f"\n=== 相對 {base} 的差 ===")
print(f"{'run':<15}{'Δ全體':>9}{'Δ水面':>9}{'Δ建築':>9}{'Δ建築紋理比':>13}")
for run in RUNS[1:]:
    d = lambda k, i: res[run][k][i] - res[base][k][i]
    print(f"{run:<15}{d('全體',0):>+9.3f}{d('水面',0):>+9.3f}{d('建築',0):>+9.3f}{d('建築',1):>+13.4f}")
print("\n  若 Δ水面 為大負、Δ建築 ≈0 或正 ⇒ noprior 的 PSNR 優勢是水面買來的，PSNR 在騙人")
print("  若 Δ建築 也是負 ⇒ 那是真的品質退步，dssim 0.5 不該採用")
