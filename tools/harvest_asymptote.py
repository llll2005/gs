"""LR 積分是不是收割期的資源？用現有 tensorboard log 驗，不碰 GPU。

收割期（densify_until 之後）不動密度 => 純最佳化。若 LR 積分是資源，
PSNR(t) = A - B*exp(-c * I(t))，I(t)=∫lr。檢查：
  (1) 每個跑次擬合得好不好（R²）
  (2) 時間常數 c 跨跑次一不一致 —— 一致才代表它是共通機制而非曲線硬湊
  (3) 對照組：改用「步數」當自變數，看誰擬得好
若 (2) 散掉，這個模型只是 3 參數在硬湊 4 個點，不能拿來外推 sched30。
"""
import glob, re, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.optimize import curve_fit
DU = 42000
def get(run, blk):
    f = sorted(glob.glob("outputs/%s/blocks/block_%d/lightning_logs/version_*/events*" % (run, blk)))
    if not f: return None
    ea = EventAccumulator(f[-1], size_guidance={'scalars': 0}); ea.Reload()
    t = ea.Tags()['scalars']
    if "val/psnr" not in t or "lr/0_means" not in t: return None
    ps = ea.Scalars("val/psnr"); lr = ea.Scalars("lr/0_means")
    ls, lv = np.array([x.step for x in lr]), np.array([x.value for x in lr])
    return np.array([x.step for x in ps]), np.array([x.value for x in ps]), ls, lv

RUNS = [("best_b7",7)] + [(r,12) for r in ["sh3_nonormal_b12","sh3_nodepth_b12","dssim05_b12",
        "dssim08_b12","npd_b12","sb4_b12","coarsel1_b12","sh3_viewdep_refix_b12","sb_refix_b12"]]
print("%-22s %4s %9s %9s %9s %9s" % ("run","n","A(漸近)","餘量dB","c","R²(LR)  R²(步數)"))
cs = []
for run, blk in RUNS:
    g = get(run, blk)
    if not g: print("%-22s 無 log" % run); continue
    s, p, ls, lv = g
    m = s >= DU
    if m.sum() < 4: print("%-22s 收割期 val 點只有 %d 個" % (run, m.sum())); continue
    I = np.array([np.trapz(lv[ls <= t][ls[ls <= t] >= DU], ls[(ls <= t) & (ls >= DU)]) for t in s[m]])
    I = I / lv[0]                                   # 用各自初始 LR 正規化（b7 的 LR 是 b12 的 1.53 倍）
    y = p[m]
    def f(x, A, B, c): return A - B*np.exp(-c*x)
    try:
        (A,B,c), _ = curve_fit(f, I, y, p0=[y[-1]+0.5, 1.0, 1/max(I.max(),1)], maxfev=20000)
        r2 = 1 - ((y-f(I,A,B,c))**2).sum()/((y-y.mean())**2).sum()
        x2 = (s[m]-DU).astype(float)
        (A2,B2,c2), _ = curve_fit(f, x2, y, p0=[y[-1]+0.5,1.0,1/max(x2.max(),1)], maxfev=20000)
        r2b = 1 - ((y-f(x2,A2,B2,c2))**2).sum()/((y-y.mean())**2).sum()
        cs.append(c)
        print("%-22s %4d %9.3f %9.3f %9.4f   %.4f  %.4f" % (run, m.sum(), A, A-y[-1], c, r2, r2b))
    except Exception as e:
        print("%-22s 擬合失敗 %s" % (run, e))
cs = np.array(cs)
print("\n時間常數 c：中位 %.4f，變異係數 %.2f（<0.3 才算跨跑次一致）" % (np.median(cs), cs.std()/cs.mean()))
