"""檢查每個 scripts/task_*.sh 的 `rm -rf outputs/X` 和它訓練的 `-n Y` 是不是同一個。

2026-08-17：我用 sed 從 task_noprior.sh 產生 task_b7.sh，只替換了 `-n noprior_b12`，
漏掉第 41 行的 `rm -rf outputs/noprior_b12` => b7 一啟動就把我們最好的模型整個刪光。
這支 lint 讓同一類錯誤不會再靜默發生。跑法：python tools/lint_task_scripts.py
"""
import glob, re, sys
bad = 0
for p in sorted(glob.glob("scripts/task_*.sh")):
    t = open(p).read()
    rms = re.findall(r"^\s*rm\s+-rf\s+outputs/([A-Za-z0-9_.-]+)", t, re.M)
    ns = [x for x in re.findall(r"-n\s+([A-Za-z0-9_.-]+)", t) if x != "gspl"]  # 排除 conda run -n gspl
    if not rms:
        continue
    for r in rms:
        if r not in ns:
            print("✘ %-28s rm -rf outputs/%-20s 但訓練的是 %s" % (p, r, ns or ["(無 -n)"]))
            bad += 1
print("\n%d 個不一致" % bad if bad else "\n全部一致 ✓")
sys.exit(1 if bad else 0)
