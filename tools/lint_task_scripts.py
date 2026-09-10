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
def _warn_running_scripts():
    """⛔ 偵測「正在被執行」的 task 腳本 —— 就地改它會讓跑動中的 bash 讀錯位元組偏移。

    2026-09-05 第二次踩到（`task_fpd_b12.sh`，rc=127 `lize_from: command not found`）：
    `python open(p,"w")` 是**就地截斷、同 inode**，執行中的 fd 還指著同一個 inode
    ⇒ bash 用舊偏移繼續讀 ⇒ 讀到一行的中間。
    ✅ `sed -i` 或「寫暫存檔再 `mv`」會**換 inode**，執行中的 fd 指向舊 inode，安全。
    這個檢查把「靜默損毀」變成「編輯後立刻看得到的警告」。
    """
    import subprocess
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return
    hot = sorted({ln.split("scripts/")[1].split()[0]
                  for ln in ps.splitlines()
                  if "scripts/task_" in ln and "lint_task_scripts" not in ln})
    if hot:
        print("\n⛔ 下列 task 腳本**正在執行中**，改它們必須用 `sed -i` 或「暫存檔+mv」，"
              "不可用 `python open(w)`（同 inode → 跑動中的 bash 會讀錯偏移）：")
        for h in hot:
            print(f"     scripts/{h}")


_warn_running_scripts()
sys.exit(1 if bad else 0)
