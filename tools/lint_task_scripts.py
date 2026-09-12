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



def _warn_rc_masked():
    """⛔ 偵測「訓練崩了但台帳會顯示 ✔ DONE rc=0」的腳本。

    runner 記到台帳的 rc 是**腳本最後一個指令**的 rc。絕大多數 task 腳本在訓練之後
    還會接一行評測/稽核（`python tools/...`、`echo`），那一行成功就把訓練的 rc 蓋掉了。

    2026-09-12 實際發生：`task_normal_b12.sh` 的訓練因 ColmapBlock 靜默 bug 載入全部
    5,621 張影像、RAM 被吃爆、**0 步 0 ckpt**，而台帳寫的是「✔ DONE (rc=0, 336s)」。
    這與 `verify_before_comparing` 記的是同一個家族：**失敗的跑次留下看起來正常的紀錄**。

    修法（本檢查要找的就是這個形狀）：
        conda run ... python -u main.py fit ... ; RC=$?
        [ "$RC" -ne 0 ] && exit "$RC"
    """
    import glob, re
    hits = []
    for fp in sorted(glob.glob("scripts/task_*.sh")):
        lines = open(fp, errors="ignore").read().split("\n")
        # 找最後一個訓練指令，並跟著反斜線續行走到結尾
        last = None
        for i, ln in enumerate(lines):
            if re.search(r"main\.py\s+fit|train_citygs_partitions\.py", ln):
                last = i
        if last is None:
            continue
        j = last
        while j < len(lines) - 1 and lines[j].rstrip().endswith("\\"):
            j += 1
        tail = lines[j + 1:]
        after = [t for t in tail if t.strip() and not t.strip().startswith("#")]
        if not after:
            continue                      # 訓練是最後一個指令 => rc 本來就是對的
        guard = "\n".join(tail[:6])
        if re.search(r"RC=\$\?|\|\|\s*exit|set\s+-e", guard) or re.search(r"set\s+-e", lines[0:last] and "\n".join(lines[:last])):
            continue
        hits.append((fp, len(after)))
    if hits:
        print("\n⚠ 下列腳本**訓練之後還有指令，且沒有傳遞訓練的 rc** "
              "=> 訓練崩掉時台帳仍會寫 ✔ DONE rc=0（比 PSNR 前務必配合 tools/run_status.py）：")
        for fp, n in hits[:40]:
            print(f"     {fp:<44} 之後 {n} 行")
        print(f"   共 {len(hits)} 支。修法：訓練後加 `RC=$?` 與 `[ \"$RC\" -ne 0 ] && exit \"$RC\"`。")


_warn_rc_masked()

_warn_running_scripts()
sys.exit(1 if bad else 0)
