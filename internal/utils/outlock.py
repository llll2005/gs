"""對「會寫共用輸出」的工具上鎖 —— 雙開在本專案已經發生三次。

```
2026-09-13 ①  使用者手動啟動 task_speed3.sh + runner 也排了同一支
               => 兩支都在 6GB 上長 N、寫同一個輸出目錄，一支在 step ~6,400 死掉（白燒 1h42m）
2026-09-13 ②  同上（第二次）=> run_fit 加了 flock
2026-09-13 ③  兩個 make_sfm_fill_init 同時寫 sfmfill_init/
               => 產出的是**舊參數**的結果，而檔案大小一模一樣，差點被當成新的
```
③ 特別陰險：沒有錯誤、沒有崩潰，只是**內容不是你以為的那個**。
⇒ 凡是寫到固定路徑（而不是以 PID/時間戳命名）的工具，都該用這個。

用法:
    from internal.utils.outlock import exclusive
    with exclusive("sfmfill_init"):
        ...寫檔...
"""
import fcntl
import os
import sys
from contextlib import contextmanager


@contextmanager
def exclusive(name, hint=""):
    d = os.path.join("logs", "locks")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name.replace("/", "_").replace(os.sep, "_") + ".lock")
    f = open(p, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"⛔ 已經有另一個行程在寫「{name}」（鎖 {p}）=> 放棄，不重複執行")
        if hint:
            print(f"   {hint}")
        print(f"   要查是誰：fuser -v {p}")
        f.close()
        sys.exit(9)
    try:
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()
