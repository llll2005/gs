#!/usr/bin/env python
"""在 `紀錄/*.md` 的表頭補一行**自動產生**的最後更新時間（來自 git）。

## 為什麼要自動

手維護的時間戳會漂，漂了就變成謊言 —— 而 git 本來就知道每個檔的最後修改時間。
⇒ 這支工具每次整理完跑一次就好，不手打。

## 為什麼**不**把「在跑的章節」也寫進表頭

那是每小時都在變的狀態；寫進 20 個檔的表頭 = 製造 20 份會過期的副本。
它只存在一個地方：`研究總覽_v2.md` §0「現況一頁」——那一節的設計目的就是
「唯一會頻繁改的一節」。其他檔案不碰。（單一來源原則）

用法: python tools/stamp_records.py          # 更新
      python tools/stamp_records.py --dry    # 只看會改什麼
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

MARK = "最後更新: "


def git_date(path):
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", path],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    n = 0
    for p in sorted(glob.glob("紀錄/*.md")):
        d = git_date(p)
        if not d:
            print(f"  ⚠ {p}：git 不知道（未追蹤？）")
            continue
        s = open(p, encoding="utf-8").read()
        line = f"{MARK}{d}（由 tools/stamp_records.py 從 git 產生，勿手改）\n"
        if MARK in s:
            s2 = re.sub(rf"^{re.escape(MARK)}.*\n", line, s, count=1, flags=re.M)
        elif s.startswith("---\n"):
            # 有 YAML 表頭 => 插在表頭結尾之前
            end = s.index("\n---\n", 4) + 1
            s2 = s[:end] + line + s[end:]
        else:
            # 沒表頭 => 插在第一個標題之後
            i = s.find("\n")
            s2 = s[:i + 1] + line + s[i + 1:]
        if s2 == s:
            continue
        n += 1
        if a.dry:
            print(f"  （dry）{p} -> {d}")
            continue
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(p)))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(s2)
        os.replace(tmp, p)
        print(f"  ✔ {p} -> {d}")
    print(f"\n{'會改' if a.dry else '改了'} {n} 檔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
