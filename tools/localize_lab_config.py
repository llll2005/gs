#!/usr/bin/env python
"""把從 lab 拉回來的 resolved config 在地化（絕對路徑改成本機的）。

## 為什麼需要

lab 的 `lightning_logs/version_*/config.yaml` 裡 `output_path` 是**絕對路徑**
`/workspace/data/hdd/11213/gs/outputs/...`，而 `main.py test --config <它>` 會直接拿去建目錄：
    PermissionError: [Errno 13] Permission denied: '/workspace'
（2026-09-13 實測。我原本以為 cli 會從 `config.output/config.name` 重算 —— `fit` 會，
  但 `test` 用的是存下來的絕對路徑。）

⚠ **只改路徑前綴，不改 `name`** —— lab 跑次的 name 就是 `lab/speed3`，
  而它在本機也確實躺在 `outputs/lab/speed3/`，所以 name 是對的、不該動。
  （對照：2026-09-13 把 b13 從 outputs/lab/ 搬到 outputs/ 時，name 才需要跟著改。）

用法: python tools/localize_lab_config.py outputs/lab/speed3/blocks/block_6
      python tools/localize_lab_config.py --all          # 掃 outputs/lab/ 底下全部
"""
import argparse
import glob
import os
import sys
import tempfile

LAB_ROOTS = ("/workspace/data/hdd/11213/gs/", "/hdd/11213/gs/")


def localize(path, repo, dry=False):
    n = 0
    for p in glob.glob(os.path.join(path, "lightning_logs", "version_*", "*.yaml")):
        s = open(p, encoding="utf-8").read()
        s2 = s
        for r in LAB_ROOTS:
            s2 = s2.replace(r, repo + os.sep)
        if s2 == s:
            continue
        n += 1
        if dry:
            print(f"  （dry）{p}")
            continue
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(s2)
        os.replace(tmp, p)
        print(f"  ✔ {p}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="outputs/lab/<run>/blocks/block_N")
    ap.add_argument("--all", action="store_true", help="掃 outputs/lab/ 底下全部")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = a.paths
    if a.all:
        targets = [d for d in glob.glob("outputs/lab/*/blocks/block_*")
                   if os.path.isdir(d) and ".aborted_" not in d]
    if not targets:
        raise SystemExit("要給路徑或 --all")
    tot = sum(localize(t, repo, a.dry) for t in targets)
    print(f"\n改了 {tot} 個 yaml（前綴 {LAB_ROOTS} -> {repo}/）")
    if tot == 0:
        print("（沒有需要改的：可能已在地化，或那些 config 沒被拉回來）")


if __name__ == "__main__":
    main()
