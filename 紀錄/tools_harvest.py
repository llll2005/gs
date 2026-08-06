# -*- coding: utf-8 -*-
"""Scan outputs/**/results.txt -> 紀錄/harvest.csv (input of tools_make_master_csv.py).

Rebuilt 2026-07-17 (the original one-off harvester was lost in the 07-06 cleanup).
Rows: date (results.txt mtime), run (outputs subdir), block (from blocks/block_N/
path or trailing _bN in run name), psnr/ssim/lpips (val/*).
"""
import csv
import datetime
import glob
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harvest.csv")


def parse_results(path):
    vals = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*([\w/]+):\s*([-\d.eE]+)", line)
            if m:
                vals[m.group(1)] = m.group(2)
    return vals


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "outputs", "**", "results.txt"), recursive=True)):
        rel = os.path.relpath(path, os.path.join(REPO, "outputs"))
        parts = rel.split(os.sep)
        run = parts[0]
        block = ""
        m = re.search(r"blocks[/\\]block_(\d+)", rel)
        if m:
            block = m.group(1)
        else:
            m = re.search(r"_b(\d+)$", run)
            if m:
                block = m.group(1)
        vals = parse_results(path)
        date = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
        rows.append({
            "date": date, "run": run, "block": block,
            "psnr": vals.get("val/psnr", ""),
            "ssim": vals.get("val/ssim", ""),
            "lpips": vals.get("val/lpips", ""),
        })
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "run", "block", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
