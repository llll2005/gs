#!/usr/bin/env python
"""圖表資料包的彙整表（2026-10-03）：把 outputs/**/chart_data/*.txt（scripts/lab/post_run_data.sh 產生）
解析成一張表，畫 `紀錄/實驗分析/` 的圖時從這裡取數字，不必回頭重量。

欄位（缺的留空）：
  跑次 塊 終點步數 | val PSNR SSIM LPIPS 紋理比（train_status 最後一個 val）
  | 精確 Load 中位 精確 Load max 代理 Load 中位 代理 Load 平均 | held-out PSNR SSIM LPIPS 紋理比 ms／幀
  | held-out@1499 PSNR | slope corr | slope@1499 corr@1499 | 失敗率% | tau_mean | ckpt GB N
  | fwd ms  fwd+bwd ms（[solo]）
用法：python tools/chart_data_summary.py [--filter 子字串] [--tsv 輸出路徑]
"""
import argparse
import glob
import os
import re

NUM = r"([-+]?[\d,]*\.?\d+)"


def num(s):
    return float(s.replace(",", "")) if s is not None else None


def grab(text, pat, k=1):
    m = re.search(pat, text, re.M)
    return num(m.group(k)) if m else None


def read(p):
    return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""


def heldout(t):
    return (grab(t, r"^PSNR\s+" + NUM), grab(t, r"^SSIM\s+" + NUM), grab(t, r"^LPIPS\s+" + NUM),
            grab(t, r"^紋理比 逐張平均\s+" + NUM), grab(t, r"^渲染\s+" + NUM + r"\s*ms"))


def geom(t):
    return grab(t, r"slope \(渲染/真實\) = " + NUM), grab(t, r"corr = " + NUM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--tsv", default="")
    a = ap.parse_args()
    cols = ["跑次", "塊", "步數", "valPSNR", "valSSIM", "valLPIPS", "val紋理比",
            "精確Load中位", "精確Loadmax", "代理Load中位", "代理Load平均",
            "hoPSNR", "hoSSIM", "hoLPIPS", "ho紋理比", "ho_ms(非獨佔不準)", "ho@1499PSNR",
            "slope", "corr", "slope@1499", "corr@1499", "失敗率%", "tau_mean", "ckptGB", "N", "fwd_ms", "fwdbwd_ms"]
    rows = []
    for cd in sorted(glob.glob("outputs/**/blocks/block_*/chart_data", recursive=True)):
        bd = os.path.dirname(cd)
        run = bd.split("outputs/", 1)[1].rsplit("/blocks/", 1)[0]
        if a.filter and a.filter not in run:
            continue
        blk = bd.rsplit("block_", 1)[1]
        ts = read(os.path.join(bd, "train_status.txt"))
        vals = re.findall(r"val step([\d,]+): psnr" + NUM + r" ssim" + NUM + r" lpips" + NUM + r" tex" + NUM, ts)
        v = vals[-1] if vals else (None,) * 5
        L = read(os.path.join(cd, "load.txt"))
        ex = re.search(r"③ 精確 Σtiles（光柵器）\s+" + NUM + r"\s+" + NUM, L)
        ho, ho1 = heldout(read(os.path.join(cd, "heldout.txt"))), heldout(read(os.path.join(cd, "heldout1499.txt")))
        g, g1 = geom(read(os.path.join(cd, "geom.txt"))), geom(read(os.path.join(cd, "geom1499.txt")))
        S = read(os.path.join(cd, "storage.txt"))
        sm = re.search(r"檔案 ([\d.]+) (GB|MB)\s+N = ([\d,]+)", S)
        T = read(os.path.join(cd, "timing.txt"))
        rows.append([run, blk, num(v[0]) if v[0] else None, *(num(x) if x else None for x in v[1:]),
                     num(ex.group(2)) if ex else None, num(ex.group(1)) if ex else None,   # 報告欄序是 max、中位
                     grab(L, r"^\s+中位\s+" + NUM), grab(L, r"^\s+平均\s+" + NUM),
                     *ho, ho1[0], *g, *g1,
                     grab(read(os.path.join(cd, "failure.txt")), r"失敗率 \*\*" + NUM),
                     grab(read(os.path.join(cd, "tau.txt")), r"tau_mean = " + NUM),
                     (num(sm.group(1)) / (1024 if sm.group(2) == "MB" else 1)) if sm else None,
                     num(sm.group(3)) if sm else None,
                     grab(T, r"光柵化 forward\s+" + NUM), grab(T, r"forward\+backward\(L1\)\s+" + NUM)])

    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"
        return str(x)
    lines = ["\t".join(cols)] + ["\t".join(fmt(x) for x in r) for r in rows]
    print("\n".join(lines))
    if a.tsv:
        open(a.tsv, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"-> {a.tsv}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
