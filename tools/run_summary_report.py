"""★ 2026-11-14 彙整每塊的 run_summary.json（demo 模式不產生 log，論文要的時間／顆數從這裡取）。
用法：python tools/run_summary_report.py <跑次目錄或名稱>（例：lab/g44tn_..._best1maskdistrfed10b4 或 outputs/lab/...）
欄位：啟動＝行程開始 → 第一步；純訓練＝第一步 → 最後一步；存檔＝最後一步 → 行程結束；任務＝shell 的 conda run 前 → python 結束。
分數不在這裡：單塊用 task_tune25.sh eval（換塊、F1…），全場景用 task_fullscene.sh 的合併＋評估（評估不算訓練時間）。
"""
import glob
import json
import os
import sys


def main():
    r = sys.argv[1].rstrip("/")
    d = r if r.startswith("outputs/") or os.path.isabs(r) else f"outputs/{r}"
    fs = sorted(glob.glob(f"{d}/blocks/block_*/run_summary.json"), key=lambda p: int(p.split("block_")[-1].split("/")[0]))
    if not fs:
        print(f"⛔ {d}/blocks/block_*/run_summary.json 一個都沒有"); sys.exit(2)
    rows = [(int(f.split("block_")[-1].split("/")[0]), json.load(open(f, encoding="utf-8"))) for f in fs]
    g = lambda s, k: s.get(k) if s.get(k) is not None else float("nan")
    print(f"{d}｜{len(rows)} 塊｜demo 模式：{sorted({bool(s.get('demo')) for _, s in rows})}")
    print(f"{'塊':>3} {'步數':>7} {'最終顆數':>11} {'啟動 s':>8} {'純訓練 s':>9} {'存檔 s':>7} {'行程 s':>8} {'任務 s':>8} {'峰值 GB':>8}")
    tot = dict(n=0, st=0.0, tr=0.0, sv=0.0, py=0.0, tk=0.0); pk = 0.0
    for b, s in rows:
        print(f"{b:>3} {g(s, 'steps'):>7} {g(s, 'n_final'):>11,} {g(s, 'startup_s'):>8.1f} {g(s, 'train_s'):>9.1f} {g(s, 'save_exit_s'):>7.1f} "
              f"{g(s, 'python_total_s'):>8.1f} {g(s, 'task_total_s'):>8.1f} {g(s, 'peak_alloc_gb'):>8.2f}")
        tot["n"] += s.get("n_final") or 0
        for k, kk in (("st", "startup_s"), ("tr", "train_s"), ("sv", "save_exit_s"), ("py", "python_total_s"), ("tk", "task_total_s")):
            tot[k] += s.get(kk) or 0.0
        pk = max(pk, s.get("peak_alloc_gb") or 0.0)
    print(f"{'合計':>3} {'':>7} {tot['n']:>11,} {tot['st']:>8.1f} {tot['tr']:>9.1f} {tot['sv']:>7.1f} {tot['py']:>8.1f} {tot['tk']:>8.1f} {pk:>8.2f}（峰值取最大）")
    print(f"=> 單卡依序跑完全部塊：純訓練 {tot['tr'] / 3600:.2f} h、含啟動與存檔 {tot['py'] / 3600:.2f} h、整個訓練任務 {tot['tk'] / 3600:.2f} h（不含合併與評估）")


if __name__ == "__main__":
    main()
