"""每個 run 到底跑完了沒 —— 比較任何 PSNR 之前先跑這支。

為什麼需要：`results.txt` 只寫指標，**不寫步數**。半途 OOM 死掉的跑次一樣會留下一個
看起來正常的 PSNR。2026-08-16 我拿 `mcmc_2dgs_60k_sh3_aggr17_aerial` 的 8 塊擬「內容密度
vs PSNR」，擬完才發現**只有 4 塊真的跑到 60k**（block 2/3 死在 1499、block 7 死在 14999、
block 5 死在 29999），那條斜率其實在描述「哪些塊會死」，不是內容難度。整個模型與由它
外推的合併分數推估全部作廢。

這是同一個陷阱的第二次（見記憶 feedback_metric_resolution：「拿中途死掉的跑次當已知的
壞模型去校驗指標」）。`logs/quad_progress.log` 對**現行**跑次有 DONE/DIED，但舊跑次沒有，
而 ckpt 檔名裡的 step 是一直都在、可回溯的權威。

用法：python tools/run_status.py [--runs a b c] [--block 12] [--all]
"""
import argparse, glob, os, re, sys

def final_step(block_dir):
    steps = [int(m.group(1)) for p in glob.glob(block_dir + "/checkpoints/*.ckpt")
             if (m := re.search(r"step=(\d+)", p))]
    return max(steps) if steps else None

def max_steps_of(run_dir):
    for cfg in glob.glob(run_dir + "/**/config.yaml", recursive=True):
        m = re.search(r"^\s*max_steps:\s*([0-9_]+)", open(cfg).read(), re.M)
        if m:
            return int(m.group(1).replace("_", ""))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="掃 outputs/ 底下全部")
    ap.add_argument("--block", type=int, default=None, help="只看某一塊；預設每塊都列")
    a = ap.parse_args()
    runs = a.runs or (sorted(os.path.basename(d) for d in glob.glob("outputs/*")
                             if os.path.isdir(d + "/blocks")) if a.all else None)
    if not runs:
        ap.error("給 --runs 或 --all")
    print("%-30s %-8s %9s %9s  %s" % ("run", "block", "步數", "應到", "PSNR"))
    bad = 0
    for run in runs:
        for bd in sorted(glob.glob("outputs/%s/blocks/block_*" % run),
                         key=lambda p: int(re.search(r"block_(\d+)", p).group(1))):
            b = int(re.search(r"block_(\d+)", bd).group(1))
            if a.block is not None and b != a.block:
                continue
            s = final_step(bd); mx = max_steps_of("outputs/" + run)
            f = bd + "/results.txt"
            m = re.search(r"val/psnr: ([0-9.]+)", open(f).read()) if os.path.exists(f) else None
            ok = s is not None and mx is not None and s >= mx - 1
            bad += (not ok)
            print("%-30s %-8d %9s %9s  %-8s%s" % (run[:30], b, s or "無", mx or "?",
                  ("%.3f" % float(m.group(1))) if m else "-", "" if ok else "   ⚠ 沒跑完，PSNR 不可比"))
    if bad:
        print("\n  ⚠ %d 筆沒跑完。它們的 PSNR **不可以**跟跑完的放在同一張表比較，" % bad)
        print("    也不可以拿去擬任何關係 —— 那會擬到「哪些跑次會死」而不是你以為的變數。")

if __name__ == "__main__":
    main()
