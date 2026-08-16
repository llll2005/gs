"""刪除失效世代的 ckpt，保留資料夾內其他檔案（results/config/tensorboard/test圖/ply）供復盤。

三個失效世代：
  舊資料集    最後 ckpt < 2026-05-29        SfM 重生，座標系不同
  GT-bug     **開始** ckpt < 2026-08-12 10:08:37（faeb4d4）  每張影像對到鄰幀
  OOM/未跑完  最後 step < config 的 max_steps

⚠ 用「開始時刻」而非「最後修改時刻」分類 —— b3x3_full(34GB) 與 sh3blur_b12 的最後檔案在
  修正之後，但兩者都是修正前開跑的，按 mtime 會被誤判為有效。
⚠ 白名單制：只有明確判定為「修正後且跑完」的 run 才保留，其餘一律刪 ckpt。
⚠ 跳過 60 分鐘內動過的檔案（正在訓練的 run）。

用法：python tools/purge_invalid_ckpts.py            # dry-run
     python tools/purge_invalid_ckpts.py --apply    # 真的刪
"""
import argparse, datetime, glob, os, re, sys, time
FIX = datetime.datetime(2026, 8, 12, 10, 8, 37).timestamp()
OLD = datetime.datetime(2026, 5, 29).timestamp()
pat = re.compile(r"step=(\d+)")

def classify(run_dir):
    ck = glob.glob(run_dir + "/**/*.ckpt", recursive=True)
    if not ck:
        return None, []
    t0, t1 = min(os.path.getmtime(p) for p in ck), max(os.path.getmtime(p) for p in ck)
    if t1 < OLD:
        return "舊資料集", ck
    if t0 < FIX:
        return "GT-bug", ck
    mx = None
    for cfg in glob.glob(run_dir + "/**/config.yaml", recursive=True):
        m = re.search(r"^\s*max_steps:\s*([0-9_]+)", open(cfg).read(), re.M)
        if m:
            mx = int(m.group(1).replace("_", "")); break
    for b in (sorted(glob.glob(run_dir + "/blocks/block_*")) or [run_dir]):
        st = [int(pat.search(p).group(1)) for p in glob.glob(b + "/**/*.ckpt", recursive=True) if pat.search(p)]
        if st and mx and max(st) < mx - 1:
            return "OOM/未跑完", ck
    return None, ck

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    now = time.time()
    tot, keep, groups = 0, [], {}
    for rd in sorted(glob.glob("outputs/*")):
        if not os.path.isdir(rd):
            continue
        era, ck = classify(rd)
        run = os.path.basename(rd)
        if era is None:
            if ck: keep.append(run)
            continue
        if any(now - os.path.getmtime(p) < 3600 for p in ck):
            print("⏭  跳過 %s（60 分鐘內動過，可能正在訓練）" % run); keep.append(run); continue
        sz = sum(os.path.getsize(p) for p in ck)
        groups.setdefault(era, []).append((run, len(ck), sz)); tot += sz
    for era, g in groups.items():
        print("\n=== %s：%d run，%d 個 ckpt，%.1f GB ===" % (era, len(g), sum(x[1] for x in g), sum(x[2] for x in g)/2**30))
        for run, n, sz in sorted(g, key=lambda x: -x[2])[:6]:
            print("   %-44s %3d 個 %7.2f GB" % (run[:44], n, sz/2**30))
        if len(g) > 6: print("   ... 另 %d 個" % (len(g)-6))
    print("\n保留 ckpt 的 run（%d 個）：%s" % (len(keep), ", ".join(sorted(keep))))
    print("\n合計可釋放 %.1f GB" % (tot/2**30))
    if not a.apply:
        print("\n（dry-run。加 --apply 才會真的刪）"); return
    n = 0
    for rd in sorted(glob.glob("outputs/*")):
        if not os.path.isdir(rd): continue
        era, ck = classify(rd)
        if era is None or any(now - os.path.getmtime(p) < 3600 for p in ck): continue
        for p in ck:
            os.remove(p); n += 1
    print("\n已刪除 %d 個 ckpt 檔。" % n)

if __name__ == "__main__":
    main()
