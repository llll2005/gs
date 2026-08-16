"""把 紀錄/實驗總表.csv 的 era/note 依實際檔案時間與逐塊完成度回填。

三個失效世代（分界見 研究總覽 §12.15）：
  舊資料集  最後檔案 < 2026-05-29           SfM 重生，座標系不同
  GT錯位    **最早**檔案 < 2026-08-12 10:08:37（faeb4d4）  每張影像對到鄰幀
  修正後    其餘

⚠ 用「最早」而非「最後」判世代：b3x3_full / sh3blur_b12 的最後檔案在修正後，但都是修正前開跑的。
⚠ 完成度要**逐塊**判。第一版我按 run 判，結果 citygsv2_mc_aerial_sh0_trim 只有 1 塊沒跑完
  卻把 25 列全標成未跑完。
⚠ ckpt 已於 2026-08-17 刪除失效世代（保留其他檔案），所以步數改讀同名的 *.ply。
"""
import csv, datetime, glob, os, re, shutil, collections
FIX = datetime.datetime(2026, 8, 12, 10, 8, 37).timestamp()
OLD = datetime.datetime(2026, 5, 29).timestamp()
pat = re.compile(r"step=(\d+)")
info, blk_ok = {}, {}
for rd in sorted(glob.glob("outputs/*")):
    if not os.path.isdir(rd): continue
    run = os.path.basename(rd)
    fs = glob.glob(rd + "/**/*.ply", recursive=True) + glob.glob(rd + "/**/events*", recursive=True) \
         + glob.glob(rd + "/**/*.ckpt", recursive=True)
    if not fs: continue
    t0, t1 = min(os.path.getmtime(p) for p in fs), max(os.path.getmtime(p) for p in fs)
    era = "舊資料集" if t1 < OLD else ("GT錯位" if t0 < FIX else "修正後")
    mx = None
    for cfg in glob.glob(rd + "/**/config.yaml", recursive=True):
        m = re.search(r"^\s*max_steps:\s*([0-9_]+)", open(cfg).read(), re.M)
        if m: mx = int(m.group(1).replace("_", "")); break
    info[run] = (era, len(glob.glob(rd + "/**/*.ckpt", recursive=True)) == 0)
    for b in sorted(glob.glob(rd + "/blocks/block_*")):
        bid = re.search(r"block_(\d+)", b).group(1)
        st = [int(pat.search(p).group(1)) for p in glob.glob(b + "/**/*", recursive=True) if pat.search(p)]
        blk_ok[(run, bid)] = (not st) or (mx is None) or (max(st) >= mx - 1)
src = "紀錄/實驗總表.csv"; shutil.copy(src, src + ".bak")
rows = list(csv.DictReader(open(src))); cnt = collections.Counter()
FIELDS = list(rows[0].keys())

# outputs/ 有但 CSV 沒有的 run -> 補列（從 results.txt 收指標）
known = {(r["run"], (r["block"] or "").strip()) for r in rows}
for run, (era, purged) in sorted(info.items()):
    for bd in sorted(glob.glob("outputs/%s/blocks/block_*" % run)) or [None]:
        bid = re.search(r"block_(\d+)", bd).group(1) if bd else ""
        if (run, bid) in known: continue
        m = {}
        f = (bd or ("outputs/" + run)) + "/results.txt"
        if os.path.exists(f):
            for k in ["psnr", "ssim", "lpips"]:
                g = re.search(r"val/%s: ([0-9.]+)" % k, open(f).read())
                if g: m[k] = "%.4f" % float(g.group(1))
        fs = glob.glob((bd or ("outputs/" + run)) + "/**/*", recursive=True)
        d = datetime.datetime.fromtimestamp(min((os.path.getmtime(x) for x in fs), default=0)).strftime("%Y-%m-%d")
        rows.append({**{k: "" for k in FIELDS}, "date": d, "run": run, "block": bid,
                     "psnr": m.get("psnr", ""), "ssim": m.get("ssim", ""), "lpips": m.get("lpips", ""),
                     "era": era, "status": "待補", "note": "2026-08-17 自動補列"})
        cnt["自動補列"] += 1
for r in rows:
    run, bid = r["run"], (r["block"] or "").strip()
    if run not in info:
        # outputs/ 已無此 run（清理或改名）=> 只能用 date 欄分類，仍要分清世代
        try:
            ts = datetime.datetime.strptime(r["date"][:10], "%Y-%m-%d").timestamp()
        except Exception:
            cnt["無法分類(無日期)"] += 1; continue
        r["era"] = "舊資料集" if ts < OLD else ("GT錯位" if ts < FIX else "修正後")
        cnt["依日期分類(outputs已無)"] += 1
        tag = {"舊資料集": "舊SfM座標系，數據作廢", "GT錯位": "GT錯位期(影像對到鄰幀)，數據作廢"}.get(r["era"])
        if tag and tag not in r["note"]: r["note"] = (r["note"] + "; " + tag).strip("; ")
        continue
    era, purged = info[run]
    r["era"] = era; cnt[era] += 1
    tags = []
    if era == "GT錯位": tags.append("GT錯位期(影像對到鄰幀)，數據作廢")
    if era == "舊資料集": tags.append("舊SfM座標系，數據作廢")
    if blk_ok.get((run, bid), True) is False:
        tags.append("此塊未跑完/OOM，PSNR不可比"); cnt["未跑完(逐塊)"] += 1
    if purged: tags.append("ckpt已刪2026-08-17(留其他檔案供復盤)")
    for t in tags:
        if t not in r["note"]: r["note"] = (r["note"] + "; " + t).strip("; ")
rows.sort(key=lambda r: (r["date"], r["run"], r["block"]))
w = csv.DictWriter(open(src, "w", newline=""), fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
for k, v in cnt.most_common(): print("  %-20s %d" % (k, v))
