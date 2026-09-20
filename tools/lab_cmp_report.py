#!/usr/bin/env python
"""lab 對比家族的一頁式報表 —— 在 lab 上跑，一個指令印完所有臂。

為什麼要有這支：2026-09-13 使用者要求「跑完一批再一次統計比較，省 token」。
沒有它的話，每個臂要分別查 train_status / best_val / 台帳 / resolved config /
slot log，來回十幾次；有它就是一次。

它報四組數字，缺一不可：
```
品質    最後一次 val 的 PSNR / SSIM / LPIPS / 紋理比（**同一個 step** 才可比）
成本    顆數 N、峰值實佔 VRAM   <- 命題要的收益在這一欄，不是 PSNR
約束    cost_budget 臂要報**最終 Load vs 預算**
        （2026-09-13 實測：顆數平的時候 Load 仍可漲 173 倍 —— 約束只擋 add，
          擋不到 scale 成長 => 不報這個就不知道約束到底有沒有綁住）
組態    該臂與基準差在哪（從 resolved config 讀，不靠腳本名）
```
⚠ **`it/s` 不可用來比**：lab 三槽平行，吞吐量取決於鄰居在做什麼。只印出來當參考。
⚠ 只比**相同 step** 的 val。不同步數的兩個跑次不可並列（收割期會改變名次 —— trimvpc
  在 10,960 輸 0.57 dB，到 21,920 變平手）。

用法: python tools/lab_cmp_report.py [--blk 6] [--prefix cs_]
"""
import argparse
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = re.compile(r"val step([\d,]+): psnr([\d.]+) ssim([\d.]+) lpips([\d.]+) tex([\d.]+)")
STEP = re.compile(r"^step ([\d,]+)/([\d,]+) \((\d+)%\)", re.M)


def num(s):
    return int(s.replace(",", ""))


def vals(path):
    """回傳 {step: (psnr, ssim, lpips, tex)}。"""
    out = {}
    if not os.path.exists(path):
        return out
    for m in VAL.finditer(open(path, errors="ignore").read()):
        out[num(m.group(1))] = tuple(float(m.group(i)) for i in range(2, 6))
    return out


def progress(path):
    if not os.path.exists(path):
        return None
    m = STEP.search(open(path, errors="ignore").read())
    return (num(m.group(1)), num(m.group(2)), int(m.group(3))) if m else None


def ledger_rows():
    """台帳所有 DONE 行的 (step, PSNR, N, peak, it/s)。

    ⛔⛔ **不可以用「同一行程的前一個 START」來配對。** lab 是**平行三槽**，
    START 與 DONE 在台帳裡是**交錯**的 => 前一個 START 通常是別的跑次的。
    2026-09-13 第一版就是這樣寫的，結果把 60k 跑次的 N=2.34M / 峰值 4.49 掛到
    21,920 步的 cs_base 上（它其實是 1.85M / 3.32）。
    正解：DONE 行自己帶 `step=` 與 `PSNR=@step`，用它去對跑次自己的 val => 唯一。"""
    p = os.path.join(ROOT, "logs", "quad_progress.log")
    rows = []
    if not os.path.exists(p):
        return rows
    for ln in open(p, errors="ignore"):
        if "| DONE" not in ln:
            continue
        d = {}
        for k, pat in (("N", r"N=([\d.]+)M"), ("its", r"([\d.]+)it/s"),
                       ("peak", r"峰值實佔([\d.]+)"), ("step", r"step=([\d,]+)/"),
                       ("psnr", r"PSNR=([\d.]+)@")):
            m = re.search(pat, ln)
            if m:
                d[k] = m.group(1)
        if "psnr" in d and "step" in d:
            rows.append(d)
    return rows


def best_psnr(bd):
    """best_val.txt 的 PSNR —— **四位小數**，是消歧的關鍵。

    ⛔ 不可以用 train_status 的 val 行來配台帳：它只印**兩位**小數（29.01 / 29.00），
    而 cs_base=29.0056 與 cs_trimvpc=29.0045 在兩位小數下差 0.01 => 容差一放寬就
    **兩行都匹配**，取最後一個就把 trimvpc 的 VRAM 掛到 base 頭上（2026-09-13 實際發生）。
    """
    p = os.path.join(bd, "best_val.txt")
    if not os.path.exists(p):
        return None
    m = re.search(r"best_val_psnr:\s*([\d.]+)", open(p, errors="ignore").read())
    return float(m.group(1)) if m else None


def match_ledger(rows, bd, pr):
    """用 (最終步數, best_val PSNR 四位小數) 認領台帳那一行。
    **配不出唯一解就回空** —— 寧可印「—」也不要掛錯人。"""
    bp = best_psnr(bd)
    if not pr or bp is None:
        return {}
    st = pr[0]
    hit = [r for r in rows if num(r["step"]) == st and abs(float(r["psnr"]) - bp) < 0.0015]
    return hit[0] if len(hit) == 1 else {}


def cfg_flags(run_dir):
    """從 resolved config 讀「這個臂到底開了什麼」——不靠腳本名或跑次名。"""
    g = glob.glob(os.path.join(run_dir, "lightning_logs", "version_*", "config.yaml"))
    if not g:
        return {}
    txt = open(sorted(g)[-1], errors="ignore").read()
    keys = ["cost_add_densify", "cost_budget", "trim_by_value_per_cost",
            "trim_value_per_cost_alpha", "lambda_dssim", "cap_max",
            "densify_until_iter", "max_steps"]
    d = {}
    for k in keys:
        m = re.search(rf"^\s*{k}:\s*(\S+)", txt, re.M)
        if m:
            d[k] = m.group(1)
    return d


def last_load(label_run):
    """從 slot log 撈該跑次最後一次 `[cost-budget]` —— 約束到底有沒有綁住。"""
    best = None
    for p in glob.glob(os.path.join(ROOT, "logs", "runner.slot*.log")):
        sect = None
        for ln in open(p, errors="ignore"):
            if ln.startswith("====="):
                sect = ln
            elif "[cost-budget]" in ln and sect and label_run in sect:
                best = ln.strip()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blk", type=int, default=6)
    ap.add_argument("--prefix", default="cs_")
    a = ap.parse_args()

    led = ledger_rows()
    runs = sorted(glob.glob(os.path.join(ROOT, "outputs", "lab", a.prefix + "*")))
    rows = []
    for d in runs:
        bd = os.path.join(d, f"blocks/block_{a.blk}")
        if not os.path.isdir(bd):
            continue
        rows.append((os.path.basename(d), bd, vals(os.path.join(bd, "train_status.txt")),
                     progress(os.path.join(bd, "train_status.txt")),
                     None, cfg_flags(bd)))
    rows = [(n, b, v, pr, match_ledger(led, b, pr), cf) for n, b, v, pr, _, cf in rows]
    if not rows:
        print(f"outputs/lab/{a.prefix}* 底下找不到 block_{a.blk}")
        return

    # 只在**所有已完賽跑次都有**的那個 step 上比較
    done = [r for r in rows if r[3] and r[3][2] == 100]
    common = None
    for r in done:
        common = set(r[2]) if common is None else (common & set(r[2]))
    cmp_step = max(common - {0}) if common and (common - {0}) else None
    if cmp_step is None:
        # ⚠ 沒有共同的比較點時**不要**各報各的最後一次 val —— 那會把 step 0 的 12.81
        #   跟 21,920 的 29.01 並排（第一版就是這樣，正是本工具要防的那種錯）。
        print("\n⚠⚠ 目前沒有「所有完賽跑次都有」的共同 val 步數 => 品質欄只印進度，不並列分數。")

    print(f"\n═══ lab 對比家族 block {a.blk} ═══   完賽 {len(done)}/{len(rows)}"
          f"   比較點 step {cmp_step:,}\n" if cmp_step else
          f"\n═══ lab 對比家族 block {a.blk} ═══   完賽 {len(done)}/{len(rows)}\n")
    hdr = f"{'臂':<16}{'進度':>12} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'紋理比':>7} {'N':>7} {'峰值GB':>7} {'it/s':>6}"
    print(hdr); print("-" * len(hdr))
    base = None
    for name, bd, v, pr, lg, _ in rows:
        p = f"{pr[0]:,}/{pr[1]:,}" if pr else "?"
        vv = v.get(cmp_step) if cmp_step else None
        q = (f"{vv[0]:>8.3f} {vv[1]:>7.3f} {vv[2]:>7.3f} {vv[3]:>7.3f}"
             if vv else f"{'—':>8} {'—':>7} {'—':>7} {'—':>7}")
        print(f"{name:<16}{p:>12} {q} {lg.get('N','—'):>7} "
              f"{lg.get('peak','—'):>7} {lg.get('its','—'):>6}")
        if name.endswith("_base"):
            base = vv
    if base:
        print("\n── 與基準的差（噪音底：PSNR 3sd=0.24 / SSIM 3sd=0.0015）──")
        for name, bd, v, pr, lg, _ in rows:
            vv = v.get(cmp_step) if cmp_step else None
            if not vv or name.endswith("_base"):
                continue
            d = [vv[i] - base[i] for i in range(4)]
            flag = "平手" if abs(d[0]) < 0.24 else ("**贏**" if d[0] > 0 else "輸")
            print(f"{name:<16} ΔPSNR {d[0]:+7.3f} {flag:<6} ΔSSIM {d[1]:+7.4f} "
                  f"ΔLPIPS {d[2]:+7.4f} Δ紋理比 {d[3]:+7.4f}")

    print("\n── 組態：這個臂到底開了什麼（讀 resolved config，不靠跑次名）──")
    for name, bd, v, pr, lg, cf in rows:
        keys = ["cost_add_densify", "cost_budget", "trim_by_value_per_cost", "lambda_dssim"]
        # alpha 只有在 trim 判準真的開著時才有意義（它的預設值 1.0 不代表「開了」）
        if cf.get("trim_by_value_per_cost", "false").lower() == "true":
            keys.insert(3, "trim_value_per_cost_alpha")
        on = {k: cf[k] for k in keys
              if k in cf and cf[k] not in ("0.0", "0", "false", "False")}
        print(f"{name:<16} " + ("、".join(f"{k}={x}" for k, x in on.items()) or "（基準）"))

    print("\n── 約束端：最終 Load vs 預算（只有開 cost_budget 的臂會有）──")
    any_load = False
    for name, bd, v, pr, lg, cf in rows:
        if cf.get("cost_budget", "0.0") in ("0.0", "0"):
            continue
        any_load = True
        print(f"{name:<16} {last_load(name) or '⚠ 沒撈到 [cost-budget] 行（該臂沒開 cost_budget_report）'}")
    if not any_load:
        print("（沒有開 cost_budget 的臂）")
    print("""
⚠ `it/s` **不可用來比** —— lab 三槽平行，吞吐量取決於鄰居。
⚠ 只比同一個 step 的 val；收割期會改變名次（trimvpc 在 10,960 輸 0.57 dB，21,920 變平手）。
⚠ 命題的收益在**峰值GB**那一欄，不是 PSNR：PSNR 平手 + VRAM 下降就是勝。""")


if __name__ == "__main__":
    main()
