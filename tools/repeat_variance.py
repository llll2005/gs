#!/usr/bin/env python
"""找出「組態相同的重複跑次」，量真正的 run-to-run 變異。

起因（2026-08-26）：`err_guided_densify` 這個旗標**從來沒有被實作**（全 repo 只出現在
自己的欄位定義與 docstring 裡，沒有任何一行程式碼讀它）=> `egd_b12` 其實就是
`sched30_b12` 又跑了一次。兩者實測差 PSNR **0.106**，而 `_ctx.md` 記載的噪音底是 0.0325
（n=2 單對，自述「是差值的一次抽樣不是 sigma」）=> **低估 3.3 倍**。

若 sigma_PSNR ≈ 0.1，今天所有的推論都要重算，例如
  notrim2 vs sched30 +0.158 = 1.5 sigma（不顯著）
  sched30 vs noprior +0.327 = 3.1 sigma（勉強，而這是「本 session 最強機制」）

本工具掃描 `outputs/`，把「除了無效旗標之外組態完全相同」的跑次分組，報每組的全距。
⚠ 只納入 GT 修正後（2026-08-12 10:08:37）且跑到 max_steps 的跑次 —— 跨世代不可混。
"""
import argparse
import glob
import os
import re

import numpy as np
import yaml
from tensorboard.backend.event_processing import event_accumulator as ea

GT_FIX_TS = 1786529317
# 這些鍵不影響行為：純診斷旗標、未實作的旗標、路徑、以及「舊 config 沒有但新版有預設值」的欄位
# ⚠ 只能忽略「確認未實作」或「純診斷」的鍵。2026-08-26 第一版誤把 err_unlock /
# transparent_corrector 放進來（兩者**有實作**），結果把 errunlock_b12 併進 sched30 那組，
# 量出 0.635 的假全距。實作與否用
#   grep -rn '"<flag>"\|<flag>' internal/ | grep -v docstring
# 逐個確認過：err_guided_densify **零引用**（寫了文件從未實作）；add_ratio 是
# getattr(self.config,"add_ratio",1.05) 存取的，**有實作**，不可忽略。
IGNORE = ("name", "version", "dir", "logger", "ckpt", "seed", "output_path", "save_dir",
          "absgrad_report", "vpc_report", "err_guided_densify", "path")
METRICS = ["val/psnr", "val/ssim", "val/lpips", "val/texratio"]


def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(d, list):
        o[p] = str(d)
    else:
        o[p] = d
    return o


def load(run):
    evs = sorted(glob.glob(f"outputs/{run}/**/events.out.tfevents.*", recursive=True))
    if not evs:
        return None
    m = re.search(r"tfevents\.(\d+)\.", os.path.basename(evs[-1]))
    t0 = int(m.group(1)) if m else 0
    a = ea.EventAccumulator(evs[-1], size_guidance={ea.SCALARS: 0})
    a.Reload()
    tags = set(a.Tags()["scalars"])
    if "val/psnr" not in tags:
        return None
    out = {t: a.Scalars(t)[-1].value for t in METRICS if t in tags}
    out["_step"] = a.Scalars("val/psnr")[-1].step
    out["_t0"] = t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-step", type=int, default=59000, help="只納入跑到這麼多步的")
    ap.add_argument("--min-t0", type=int, default=GT_FIX_TS)
    args = ap.parse_args()

    # ⚠ 不可用「雜湊整份 config」分組：舊跑次的 config 缺新欄位（None），新的有預設值（0.0），
    # 文字不同但功能相同。也不可「一律忽略 0」—— `opacity_reg: 0.0`、`lambda_normal: 0.0`
    # 都是有意義的設定。=> **逐對比較兩者共有且皆非 None 的鍵**（120 個跑次只有 7 千對，很便宜）。
    cfgs = {}
    for c in glob.glob("outputs/*/blocks/block_*/lightning_logs/version_*/config.yaml"):
        run = c.split("/")[1]
        try:
            f = flat(yaml.safe_load(open(c)))
        except Exception:
            continue
        cfgs[run] = {k: v for k, v in f.items()
                     if not any(i in k.lower() for i in IGNORE) and v is not None}

    ok = {}
    for r in cfgs:
        d = load(r)
        if d and d["_step"] >= args.min_step and d["_t0"] >= args.min_t0:
            ok[r] = d
    names = sorted(ok)

    def same(a, b):
        # ⚠ 「跳過 None」讓「未設定」與任何值相容 => 曾把 sfminit_b12（initialize_from: null，
        # 被濾掉）誤配成 coarseft_b12（ckpt 路徑）。這幾個鍵即使是 None 也有語意，必須比。
        for k in ("model.initialize_from",):
            if str(cfgs[a].get(k)) != str(cfgs[b].get(k)):
                return False
        shared = set(cfgs[a]) & set(cfgs[b])
        return bool(shared) and all(str(cfgs[a][k]) == str(cfgs[b][k]) for k in shared)

    seen, groups = set(), []
    for i, a in enumerate(names):
        if a in seen:
            continue
        grp = [a] + [b for b in names[i+1:] if b not in seen and same(a, b)]
        if len(grp) >= 2:
            groups.append([(r, ok[r]) for r in grp])
            seen.update(grp)

    if not groups:
        print("沒有找到同世代、跑滿、組態相同的重複跑次")
        return
    print(f"同世代（GT 修正後）+ 跑滿 + 組態相同的組別：{len(groups)}\n")
    spreads = {t: [] for t in METRICS}
    for vals in groups:
        print("  " + " | ".join(f"{r} {d['val/psnr']:.3f}" for r, d in vals))
        for t in METRICS:
            xs = [d[t] for _, d in vals if t in d]
            if len(xs) >= 2:
                spreads[t].append(max(xs) - min(xs))
                if t == "val/psnr":
                    print(f"      全距 PSNR {max(xs)-min(xs):.3f}")
    print("\n=== 實測 run-to-run 全距 vs `_ctx.md` 記載的噪音底 ===")
    doc = {"val/psnr": 0.0325, "val/ssim": 0.00091, "val/lpips": 0.00200, "val/texratio": 0.00314}
    print(f"{'指標':>10} {'n':>3} {'中位全距':>10} {'最大全距':>10} {'文件值':>10} {'中位/文件':>9}")
    for t in METRICS:
        s = spreads[t]
        if not s:
            continue
        print(f"{t.split('/')[1]:>10} {len(s):>3} {np.median(s):>10.4f} {max(s):>10.4f} "
              f"{doc[t]:>10.5f} {np.median(s)/doc[t]:>8.1f}x")
    print("""
⚠ 全距不是 sigma（n 很小時全距系統性低估 sigma）。這是**下界**：
   真正的 run-to-run 標準差至少這麼大。任何小於它的「效應」都不可宣稱。""")


if __name__ == "__main__":
    main()
