#!/usr/bin/env python
"""兩個跑次的 **resolved config** 差在哪 —— 比分數之前必跑（純 CPU，秒級）。

## 為什麼需要這支

`speed3_sfminit_b12` 是**手動啟動**的，以為唯一變數是 `initialize_from`，
實際 resolved config 差**四項**：
```
initialize_from      depth_init.ply -> null      <- 想測的變數
dynamic_strips       false -> **true**            <- 污染源
strip_vram_target_gb 5.4 -> 5.2
strip_safety         0.6 -> 0.85
skip_surf_normal     false -> true
```
而 `dynamic_strips=true` 讓 **77.7% 的訓練步跑在被改過的 loss 上**
（逐條帶 SSIM 是 docstring 自承的 "boundary-window approximation"，K=6 時 SSIM 偏高到
噪音底的 11 倍），且 absgrad 只讀得到最後一條帶 ⇒ **那個 60k 跑次的分數全部作廢**。
一整趟 9 小時的 GPU，因為沒做這個秒級的檢查而白費。

⇒ 規則（記憶 `dynamic_strips_confound`）：**比較前 diff 兩份 resolved config。**

## 用法

```
python tools/diff_resolved_config.py speed3:13 gate15000:12
python tools/diff_resolved_config.py lab/speed3:6 lab/costbudget:6 --expect 1
```
`--expect N` = 宣稱的變數個數；實際差異超過 N 就 **exit 1**（可放進腳本當守門）。

## 讀哪個檔

`outputs/<run>/blocks/block_<N>/lightning_logs/version_*/config.yaml`，取 version 最大的那個
—— 那是 Lightning 寫下的**實際生效值**，不是我們下的旗標。
⚠ 差別很重要：CLI 旗標打錯字會被 jsonargparse 吃掉而不報錯，只有 resolved config 看得出來。
"""
import argparse
import glob
import os
import re
import sys

# 這些鍵**本來就會**不同，不是污染；但仍會列在最後讓人看到，不會靜默藏起來
EXPECTED = (
    "name", "version", "output_path", "ckpt_path", "save_dir",
    "data.parser.init_args.block_id", "model.initialize_from",
    "trainer.logger", "trainer.callbacks",
)


def latest_config(spec):
    """spec = '<run>' 或 '<run>:<block>'"""
    run, _, blk = spec.partition(":")
    pats = []
    if blk:
        pats.append(f"outputs/{run}/blocks/block_{blk}/lightning_logs/version_*/config.yaml")
    pats.append(f"outputs/{run}/blocks/block_*/lightning_logs/version_*/config.yaml")
    pats.append(f"outputs/{run}/lightning_logs/version_*/config.yaml")
    for pat in pats:
        hits = glob.glob(pat)
        if not hits:
            continue
        if len(set(os.path.dirname(os.path.dirname(h)) for h in hits)) > 1 and not blk:
            raise SystemExit(f"⛔ {spec} 有多個 block，請指定：{spec}:<block>\n  "
                             + "\n  ".join(sorted(hits)))
        def ver(h):
            m = re.search(r"version_(\d+)", h)
            return int(m.group(1)) if m else -1
        return max(hits, key=ver)
    raise SystemExit(f"⛔ 找不到 {spec} 的 resolved config（outputs/{run}/...）")


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        # list 直接當成一個值比對：順序與長度的改變都算差異
        out[prefix] = repr(obj)
    else:
        out[prefix] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="跑次 A，格式 <run>[:<block>]")
    ap.add_argument("b", help="跑次 B，格式 <run>[:<block>]")
    ap.add_argument("--expect", type=int, default=None,
                    help="宣稱的變數個數；實際差異超過它就 exit 1")
    ap.add_argument("--all", action="store_true", help="連「本來就會不同」的鍵也一起算進差異")
    args = ap.parse_args()

    import yaml
    pa, pb = latest_config(args.a), latest_config(args.b)
    print(f"A  {pa}")
    print(f"B  {pb}")
    fa = flatten(yaml.safe_load(open(pa, encoding="utf-8")))
    fb = flatten(yaml.safe_load(open(pb, encoding="utf-8")))

    keys = sorted(set(fa) | set(fb))
    real, expected = [], []
    for k in keys:
        va, vb = fa.get(k, "<缺>"), fb.get(k, "<缺>")
        if repr(va) == repr(vb):
            continue
        (expected if (not args.all and any(k == e or k.endswith("." + e) or e in k
                                           for e in EXPECTED)) else real).append((k, va, vb))

    def show(title, rows):
        print(f"\n{title}（{len(rows)} 項）")
        if not rows:
            print("    （無）")
        w = max([len(k) for k, _, _ in rows], default=10)
        for k, va, vb in rows:
            print(f"    {k:<{w}}  {va!r}  ->  {vb!r}")

    show("★ 實質差異", real)
    show("（本來就會不同，僅供核對）", expected)

    print(f"\n實質差異 = **{len(real)}** 項")
    if args.expect is not None:
        if len(real) > args.expect:
            print(f"⛔ 宣稱只有 {args.expect} 個變數，實際差 {len(real)} 項 "
                  f"=> **不是單變數，比較無效**（speed3_sfminit_b12 就是這樣廢掉的）")
            return 1
        print(f"✔ 與宣稱的 {args.expect} 個變數一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
