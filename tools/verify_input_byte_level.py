#!/usr/bin/env python
"""`input/` 的每一張，到底是哪個 block 的哪一張？—— **位元級**比對（純 CPU）

官方 `transform_json2txt_mc_aerial.py` 用 `cp` 把 per-block 影像複製成 `0000.png..`，
所以 `input/` 的檔案與來源**逐位元相同** ⇒ 用「檔案大小 + md5」就能給出**確定**的對應，
比姿態比對強得多（姿態比對抓不到「兩個全域檔名共用同一組重複姿態」那一類，
見研究總覽 §16.13 的 5 對缺陷）。

用法:
  python tools/verify_input_byte_level.py                       # 全部 5,621 張
  python tools/verify_input_byte_level.py --only 000253.png,000897.png
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial")
    ap.add_argument("--only", default="", help="逗號分隔的檔名；空＝全部")
    ap.add_argument("--map", default="")
    args = ap.parse_args()
    D = args.data
    IN = os.path.join(D, "train", "block_all", "input")

    # 1) per-block 影像：先按大小分桶（PNG 大小離散度高 => 很強的過濾）
    by_size = defaultdict(list)
    nsrc = 0
    for n in range(1, 11):
        d = os.path.join(D, "train", f"block_{n}", "input")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".png"):
                continue
            p = os.path.join(d, f)
            by_size[os.path.getsize(p)].append((n, int(os.path.splitext(f)[0]), p))
            nsrc += 1
    print(f"per-block 影像 {nsrc:,} 張，{len(by_size):,} 種檔案大小")

    targets = sorted(os.listdir(IN)) if not args.only else args.only.split(",")
    print(f"要比對 {len(targets):,} 張\n")
    cache = {}
    res, amb, none_ = {}, [], []
    for k, t in enumerate(targets):
        tp = os.path.join(IN, t)
        sz = os.path.getsize(tp)
        cand = by_size.get(sz, [])
        if not cand:
            none_.append(t); continue
        if len(cand) == 1:
            res[t] = (cand[0][0], cand[0][1] + 1)      # frame_index 從 1 起算
        else:
            tm = md5(tp)
            hit = []
            for n, idx, p in cand:
                if p not in cache:
                    cache[p] = md5(p)
                if cache[p] == tm:
                    hit.append((n, idx + 1))
            if len(hit) == 1:
                res[t] = hit[0]
            elif not hit:
                none_.append(t)
            else:
                amb.append((t, hit)); res[t] = hit[0]
        if (k + 1) % 1000 == 0:
            print(f"  {k+1:,}/{len(targets):,}", flush=True)

    print(f"\n唯一確定 {len(res):,} ／ 多重相同內容 {len(amb):,} ／ 找不到來源 {len(none_):,}")
    if none_[:5]:
        print("  找不到來源的前五：", none_[:5])
    if amb[:5]:
        print("  內容完全相同的多個來源（無害，任一皆可）前五：")
        for t, h in amb[:5]:
            print(f"    {t}: {h}")

    mp = args.map or os.path.join(D, "global_frame_map.json")
    if os.path.exists(mp) and not args.only:
        old = json.load(open(mp))
        agree = dis = 0
        bad = []
        for t, (b, fi) in res.items():
            o = old.get(t)
            if o is None:
                continue
            if (o["block"], o["frame_index"]) == (b, fi):
                agree += 1
            else:
                dis += 1
                if len(bad) < 12:
                    bad.append((t, (o["block"], o["frame_index"]), (b, fi)))
        print(f"\n與姿態比對的映射：一致 {agree:,} ／ **不一致 {dis:,}**")
        for t, o, n in bad:
            print(f"    {t}: 姿態說 block_{o[0]}#{o[1]}  位元說 **block_{n[0]}#{n[1]}**")
        out = mp.replace(".json", "_bytelevel.json")
        json.dump({t: {"block": b, "frame_index": fi} for t, (b, fi) in res.items()},
                  open(out, "w"), indent=0)
        print(f"\n位元級映射已寫出 {out}")
        print("""
判讀：
  不一致 = 0            => 姿態比對的映射也正確
  不一致 > 0            => **位元級的才是對的**（姿態比對抓不到重複姿態那一類）
                           而不一致的那些，就是姿態錯掉的幀""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
