#!/usr/bin/env python
"""速度第二批（record_reduce／tile_cull）的自動判定（2026-10-03）。

讀 `scripts/lab/task_speed2_check.sh` 的報告，依**事先寫好的規則**決定要不要把新光柵器裝進共用環境、
以及 4x4 全場景（`task_full44.sh`）要開哪些旗標。使用者 10-03 的決定是「等速度第二批裝好再開 4x4」，
而 lab 的狀態由使用者自己看、Claude 不輪詢 => 判定必須能在佇列裡自己跑，不能等人讀 log。

規則（與 `紀錄/實驗分析/10` §4 的判定規則相同，寫在結果出來之前）：
  B 共用環境 vs 新 .so（lean 關）  渲染／radii／覆蓋數逐位元相同，梯度 <= 噪音底容許值   => 舊路徑沒變，否則什麼都不裝
  R lean vs lean+record_reduce     渲染／radii／覆蓋數逐位元相同，梯度在容許值內，trim 遮罩重疊 >= 99%
  T lean vs lean+rr+tile_cull      同上，且 binning 配對要變少
  梯度容許值 = max(10 x 噪音底 N 那一組的同名相對差, 2e-5)   （lean 驗證時 C 組最差是噪音底的 5 倍）
  真實迴圈（同一個起點 1,200 步）：旗標要讓真實步時間至少快 1% 才開
任何一組正確性沒過 => status=fail（不裝、不開 4x4，等使用者看）；正確但都沒變快 => status=no_gain（不裝，4x4 只開 lean）。

輸出：印出逐項判定；寫 logs/speed2_gate.json {status, flags, real_ms, checks}；fail 時 exit 1。
用法：python tools/speed2_gate.py [報告路徑]（省略＝最新的 logs/speed2_check_*.log）
"""
import glob
import json
import os
import re
import sys

GRAD = re.compile(r"^\s{4}(\S+)\s+最大絕對差 (\S+)\s+相對(?:（÷最大幅度）)?\s*(\S+)")
REAL = re.compile(r"（真實步時間·含重疊）\s+([\d.]+)")


def parse(text):
    blocks, cur, loops, cfg = {}, None, {}, None
    for ln in text.split("\n"):
        m = re.match(r"^══ (\S+) ", ln)
        if m:
            cur = blocks.setdefault(m.group(1), {"render": None, "radii": None, "cover": None,
                                                 "overlap": None, "pairs": None, "grads": {}})
            continue
        m = re.match(r"^── step_cost（(\S+)）", ln)
        if m:
            cfg, cur = m.group(1), None
            continue
        if cfg and REAL.search(ln):
            loops[cfg] = float(REAL.search(ln).group(1)); cfg = None
            continue
        if cur is None:
            continue
        if "渲染" in ln and "不同像素值" in ln:
            cur["render"] = "✅" in ln
        elif ln.strip().startswith("radii："):
            cur["radii"] = "✅" in ln
        elif "覆蓋數" in ln and "不同" in ln:
            cur["cover"] = "✅" in ln
        elif "遮罩重疊" in ln:
            m = re.search(r"遮罩重疊 ([\d.]+)%", ln); cur["overlap"] = float(m.group(1)) if m else None
        elif "binning 配對" in ln:
            m = re.search(r"([\d,]+) -> ([\d,]+)", ln)
            if m:
                cur["pairs"] = (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")))
        else:
            m = GRAD.match(ln)
            if m:
                cur["grads"][m.group(1)] = float(m.group(3))
    return blocks, loops


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("logs/speed2_check_*.log") or [""], key=os.path.getmtime, default="")
    if not path or not os.path.exists(path):
        print("⛔ 找不到 speed2_check 報告"); return 1
    blocks, loops = parse(open(path, encoding="utf-8", errors="ignore").read())
    print(f"報告：{path}")
    need = ["B", "N", "R", "T"]
    miss = [k for k in need if k not in blocks]
    out = {"report": path, "status": "fail", "flags": [], "real_ms": loops, "checks": {}}
    if miss or not all(k in loops for k in ("L", "LR", "LRT")):
        print(f"⛔ 報告不完整：缺比對 {miss}、缺真實迴圈 {[k for k in ('L', 'LR', 'LRT') if k not in loops]}")
        json.dump(out, open("logs/speed2_gate.json", "w"), ensure_ascii=False, indent=1); return 1
    noise = blocks["N"]["grads"]

    def check(k, need_overlap=False, need_fewer_pairs=False):
        b, why = blocks[k], []
        for f in ("render", "radii", "cover"):
            if b[f] is not True:
                why.append(f"{f} 不是逐位元相同")
        for g, rel in b["grads"].items():
            tol = max(10 * noise.get(g, 0.0), 2e-5)
            if rel > tol:
                why.append(f"梯度 {g} 相對差 {rel:.2e} > 容許 {tol:.2e}")
        if need_overlap and (b["overlap"] is None or b["overlap"] < 99.0):
            why.append(f"trim 遮罩重疊 {b['overlap']}% < 99%")
        if need_fewer_pairs and (b["pairs"] is None or b["pairs"][1] >= b["pairs"][0]):
            why.append(f"binning 配對沒有變少 {b['pairs']}")
        ok = not why
        print(f"  {'✅' if ok else '⛔'} {k}" + ("" if ok else "：" + "；".join(why)))
        out["checks"][k] = {"ok": ok, "why": why}
        return ok

    print("── 正確性")
    okB = check("B")
    okR = check("R", need_overlap=True)
    okT = check("T", need_overlap=True, need_fewer_pairs=True)
    L, LR, LRT = loops["L"], loops["LR"], loops["LRT"]
    print(f"── 真實迴圈（真實步 ms）：lean {L:.2f}／+record_reduce {LR:.2f}（{100 * (LR / L - 1):+.1f}%）"
          f"／+record_reduce+tile_cull {LRT:.2f}（{100 * (LRT / L - 1):+.1f}%）")
    if not (okB and okR and okT):
        print("⛔ 判定：正確性沒過 => 不裝、不自動開 4x4（等使用者看報告）")
        json.dump(out, open("logs/speed2_gate.json", "w"), ensure_ascii=False, indent=1); return 1
    flags = []
    if LRT <= 0.99 * min(L, LR):
        flags = ["record_reduce", "tile_cull"]
    elif LR <= 0.99 * L:
        flags = ["record_reduce"]
    out["flags"] = flags
    out["status"] = "install" if flags else "no_gain"
    print(f"✅ 判定：{'安裝新光柵器，4x4 開 ' + '＋'.join(flags) if flags else '正確但沒有變快 => 不裝，4x4 只開 lean'}")
    json.dump(out, open("logs/speed2_gate.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
