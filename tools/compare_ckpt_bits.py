#!/usr/bin/env python
"""兩個 ckpt 是否逐位元相同（參數＋優化器狀態）。用法：compare_ckpt_bits.py A.ckpt B.ckpt [--label 名稱]
結束碼 0 = 完全相同；1 = 有差異。"""
import argparse, os, sys
import torch
# ckpt 的 hyper_parameters 會序列化 internal.* 類別 => 必須能 import 專案模組（09-18 踩過：No module named 'internal'）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def flat(x, prefix=""):
    if torch.is_tensor(x):
        yield prefix, x
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            yield from flat(x[k], f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from flat(v, f"{prefix}[{i}]")

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--label", default="")
args = ap.parse_args()
A = torch.load(args.a, map_location="cpu"); B = torch.load(args.b, map_location="cpu")
bad = 0; n = 0
for part in ("state_dict", "optimizer_states"):
    fa = dict(flat(A.get(part, {}))); fb = dict(flat(B.get(part, {})))
    if set(fa) != set(fb):
        print(f"⛔ {part} 鍵不同：只在 A {sorted(set(fa)-set(fb))[:5]}／只在 B {sorted(set(fb)-set(fa))[:5]}"); bad += 1; continue
    for k in fa:
        n += 1
        x, y = fa[k], fb[k]
        if x.shape != y.shape:
            print(f"⛔ {part}:{k} 形狀不同 {tuple(x.shape)} vs {tuple(y.shape)}"); bad += 1
        elif not torch.equal(x, y):
            d = (x.double() - y.double()).abs().max().item() if x.is_floating_point() else float("nan")
            print(f"⛔ {part}:{k} 不同，最大差 {d:.3e}"); bad += 1
print(f"{'✅' if bad == 0 else '⛔'} {args.label} 比較 {n} 個張量：{'逐位元完全相同' if bad == 0 else f'{bad} 個不同'}"
      f"（global_step {A.get('global_step')} vs {B.get('global_step')}）")
sys.exit(0 if bad == 0 else 1)   # 1 = 有差異；其他非 0（例如例外）= 工具本身出錯，不代表不同
