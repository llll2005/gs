#!/usr/bin/env python
"""把一個 60k 的組態等比例壓縮成 proxy 組態（單一旋鈕：時間尺度 s）。

為什麼要等比例而不是截斷（2026-08-25 實測）：
  每個跑次的 val 曲線都在 `densify_until` 那一刻跳 +1 dB，之前幾乎是平的。
  截斷會停在「族群蓋好但還沒收割」的位置 => 量到的不是配方的品質。
  等比例縮放保留整個排程形狀（含收割段），只是每段變短。

為什麼連 interval 也要縮：
  縮 interval 才能保住**事件次數**。densify/trim 的族群軌跡是事件空間的
  （每事件 x1.05 / x0.9），事件次數一樣 => 族群軌跡一樣，破平衡關係也一樣
  （break-even = prune_interval x ln(add_ratio) / (-ln(1-prune_ratio))，
   兩個 interval 同乘 s 時該式左右同乘 s，關係不變）。
  只縮端點不縮 interval 的話，事件次數掉 1/s 倍，靠事件生效的機制會被系統性低估。

⚠ 已知且**故意不補償**的兩個失真（補償會多出旋鈕，失敗時無法歸因）：
  1. LR 積分變 s 倍 —— 收割段的增益實測約 +0.38 dB / 每倍步數，所以 proxy 的絕對分數
     一定較低。proxy 是**篩子不是成品**。
  2. MCMC noise 的總擴散量變 s 倍（noise_lr x lr_t 每步注入，步數變少）。
  這兩個是驗證實驗要回答的對象，不是要在設計階段猜的。

用法:
  python tools/make_proxy_config.py --base configs/X.yaml --scale 0.25 \
      --out configs/proxy/X_s25.yaml --set model.density.init_args.densify_until_iter=30000
"""
import argparse
import copy
import os
import sys

import yaml

# 需要按 s 縮放的鍵（以「路徑最後一段」比對）。
SCALE_KEYS = {
    "max_steps",
    "densify_from_iter",
    "densify_until_iter",
    "densification_interval",
    "normal_regularization_from_iter",
    "dist_regularization_from_iter",   # 2DGS distortion loss 起始點；稽核抓到的漏網
    "sh_degree_up_interval",
    "contribution_prune_from_iter",
    "contribution_prune_interval",
    "contribution_prune_until_iter",
    "min_opacity_anneal_end_iter",
    "min_opacity_anneal_start_iter",
    "harvest_dust_trim_interval",
    "vpc_interval",
    "check_val_every_n_epoch",
    "opacity_reset_interval",
    "densify_grad_threshold_iter",
}

# 這些即使是 -1 / 0 也要原樣保留（哨兵值，縮放會改變語意）
SENTINELS = {-1, 0, None, False, True}

# config 裡沒寫、走程式預設、但一定要一起縮的（否則 proxy 的排程比例會跑掉）
DEFAULTS_TO_PIN = {
    "model.renderer.init_args.contribution_prune_from_iter": 1000,
    "model.renderer.init_args.contribution_prune_interval": 500,
    "model.gaussian.init_args.optimization.sh_degree_up_interval": 1000,
}

# 看起來像步數但**不該**縮的（避免誤判用；出現時只提示不動作）
SUSPICIOUS_HINTS = ("iter", "step", "interval", "epoch", "warm")


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, node


def set_path(cfg, dotted, value):
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_path(cfg, dotted):
    cur = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def scale_int(v, s, minimum=1):
    return max(minimum, int(round(v * s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--scale", type=float, required=True, help="時間尺度 s，例如 0.25")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[],
                    help="縮放**之前**先套用的覆寫（把 CLI 覆寫烤進來），key=value")
    ap.add_argument("--drop", action="append", default=[],
                    help="移除的鍵（跑次專屬路徑等），dotted path")
    args = ap.parse_args()
    s = args.scale
    assert 0 < s <= 1, "s 必須在 (0, 1]"

    with open(args.base) as f:
        cfg = yaml.safe_load(f)

    # 1) 先把 CLI 覆寫烤進 config（否則 proxy 腳本又要傳未縮放的步數，必漏）
    applied = []
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            val = yaml.safe_load(v)
        except Exception:
            val = v
        set_path(cfg, k, val)
        applied.append((k, val))

    dropped = []
    for k in args.drop:
        parts = k.split(".")
        cur = cfg
        for p in parts[:-1]:
            cur = cur.get(p) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, dict) and parts[-1] in cur:
            dropped.append((k, cur.pop(parts[-1])))

    # 2) 把「走預設但必須一起縮」的參數顯式寫進來
    pinned = []
    for k, dv in DEFAULTS_TO_PIN.items():
        if get_path(cfg, k) is None:
            set_path(cfg, k, dv)
            pinned.append((k, dv))

    before = copy.deepcopy(cfg)

    # 3) 縮放
    scaled, skipped, suspicious = [], [], []
    for path, val in list(walk(cfg)):
        leaf = path.split(".")[-1].split("[")[0]
        if leaf in SCALE_KEYS:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                skipped.append((path, val, "非數值"))
            elif val in SENTINELS:
                skipped.append((path, val, "哨兵值，語意會變"))
            else:
                nv = scale_int(val, s)
                set_path(cfg, path, nv)
                scaled.append((path, val, nv))
        elif path.split("[")[0] == "save_iterations":
            pass  # 在 walk 之後另外整批處理
        elif any(h in leaf.lower() for h in SUSPICIOUS_HINTS):
            suspicious.append((path, val))

    # save_iterations 是純數字 list，另外處理
    if isinstance(cfg.get("save_iterations"), list):
        old = cfg["save_iterations"]
        new = sorted({scale_int(v, s) for v in old if isinstance(v, int)})
        cfg["save_iterations"] = new
        scaled.append(("save_iterations", old, new))

    # 4) 一致性檢查
    problems = []
    T = get_path(cfg, "trainer.max_steps")
    du = get_path(cfg, "model.density.init_args.densify_until_iter")
    di = get_path(cfg, "model.density.init_args.densification_interval")
    ci = get_path(cfg, "model.renderer.init_args.contribution_prune_interval")
    # resolved config 會多包一層 init_args，所以用「葉節點名稱」找而不是寫死路徑。
    # 所有 max_steps 原本都等於 trainer.max_steps；縮放後必須仍然相等，否則 LR/權重
    # 排程會停在錯的位置（踩過的坑：只改 trainer 那個，LR 停在 10% 不再衰減）。
    # v is None => jsonargparse 的未設值連結鍵，不是排程
    all_ms = [(p, v) for p, v in walk(cfg)
              if p.split(".")[-1] == "max_steps" and v is not None]
    bad = [(p, v) for p, v in all_ms if v != T]
    if bad:
        for p, v in bad:
            problems.append(f"{p} = {v} != trainer.max_steps({T}) => 排程會停在錯的位置")
    else:
        print(f"（max_steps 一致性：{len(all_ms)} 處全部 = {T}）")
    if du and T and du >= T:
        problems.append(f"densify_until({du}) >= max_steps({T}) => 沒有收割段，跳點量不到")
    for nm, v in (("densification_interval", di), ("contribution_prune_interval", ci)):
        if v is not None and v < 10:
            problems.append(f"{nm}={v} < 10 步 => 事件幾乎每步發生，動力學已質變，s 太小")
    # 事件次數對照
    def n_events(c, frm, until, inter):
        f, u, i = get_path(c, frm), get_path(c, until), get_path(c, inter)
        if None in (f, u, i) or i <= 0:
            return None
        return int((u - f) / i)
    for label, frm, until, inter in [
        ("densify", "model.density.init_args.densify_from_iter",
         "model.density.init_args.densify_until_iter",
         "model.density.init_args.densification_interval"),
    ]:
        a = n_events(before, frm, until, inter)
        b = n_events(cfg, frm, until, inter)
        if a and b and abs(b - a) / a > 0.05:
            problems.append(f"{label} 事件次數 {a} -> {b}（差 {100*(b-a)/a:+.1f}%）=> 族群軌跡會變")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"base = {args.base}\nscale s = {s}\nout  = {args.out}\n")
    if applied:
        print("烤進來的 CLI 覆寫:")
        for k, v in applied:
            print(f"  {k} = {v}")
        print()
    if pinned:
        print("原本走程式預設、已顯式寫入以便縮放:")
        for k, v in pinned:
            print(f"  {k} = {v}")
        print()
    print(f"{'已縮放的參數':<62} {'原值':>12} -> {'新值':>10}")
    for p, o, n in scaled:
        print(f"  {p:<60} {str(o):>12} -> {str(n):>10}")
    if skipped:
        print("\n未縮放（哨兵/非數值）:")
        for p, v, why in skipped:
            print(f"  {p} = {v}   ({why})")
    if suspicious:
        print("\n⚠ 看起來像步數但不在白名單 —— 請人工確認是否該縮:")
        for p, v in suspicious:
            print(f"  {p} = {v}")
    if problems:
        print("\n⛔ 一致性檢查未通過:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\n✅ 一致性檢查通過")
    # 事件次數摘要
    for label, frm, until, inter in [
        ("densify", "model.density.init_args.densify_from_iter",
         "model.density.init_args.densify_until_iter",
         "model.density.init_args.densification_interval"),
    ]:
        print(f"   {label} 事件次數: {n_events(before, frm, until, inter)} "
              f"-> {n_events(cfg, frm, until, inter)}")
    print(f"   收割步數: {get_path(before,'trainer.max_steps') - get_path(before,'model.density.init_args.densify_until_iter')} "
          f"-> {T - du}")


if __name__ == "__main__":
    main()
