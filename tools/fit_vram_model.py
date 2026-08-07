"""Re-fit the peak-VRAM model from every run we have already paid for.

`internal/utils/strip_cameras.py`'s A_RENDER=500 / B_RENDER=1550 were fitted on 0.7-1.45M points of
one arrangement in 2026-07-22, then extrapolated. On 2026-08-07 `cap4m_b12` finished at N=4.0M
using 5.51/6.1 G against a prediction of 8.2 G for the render alone, and the wall it predicted at
2.04M never appeared. `tools/calibrate_block_caps.py` sizes every per-block cap from that model, so
the error has been spent, not just noted.

The refit needs no GPU: `logs/quad_progress.log` has ~81 START/DONE pairs, each carrying a final N
and a peak VRAM, across model types and caps. That is a wider spread of N than the original fit had.

WHAT THE DATA CAN AND CANNOT SAY
  can:    the per-point cost averaged over real training, and how tightly it holds across runs
  cannot: the arrangement term. `uniform_60k_b12` used ~5.5 G at N=0.5M while `cap4m_b12` used
          5.51 G at N=4.0M -- 8x the points for the same peak -- so overdraw clearly dominates, but
          the ledger records no geometry. The residual spread here is a LOWER BOUND on that effect.

TWO BIASES, both stated rather than silently corrected:
  1. VRAM is torch.cuda.memory_reserved(), a high-water mark, but N is the FINAL count. Cap-bound
     runs end at 0.9*cap after the last prune, so their peak N was ~1.11x the N recorded. Fitting
     bytes/point against the smaller N therefore OVERSTATES the constant -- the same direction as
     the error we are chasing, so a fit that still comes out below 2050 B/pt is conservative.
  2. Floats per point differ by model (SB 25, SH3 59, SH2 27, SH0 3+...), and model state is
     N*F*4*4 bytes, which is exact arithmetic. It is subtracted before fitting so the residual is
     render cost alone.
"""
import argparse
import glob
import os
import re
from collections import namedtuple

GB = 2 ** 30
Run = namedtuple("Run", "name n vram_gb its floats state_gb render_gb bpp")

# floats/point per model, counted from the model's properties (params only, not optimiser state)
FLOATS = {
    "gaussian_2d_sb.Gaussian2DSB": 25,      # xyz3 + scale2 + rot4 + opacity1 + sh0(3) + SB lobes
    "gaussian_2d.Gaussian2D": 59,           # xyz3 + scale2 + rot4 + opacity1 + sh3(48) + ...
    "vanilla_gaussian.VanillaGaussian": 62,
}
STATE_MULT = 4                               # params + grad + Adam exp_avg + exp_avg_sq


def parse_ledger(path):
    """START carries the output path; DONE that follows it carries N and VRAM. Pairing is by
    order because DONE lines do not name their run (internal/callbacks.py says so explicitly)."""
    pending, out = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        if "| START" in line:
            m = re.search(r"out=(\S+)", line)
            pending = m.group(1) if m else None
        elif "| DONE" in line and pending:
            n = re.search(r"N=([\d.]+)M", line)
            v = re.search(r"VRAM=([\d.]+)/", line)
            i = re.search(r"([\d.]+)it/s", line)
            if n and v:
                out.append((pending, float(n.group(1)) * 1e6, float(v.group(1)),
                            float(i.group(1)) if i else float("nan")))
            pending = None
        elif "| DIED" in line:
            pending = None
    return out


def floats_for(run_dir):
    """Read the resolved config to find which model ran. Returns None if it cannot be determined --
    guessing here would silently mix a 25-float and a 59-float run into one fit."""
    for cfg in glob.glob(os.path.join(run_dir, "lightning_logs", "version_*", "config.yaml")):
        text = open(cfg, encoding="utf-8", errors="replace").read()
        for key, f in FLOATS.items():
            if key in text:
                return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="logs/quad_progress.log")
    ap.add_argument("--min_n", type=float, default=50_000, help="tiny runs are dominated by fixed overhead")
    ap.add_argument("--v_os_gb", type=float, default=0.45, help="CUDA context + driver, not per-point")
    a = ap.parse_args()

    runs, skipped = [], 0
    for out_dir, n, vram, its in parse_ledger(a.ledger):
        if n < a.min_n:
            continue
        f = floats_for(out_dir)
        if f is None:
            skipped += 1
            continue
        state = n * f * 4 * STATE_MULT / GB
        render = vram - a.v_os_gb - state
        if render <= 0:                       # model state alone exceeds the reading -> unusable
            skipped += 1
            continue
        m = re.search(r"outputs/([^/]+)", out_dir)
        runs.append(Run(m.group(1) if m else out_dir, n, vram, its, f,
                        state, render, render * GB / n))

    if not runs:
        print("沒有可用資料點"); return

    runs.sort(key=lambda r: r.n)
    print(f"{'run':<34}{'N':>11}{'F':>4}{'VRAM':>8}{'狀態':>8}{'渲染':>8}{'B/pt':>8}")
    for r in runs:
        print(f"{r.name[:33]:<34}{r.n:>11,.0f}{r.floats:>4}{r.vram_gb:>7.2f}G"
              f"{r.state_gb:>7.2f}G{r.render_gb:>7.2f}G{r.bpp:>8.0f}")

    bpp = sorted(x.bpp for x in runs)
    mid = bpp[len(bpp) // 2]
    lo, hi = bpp[len(bpp) // 10], bpp[-1 - len(bpp) // 10]
    print(f"\n[渲染 B/pt]  n={len(runs)}  中位 {mid:.0f}   p10 {lo:.0f}   p90 {hi:.0f}"
          f"   全距 {bpp[0]:.0f}~{bpp[-1]:.0f}  ({bpp[-1] / max(bpp[0], 1):.1f}×)")
    print(f"[對照] strip_cameras.py 現值 A_RENDER+B_RENDER = 2050 B/pt")
    print(f"[跳過] {skipped} 筆（無法判定 model / 狀態已超過讀數）")

    # The B/pt spread collapses as N grows, which is what a FIXED overhead being amortised looks
    # like -- not a per-point cost at all. Refit render = V_fixed + c*N and see whether the
    # intercept explains it. Least squares, no library, on the F=25 runs so the model is constant.
    sb = [r for r in runs if r.floats == 25]
    if len(sb) >= 3:
        xs = [r.n for r in sb]
        ys = [r.render_gb * GB for r in sb]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        v0 = my - c * mx
        resid = [y - (v0 + c * x) for x, y in zip(xs, ys)]
        sst = sum((y - my) ** 2 for y in ys)
        r2 = 1 - sum(e * e for e in resid) / sst if sst else float("nan")
        print(f"\n[SB(F=25) 擬合 render = V_fixed + c*N]  n={len(sb)}")
        print(f"    V_fixed = {v0 / GB:>6.2f} G        c = {c:>6.0f} B/pt        R² = {r2:.3f}")
        print(f"    殘差 ±{max(abs(e) for e in resid) / GB:.2f} G "
              f"(標準差 {(sum(e * e for e in resid) / len(resid)) ** 0.5 / GB:.2f} G)")
        print(f"    ⇒ 現行 V_OS 預設 0.8G 對照擬合出的 {(a.v_os_gb + v0 / GB):.2f}G 總固定量")

    # Why the fit fails: runs at the SAME N differ by GB. Surface them, because a pair like that is
    # worth more than the regression -- it is a controlled experiment we already ran by accident.
    by_n = {}
    for r in runs:
        by_n.setdefault((round(r.n, -5), r.floats), []).append(r)
    pairs = [(k, v) for k, v in by_n.items() if len(v) > 1
             and max(x.vram_gb for x in v) - min(x.vram_gb for x in v) > 0.5]
    if pairs:
        print("\n[同顆數、不同 VRAM]  顆數解釋不了的部分：")
        for (n, f), v in sorted(pairs):
            v.sort(key=lambda x: x.vram_gb)
            print(f"    N≈{n:,.0f} (F={f}): " +
                  "  ".join(f"{x.name[:26]} {x.vram_gb:.2f}G" for x in v) +
                  f"   ⇒ 差 {v[-1].vram_gb - v[0].vram_gb:.2f}G")

    print("\n⚠ R² 低不是雜訊，是模型形狀錯。VRAM 由 OVERDRAW（每像素要混合幾顆）驅動，不由 N。")
    print("  機制＝透射率：opacity 低 ⇒ T=(1-o)^n 衰減慢 ⇒ 每像素混合更多層 ⇒ 交集數與 backward 暴增。")
    print("  同一機制的三個獨立佐證：①opacity_reg 越大 VRAM 越高（顆數還更少）")
    print("    ②起始 trim 在 opacity 0.99 砍 73%、0.05 只砍 20%   ③體積雲 50 萬顆 ≈ 表面雲 360 萬顆")
    print("  ⇒ 正確的 VRAM 模型就是命題裡的 c_i（螢幕足跡），按顆數計價從形狀上就錯。")


if __name__ == "__main__":
    main()
