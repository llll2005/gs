"""解析 scripts/lab/task_best1_time.sh loop 的 log（10-15）：每組配方 x 階段的逐段 ms／步。
trim 的換算：profiler 在 1,000 步窗內只抓到 1 次 trim（攤成每 1,000 步一次），真實週期是每 500 步 => 增生期每步 = 真實步時間 − trim/1000 ＋ trim單次/500。
一塊 60k 估計 = 30,000 x 增生期 ＋ 30,000 x 收割期（不含驗證與存檔）。
用法：python tools/parse_best1_time.py <log>..."""
import re, sys
SEG = {"6 backward": "backward", "3 forward（光柵化）": "forward", "9 optimizer.step": "optimizer", "4 loss（L1+SSIM+正則）": "loss",
       "0 迴圈外（dataloader/Lightning）": "loop"}
def parse(path):
    out, cur = {}, None
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"════ (\w+)／(\w+)", ln)
        if m:
            cur = (m.group(1), m.group(2)); out[cur] = {}; continue
        if cur is None:
            continue
        m = re.match(r"(\d+ [^\s].*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d,]+)\s+([\d.]+)", ln)
        if m:
            name, per_step, calls, per_call = m.group(1).strip(), float(m.group(3)), int(m.group(4).replace(",", "")), float(m.group(5))
            if name.startswith("11 週期性 trim"):
                out[cur]["trim_once_ms"] = per_call; out[cur]["trim_in_window"] = per_step
            for k, v in SEG.items():
                if name == k:
                    out[cur][v] = per_step
        m = re.search(r"（真實步時間·含重疊）\s+([\d.]+)", ln)
        if m:
            out[cur]["real"] = float(m.group(1))
    for k, d in out.items():
        if "real" in d:
            t = d.get("trim_once_ms", 0.0)
            d["step"] = d["real"] - d.get("trim_in_window", 0.0) + t / 500.0
            d["trim_per_step"] = t / 500.0
    return out
if __name__ == "__main__":
    for p in sys.argv[1:]:
        o = parse(p); print("==", p)
        for arm in ("best0", "best0fast", "best1"):
            g, h = o.get((arm, "grow"), {}), o.get((arm, "harvest"), {})
            if "step" in g and "step" in h:
                tot = 30000 * (g["step"] + h["step"]) / 1000 / 60
                print(f"  {arm:<9} 增生期 {g['step']:6.1f} ms（bwd {g.get('backward',0):5.1f}／fwd {g.get('forward',0):5.1f}／trim {g['trim_per_step']:5.1f}／opt {g.get('optimizer',0):5.1f}）"
                      f"  收割期 {h['step']:6.1f} ms（bwd {h.get('backward',0):5.1f}）  => 60k 約 {tot:5.1f} 分鐘")
