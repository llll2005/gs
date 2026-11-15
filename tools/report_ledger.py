"""★ 2026-11-15 逐跑次總帳產生器（使用者：「之前網頁還有各項分數、備註、問題的參考數據」，接續 11-09 的「實驗總帳」頁）。
收集：
  - 評估 log（scripts/lab/eval_arms_lab.sh 的輸出 logs/eval_arms_*.log；lab 與本機）：時間、held-out、F1、地面／地下、立面、朝向、海面、小物件
  - 換塊 log（tools/swap_block_eval.py 的 logs/swap_b<B>_*.log；基準一律 full44_best => 不同天的數字可直接比）
  - 11-09 總帳（outputs/local/scratch_1109/report_data_1109.json，11-06～08 本機舊底 30k 的 57 個跑次）
同一個 (機器, 塊, 跑次) 在多份 log 出現時取最新的（ckpt 的 CPU 量測與渲染都是決定性的；時間取最後一次）。
輸出：把 <script id="ledger"> 的 JSON 換掉（紀錄/流程與決策.html），其餘版面不動。
用法：python tools/report_ledger.py --html 紀錄/流程與決策.html --logs logs outputs/local/lablogs_1115/logs [--old outputs/local/scratch_1109/report_data_1109.json]
"""
import argparse, glob, json, os, re

BASE3 = "g44tn_s20g70np16tricap0p5cmflattrimasklowresgeo8log0_best1maskdistrfed10b4"
ADIST4 = "g44s30ksfm10fssh16di4_best1maskdistrfed10b4"
NB30 = "g44s30ksfm10lrx100nov4_best1maskdistrfed10b4"
NUM = r"[-+]?[\d,]*\.?\d+"


def fnum(s):
    try:
        return float(str(s).replace(",", "").replace("%", "").replace("+", ""))
    except ValueError:
        return None


def where(path):
    """ckpt 路徑 => (機器, 跑次, 塊)；CityGS 參考 => ('ref', 'CityGS', 塊)。"""
    p = path.strip()
    m = re.search(r"block_(\d+)", p)
    blk = int(m.group(1)) if m else None
    if "cityGS_origin" in p or "citygs_ref" in p or "official_release" in p:
        m2 = re.search(r"citygs_inbox_b(\d+)", p)
        return "ref", "CityGS", blk if blk is not None else (int(m2.group(1)) if m2 else None)
    m = re.search(r"outputs/(lab|local)/([^/]+)/blocks/block_(\d+)", p)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, blk


def parse_eval(path):
    """一份 eval_arms log => {label: {metric: value}}、{label: ckpt 路徑}、塊。"""
    L = open(path, encoding="utf-8", errors="replace").read().split("\n")
    lab2path, out, sec = {}, {}, None
    mb = re.search(r"_b(\d+)\.log$", path)
    blk = int(mb.group(1)) if mb else None
    cur = None
    for raw in L:
        l = raw.rstrip()
        if l.startswith("### "):
            sec = l[4:].split("（")[0].strip(); cur = None; continue
        if sec is None:
            if "=" in l and ("/checkpoints/" in l or ".ckpt" in l):
                k, v = l.split("=", 1); lab2path[k.strip()] = v.strip()
            continue
        if not l.strip():
            continue
        d = None
        if sec == "時間" and "純訓練" in l:
            lab = l.split("純訓練")[0].strip()
            m = re.search(r"純訓練（[^）]*）\s*(" + NUM + r")\s*分", l)
            d = out.setdefault(lab, {})
            if m and fnum(m.group(1)) > 0: d["time_min"] = fnum(m.group(1))   # 接續跑次（只跑 1000 步）記成 0.0 => 當缺值
            m = re.search(r"整個任務\s*(" + NUM + r")\s*分", l)
            if m: d["task_min"] = fnum(m.group(1))
            m = re.search(r"｜N\s*([\d,]+)", l)
            if m: d["N"] = fnum(m.group(1))
            m = re.search(r"最後一步\s*(\d+)", l)
            if m: d["steps"] = fnum(m.group(1))
        elif sec == "heldout":
            if l.startswith("== "):
                cur = l[3:].strip(); continue
            if cur:
                m = re.match(r"^(PSNR|SSIM|LPIPS)\s+(" + NUM + ")", l.strip())
                if m:
                    out.setdefault(cur, {})["ho_" + m.group(1).lower()] = fnum(m.group(2))
        elif sec == "F1" and " F1 " in l:
            t = l.split(); lab = t[0]
            m = re.search(r"P\s+(" + NUM + r")\s+R\s+(" + NUM + r")\s+F1\s+(" + NUM + ")", l)
            if m:
                d = out.setdefault(lab, {}); d["f1_p"], d["f1_r"], d["f1"] = (fnum(x) for x in m.groups())
        elif sec == "ground":
            t = l.split()
            if len(t) == 10 and all(fnum(x) is not None for x in t[1:]):
                d = out.setdefault(t[0], {})
                for k, v in zip(["g_cov", "g_full", "g_cov_o9", "g_flat", "g_above", "g_under", "g_under_o9", "g_huge", "g_huge_under"], t[1:]):
                    d[k] = fnum(v)
        elif sec == "facade":
            if l.startswith("== "):
                cur = l[3:].strip(); continue
            m = re.search(r"塊內不透明\s+([\d,]+)\s+立面\s+(" + NUM + r")%\s+貼牆中位\s+(" + NUM + r")\s+側視\s+(" + NUM + r")%\s+平躺\s+(" + NUM + ")%", l)
            if m and cur:
                d = out.setdefault(cur, {})
                d["n_opaque"], d["facade_pct"], d["wall_fit"], d["side_pct"], d["flat_pct"] = (fnum(x) for x in m.groups())
        elif sec == "orient bands" and "｜" in l and "50~150m" in l:
            lab = l.split("｜")[0].strip()
            d = out.setdefault(lab, {})
            for band, key in (("50~150m", "50"), ("150~250m", "150"), (">250m", "250")):
                m = re.search(re.escape(band) + r"\s+(" + NUM + r")／\s*(" + NUM + r")／\s*(" + NUM + r")／\s*(" + NUM + ")", l)
                if m:
                    d["hole_" + key], d["al_" + key] = fnum(m.group(1)), fnum(m.group(2))
        elif sec == "sea":
            t = l.split()
            if len(t) == 11 and t[0].endswith("塊內") and all(fnum(x) is not None for x in t[1:]):
                d = out.setdefault(t[0][:-2], {})
                d["sea_n"], d["sea_area"], d["sea_h"], d["sea_20"], d["sea_50"] = fnum(t[1]), fnum(t[4]), fnum(t[5]), fnum(t[9]), fnum(t[10])
        elif sec == "smallobj":
            m = re.match(r"^(\S+)\s+(\d+)/(\d+)\s*%\s+(\d+)/(\d+)\s*%\s+(\d+)/(\d+)\s*%\s+([\d,]+)\s+([\d,]+)", l)
            if m:
                d = out.setdefault(m.group(1), {})
                d["lamp_gap"], d["iso_gap"], d["car_gap"] = fnum(m.group(2)), fnum(m.group(4)), fnum(m.group(6))
                d["lamp_n"], d["car_n"] = fnum(m.group(8)), fnum(m.group(9))
    return out, lab2path, blk


def parse_swap(path):
    """一份換塊 log => {跑次名: {swap_*}}、塊。只讀「第 B 塊視角」那張表。"""
    mb = re.search(r"swap_(?:local_)?b(\d+)_", os.path.basename(path))
    blk = int(mb.group(1)) if mb else None
    out, on = {}, False
    for l in open(path, encoding="utf-8", errors="replace"):
        if l.startswith("=== 第"):
            on = True; continue
        if on and l.startswith("==="):
            break
        if not on:
            continue
        t = l.split()
        if len(t) >= 11 and fnum(t[-10]) is not None and re.match(r"^\d+/\d+$", t[-5] or ""):
            name = " ".join(t[:-10])
            v = [fnum(x) for x in t[-10:]]
            out[name] = {"sw_psnr": v[0], "sw_ssim": v[1], "sw_lpips": v[2], "sw_fail": v[3], "sw_d": v[4],
                         "sw_reg": v[6], "sw_reg_d": v[7], "sw_geo": v[8], "sw_geo_d": v[9]}
    return out, blk


def status_of(machine, run, label):
    s = run + " " + (label or "")
    if run == BASE3:
        return "參考", "新底 3（20k；總帳差值的基準；11-15 起現行是 17k 的新底 4）"
    if run == ADIST4:
        return "參考", "A＋dist4（全場景最佳 28.058 的配方）"
    if run == NB30:
        return "參考", "11-10 新基準 30k"
    if run == "CityGS":
        return "參考", "CityGS 公開模型（60k）"
    R3 = "np16tricap0p5cmflattrimasklowresgeo8log0"            # 新底 3 的配方字串（只有建在它上面的才算現在的候選）
    on3 = run.startswith("g44tn_s20g70" + R3) or run.startswith("g44tn_s17g70" + R3) or run.startswith("g44tn_s15g70" + R3) \
        or run.startswith("g44tn_s15g80" + R3) or run.startswith("g44tn_s20g0" + R3)
    if "capfc" in run and "tricap2g" not in run and not run.startswith("g44tn_s17g70"):   # 新底 5（17k＋dense＋capfc）系列在下面分
        return "建議採用", "只補前面：換塊打平、地下 −21～−43%、F1 −.002～−.018 => 11-15 晚進底（新底 5）"
    if "tricap2g" in run:
        return "不採用", "上限只在增生期：沒拿回畫質"
    if run.startswith("seadose"):
        return "待驗證", "海面往下沉（接續 1000 步）：b1 打平；b4 不保護 −0.022、保護 H200 仍 −0.019（地下不再增加、但海面也幾乎沒降）"
    if run.startswith("probe_sea"):
        return "參考", "海面接續 500 步（11-13）"
    if on3:
        tail = run[len("g44tn_s20g70" + R3):] if run.startswith("g44tn_s20g70" + R3) else ""
        if run.startswith("g44tn_s20g0"):
            return "不採用", "EDGS 式不增生：立面減半、地下變多"
        for key, st, note in [("fp16", "不採用", "完整路徑每 16 步：只省 2%、F1 掉"), ("cskip", "不採用", "C 區不跑：不省時間、畫質掉"),
                              ("pxdelf", "打平", "點精靈全解析度才刪：與撐大打平"), ("pxabsf", "不採用", "點精靈全解析度才吸收：與撐大打平但慢 8%"),
                              ("pxdel", "不採用", "點精靈刪除（低解析度期也刪，第 1,000 步判掉 82%）"), ("pxabs", "不採用", "點精靈吸收（低解析度期也做）"),
                              ("mg", "不採用", "合併系列（本機）：不省時間（+0～5%）、最終顆數沒降（cap 補回；mgpx／mgrel +15～21%）、畫質打平"),
                              ("noip", "不採用", "拿掉面內增生：海面高處 −61／−96%，但使用者 viewer 看 b1 那團還在（不是來源；擋住的是 v4 投票）"),
                              ("dense", "建議採用", "三角式稠密起點（照常增生）：四塊換塊打平、F1 都升 => 11-15 晚進底（新底 5）"),
                              ("r2", "不採用", "兩輪（名額 85%）：被 17k 壓過（省得少、換塊較差、幾何代價相同）、顆數 +15%")]:
            if key in tail:
                return st, note
        if run.startswith("g44tn_s17g70" + R3) and any(k in run[len("g44tn_s17g70" + R3):] for k in ("v4f", "v5", "dsf")):
            return "不採用", "v5 系列（v4f／v5／v5e8／v5s4／dsf）：b1 那團都還在、塊內高處 −14～+10%、時間 +11～25% => 11-15 晚關線"
        if run.startswith("g44tn_s17g70") and "r2" in run:
            return "不採用", "17k＋兩輪名額 85%：時間 −18～19%，但換塊 −0.021／−0.028、本塊區域 −0.06～−0.08、F1 −.030／−.027（比 17k 單獨多掉一截）"
        if run.startswith("g44tn_s17g70" + R3 + "densecapfcrs"):
            return "待驗證", "凸包外也用校正深度（rs4／rs4+v4f）：b1 那團的根因修法（11-15 深夜）"
        if run.startswith("g44tn_s17g70" + R3 + "densecapfcncs"):
            return "不採用", "無約束往下沉：b1／b4 換塊 −0.07～−0.10、塔頂對齊掉 => 11-15 深夜關線"
        if run == "g44tn_s17g70" + R3 + "densecapfc_best1maskdistrfed10b4":
            return "現行", "新底 5（17k＋稠密起點＋只補前面；11-15 晚 使用者採用）"
        if run == "g44tn_s17g70" + R3 + "_best1maskdistrfed10b4":
            return "參考", "新底 4（17k；11-15 採用，晚上被新底 5 取代）"
        if run.startswith("g44tn_s17g70"):
            return "取捨", "17k：時間 −12%、換塊打平，但地下 +0.5～0.6 層、F1 −.010～−.018"
        if run.startswith("g44tn_s15g80"):
            return "不採用", "15k 增生 80%：本塊區域 b1 −0.077、F1 與地下都差"
        if run.startswith("g44tn_s15g70"):
            return "不採用", "15k 增生 70%：本機 F1 −.012／−.032（比 15k／80% 更差）"
        if not run.startswith("g44tn_s20g70"):
            return "待驗證", "步數再降（新底 3 配方）"
    if run.startswith("g44tn_s30g50") and ("r2" in run or "dense" in run):
        return "不採用", "舊底上的兩輪流程／窗口／稠密起點組合（38 §5、§9）"
    return "歷史", "11-10～11-13 調參系列（已進底或已關，見研究總覽與 38）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--old", default=None)
    a = ap.parse_args()
    rows, seen_logs = {}, {}
    files = []
    for d in a.logs:
        files += glob.glob(os.path.join(d, "eval_arms_*.log")) + glob.glob(os.path.join(d, "swap_b*_*.log")) + glob.glob(os.path.join(d, "swap_local_b*_*.log"))
    files.sort(key=os.path.getmtime)
    name2label = {}
    for f in files:
        base = os.path.basename(f)
        if base.startswith("eval_arms_"):
            vals, l2p, blk = parse_eval(f)
            for lab, v in vals.items():
                p = l2p.get(lab)
                if not p:
                    continue
                mach, run, b = where(p)
                if mach is None:
                    continue
                b = b if b is not None else blk
                key = f"{mach}|b{b}|{run}"
                r = rows.setdefault(key, {"machine": mach, "blk": b, "run": run, "label": lab, "v": {}, "src": []})
                if len(lab) > len(r["label"]) or r["label"] in ("", run):
                    r["label"] = lab
                r["v"].update({k: x for k, x in v.items() if x is not None})
                if base not in r["src"]:
                    r["src"].append(base)
                name2label[run] = r["label"]
        else:
            vals, blk = parse_swap(f)
            mach = "local" if base.startswith("swap_local_") else "lab"
            for name, v in vals.items():
                if name.startswith("基準") or name.startswith("自我檢查"):
                    continue
                key = f"{mach}|b{blk}|{name}"
                r = rows.setdefault(key, {"machine": mach, "blk": blk, "run": name, "label": name2label.get(name, name), "v": {}, "src": []})
                r["v"].update(v)
                if base not in r["src"]:
                    r["src"].append(base)
    out = []
    for key, r in rows.items():
        if r["machine"] == "ref":
            r["status"], r["note"] = "參考", "CityGS 公開模型（60k；塊內）"
        else:
            r["status"], r["note"] = status_of(r["machine"], r["run"], r["label"])
        r["id"] = key
        r["group"] = "lab" if r["machine"] == "lab" else ("本機" if r["machine"] == "local" else "CityGS")
        out.append(r)
    # 時間：新底 3 配方的 20k 跑次若比同塊新底 3 慢 1.5 倍以上 => 共用卡期間訓練的，時間不可比（拿掉、加註）
    t3 = {(r["machine"], r["blk"]): r["v"].get("time_min") for r in out if r["run"] == BASE3}
    for r in out:
        t = r["v"].get("time_min"); b = t3.get((r["machine"], r["blk"]))
        if t and b and r["run"].startswith("g44tn_s20g70") and t > 1.5 * b:
            r["v"].pop("time_min"); r["note"] = (r.get("note") or "") + f"｜時間 {t:.1f} 分不可比（共用卡期間訓練）"
    # 基準：同機器同塊的新底 3
    idx = {(r["machine"], r["blk"], r["run"]): r["id"] for r in out}
    for r in out:
        r["base"] = idx.get((r["machine"], r["blk"], BASE3))
        if r["machine"] == "ref":
            r["base"] = idx.get(("lab", r["blk"], BASE3)) or idx.get(("local", r["blk"], BASE3))
    # 11-09 總帳（本機舊底 30k）
    if a.old and os.path.exists(a.old):
        od = json.load(open(a.old, encoding="utf-8"))
        km = {"time_min": "time_min", "N": "N", "ho_PSNR": "ho_psnr", "ho_LPIPS": "ho_lpips", "F1": "f1", "g_cov": "g_cov", "g_full": "g_full",
              "g_flat": "g_flat", "g_above": "g_above", "g_under": "g_under", "g_huge_under": "g_huge_under", "sea_n": "sea_n", "sea_area": "sea_area",
              "lamp_gap": "lamp_gap", "iso_gap": "iso_gap", "car_gap": "car_gap", "facade_pct": "facade_pct", "wall_fit": "wall_fit",
              "al_50": "al_50", "al_150": "al_150", "al_250": "al_250", "hole_50": "hole_50", "hole_250": "hole_250", "lamp_n": "lamp_n", "car_n": "car_n"}
        for r in od["runs"]:
            v = {km[k]: x for k, x in (r.get("vals") or {}).items() if k in km and x is not None}
            out.append({"id": "old|" + r["code"], "machine": "local-old", "group": "11-06～08 本機舊底", "blk": int(r["blk"][1:]), "run": r["code"],
                        "label": r["name"], "v": v, "src": ["11-09 總帳（report_data_1109.json）"], "status": "歷史",
                        "note": (r.get("mech") or "") + ("｜" + r["verdict"] if r.get("verdict") else ""),
                        "base": ("old|" + r["base"]) if r.get("base") else None})
    PRI = {"現行": 0, "建議採用": 0.5, "待驗證": 1, "取捨": 2, "待評": 3, "打平": 4, "不採用": 5, "參考": 6, "歷史": 7}
    out.sort(key=lambda r: ({"lab": 0, "local": 1, "ref": 2, "local-old": 3}.get(r["machine"], 9), PRI.get(r["status"], 9), r["blk"] or 0, r["label"]))
    import math
    for r in out:                                          # NaN／inf 不是合法 JSON（瀏覽器整段讀不進來）=> 丟掉
        r["v"] = {k: x for k, x in r["v"].items() if isinstance(x, (int, float)) and math.isfinite(x)}
    data = {"generated": __import__("time").strftime("%Y-%m-%d %H:%M"), "n": len(out), "rows": out}
    s = open(a.html, encoding="utf-8").read()
    js = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    s2, n = re.subn(r'(<script id="ledger" type="application/json">)(.*?)(</script>)', lambda m: m.group(1) + js + m.group(3), s, flags=re.S)
    assert n == 1, "html 裡找不到 <script id=\"ledger\">"
    open(a.html, "w", encoding="utf-8").write(s2)
    from collections import Counter
    print(f"總帳 {len(out)} 列：", dict(Counter(r["group"] for r in out)), "｜狀態", dict(Counter(r["status"] for r in out)))


if __name__ == "__main__":
    main()
