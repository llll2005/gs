"""★ 2026-11-13 全場景版的各項量測：把 16 塊的 eval_arms_lab.sh CPU 項（logs/eval_arms_fullprobe_b<B>.log）彙總成一張表。
使用者 11-13：「我們是否有 CityGS full blk merge 的 F1 跟各項數據？如果沒有補量測」。

彙總規則（每塊各模型都只看塊內，16 塊合起來＝全場景）：
  - 可以加總的量直接加總：塊內不透明顆數、海面上方顆數與 Σo·面積、路燈帶／車輛帶顆數。
  - 比例：用逐塊的分母加權——立面％等用塊內不透明顆數、朝向帶用 CityGS 立面參考顆數、路燈／車缺口用 CityGS 參考顆數。
  - 地面層的「層」（覆蓋 ÷ 塊內地面格面積）：log 裡沒有逐塊地面格數 => 16 塊等權平均，另列中位數。
用法：python tools/probe/fullscene_probe_agg.py logs/eval_arms_fullprobe_b{0..15}.log
"""
import re
import sys
from collections import defaultdict

NUM = r"[-+]?\d[\d,]*\.?\d*|nan"


def f(x):
    x = x.replace(",", "").replace("%", "")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def sections(text):
    out, cur = {}, None
    for line in text.split("\n"):
        m = re.match(r"^### (\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return out


def main(paths):
    ground = defaultdict(list)          # 模型 -> [每塊 9 欄]
    facade = defaultdict(list)          # 模型 -> [(顆數, 立面%, 貼牆中位, 側視%, 平躺%)]
    orient = defaultdict(lambda: defaultdict(lambda: [0.0, [0.0, 0.0, 0.0, 0.0]]))   # 模型 -> 帶 -> [權重和, 加權四欄]
    sea = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])   # 模型 -> [o>0.5, 0.1~0.5, 0.02~0.1, Σo·面積, >20m, >50m, 有海面的塊數]
    small = defaultdict(lambda: [[0.0] * 6, 0.0, 0.0])       # 模型 -> [加權缺口 6 欄, 路燈帶顆數, 車輛帶顆數]
    small_w = [0.0, 0.0, 0.0]                                 # CityGS 參考：路燈帶、孤立柱、車輛帶
    nblk = 0
    for p in paths:
        try:
            S = sections(open(p, encoding="utf-8", errors="replace").read())
        except OSError:
            print(f"⚠ 缺 {p}")
            continue
        nblk += 1
        for line in S.get("ground", []):
            t = line.split()
            if len(t) >= 10 and not line.startswith("b") and "貼地" not in line:
                vals = [f(x) for x in t[-9:]]
                ground[" ".join(t[:-9])].append(vals)
        cur = None
        for line in S.get("facade", []):
            m = re.match(r"^== (.+)$", line.strip())
            if m:
                cur = m.group(1).strip()
                continue
            m = re.search(r"塊內不透明\s+([\d,]+)\s+立面\s+([\d.]+)%\s+貼牆中位\s+([\d.]+)\s+側視\s+([\d.]+)%\s+平躺\s+([\d.]+)%", line)
            if m and cur:
                facade[cur].append(tuple(f(x) for x in m.groups()))
        ref = {}
        for line in S.get("orient", []):
            m = re.search(r"立面參考顆粒：50~150m ([\d,]+)、150~250m ([\d,]+)、>250m ([\d,]+)", line)
            if m:
                ref = dict(zip(("50~150m", "150~250m", ">250m"), (f(x) for x in m.groups())))
                continue
            if "｜" in line and ref:
                name = line.split("｜")[0].strip()
                for seg in line.split("｜")[1:]:
                    mm = re.match(r"\s*(50~150m|150~250m|>250m)\s+(.+)", seg)
                    if not mm:
                        continue
                    v = [f(x) for x in mm.group(2).split("／")]
                    w = ref.get(mm.group(1), 0.0)
                    if len(v) == 4 and w > 0 and all(x == x for x in v):
                        acc = orient[name][mm.group(1)]
                        acc[0] += w
                        for k in range(4):
                            acc[1][k] += w * v[k]
        for line in S.get("sea", []):
            m = re.match(r"^(\S+?)塊內\s+(.+)$", line.strip())
            if m:
                t = [f(x) for x in m.group(2).split()]
                if len(t) >= 10:
                    a = sea[m.group(1)]
                    a[0] += t[0]; a[1] += t[1]; a[2] += t[2]; a[3] += t[3]; a[4] += t[8]; a[5] += t[9]; a[6] += 1
        w = None
        for line in S.get("smallobj", []):
            m = re.search(r"路燈帶 3～15 m ([\d,]+) 顆、其中孤立柱 ([\d,]+) 顆.*車輛帶 0.5～2.5 m ([\d,]+) 顆", line)
            if m:
                w = [f(x) for x in m.groups()]
                for k in range(3):
                    small_w[k] += w[k]
                continue
            m = re.match(r"^(\S+)\s+(\d+)/(\d+)\s*%\s+(\d+)/(\d+)\s*%\s+(\d+)/(\d+)\s*%\s+([\d,]+)\s+([\d,]+)", line.strip())
            if m and w:
                g = [f(x) for x in m.groups()[1:7]]
                a = small[m.group(1)]
                for k, wk in enumerate((w[0], w[0], w[1], w[1], w[2], w[2])):
                    a[0][k] += wk * g[k]
                a[1] += f(m.group(8)); a[2] += f(m.group(9))
    print(f"彙總 {nblk} 塊（{', '.join(paths[:1])} …）")
    print("\n### 地面／地下層（16 塊等權平均；中位數在括號）")
    cols = ("貼地覆蓋", "蓋滿格%", "貼地o>0.9", "貼地躺平", "地上覆蓋", "地下覆蓋", "地下o>0.9", "巨大覆蓋", "巨大在地下")
    print(f"{'模型':16}" + "".join(f"{c:>16}" for c in cols))
    for name, rows in ground.items():
        import statistics as st
        out = []
        for k in range(9):
            v = [r[k] for r in rows if r[k] == r[k]]
            out.append(f"{st.mean(v):7.2f}({st.median(v):5.2f})" if v else "—")
        print(f"{name:16}" + "".join(f"{x:>16}" for x in out) + f"  ｜{len(rows)} 塊")
    print("\n### 立面（塊內不透明顆數加總；比例用顆數加權；貼牆中位＝逐塊中位的加權平均）")
    for name, rows in facade.items():
        n = sum(r[0] for r in rows)
        if n <= 0:
            continue
        wavg = lambda k: sum(r[0] * r[k] for r in rows) / n
        print(f"{name:16} 塊內不透明 {n:,.0f}  立面 {wavg(1):.1f}%  貼牆中位 {wavg(2):.3f}  側視 {wavg(3):.1f}%  平躺 {wavg(4):.1f}%  ｜{len(rows)} 塊")
    print("\n### 立面朝向帶（用 CityGS 立面參考顆數加權）：沒有／對齊／平躺／歪（%）")
    for name, bands in orient.items():
        segs = []
        for b in ("50~150m", "150~250m", ">250m"):
            w, acc = bands.get(b, [0.0, [0, 0, 0, 0]])
            segs.append(f"{b} " + ("／".join(f"{x / w:4.1f}" for x in acc) if w > 0 else "—"))
        print(f"{name:16}｜" + "｜".join(segs))
    print("\n### 海面上方（有海面格的塊加總）")
    print(f"{'模型':16}{'o>0.5':>10}{'0.1~0.5':>10}{'0.02~0.1':>10}{'Σo·面積 m²':>14}{'>20m':>8}{'>50m':>8}  塊數")
    for name, a in sea.items():
        print(f"{name:16}{a[0]:>10,.0f}{a[1]:>10,.0f}{a[2]:>10,.0f}{a[3]:>14,.0f}{a[4]:>8,.0f}{a[5]:>8,.0f}  {a[6]}")
    print(f"\n### 路燈與車（缺口用 CityGS 參考顆數加權；參考合計：路燈帶 {small_w[0]:,.0f}、孤立柱 {small_w[1]:,.0f}、車輛帶 {small_w[2]:,.0f}）")
    print(f"{'模型':16}{'路燈帶缺口0.5/1m':>18}{'孤立柱缺口':>14}{'車輛帶缺口':>14}{'路燈帶顆數':>12}{'車輛帶顆數':>12}")
    for name, (g, nl, nv) in small.items():
        ws = (small_w[0], small_w[0], small_w[1], small_w[1], small_w[2], small_w[2])
        r = [g[k] / ws[k] if ws[k] > 0 else float("nan") for k in range(6)]
        print(f"{name:16}{r[0]:>12.0f}/{r[1]:<4.0f}%{r[2]:>9.0f}/{r[3]:<3.0f}%{r[4]:>9.0f}/{r[5]:<3.0f}%{nl:>12,.0f}{nv:>12,.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
