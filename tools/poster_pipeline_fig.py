"""海報流程圖（論文總覽圖風格）：輸入 → 幾何先驗（改良 1、2）→ 分塊訓練（改良 3、4）→ 合併輸出。

產出 SVG（可編輯）與 PNG（貼到 Canva 用）：
  python tools/poster_pipeline_fig.py [--out 紀錄/海報圖/流程圖] [--scale 2]
圖上數字的出處同 `紀錄/海報文字稿.md` 附錄（數字出處表）。
"""
import argparse
import math
import subprocess

import numpy as np
from scipy.spatial import Delaunay

W, H = 2990, 960   # ★ 11-27 v2：海報騰出空間（約 3:1）=> 加高到 960，補細節
FONT = "Noto Sans CJK TC"
NAVY = "#1F4E79"
NAVY2 = "#2E6A9E"
TXT = "#1F3B57"
SUB = "#4A6178"
PANEL = "#F4F8FC"
EDGE = "#C5D6E6"
ORG = "#E8833A"
ORG_D = "#C2611F"
ORG_L = "#FCE9DA"
SEA = "#5BA4D6"
SEA_L = "#CFE6F5"
GREEN = "#3A9E5C"
RED = "#D64545"
ROOFS = ["#A9B4C0", "#C4B49F", "#B9A58C", "#97A5B5", "#D1C3B0", "#8F9DAE"]
GCOL = ["#E8833A", "#4A90C2", "#7FB069", "#D4A373", "#9C89B8", "#E9C46A", "#5BA4D6", "#C97B63"]

out = []
defs = []
_uid = [0]


def uid(p="u"):
    _uid[0] += 1
    return f"{p}{_uid[0]}"


def add(s):
    out.append(s)


def rect(x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0, op=1.0, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d} {extra}/>')


def text(x, y, s, size=22, fill=TXT, weight=400, anchor="middle", op=1.0, style=""):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" opacity="{op}" {style}>{s}</text>')


def line(x1, y1, x2, y2, stroke=NAVY, sw=3, dash=None, op=1.0, cap="round", marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="{cap}" opacity="{op}"{d}{m}/>')


def poly(pts, fill="none", stroke="none", sw=1.0, op=1.0, extra=""):
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    add(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}" '
        f'stroke-linejoin="round" {extra}/>')


def polyline(pts, stroke=NAVY, sw=3, op=1.0, dash=None, marker=None):
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    add(f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{sw}" opacity="{op}" '
        f'stroke-linejoin="round" stroke-linecap="round"{d}{m}/>')


def arrow(x1, y1, x2, y2, color=NAVY, sw=3.5, dash=None):
    mk = "ah_navy" if color == NAVY else ("ah_org" if color == ORG else ("ah_sea" if color == SEA else "ah_grey"))
    # 縮短讓箭頭尖端落在終點
    L = math.hypot(x2 - x1, y2 - y1)
    k = max(L - 9, 0) / L if L > 0 else 0
    line(x1, y1, x1 + (x2 - x1) * k, y1 + (y2 - y1) * k, stroke=color, sw=sw, dash=dash, marker=mk)


GRAD_IDS = {}


def grad(color):
    if color not in GRAD_IDS:
        gid = uid("g")
        GRAD_IDS[color] = gid
        defs.append(f'<radialGradient id="{gid}" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0" stop-color="{color}" stop-opacity="0.95"/>'
                    f'<stop offset="0.55" stop-color="{color}" stop-opacity="0.75"/>'
                    f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient>')
    return GRAD_IDS[color]


def surfel(cx, cy, u, v, color, r=1.0, op=1.0, outline=False):
    """平面上的 2D 高斯：單位圓經 [u v] 仿射映射到螢幕（u、v＝平面兩個切向量的螢幕向量，長度＝半徑）。"""
    a, b = u
    c, d = v
    g = grad(color)
    ol = f' stroke="{color}" stroke-width="{0.06 / max(math.hypot(a, b), 1e-6) * 18:.3f}" stroke-opacity="0.5"' if outline else ""
    add(f'<circle cx="0" cy="0" r="{r}" fill="url(#{g})" opacity="{op}"{ol} '
        f'transform="matrix({a:.3f} {b:.3f} {c:.3f} {d:.3f} {cx:.2f} {cy:.2f})"/>')


C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))


def iso(ox, oy, s):
    def f(X, Y, Z):
        return ox + (X - Y) * C30 * s, oy + (X + Y) * S30 * s - Z * s
    return f


def iso_vecs(s):
    ux = (C30 * s, S30 * s)
    uy = (-C30 * s, S30 * s)
    uz = (0.0, -s)
    return ux, uy, uz


def box(P, X0, Y0, X1, Y1, h, col, shade=True):
    top = [P(X0, Y0, h), P(X1, Y0, h), P(X1, Y1, h), P(X0, Y1, h)]
    fx = [P(X1, Y0, 0), P(X1, Y1, 0), P(X1, Y1, h), P(X1, Y0, h)]   # X=X1 面（右前）
    fy = [P(X0, Y1, 0), P(X1, Y1, 0), P(X1, Y1, h), P(X0, Y1, h)]   # Y=Y1 面（左前）
    poly(fy, fill=shade_col(col, 0.80), stroke="#ffffff", sw=0.8)
    poly(fx, fill=shade_col(col, 0.66), stroke="#ffffff", sw=0.8)
    poly(top, fill=col, stroke="#ffffff", sw=0.8)


def shade_col(hexc, k):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (int(r * k), int(g * k), int(b * k))


def mix(hexc, other, t):
    h1, h2 = hexc.lstrip("#"), other.lstrip("#")
    c1 = [int(h1[i:i + 2], 16) for i in (0, 2, 4)]
    c2 = [int(h2[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(a * (1 - t) + b * t) for a, b in zip(c1, c2))


def clip_rect(x, y, w, h, rx=6):
    cid = uid("c")
    defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/></clipPath>')
    return cid


def city_layout(seed, sea=False):
    """俯視城市（正規化座標 0..1）：道路、建物矩形、海。"""
    rng = np.random.default_rng(seed)
    xs = [0.0, 0.3, 0.62, 1.0]
    ys = [0.0, 0.36, 0.7, 1.0]
    blds = []
    for i in range(3):
        for j in range(3):
            if sea and i == 2 and j >= 1:
                continue
            x0, x1 = xs[i] + 0.04, xs[i + 1] - 0.04
            y0, y1 = ys[j] + 0.05, ys[j + 1] - 0.05
            n = rng.integers(1, 4)
            for k in range(n):
                bw = (x1 - x0) * rng.uniform(0.35, 0.8)
                bh = (y1 - y0) * rng.uniform(0.35, 0.8)
                bx = rng.uniform(x0, x1 - bw)
                by = rng.uniform(y0, y1 - bh)
                blds.append((bx, by, bw, bh, ROOFS[rng.integers(len(ROOFS))], rng.uniform(0.3, 1.0)))
    sea_poly = [(0.66, 0.40), (1.0, 0.30), (1.0, 1.0), (0.60, 1.0)] if sea else None
    return xs, ys, blds, sea_poly


def aerial(x, y, w, h, seed, sea=False, mode="rgb", blur=False, rx=6, border=True):
    """mode: rgb（空拍）、depth（單目深度，灰階、近亮遠暗）、depthc（校正後）。"""
    cid = clip_rect(x, y, w, h, rx)
    xs, ys, blds, sp = city_layout(seed, sea)
    add(f'<g clip-path="url(#{cid})">')
    if mode == "rgb":
        rect(x, y, w, h, fill="#C9D2C3")
        for xx in xs[1:-1]:
            rect(x + xx * w - 4, y, 8, h, fill="#ECEDEA")
        for yy in ys[1:-1]:
            rect(x, y + yy * h - 4, w, 8, fill="#ECEDEA")
        if sp:
            poly([(x + a * w, y + b * h) for a, b in sp], fill="#7FB3D9")
        for bx, by, bw, bh, col, ht in blds:
            rect(x + bx * w + 3, y + by * h + 3, bw * w, bh * h, fill="#7d8a7a", op=0.45)
            rect(x + bx * w, y + by * h, bw * w, bh * h, fill=col, stroke="#ffffff", sw=0.6)
    else:
        gid = uid("lg")
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
                    f'<stop offset="0" stop-color="#2b2f36"/><stop offset="1" stop-color="#8a909a"/></linearGradient>')
        rect(x, y, w, h, fill=f"url(#{gid})")
        flt = ""
        if blur:
            fid = uid("f")
            defs.append(f'<filter id="{fid}" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3.2"/></filter>')
            flt = f' filter="url(#{fid})"'
        add(f'<g{flt}>')
        if sp:
            poly([(x + a * w, y + b * h) for a, b in sp], fill="#1d2024" if not blur else "#3a3f46")
        for bx, by, bw, bh, col, ht in blds:
            g = int(150 + 100 * ht * (0.55 + 0.45 * (by + bh)))
            if blur:
                g = int(g * 0.85 + 20)
            g = min(g, 250)
            rect(x + bx * w, y + by * h, bw * w, bh * h, fill="#%02x%02x%02x" % (g, g, g))
        add('</g>')
        if blur:
            # 單目深度的低頻偏差：整片色偏
            gid2 = uid("lg")
            defs.append(f'<linearGradient id="{gid2}" x1="0" y1="0" x2="1" y2="1">'
                        f'<stop offset="0" stop-color="#ffffff" stop-opacity="0.18"/><stop offset="1" stop-color="#000000" stop-opacity="0.25"/></linearGradient>')
            rect(x, y, w, h, fill=f"url(#{gid2})")
    add('</g>')
    if border:
        rect(x, y, w, h, stroke="#8FA6BC", sw=1.5, rx=rx)
    return blds, sp


def sfm_overlay(x, y, w, h, seed):
    """SfM 點（橘點）＋三角形（Delaunay）疊在深度圖上。"""
    rng = np.random.default_rng(seed + 7)
    xs, ys, blds, sp = city_layout(seed, False)
    pts = []
    for bx, by, bw, bh, col, ht in blds:
        for _ in range(3):
            pts.append((bx + rng.uniform(0.1, 0.9) * bw, by + rng.uniform(0.1, 0.9) * bh))
    for _ in range(10):
        pts.append((rng.uniform(0.03, 0.97), rng.uniform(0.03, 0.97)))
    pts += [(0.02, 0.02), (0.98, 0.02), (0.02, 0.98), (0.98, 0.98)]
    P = np.array(pts)
    tri = Delaunay(P)
    for s in tri.simplices:
        q = [(x + P[i, 0] * w, y + P[i, 1] * h) for i in s]
        poly(q, stroke=ORG, sw=1.1, op=0.75)
    for px, py in P[:-4]:
        add(f'<circle cx="{x + px * w:.1f}" cy="{y + py * h:.1f}" r="3.2" fill="{ORG}" stroke="#ffffff" stroke-width="1"/>')


def badge(x, y, n, title, size=26):
    add(f'<circle cx="{x}" cy="{y}" r="18" fill="{ORG}"/>')
    text(x, y + 8, str(n), size=22, fill="#ffffff", weight=700)
    text(x + 30, y + 9, title, size=size, fill=ORG_D, weight=700, anchor="start")


def pill(x, y, label, size=15, fill=ORG):
    """橘底白字膠囊（例：「新方法」）；(x, y)＝左緣、文字基線。"""
    w = len(label) * size + 22
    rect(x, y - size - 3, w, size + 12, fill=fill, rx=(size + 12) / 2)
    text(x + w / 2, y + 2, label, size=size, fill="#ffffff", weight=700)


def panel(x, y, w, h, title):
    rect(x, y, w, h, fill=PANEL, stroke=EDGE, sw=2, rx=14)
    cid = clip_rect(x, y, w, 60, 14)
    add(f'<g clip-path="url(#{cid})">')
    rect(x, y, w, 74, fill=NAVY, rx=14)
    add('</g>')
    text(x + w / 2, y + 40, title, size=28, fill="#ffffff", weight=700)


def camera(cx, cy, s=1.0, ang=0.0, col=NAVY, frustum=True):
    """相機圖示：機身（含鏡頭）朝下；frustum＝畫半透明視錐（ang＝相對垂直向下的角度，度）。"""
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) rotate({ang:.1f}) scale({s:.2f})">')
    if frustum:
        poly([(-6, 6), (6, 6), (20, 34), (-20, 34)], fill=NAVY2, op=0.22)
    rect(-14, -14, 28, 16, fill=col, rx=4)
    rect(-6, 1, 12, 6, fill=col, rx=1.5)
    add(f'<circle cx="0" cy="-6" r="4.5" fill="#ffffff" opacity="0.9"/>')
    add('</g>')


def grp(tx, ty, k=1.0):
    """把一段原座標的內容整體縮放＋平移：新座標＝k×原座標＋(tx, ty)。"""
    add(f'<g transform="matrix({k} 0 0 {k} {tx:.2f} {ty:.2f})">')


def end():
    add('</g>')


def chevron(x, y):
    poly([(x, y - 34), (x + 22, y), (x, y + 34), (x + 12, y)], fill=NAVY, op=0.85)


# ─────────────────────────── 繪圖 ───────────────────────────
def draw():
    rng = np.random.default_rng(11)
    rect(0, 0, W, H, fill="#ffffff")

    # ════ 面板 1：輸入 ════
    panel(20, 20, 330, 920, "輸入")
    grp(2.25, 11.2, 0.85)
    for i, (cx, ang) in enumerate([(130, -14), (215, 0), (300, 14)]):
        camera(cx, 104, 1.1, ang)
    for k in range(3):
        aerial(55 + 16 * k, 150 + 16 * k, 250, 155, seed=3 + k, sea=(k == 2))
    text(215, 362, "空拍影像＋相機位置", size=24, weight=700)
    # 分岔到 SfM 與單目深度
    polyline([(60, 375), (60, 545)], stroke=NAVY, sw=3.5)
    polyline([(60, 545), (60, 840)], stroke=NAVY, sw=3.5)
    arrow(60, 545, 92, 545)
    arrow(60, 840, 92, 840)
    # SfM 點雲（等角投影的點）
    rect(95, 470, 290, 150, fill="#ffffff", stroke="#8FA6BC", sw=1.5, rx=6)
    _c = clip_rect(95, 470, 290, 150, 6)
    add(f'<g clip-path="url(#{_c})">')
    P = iso(232, 598, 21)
    pr = np.random.default_rng(5)
    for (X0, Y0, X1, Y1, hgt, col) in [(-2.4, -1.2, -1.2, 0.0, 2.6, "#6f8299"), (-0.8, -2.6, 0.6, -1.4, 3.6, "#9a8467"),
                                        (0.4, 0.0, 1.7, 1.1, 1.6, "#7f93ab"), (-1.9, 0.8, -0.8, 1.9, 1.2, "#a08f74")]:
        # 可見的 9 條稜邊（取點較密）＋三個可見面（較疏）
        E = [((X0, Y0, hgt), (X1, Y0, hgt)), ((X1, Y0, hgt), (X1, Y1, hgt)), ((X1, Y1, hgt), (X0, Y1, hgt)), ((X0, Y1, hgt), (X0, Y0, hgt)),
             ((X1, Y0, 0), (X1, Y0, hgt)), ((X1, Y1, 0), (X1, Y1, hgt)), ((X0, Y1, 0), (X0, Y1, hgt)),
             ((X1, Y0, 0), (X1, Y1, 0)), ((X1, Y1, 0), (X0, Y1, 0))]
        for (pa, pb) in E:
            L = math.dist(pa, pb)
            for _ in range(int(9 * L) + 2):
                t = pr.uniform()
                q = [pa[k] + (pb[k] - pa[k]) * t + pr.normal(0, 0.04) for k in range(3)]
                sx, sy = P(*q)
                add(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="1.9" fill="{col}"/>')
        for _ in range(28):
            face = pr.integers(3)
            if face == 0:
                X, Y, Z = pr.uniform(X0, X1), pr.uniform(Y0, Y1), hgt
            elif face == 1:
                X, Y, Z = X1, pr.uniform(Y0, Y1), pr.uniform(0, hgt)
            else:
                X, Y, Z = pr.uniform(X0, X1), Y1, pr.uniform(0, hgt)
            sx, sy = P(X, Y, Z)
            add(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="1.7" fill="{col}" opacity="0.7"/>')
    for _ in range(110):
        sx, sy = P(pr.uniform(-3.0, 2.6), pr.uniform(-3.0, 2.6), 0)
        add(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="1.6" fill="#9aa99a"/>')
    add('</g>')
    text(240, 652, "SfM 稀疏點", size=23, weight=700)
    text(240, 680, "（用已知相機位置三角化）", size=17, fill=SUB)
    # 單目深度
    aerial(95, 765, 290, 150, seed=4, mode="depth", blur=True)
    text(240, 946, "單目深度", size=23, weight=700)
    text(240, 976, "（Depth Anything V2）", size=19, fill=SUB)

    end()
    text(185, 888, "MatrixCity 空拍城市", size=17, fill=SUB, weight=700)
    text(185, 914, "訓練影像 5,621 張", size=16, fill=SUB)
    chevron(359, 480)

    # ════ 面板 2：幾何先驗 ════
    panel(390, 20, 800, 920, "幾何先驗：讓高斯長在真實表面上")
    grp(-6.7, 0.6, 0.92)
    # ── 改良 1 ──
    badge(490, 116, 1, "SfM 校正深度")
    pill(715, 116, "新方法")
    aerial(480, 145, 220, 135, seed=4, mode="depth", blur=True)
    sfm_overlay(480, 145, 220, 135, seed=4)
    text(590, 306, "SfM 點連成三角形", size=20, fill=SUB)
    arrow(708, 212, 752, 212, sw=4)
    text(730, 196, "逐張", size=17, fill=SUB)
    text(730, 238, "校正", size=17, fill=SUB)
    aerial(760, 145, 220, 135, seed=4, mode="depthc")
    text(870, 306, "校正後的深度", size=20, fill=SUB)
    # 校正示意（沿一條掃描線）：單目深度形狀對、尺度歪 → 用 SfM 點拉回
    text(1130, 166, "沿一條掃描線看", size=18, weight=700)
    fx0, fx1, fy0, fy1 = 1022, 1250, 180, 276          # 圖框（上＝近、下＝遠）
    line(fx0, fy0 - 4, fx0, fy1, stroke=SUB, sw=1.6, cap="butt")
    line(fx0, fy1, fx1, fy1, stroke=SUB, sw=1.6, cap="butt")
    text(fx0 - 6, fy0 + 6, "近", size=13, fill=SUB, anchor="end")
    text(fx0 - 6, fy1, "遠", size=13, fill=SUB, anchor="end")
    u = np.linspace(0, 1, 241)
    near = np.full_like(u, 0.12)                          # 地面
    near[(u > 0.22) & (u < 0.48)] = 0.78                  # 高樓
    near[(u > 0.62) & (u < 0.8)] = 0.45                   # 矮樓
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); k /= k.sum()
    mono = np.convolve(np.pad(near, 12, mode="edge"), k, mode="valid")
    mono = 0.5 * mono + 0.30 + 0.20 * (u - 0.5)           # 單目：邊緣模糊＋整體縮放、偏移、傾斜
    X = fx0 + 6 + u * (fx1 - fx0 - 10)
    Y = lambda v: fy1 - 6 - v * (fy1 - fy0 - 14)
    polyline(list(zip(X, Y(mono))), stroke="#9AA6B2", sw=2.4, dash="6 4")
    polyline(list(zip(X, Y(near))), stroke=NAVY2, sw=3)
    for uu in (0.08, 0.33, 0.70, 0.92):
        i = int(round(uu * (len(u) - 1)))
        y1, y2 = Y(mono[i]), Y(near[i])
        d = 1 if y2 > y1 else -1                         # 往下 d=1、往上 d=-1
        tip = y2 - d * 6                                 # 停在點的外緣
        line(X[i], y1, X[i], tip - d * 6, stroke=ORG, sw=2, cap="butt")
        poly([(X[i] - 4.5, tip - d * 7), (X[i] + 4.5, tip - d * 7), (X[i], tip)], fill=ORG)
    for uu in (0.08, 0.15, 0.30, 0.40, 0.70, 0.75, 0.92):
        i = int(round(uu * (len(u) - 1)))
        add(f'<circle cx="{X[i]:.1f}" cy="{Y(near[i]):.1f}" r="4.2" fill="{ORG}" stroke="#ffffff" stroke-width="1.2"/>')
    lx, ly = 1012, 302
    line(lx, ly - 5, lx + 26, ly - 5, stroke="#9AA6B2", sw=2.4, dash="6 4")
    text(lx + 31, ly, "單目", size=14, fill=SUB, anchor="start")
    add(f'<circle cx="{lx + 84}" cy="{ly - 5}" r="4.2" fill="{ORG}" stroke="#ffffff" stroke-width="1.2"/>')
    text(lx + 93, ly, "SfM 點", size=14, fill=SUB, anchor="start")
    line(lx + 152, ly - 5, lx + 178, ly - 5, stroke=NAVY2, sw=3)
    text(lx + 183, ly, "校正後", size=14, fill=SUB, anchor="start")
    text(1130, 334, "誤差中位數 2.15% → 0.15%", size=17, fill=ORG_D, weight=700)
    add('<g transform="translate(0,34)">')                # ★ v2：三個用途整列下移，騰出誤差那行
    # 分岔到三個用途
    polyline([(870, 318), (870, 336)], stroke=NAVY, sw=3.5)
    polyline([(600, 336), (1140, 336)], stroke=NAVY, sw=3.5)
    for cx in (600, 870, 1140):
        arrow(cx, 336, cx, 362)
    # a 塊外遮罩
    aerial(510, 366, 180, 96, seed=8, rx=5, border=False)
    hid = uid("h")
    defs.append(f'<pattern id="{hid}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="8" height="8" fill="#ffffff" fill-opacity="0.55"/><line x1="0" y1="0" x2="0" y2="8" stroke="#7a8796" stroke-width="2"/></pattern>')
    poly([(510, 366), (690, 366), (690, 462), (510, 462), (510, 366), (548, 384), (548, 444), (652, 444), (652, 384), (548, 384)],
         fill=f"url(#{hid})", extra='fill-rule="evenodd"')
    rect(548, 384, 104, 60, stroke=ORG, sw=3.5, rx=3)
    rect(510, 366, 180, 96, stroke="#8FA6BC", sw=1.5, rx=5)
    text(600, 494, "塊外遮罩", size=22, weight=700)
    text(600, 520, "分出屬於這一塊的像素", size=18, fill=SUB)
    text(600, 545, "塊外要透明・被擋住的不計分", size=16, fill=SUB)
    # b 深度上限（剖面）
    gx0, gy = 780, 438
    rect(gx0, gy, 180, 26, fill="#D8CDBE", rx=3)
    rect(845, 382, 52, 56, fill="#C9D3DE", stroke="#8FA6BC", sw=1.2)
    for (x1, y1, x2, y2) in [(786, 433, 810, 433), (818, 433, 840, 433), (902, 433, 926, 433), (932, 433, 954, 433),
                             (850, 377, 872, 377), (875, 377, 894, 377), (901, 390, 901, 412), (840, 395, 840, 418)]:
        line(x1, y1, x2, y2, stroke=NAVY2, sw=5)
    for (x1, y1, x2, y2) in [(796, 455, 818, 455), (920, 452, 944, 452)]:
        line(x1, y1, x2, y2, stroke=RED, sw=5)
    for cxr in (807, 932):
        line(cxr - 6, 447, cxr + 6, 461, stroke=RED, sw=2.5)
        line(cxr + 6, 447, cxr - 6, 461, stroke=RED, sw=2.5)
    text(870, 494, "逐貢獻深度上限", size=22, weight=700, fill=ORG_D)
    text(870, 520, "只罰地面、屋頂後方的貢獻", size=18, fill=SUB)
    text(870, 545, "梯度只回饋前方的高斯", size=16, fill=SUB)
    text(870, 570, "表面下的高斯約減半", size=16, fill=ORG_D, weight=700)
    # c 表面起點
    prof = [(1050, 440), (1105, 440), (1105, 384), (1160, 384), (1160, 412), (1195, 412), (1195, 440), (1230, 440)]
    rect(1050, 440, 180, 24, fill="#D8CDBE", rx=3)
    polyline(prof, stroke="#8FA6BC", sw=2.5)
    seg = np.array(prof, dtype=float)
    for i in range(len(seg) - 1):
        a, b = seg[i], seg[i + 1]
        n = max(int(np.linalg.norm(b - a) / 11), 1)
        for t in np.linspace(0, 1, n, endpoint=False):
            p = a + (b - a) * t
            add(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="4" fill="{GCOL[int(rng.integers(len(GCOL)))]}" opacity="0.9"/>')
    text(1140, 494, "表面起點", size=22, weight=700)
    text(1140, 520, "高斯從表面開始長", size=18, fill=SUB)
    text(1140, 545, "校正深度反投影成起點", size=16, fill=SUB)
    add('</g>')

    line(475, 645, 1255, 645, stroke=EDGE, sw=2, dash="8 6")

    # ── 改良 2 ──（★ v2：整段下移 72）
    add('<g transform="translate(0,95)">')
    badge(490, 590, 2, "海面當成一個平面")
    pill(740, 590, "新方法")
    blds, sp = aerial(480, 620, 220, 135, seed=21, sea=True)
    poly([(480 + a * 220, 620 + b * 135) for a, b in sp], fill=SEA, op=0.45, stroke="#1f78c1", sw=3)
    text(480 + 0.84 * 220, 620 + 0.80 * 135, "水體", size=19, fill="#ffffff", weight=700)
    text(590, 782, "偵測水體", size=20, fill=SUB)
    arrow(708, 687, 752, 687, sw=4)
    # 多視角共識平面
    Pp = iso(880, 700, 34)
    plane = [Pp(-2.2, -0.4, 0), Pp(1.2, -0.4, 0), Pp(1.2, 2.2, 0), Pp(-2.2, 2.2, 0)]
    poly(plane, fill=SEA_L, stroke=SEA, sw=2)
    for t in np.linspace(-2.2, 1.2, 6)[1:-1]:
        a, b = Pp(t, -0.4, 0), Pp(t, 2.2, 0)
        line(*a, *b, stroke=SEA, sw=1, op=0.7, cap="butt")
    for t in np.linspace(-0.4, 2.2, 5)[1:-1]:
        a, b = Pp(-2.2, t, 0), Pp(1.2, t, 0)
        line(*a, *b, stroke=SEA, sw=1, op=0.7, cap="butt")
    hits = [Pp(-0.6, 0.5, 0), Pp(0.1, 1.2, 0), Pp(-1.0, 1.4, 0)]
    cams = [(800, 615, -18), (885, 600, 0), (965, 612, 18)]
    for (cx, cy, ang) in cams:
        for hx, hy in hits:
            line(cx, cy + 14, hx, hy, stroke=NAVY2, sw=1.4, dash="5 4", op=0.75)
        camera(cx, cy, 0.85, ang, frustum=False)
    for hx, hy in hits:
        add(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="5" fill="{ORG}" stroke="#ffffff" stroke-width="1.5"/>')
    text(880, 782, "多視角共識決定海平面", size=20, fill=SUB)
    arrow(1008, 687, 1048, 687, sw=4)
    # 收回漂浮的高斯
    Pq = iso(1150, 712, 30)
    plane2 = [Pq(-2.0, -0.5, 0), Pq(1.3, -0.5, 0), Pq(1.3, 2.0, 0), Pq(-2.0, 2.0, 0)]
    poly(plane2, fill=SEA_L, stroke=SEA, sw=2)
    ux, uy, uz = iso_vecs(30)
    for (X, Y, Z) in [(-1.0, 0.2, 1.9), (0.3, 0.9, 1.5), (-0.6, 1.4, 2.4)]:
        sx, sy = Pq(X, Y, Z)
        tx, ty = Pq(X, Y, 0)
        surfel(sx, sy, (ux[0] * 0.55, ux[1] * 0.55), (uy[0] * 0.55, uy[1] * 0.55), "#9BB7CF", op=0.8)
        arrow(sx, sy + 10, tx, ty - 6, color=SEA, sw=2.5, dash="5 4")
        surfel(tx, ty, (ux[0] * 0.55, ux[1] * 0.55), (uy[0] * 0.55, uy[1] * 0.55), "#2F7FBF", op=0.95)
    text(1150, 782, "收回漂浮的高斯", size=20, fill=SUB)
    text(1150, 806, "海面層錨定在海平面上", size=16, fill=SUB)
    text(880, 840, "→ 海上的漂浮物降到 1/52 以下", size=18, fill=ORG_D, weight=700)
    add('</g>')

    end()
    chevron(1199, 480)

    # ════ 面板 3：分塊訓練（機制流程，不放數字）════
    panel(1230, 20, 1400, 920, "分塊訓練")
    # 分區（左欄上）
    grp(-15, 0)
    gx, gy0, gs = 1355, 96, 160
    aerial(gx, gy0, gs, gs, seed=31, sea=True, rx=4)
    for i in range(1, 4):
        line(gx + i * gs / 4, gy0, gx + i * gs / 4, gy0 + gs, stroke="#ffffff", sw=2.5, cap="butt")
        line(gx, gy0 + i * gs / 4, gx + gs, gy0 + i * gs / 4, stroke="#ffffff", sw=2.5, cap="butt")
    hx0, hy0 = gx + gs / 4, gy0 + gs / 4
    rect(hx0, hy0, gs / 4, gs / 4, stroke=ORG, sw=4, fill=ORG, op=1.0, extra='fill-opacity="0.25"')
    text(gx + gs / 2, 282, "將場景分塊", size=20, weight=700)
    text(gx + gs / 2, 306, "高斯數量上限依 SfM 點數分配", size=15, fill=ORG_D, weight=700)
    text(gx + gs / 2, 328, "不先訓練全場景粗模型", size=15, fill=ORG_D, weight=700)
    end()
    # 放大：一塊的 2D 高斯
    bx0, by0, bw, bh = 1565, 88, 430, 232
    # 放大線（新座標：格子底邊兩角 → 放大框頂邊兩角）
    _cx0, _cy1 = hx0 - 15, hy0 + gs / 4
    line(_cx0, _cy1, 1252, 440, stroke=ORG, sw=1.8, dash="6 5")
    line(_cx0 + gs / 4, _cy1, 1587, 440, stroke=ORG, sw=1.8, dash="6 5")
    grp(31.3, 371.4, 0.78)                          # 放大框（左欄下）：原 (1565,88) 430x232 → (1252,370) 335x181
    rect(bx0, by0, bw, bh, fill="#ffffff", stroke=ORG, sw=2.5, rx=10)
    s = 21
    Pb = iso(1800, 188, s)
    ux, uy, uz = iso_vecs(s)
    gr = np.random.default_rng(17)
    ground = [Pb(-3.4, -2.6, 0), Pb(3.0, -2.6, 0), Pb(3.0, 3.4, 0), Pb(-3.4, 3.4, 0)]
    poly(ground, fill="#EEF1EC", stroke="#D5DBD0", sw=1)
    gpts = [(gr.uniform(-3.2, 2.8), gr.uniform(-2.4, 3.2)) for _ in range(70)]
    B = (-1.2, -1.2, 1.0, 0.8, 3.2)

    def in_b(X, Y):
        return B[0] - 0.3 < X < B[2] + 0.3 and B[1] - 0.3 < Y < B[3] + 0.3

    gcols = ["#A7B59B", "#BFC7B3", "#9FB0A0", "#C9C2B0", "#B5BFC9"]
    behind = [(X, Y) for X, Y in gpts if not in_b(X, Y) and (X + Y) < (B[2] + B[3])]
    front = [(X, Y) for X, Y in gpts if not in_b(X, Y) and (X + Y) >= (B[2] + B[3])]
    for X, Y in behind:
        sx, sy = Pb(X, Y, 0)
        r = gr.uniform(0.28, 0.5)
        surfel(sx, sy, (ux[0] * r, ux[1] * r), (uy[0] * r, uy[1] * r), gcols[gr.integers(len(gcols))])
    box(Pb, *B, "#C3CCD6")
    X0, Y0, X1, Y1, hh = B
    bcols = ["#9FB1C6", "#B7C4D2", "#8EA3BA", "#C7B9A3", "#A9BACB", "#E0C9A6"]
    for _ in range(26):
        X, Y = gr.uniform(X0 + 0.25, X1 - 0.25), gr.uniform(Y0 + 0.25, Y1 - 0.25)
        sx, sy = Pb(X, Y, hh)
        r = gr.uniform(0.22, 0.38)
        surfel(sx, sy, (ux[0] * r, ux[1] * r), (uy[0] * r, uy[1] * r), bcols[gr.integers(len(bcols))])
    for _ in range(36):
        Y, Z = gr.uniform(Y0 + 0.2, Y1 - 0.2), gr.uniform(0.25, hh - 0.25)
        sx, sy = Pb(X1, Y, Z)
        r = gr.uniform(0.2, 0.34)
        surfel(sx, sy, (uy[0] * r, uy[1] * r), (uz[0] * r * 0.9, uz[1] * r * 0.9), bcols[gr.integers(len(bcols))])
    for _ in range(40):
        X, Z = gr.uniform(X0 + 0.2, X1 - 0.2), gr.uniform(0.25, hh - 0.25)
        sx, sy = Pb(X, Y1, Z)
        r = gr.uniform(0.2, 0.34)
        surfel(sx, sy, (ux[0] * r, ux[1] * r), (uz[0] * r * 0.9, uz[1] * r * 0.9), bcols[gr.integers(len(bcols))])
    for X, Y in front:
        sx, sy = Pb(X, Y, 0)
        r = gr.uniform(0.28, 0.5)
        surfel(sx, sy, (ux[0] * r, ux[1] * r), (uy[0] * r, uy[1] * r), gcols[gr.integers(len(gcols))])
    zx, zy = 1630, 134
    surfel(zx, zy, (30, 0), (0, 18), ORG)
    add(f'<ellipse cx="{zx}" cy="{zy}" rx="30" ry="18" fill="none" stroke="{ORG_D}" stroke-width="1.2" stroke-dasharray="3 3"/>')
    text(zx, zy + 42, "一個 2D 高斯", size=19, fill=ORG_D, weight=700)
    end()
    text(1420, 650, "一塊的 2D 高斯", size=19, weight=700)
    text(1420, 674, "（帶顏色的半透明小圓片）", size=15, fill=SUB)
    text(1420, 704, "各塊獨立訓練後合併", size=15, fill=SUB)

    # ── 訓練迴圈（機制）：中欄 ──
    grp(108.6, -175.8, 1.12)                       # 原 (1345,340) → (1615,205)，放大 1.12
    def node(x, y, w, h, title, tsize=20):
        rect(x, y, w, h, fill="#ffffff", stroke=NAVY2, sw=2.2, rx=10)
        if title:
            text(x + w / 2, y + 28, title, size=tsize, weight=700)

    def tag(x, y, n):
        rect(x, y - 15, 58, 21, fill=ORG, rx=10)
        text(x + 29, y + 1, f"改良 {n}", size=14, fill="#ffffff", weight=700)

    R1, R1H = 340, 200
    # 2D 高斯
    node(1345, R1, 185, R1H, "2D 高斯")
    for (dx, dy, c) in [(-38, 0, "#4A90C2"), (0, -6, "#E8833A"), (38, 2, "#7FB069")]:
        surfel(1437 + dx, 408 + dy, (22, 0), (6, 13), c)
    text(1437, 458, "位置・形狀", size=15, fill=SUB)
    text(1437, 480, "不透明度・顏色", size=15, fill=SUB)
    text(1437, 512, "參數 58 → 34 個", size=15, fill=ORG_D, weight=700)
    # 渲染器（改良 4）
    node(1572, R1, 220, R1H, None)
    add(f'<circle cx="1596" cy="{R1 + 23}" r="14" fill="{ORG}"/>')
    text(1596, R1 + 29, "4", size=17, fill="#ffffff", weight=700)
    text(1690, R1 + 30, "算繪器", size=20, weight=700)
    for i, sc in enumerate([0.35, 0.5, 0.75, 1.0]):
        hgt = 40 * sc
        rect(1590 + i * 22, 436 - hgt, 18, hgt, fill=NAVY2, op=0.35 + 0.65 * sc, rx=2)
    text(1634, 458, "先低後高", size=13, fill=SUB)
    text(1634, 474, "解析度", size=13, fill=SUB)
    ts = 12
    ell = (1700 + 1.6 * ts, 396 + 1.5 * ts, 1.3 * ts, 0.75 * ts)
    for i in range(6):
        for j in range(3):
            x, y = 1700 + i * ts, 396 + j * ts
            cxp = min(max(ell[0], x), x + ts)
            cyp = min(max(ell[1], y), y + ts)
            hit = ((cxp - ell[0]) / (0.85 * ell[2])) ** 2 + ((cyp - ell[1]) / (0.85 * ell[3])) ** 2 <= 1
            rect(x, y, ts, ts, fill="#ffffff" if hit else "#D3D8DE", stroke="#8FA6BC", sw=0.8)
    surfel(ell[0], ell[1], (ell[2], 0), (0, ell[3]), ORG, op=0.9)
    text(1736, 458, "精確包圍盒", size=13, fill=SUB)
    text(1736, 474, "碰不到的格子不算", size=13, fill=SUB)
    line(1584, 484, 1780, 484, stroke=EDGE, sw=1.2, cap="butt")
    text(1682, 499, "前 75% 步數用 0.35～0.75 倍解析度", size=13, fill="#7F93A8")
    text(1682, 518, "幾何項每 8 步・數學化簡・單精度", size=13, fill="#7F93A8")
    # 渲染結果
    node(1832, R1, 163, R1H, "算繪結果")
    aerial(1843, 388, 68, 52, seed=8, rx=3)
    aerial(1917, 388, 68, 52, seed=8, mode="depthc", rx=3)
    text(1877, 462, "影像", size=14, fill=SUB)
    text(1951, 462, "深度", size=14, fill=SUB)
    text(1914, 496, "（法向每 8 步算一次）", size=13, fill="#7F93A8")
    arrow(1530, 420, 1570, 420, sw=3.5)
    arrow(1792, 420, 1830, 420, sw=3.5)

    R2, R2H = 600, 225
    # 密度控制（MCMC）
    node(1345, R2, 290, R2H, '密度控制（<tspan fill="#C2611F" font-weight="700">MCMC</tspan>）')
    text(1362, R2 + 59, '<tspan fill="#C2611F" font-weight="700">往誤差大的地方增生</tspan>・搬家・刪除', size=15, fill=SUB, anchor="start")
    text(1362, R2 + 87, '<tspan fill="#C2611F" font-weight="700">新點在母點的平面內錯開</tspan>', size=15, fill=SUB, anchor="start")
    text(1362, R2 + 115, '<tspan fill="#C2611F" font-weight="700">刪掉透明的高斯</tspan>・<tspan fill="#C2611F" font-weight="700">最小 1 像素</tspan>', size=15, fill=SUB, anchor="start")
    text(1362, R2 + 143, '<tspan fill="#C2611F" font-weight="700">依貢獻剪枝</tspan>・<tspan fill="#C2611F" font-weight="700">合併相近的高斯</tspan>', size=15, fill=SUB, anchor="start")
    text(1362, R2 + 171, "收回海面漂浮的高斯", size=15, fill=SUB, anchor="start")
    tag(1505, R2 + 167, 2)
    # 損失
    node(1660, R2, 335, R2H, "損失（與照片比較）")
    text(1678, R2 + 59, '顏色：L1＋<tspan fill="#C2611F" font-weight="700">L2</tspan>＋SSIM', size=15, fill=SUB, anchor="start")
    text(1678, R2 + 87, '幾何：<tspan fill="#C2611F" font-weight="700">逐貢獻深度上限</tspan>', size=15, fill=SUB, anchor="start")
    tag(1840, R2 + 83, 1)
    text(1723, R2 + 115, '法向・<tspan fill="#C2611F" font-weight="700">深度失真</tspan>・<tspan fill="#C2611F" font-weight="700">鄰域平面</tspan>', size=15, fill=SUB, anchor="start")
    text(1678, R2 + 143, '正則：<tspan fill="#C2611F" font-weight="700">不透明度二值化</tspan>・<tspan fill="#C2611F" font-weight="700">尺度</tspan>', size=15, fill=SUB, anchor="start")
    text(1723, R2 + 171, '<tspan fill="#C2611F" font-weight="700">只算畫面內的高斯</tspan>', size=15, fill=SUB, anchor="start")
    text(1678, R2 + 199, '<tspan fill="#C2611F" font-weight="700">幾何項與上限每 8 步算一次</tspan>', size=15, fill=SUB, anchor="start")

    arrow(1913, R1 + R1H, 1913, R2 - 2, sw=3.5)                       # 渲染結果 → 損失
    lane = 570
    polyline([(1700, R2), (1700, lane), (1400, lane), (1400, R1 + R1H + 8)], stroke=ORG, sw=3, dash="8 6", marker="ah_org")
    text(1560, lane - 8, "梯度", size=15, fill=ORG_D, weight=700)
    arrow(1480, R2 - 2, 1480, R1 + R1H + 4, sw=3.5)                   # 密度控制 → 2D 高斯
    # 圖例
    text(1345, 856, '<tspan fill="#C2611F" font-weight="700">橘字</tspan>：與 CityGaussianV2 不同的設計（改動或新增）', size=14, fill=SUB, anchor="start")
    line(1810, 850, 1845, 850, stroke=NAVY, sw=3)
    text(1851, 856, "運算", size=14, fill=SUB, anchor="start")
    line(1900, 850, 1935, 850, stroke=ORG, sw=3, dash="8 6")
    text(1941, 856, "梯度", size=14, fill=SUB, anchor="start")

    end()
    # 改良 3：右欄（直式），連到 2D 高斯的「顏色」
    c3x, c3y, c3w, c3h = 2370, 175, 240, 595
    rect(c3x, c3y, c3w, c3h, fill="#ffffff", stroke=EDGE, sw=1.8, rx=10)
    badge(c3x + 22, c3y + 32, 3, "顏色只學看得到的方向", size=18)
    scx, scy, sr = c3x + c3w / 2, c3y + 180, 48
    sg = uid("sg")
    defs.append(f'<radialGradient id="{sg}" cx="0.38" cy="0.35" r="0.75"><stop offset="0" stop-color="#ffffff"/>'
                f'<stop offset="1" stop-color="#9FB3C8"/></radialGradient>')
    add(f'<path d="M {scx - sr} {scy} A {sr} {sr} 0 0 0 {scx + sr} {scy} A {sr} {sr * 0.3} 0 0 1 {scx - sr} {scy} Z" fill="#D7DDE3"/>')
    add(f'<path d="M {scx - sr} {scy} A {sr} {sr} 0 0 1 {scx + sr} {scy} A {sr} {sr * 0.3} 0 0 1 {scx - sr} {scy} Z" fill="url(#{sg})"/>')
    add(f'<ellipse cx="{scx}" cy="{scy}" rx="{sr}" ry="{sr * 0.3}" fill="none" stroke="#7F93A8" stroke-width="1.3" stroke-dasharray="3 3"/>')
    add(f'<circle cx="{scx}" cy="{scy}" r="{sr}" fill="none" stroke="#7F93A8" stroke-width="1.6"/>')
    for ang in (-60, -30, 0, 30, 60):
        a_ = math.radians(ang)
        cx, cy = scx + 92 * math.sin(a_), scy - 92 * math.cos(a_)
        ex, ey = scx + (sr + 5) * math.sin(a_), scy - (sr + 5) * math.cos(a_)
        arrow(cx - 12 * math.sin(a_), cy + 12 * math.cos(a_), ex, ey, color=ORG, sw=2.2)
        camera(cx, cy, 0.55, ang, frustum=False)
    text(scx, scy + sr + 22, "沒有相機從下方看", size=14, fill="#7F93A8")
    text(scx, c3y + 300, "每個色版的顏色係數", size=15, fill=SUB)
    sq, gap = 16, 4
    x0 = scx - (8 * sq + 7 * gap) / 2
    for i in range(16):
        col = ORG if i < 8 else ("#F6C9A4" if i < 10 else "#D3D8DE")
        rect(x0 + (i % 8) * (sq + gap), c3y + 314 + (i // 8) * (sq + gap), sq, sq, fill=col, rx=2)
    lx = c3x + 18
    for col, lab, w_ in [(ORG, "學習", 64), ("#F6C9A4", "看得到", 80), ("#D3D8DE", "看不到", 80)]:
        rect(lx, c3y + 376, 12, 12, fill=col, rx=2)
        text(lx + 17, c3y + 387, lab, size=14, fill=SUB, anchor="start")
        lx += w_
    text(scx, c3y + 428, "空拍相機只從上方看，", size=15, fill=SUB)
    text(scx, c3y + 451, "所以只學看得到的方向", size=15, fill=SUB)
    line(c3x + 20, c3y + 474, c3x + c3w - 20, c3y + 474, stroke=EDGE, sw=1.2, cap="butt")
    text(scx, c3y + 502, "16 個係數：看得到 10、學 8", size=15, fill=SUB)
    text(scx, c3y + 528, "每個高斯參數 58 → 34", size=15, fill=SUB)
    text(scx, c3y + 556, "訓練顯存峰值 −30%", size=16, fill=ORG_D, weight=700)
    # 連線：2D 高斯（中欄左上）→ 改良 3
    polyline([(1718, 205), (1718, 150), (c3x + c3w / 2, 150), (c3x + c3w / 2, c3y - 4)], stroke=ORG, sw=2.2, dash="5 4", marker="ah_org")
    text(2060, 142, "作用在高斯的顏色", size=15, fill=ORG_D, weight=700)

    # ★ v2：整塊的訓練設定（面板 3 底部）
    rect(1250, 805, 1360, 56, fill="#ffffff", stroke=EDGE, sw=1.5, rx=10)
    text(1930, 841, '每塊 <tspan fill="#C2611F" font-weight="700">17,000 步</tspan>・前 75% 低解析度・平均 <tspan fill="#C2611F" font-weight="700">150 萬個高斯</tspan>・'
         '<tspan fill="#C2611F" font-weight="700">1 張 RTX 3090</tspan> 依序訓練 16 塊', size=19, fill=TXT)
    chevron(2639, 480)

    # ════ 面板 4：合併輸出 ════
    panel(2670, 20, 300, 920, "合併輸出")
    grp(711, -45.6, 0.95)
    add('<g transform="translate(0,38)">')
    t0x, t0y, tg, off = 2220 - (4 * 30 + 3 * 4) / 2, 110, 30, 4
    for i in range(4):
        for j in range(4):
            rect(t0x + i * (tg + off), t0y + j * (tg + off), tg, tg, fill="#B9CCE0" if (i, j) != (1, 1) else ORG_L,
                 stroke=NAVY2 if (i, j) != (1, 1) else ORG, sw=1.6, rx=3)
    text(2220, 268, "分塊合併", size=20, fill=SUB)
    text(2220, 294, "每塊只留盒內的高斯", size=16, fill=SUB)
    arrow(2220, 308, 2220, 348, sw=4)
    add('<g transform="translate(0,30)">')
    s2 = 13
    Pc = iso(2220, 470, s2)
    ux2, uy2, uz2 = iso_vecs(s2)
    gnd = [Pc(-6, -5, 0), Pc(6, -5, 0), Pc(6, 6, 0), Pc(-6, 6, 0)]
    poly(gnd, fill="#E3E8DF", stroke="#C9D1C3", sw=1)
    sea_q = [Pc(3.0, 2.0, 0), Pc(6, 2.0, 0), Pc(6, 6, 0), Pc(1.5, 6, 0)]
    poly(sea_q, fill=SEA_L, stroke=SEA, sw=1.2)
    cr = np.random.default_rng(23)
    for _ in range(40):
        X, Y = cr.uniform(2.6, 5.8), cr.uniform(2.4, 5.8)
        sx, sy = Pc(X, Y, 0)
        r = cr.uniform(0.5, 0.9)
        surfel(sx, sy, (ux2[0] * r, ux2[1] * r), (uy2[0] * r, uy2[1] * r), "#7FB3D9", op=0.8)
    blist = []
    for X in np.arange(-5.2, 2.6, 1.9):
        for Y in np.arange(-4.4, 3.0, 1.9):
            if X > 1.2 and Y > 0.8:
                continue
            h = float(cr.choice([1.0, 1.5, 2.2, 3.0, 4.2, 5.5]))
            blist.append((X, Y, X + 1.3, Y + 1.3, h, ROOFS[int(cr.integers(len(ROOFS)))]))
    blist.sort(key=lambda b: b[0] + b[1])
    for (X0, Y0, X1, Y1, h, col) in blist:
        box(Pc, X0, Y0, X1, Y1, h, col)
        for _ in range(3):
            sx, sy = Pc(cr.uniform(X0 + 0.3, X1 - 0.3), cr.uniform(Y0 + 0.3, Y1 - 0.3), h)
            surfel(sx, sy, (ux2[0] * 0.35, ux2[1] * 0.35), (uy2[0] * 0.35, uy2[1] * 0.35), shade_col(col, 0.9), op=0.8)
    text(2220, 585, "城市級 2D 高斯場景", size=21, weight=700)
    text(2220, 611, "約 1,600 萬個高斯", size=16, fill=SUB)
    add('</g>')
    add('<g transform="translate(0,70)">')
    # 兩個輸出：新視角影像、表面幾何
    polyline([(2220, 600), (2220, 618)], stroke=NAVY, sw=3.5)
    polyline([(2148, 618), (2292, 618)], stroke=NAVY, sw=3.5)
    arrow(2148, 618, 2148, 645)
    arrow(2292, 618, 2292, 645)
    aerial(2080, 650, 136, 100, seed=41, sea=True, rx=5)
    # 表面幾何：灰色著色的等角網格
    rect(2224, 650, 136, 100, fill="#ffffff", stroke="#8FA6BC", sw=1.5, rx=5)
    gc = clip_rect(2224, 650, 136, 100, 5)
    add(f'<g clip-path="url(#{gc})">')
    Pm = iso(2292, 712, 6.2)
    poly([Pm(-6, -5, 0), Pm(6, -5, 0), Pm(6, 6, 0), Pm(-6, 6, 0)], fill="#E6E8EB", stroke="#C9CED4", sw=0.8)
    for (X0, Y0, X1, Y1, h, col) in blist:
        box(Pm, X0, Y0, X1, Y1, h, "#C7CDD4")
    add('</g>')
    text(2148, 778, "新視角影像", size=18, weight=700)
    text(2292, 778, "表面幾何", size=18, weight=700)
    text(2148, 802, "741 張測試視角", size=14, fill=SUB)
    text(2148, 822, "每秒算繪 109 張", size=14, fill=SUB)
    text(2292, 802, "對 MatrixCity", size=14, fill=SUB)
    text(2292, 822, "點雲算 F1", size=14, fill=SUB)
    add('</g>')
    add('</g>')
    end()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="紀錄/海報圖/流程圖_v2")
    ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()
    draw()
    mk = []
    for name, col in [("ah_navy", NAVY), ("ah_org", ORG), ("ah_sea", SEA), ("ah_grey", SUB)]:
        mk.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse">'
                  f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<defs>{"".join(mk)}{"".join(defs)}</defs>{"".join(out)}</svg>')
    with open(a.out + ".svg", "w") as f:
        f.write(svg)
    subprocess.run(["rsvg-convert", "-z", str(a.scale), "-o", a.out + ".png", a.out + ".svg"], check=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", a.out + ".pdf", a.out + ".svg"], check=True)
    print("wrote", a.out + ".svg", a.out + ".png", a.out + ".pdf")


if __name__ == "__main__":
    main()
