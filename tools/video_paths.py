"""產生影片用的相機路徑（純 CPU）＋俯視路線圖。2026-11-25 新增。

路徑存成一個 npz：每個鍵＝一條路線，值＝(N,4,4) float32 相機到世界（OpenCV：x 右、y 下、z 前；世界 +z 朝上）。
渲染用 tools/render_paths.py（GPU）；兩個模型讀同一個檔 => 每一幀視角完全相同，可以並排對照。

路線（MatrixCity 空拍，座標＝sparse/0，1 單位＝100 m）：
  capture   沿拍攝航線：直接取訓練相機（同一條南北掃描線、朝向＝前進方向）的位置串起來，蛇行一去一回；
            高度與俯角照原始拍攝（北區 250 m／45°、高樓區 500 m／60°），兩區之間平滑爬升 => 最接近訓練視角
  street    沿街道飛行：最低一層訓練高度 150 m，沿北區棋盤街道轉彎，最後朝南看高樓區天際線
  low       低空沿街：約 25 m，沿北區一條東西向街道直飛（比所有訓練相機都低 => 看模型沒見過的角度）
  towers    高樓區環繞：繞高樓群一圈、鏡頭對著中心
  coast     沿海岸：在海面上沿城市外緣飛，鏡頭斜看向城市
街道中線、高樓群中心、海岸線都由模型的俯視高度圖算出（不手填座標）；每條路線印出最小離地／離樓高度。

python tools/video_paths.py --ckpt outputs/lab/view_base10/checkpoints/view_base10.ckpt \
    --sparse data/matrix_city/aerial/train/block_all/sparse/0 \
    --out configs/video_paths_matrixcity_aerial.npz --fig outputs/lab/videos_1125/routes.png
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

FPS = 30


# ───────────────────────────── 俯視圖 ─────────────────────────────
class TopMap:
    def __init__(self, ckpt, cs=0.02):
        import torch
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        m = sd["gaussian_model.gaussians.means"].numpy()
        op = torch.sigmoid(sd["gaussian_model.gaussians.opacities"][:, 0]).numpy()
        sc = torch.exp(sd["gaussian_model.gaussians.scales"]).max(1).values.numpy()
        dc = sd["gaussian_model.gaussians.shs_dc"][:, 0].numpy()
        rgb = np.clip(dc * 0.28209479177387814 + 0.5, 0, 1)
        lo, hi = np.percentile(m[:, :2], 0.5, axis=0) - 1.0, np.percentile(m[:, :2], 99.5, axis=0) + 1.0
        keep = (op > 0.5) & (sc < 0.2) & np.all(m[:, :2] > lo, 1) & np.all(m[:, :2] < hi, 1)
        m, rgb = m[keep], rgb[keep]
        self.cs, self.x0, self.y1 = cs, lo[0], hi[1]
        self.W, self.H = int((hi[0] - lo[0]) / cs) + 1, int((hi[1] - lo[1]) / cs) + 1
        ix, iy = self.to_px(m[:, 0], m[:, 1])
        cell = iy * self.W + ix
        o = np.lexsort((m[:, 2], cell))
        cs_ = cell[o]
        top = o[np.r_[cs_[1:] != cs_[:-1], True]]
        h = np.full(self.H * self.W, np.nan, np.float32)
        col = np.zeros((self.H * self.W, 3), np.float32)
        h[cell[top]], col[cell[top]] = m[top, 2], rgb[top]
        self.h = h.reshape(self.H, self.W)
        self.rgb = col.reshape(self.H, self.W, 3)
        self.count = np.bincount(cell, minlength=self.H * self.W).reshape(self.H, self.W)
        self.ground = np.nanpercentile(self.h, 25)
        self._maxf = {}

    def to_px(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / self.cs).astype(int), 0, self.W - 1)
        iy = np.clip(((self.y1 - np.asarray(y)) / self.cs).astype(int), 0, self.H - 1)
        return ix, iy

    def to_world(self, ix, iy):
        return self.x0 + (np.asarray(ix) + 0.5) * self.cs, self.y1 - (np.asarray(iy) + 0.5) * self.cs

    def max_height_near(self, x, y, r):
        k = max(1, int(round(r / self.cs)))
        if k not in self._maxf:
            fp = np.hypot(*np.mgrid[-k:k + 1, -k:k + 1]) <= k
            self._maxf[k] = ndimage.maximum_filter(np.nan_to_num(self.h, nan=-9.0), footprint=fp)
        ix, iy = self.to_px(x, y)
        return self._maxf[k][iy, ix]


# ───────────────────────────── 幾何工具 ─────────────────────────────
def rounded_polyline(pts, r, n_arc=40):
    """折線在每個轉角用二次 Bezier 圓角（半徑上限＝相鄰兩段較短者的 45%），回傳密集點列。"""
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, p, b = pts[i - 1], pts[i], pts[i + 1]
        la, lb = np.linalg.norm(p - a), np.linalg.norm(b - p)
        rr = min(r, 0.45 * la, 0.45 * lb)
        A, B = p - (p - a) / la * rr, p + (b - p) / lb * rr
        t = np.linspace(0, 1, n_arc)[:, None]
        out.extend((1 - t) ** 2 * A + 2 * (1 - t) * t * p + t ** 2 * B)
    out.append(pts[-1])
    return densify(np.array(out), 0.005)


def densify(P, step):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    keep = np.r_[True, seg > 1e-9]
    P, s = P[keep], s[keep]
    ss = np.arange(0, s[-1], step)
    return np.stack([np.interp(ss, s, P[:, k]) for k in range(3)], 1)


def time_resample(P, v_max, v_turn_frac=0.45, ease_s=1.2):
    """依弧長＋轉彎曲率給速度（轉彎減速），頭尾 ease_s 秒加減速，回傳每幀位置與每幀切線水平航向。"""
    d = np.diff(P, axis=0)
    seg = np.linalg.norm(d, axis=1)
    s = np.r_[0, np.cumsum(seg)]
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    hd = np.r_[hd, hd[-1]]
    hd_s = ndimage.gaussian_filter1d(hd, 30)
    curv = np.abs(np.gradient(hd_s, s, edge_order=1))          # rad / 單位長
    v = np.clip(v_max / (1.0 + curv / 1.5), v_max * v_turn_frac, v_max)   # 曲率 1.5 rad/單位 時速度減半
    v = ndimage.gaussian_filter1d(v, 40)
    # 頭尾加減速（依弧長近似：ease 距離＝v_max*FPS*ease_s/2）
    de = v_max * FPS * ease_s / 2
    v = v * np.clip(np.minimum(s, s[-1] - s) / de, 0.08, 1.0) ** 0.5
    t = np.r_[0, np.cumsum(seg / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-6))]   # 單位：幀
    n = int(t[-1]) + 1
    tt = np.arange(n, dtype=float)
    Pf = np.stack([np.interp(tt, t, P[:, k]) for k in range(3)], 1)
    hf = np.interp(tt, t, hd_s)
    return Pf, hf


def fwd_from(yaw, pitch_deg):
    p = np.radians(pitch_deg)
    return np.stack([np.cos(yaw) * np.cos(p), np.sin(yaw) * np.cos(p), -np.sin(p) * np.ones_like(yaw)], -1)


def c2w_from(P, F):
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    up = np.array([0, 0, 1.0])
    R = np.cross(F, up)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = np.cross(F, R)
    M = np.tile(np.eye(4), (len(P), 1, 1))
    M[:, :3, 0], M[:, :3, 1], M[:, :3, 2], M[:, :3, 3] = R, D, F, P
    return M.astype(np.float32)


def smooth_fwd(F, sigma):
    F = ndimage.gaussian_filter1d(F, sigma, axis=0, mode="nearest")
    return F / np.linalg.norm(F, axis=1, keepdims=True)


# ───────────────────────────── 場景分析 ─────────────────────────────
def street_lines(tm, axis, region, frac=0.6):
    """在 region=(x0,x1,y0,y1) 內，沿 axis（'x'＝南北向街道的 x 座標）找地面格比例的峰值。"""
    grd = np.nan_to_num(tm.h, nan=9) < tm.ground + 0.04
    ix0, iy1 = tm.to_px(region[0], region[2])
    ix1, iy0 = tm.to_px(region[1], region[3])
    sub = grd[iy0:iy1, ix0:ix1]
    p = ndimage.uniform_filter1d(sub.mean(0 if axis == "x" else 1), 5)
    out = []
    for i in range(2, len(p) - 2):
        if p[i] >= frac and p[i] == p[max(0, i - 8):i + 9].max():
            out.append(i)
    if axis == "x":
        v = [float(tm.to_world(ix0 + i, 0)[0]) for i in out]
    else:
        v = [float(tm.to_world(0, iy0 + i)[1]) for i in out]
    ded = []                                   # 同一條街的兩個相鄰峰（< 0.3）只留一個
    for a in sorted(v):
        if not ded or a - ded[-1] > 0.3:
            ded.append(a)
    return ded


def tower_cluster(tm, thr=1.5):
    tall = np.nan_to_num(tm.h, nan=0) > thr
    lab, n = ndimage.label(ndimage.binary_dilation(tall, iterations=40))
    sizes = ndimage.sum(tall, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    iy, ix = np.nonzero(tall & (lab == k))
    x, y = tm.to_world(ix, iy)
    cx, cy = x.mean(), y.mean()
    rad = np.percentile(np.hypot(x - cx, y - cy), 95)
    return cx, cy, rad


def land_mask(tm):
    dens = ndimage.uniform_filter(tm.count.astype(float), 7) > 4
    dens = ndimage.binary_closing(dens, iterations=6)
    dens = ndimage.binary_fill_holes(dens)
    lab, n = ndimage.label(dens)
    k = int(np.argmax(ndimage.sum(dens, lab, range(1, n + 1)))) + 1
    land = lab == k
    # 碼頭／突堤用開運算拿掉，只留城市本體外緣
    return ndimage.binary_opening(land, iterations=25)


# ───────────────────────────── 五條路線 ─────────────────────────────
def path_capture(sparse, tm):
    from internal.utils.colmap import read_images_binary
    ims = read_images_binary(os.path.join(sparse, "images.bin"))
    C, Fw = [], []
    for im in ims.values():
        R = im.qvec2rotmat()
        C.append(-R.T @ im.tvec)
        Fw.append(R.T @ np.array([0, 0, 1.0]))
    C, Fw = np.array(C), np.array(Fw)
    yaw = np.round(np.degrees(np.arctan2(Fw[:, 1], Fw[:, 0])))

    def line(z, yw, x_near):
        s = (np.abs(C[:, 2] - z) < 0.05) & (yaw == yw)
        xs = np.round(C[s, 0], 2)
        ux = np.unique(xs)
        best = ux[np.argmin(np.abs(ux - x_near) - 0.001 * (xs[:, None] == ux).sum(0))]
        pts = C[s][xs == best]
        return pts[np.argsort(-pts[:, 1] if yw == -90 else pts[:, 1])]

    cx, cy, _ = tower_cluster(tm)
    # 去程（朝南、鏡頭朝南）：北區 250 m 線 -> 高樓區 500 m 線；回程（朝北）在東邊下一條線
    hi1 = line(5.0, -90, cx - 0.6)
    lo1 = line(2.5, -90, hi1[0, 0])
    hi2 = line(5.0, 90, hi1[0, 0] + 0.8)
    lo2 = line(2.5, 90, hi2[0, 0])
    yS = min(hi1[:, 1].min(), hi2[:, 1].min()) - 0.6
    pts = np.vstack([lo1, hi1, [[hi1[0, 0], yS, 5.0], [hi2[0, 0], yS, 5.0]], hi2, lo2])
    P = rounded_polyline(pts, 0.6)
    Pf, hd = time_resample(P, v_max=0.032)
    z = Pf[:, 2]
    pitch = 45 + np.clip((z - 2.5) / 2.0, 0, 1) * 15        # 照原始拍攝：250 m 45°、450～500 m 60°
    F = smooth_fwd(fwd_from(hd, pitch), 6)
    return c2w_from(Pf, F), pts


def path_street(tm, end_rise=3.5):
    xs = street_lines(tm, "x", (-4.5, 6.5, 0.5, 6.0))
    ys = street_lines(tm, "y", (-3.5, 5.0, -1.0, 6.5))
    nx = lambda v: min(xs, key=lambda a: abs(a - v))
    ny = lambda v: min(ys, key=lambda a: abs(a - v))
    x1, x2, x3 = nx(-1.3), nx(1.3), nx(3.9)
    y0, y1, y2 = max(ys) + 0.3, ny(3.3), ny(0.06)
    z = 1.5
    wp = [(x1, y0), (x1, y1), (x2, y1), (x2, y2), (x3, y2)]
    pts = np.array([(a, b, z) for a, b in wp])
    P = rounded_polyline(pts, 0.5)
    Pf, hd = time_resample(P, v_max=0.026)
    # 收尾：最後 5 秒爬升到 end_rise、俯角從 30° 壓到 60°，往下俯瞰街道網格
    # （試過轉向高樓群中心看天際線：高樓區太密，停在哪裡都被前排高樓整面擋住）
    n = len(Pf)
    k = min(n - 1, int(5.0 * FPS))
    w = np.clip((np.arange(n) - (n - 1 - k)) / k, 0, 1)
    w = w * w * (3 - 2 * w)
    Pf = Pf.copy()
    Pf[:, 2] = z + (end_rise - z) * w
    F = smooth_fwd(fwd_from(hd, 30.0 + 30.0 * w), 4)
    return c2w_from(Pf, F), pts


def path_low(tm, z=0.25):
    """在北區東西向街道裡，找「沿街中線、高度 z 一路不撞東西」最長的一段（最北那條臨海不算）。"""
    ys = street_lines(tm, "y", (-3.5, 5.0, -1.0, 6.5))
    xs = np.arange(-4.5, 6.5, 0.02)
    best = None
    for yy in sorted(ys)[:-1]:
        # 只算真正在地面的街段（高架橋也算障礙，不然會選到「貼著高架橋飛」）
        clear = tm.max_height_near(xs, np.full_like(xs, yy), 0.04) < tm.ground + 0.10
        lab, n = ndimage.label(clear)
        for k in range(1, n + 1):
            idx = np.nonzero(lab == k)[0]
            L = xs[idx[-1]] - xs[idx[0]]
            if best is None or L > best[0]:
                best = (L, yy, xs[idx[0]] + 0.15, xs[idx[-1]] - 0.15)
    _, yy, xa, xb = best
    if xb - xa > 7.0:                          # 最多 7 單位（約 21 秒），取中段
        xm = 0.5 * (xa + xb)
        xa, xb = xm - 3.5, xm + 3.5
    pts = np.array([(xa, yy, z), (xb, yy, z)])
    P = rounded_polyline(pts, 0.1)
    Pf, hd = time_resample(P, v_max=0.011)
    F = smooth_fwd(fwd_from(hd, 8.0), 4)
    return c2w_from(Pf, F), pts


def path_towers(tm, z=4.5, n=720):
    cx, cy, rad = tower_cluster(tm)
    R = rad + 0.8
    th = np.linspace(-np.pi / 2, 1.5 * np.pi, n, endpoint=False)
    P = np.stack([cx + R * np.cos(th), cy + R * np.sin(th), np.full(n, z)], 1)
    tgt = np.array([cx, cy, 0.5])
    F = tgt[None] - P
    return c2w_from(P, F), P[::60]


def path_coast(tm, off=1.0, z=1.5):
    from skimage import measure
    land = land_mask(tm)
    pad = int(2 * off / tm.cs)                  # 俯視圖外圍只多 ~1 單位，補邊讓外推的海岸線能閉合
    dist = ndimage.distance_transform_edt(~np.pad(land, pad)) * tm.cs
    cs_ = measure.find_contours(dist, off)
    c = max(cs_, key=len) - pad                 # (row, col)
    x, y = tm.x0 + (c[:, 1] + 0.5) * tm.cs, tm.y1 - (c[:, 0] + 0.5) * tm.cs
    P = np.stack([x, y, np.full(len(x), z)], 1)
    P = densify(P, 0.005)
    P[:, :2] = ndimage.gaussian_filter1d(P[:, :2], 120, axis=0, mode="wrap")
    # 取東岸 -> 東南（高樓區外海、長橋）-> 南岸：從城市中心看方位角 30° 順時針掃 180°
    ccx, ccy = tm.to_world(*np.mean(np.nonzero(land)[::-1], axis=1))
    ang = np.arctan2(P[:, 1] - ccy, P[:, 0] - ccx)
    if np.mean(np.diff(np.unwrap(ang))) > 0:     # 統一成順時針（方位角遞減）
        P, ang = P[::-1], ang[::-1]
    i0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - np.radians(30)))))))
    P, ang = np.roll(P, -i0, axis=0), np.roll(ang, -i0)
    a = np.unwrap(ang)
    i0, i1 = 0, int(np.argmax(a < a[0] - np.radians(180)))
    seg = P[i0:i1]
    Pf, hd = time_resample(seg, v_max=0.034)
    # 鏡頭：前進方向往陸地側轉 35°，俯 20°
    inland = np.stack([ccx - Pf[:, 0], ccy - Pf[:, 1]], 1)
    side = np.sign(np.cross(np.stack([np.cos(hd), np.sin(hd)], 1), inland))   # +1＝陸地在左
    yaw = hd + side * np.radians(35)
    yaw = ndimage.gaussian_filter1d(np.unwrap(yaw), 20)
    F = smooth_fwd(fwd_from(yaw, 20.0), 4)
    return c2w_from(Pf, F), seg[::400]


# ───────────────────────────── 圖與檢查 ─────────────────────────────
def report(name, M, tm):
    P = M[:, :3, 3]
    hmax = tm.max_height_near(P[:, 0], P[:, 1], 0.05)
    clr = P[:, 2] - hmax
    print(f"[{name}] {len(M)} 幀 = {len(M) / FPS:.1f} 秒；高度 {P[:, 2].min():.2f}～{P[:, 2].max():.2f}；"
          f"與正下方 5 m 內最高物的最小間距 {clr.min():.3f}（第 {int(np.argmin(clr))} 幀）；"
          f"每幀最大位移 {np.linalg.norm(np.diff(P, axis=0), axis=1).max():.4f}")
    return clr.min()


def draw(tm, paths, fig):
    from PIL import Image, ImageDraw, ImageFont
    S = 2
    img = Image.fromarray((tm.rgb * 255).astype(np.uint8)).resize((tm.W * S, tm.H * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    cols = {"capture": (255, 80, 80), "street": (255, 210, 0), "low": (0, 230, 255),
            "towers": (255, 120, 255), "coast": (120, 255, 120), "orbit": (255, 255, 255)}
    label = {"capture": "1 沿拍攝航線", "street": "2 沿街道飛行", "low": "3 低空沿街",
             "towers": "4 高樓區環繞", "coast": "5 沿海岸", "orbit": "0 全城繞圈（上次）"}
    for name, M in paths.items():
        P = M[:, :3, 3]
        ix, iy = tm.to_px(P[:, 0], P[:, 1])
        xy = list(zip(ix * S, iy * S))
        col = cols.get(name, (255, 255, 255))
        d.line(xy, fill=col, width=5)
        # 起點圓點＋方向箭頭
        d.ellipse([xy[0][0] - 10, xy[0][1] - 10, xy[0][0] + 10, xy[0][1] + 10], fill=col)
        for k in range(len(xy) // 6, len(xy), max(1, len(xy) // 6)):
            (x0, y0), (x1, y1) = xy[max(0, k - 3)], xy[k]
            a = np.arctan2(y1 - y0, x1 - x0)
            tip = np.array([x1, y1])
            l, r = tip - 22 * np.array([np.cos(a - 0.5), np.sin(a - 0.5)]), tip - 22 * np.array([np.cos(a + 0.5), np.sin(a + 0.5)])
            d.polygon([tuple(tip), tuple(l), tuple(r)], fill=col)
        tx, ty = xy[0]
        d.text((tx + 14, ty - 40), label.get(name, name), fill=col, font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    os.makedirs(os.path.dirname(fig) or ".", exist_ok=True)
    img.save(fig)
    print("路線圖 ->", fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="任一個合併後的 ckpt（只讀高斯位置／顏色做俯視圖）")
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    tm = TopMap(a.ckpt)
    print(f"俯視圖 {tm.W}x{tm.H}，地面高度 {tm.ground:.3f}；高樓群中心／半徑 {tower_cluster(tm)}")
    print("南北向街道 x =", [round(v, 2) for v in street_lines(tm, "x", (-4.5, 6.5, 0.5, 6.0))])
    print("東西向街道 y =", [round(v, 2) for v in street_lines(tm, "y", (-3.5, 5.0, -1.0, 6.5))])
    builders = {"capture": lambda: path_capture(a.sparse, tm), "street": lambda: path_street(tm),
                "low": lambda: path_low(tm), "towers": lambda: path_towers(tm), "coast": lambda: path_coast(tm)}
    paths = {}
    for k, f in builders.items():
        if a.only and k not in a.only:
            continue
        M, _ = f()
        report(k, M, tm)
        paths[k] = M
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **paths)
    print("路徑 ->", a.out, {k: v.shape for k, v in paths.items()})
    if a.fig:
        draw(tm, paths, a.fig)


if __name__ == "__main__":
    main()
