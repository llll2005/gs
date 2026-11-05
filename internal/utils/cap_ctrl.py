"""採樣控制顆數（cap_ctrl；2026-11-05 使用者：從某步起抽幾張驗證視角看破洞與品質，決定加點或減點）——選驗證視角＋每次的決定。

用 SfM 的觀測紀錄近似「資訊」：可靠點（track >= 3、重投影誤差 < 1.5）只算本塊訓練相機的觀測數 deg。
每台相機的風險 = 它看到的點裡 deg <= weak_deg 的比例（越低 => 拿掉它，別人還看得到）。
從最安全的 pool_mult x K 台裡隨機抽 K 台（K = max(k_min, round(frac x 相機數))）。看到的可靠點 < min_pts 的相機不選（例：只看海，內容可能是獨有的）。
11-05 b3／b12 模擬（scratchpad holdout_cover_1105.py，16 台）：從 >= 3 台掉到 < 3 台的點，隨機留 1.47%／3.02% => 這樣挑 0.63%／0.73%。
"""
import os

import numpy as np


def select_safe_holdout(dataset_path: str, train_names, frac: float = 0.015, k_min: int = 6, pool_mult: int = 8,
                        weak_deg: int = 4, min_pts: int = 50, seed: int = 0):
    from internal.utils.colmap import read_images_binary, read_points3D_binary
    sp = next((d for d in (os.path.join(dataset_path, "sparse", "0"), os.path.join(dataset_path, "sparse"))
               if os.path.isfile(os.path.join(d, "points3D.bin"))), None)
    if sp is None:
        raise FileNotFoundError(f"找不到 {dataset_path}/sparse(/0)/points3D.bin")
    base = [os.path.basename(str(n)) for n in train_names]
    idx = {n: i for i, n in enumerate(base)}
    imgs = read_images_binary(os.path.join(sp, "images.bin"))
    id2cam = {i: idx.get(os.path.basename(v.name), -1) for i, v in imgs.items()}
    pts = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    rows, cols, n_pt = [], [], 0
    for p in pts.values():
        if len(p.image_ids) < 3 or p.error >= 1.5:
            continue
        cs = {id2cam.get(int(i), -1) for i in p.image_ids}
        cs.discard(-1)
        if not cs:
            continue
        rows.extend([n_pt] * len(cs)); cols.extend(cs); n_pt += 1
    rows = np.asarray(rows, np.int64); cols = np.asarray(cols, np.int64)
    deg = np.bincount(rows, minlength=n_pt)
    n_cam = len(base)
    seen = np.bincount(cols, minlength=n_cam)
    weak = np.bincount(cols, weights=(deg[rows] <= weak_deg).astype(np.float64), minlength=n_cam) / np.maximum(seen, 1)
    K = max(int(k_min), int(round(frac * n_cam)))
    ok = np.nonzero(seen >= min_pts)[0]
    pool = ok[np.argsort(weak[ok], kind="stable")][: max(K, K * int(pool_mult))]
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, size=min(K, len(pool)), replace=False)
    info = {"K": int(len(pick)), "n_cam": n_cam, "n_pts": int(n_pt), "pool": int(len(pool)),
            "weak_pick": float(weak[pick].mean()) if len(pick) else 0.0, "weak_all": float(weak[ok].mean()) if len(ok) else 0.0}
    return {base[i] for i in pick}, info


def decide(prev, L1: float, hole: float, N: int, cap: int, cap_hard: int, cap_min: int, step_frac: float,
           hole_hi: float, hole_lo: float, gain_hi: float, gain_lo: float):
    """採樣控制器的一次決定（純函式）。prev＝上一次檢查的紀錄 dict（L1、act）或 None。回傳 (動作, 新 cap, gain)。
      gain = (上次 L1 − 這次 L1) / 上次 L1          驗證視角塊內誤差的相對進步
      要點：透空 >= hole_hi，或 gain >= gain_hi（還在明顯進步）；但上次「加」了之後 gain < gain_lo => 加的點沒用，不再加
            且 cap 綁住了（N >= 0.95 cap）才加 —— 沒綁住時加 cap 不會多長點
      減點：透空 <= hole_lo 且 gain <= gain_lo（平了、也沒有破洞）=> cap 減 step_frac（MCMC 不補、trim 剪到新 cap）
    """
    if prev is None:
        return "持平", cap, None
    gain = (prev["L1"] - L1) / max(prev["L1"], 1e-12)
    hungry = hole >= hole_hi or gain >= gain_hi
    if prev.get("act") == "加" and gain < gain_lo:
        hungry = False
    if hungry and N >= 0.95 * cap and cap < cap_hard:
        return "加", min(cap_hard, int(cap * (1 + step_frac))), gain
    if hole <= hole_lo and gain <= gain_lo and cap > cap_min:
        return "減", max(cap_min, int(cap * (1 - step_frac))), gain
    return "持平", cap, gain
