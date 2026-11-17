"""11-17 scripts/prep_pipeline.sh 的驗收：新流程的產出（<目標>）對現行檔案（<現行>）逐項比對。
應逐位元相同：預縮影像、校正深度（＋_rel）、分區清單、SfM＋補點底（裁盒後）、校正深度版塊外遮罩。
只比數值：單目深度（lab 重算 vs 原本的檔案）、單目深度尺度。稠密起點用 --seed_per_block（差一個亂數種子）：只比點數與最近鄰距離。
用法：python tools/prep_compare.py <現行資料夾> <目標資料夾>
"""
import hashlib, json, os, sys
import numpy as np

S, T = sys.argv[1], sys.argv[2]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
rng = np.random.default_rng(0)


def same_files(sub, names, label):
    ok = miss = bad = 0
    for n in names:
        a, b = os.path.join(S, sub, n), os.path.join(T, sub, n)
        if not (os.path.exists(a) and os.path.exists(b)):
            miss += 1; continue
        if md5(a) == md5(b):
            ok += 1
        else:
            bad += 1
    print(f"[{label}] 逐位元相同 {ok}／{ok + bad}（缺 {miss}）{'✅' if bad == 0 and ok > 0 else '⚠⚠'}", flush=True)


def sample(sub, k, ext):
    fs = sorted(f for f in os.listdir(os.path.join(T, sub)) if f.endswith(ext))
    return [fs[i] for i in sorted(rng.choice(len(fs), size=min(k, len(fs)), replace=False))] if fs else []


# 單目深度（數值）
if os.path.isdir(os.path.join(T, "estimated_depths_new")):
    rel = []
    for n in sample("estimated_depths_new", 40, ".npy"):
        a = np.load(os.path.join(S, "estimated_depths", n)).astype(np.float64); b = np.load(os.path.join(T, "estimated_depths_new", n)).astype(np.float64)
        if a.shape != b.shape:
            print(f"[單目深度] {n} 形狀不同 {a.shape} vs {b.shape}"); continue
        m = np.abs(a) > 1e-6
        rel.append(np.abs(b[m] - a[m]) / np.abs(a[m]))
    if rel:
        r = np.concatenate(rel)
        print(f"[單目深度] 40 張的相對差：中位 {np.median(r):.2e}、p99 {np.percentile(r, 99):.2e}、最大 {r.max():.2e}", flush=True)
if os.path.exists(os.path.join(T, "estimated_depth_scales_new.json")):
    A = json.load(open(os.path.join(S, "estimated_depth_scales.json"))); B = json.load(open(os.path.join(T, "estimated_depth_scales_new.json")))
    ks = [k for k in A if k in B]
    r = np.array([abs(B[k]["scale"] / A[k]["scale"] - 1) for k in ks if A[k]["scale"]])
    print(f"[單目尺度] {len(ks)} 張；scale 相對差 中位 {np.median(r):.2e}、最大 {r.max():.2e}", flush=True)
# 預縮影像
same_files("input_r1.2", sample("input_r1.2", 40, ""), "預縮影像（抽 40）")
# 校正深度
names = sample("estimated_depths_tri", 200, ".npy")
same_files("estimated_depths_tri", names, "校正深度（抽 200）")
same_files("estimated_depths_tri_rel", names, "校正深度 _rel（抽 200）")
# 分區
P = "partition/partitions-dim_4_4_visibility_0.08"
same_files(P, sorted(f for f in os.listdir(os.path.join(T, P)) if f.endswith(".txt")), "分區清單（16 塊）")
try:
    import torch
    pa = torch.load(os.path.join(S, P, "partitions.pt"), map_location="cpu", weights_only=False)
    pb = torch.load(os.path.join(T, P, "partitions.pt"), map_location="cpu", weights_only=False)
    def eq(x, y):
        if isinstance(x, dict):
            return all(k in y and eq(x[k], y[k]) for k in x)
        if torch.is_tensor(x):
            return torch.is_tensor(y) and x.shape == y.shape and bool(torch.equal(x, y))
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(eq(u, v) for u, v in zip(x, y))
        return True
    print(f"[分區 partitions.pt] 張量全部相同：{'✅' if eq(pa, pb) else '⚠⚠ 不同'}", flush=True)
except Exception as e:
    print(f"[分區 partitions.pt] 比不了：{e}")
# SfM＋補點底（裁盒後）
same_files("sfmfill_sweep44_crop/dup1", [f"block_{b}.ply" for b in range(16)], "SfM＋補點底（裁盒後，16 塊）")
# 塊外遮罩
for b in range(16):
    d = f"block_masks44_tri/block_{b}"
    if os.path.isdir(os.path.join(T, d)):
        fs = sorted(f for f in os.listdir(os.path.join(T, d)) if f.endswith(".png"))
        ok = sum(1 for f in fs if os.path.exists(os.path.join(S, d, f)) and md5(os.path.join(S, d, f)) == md5(os.path.join(T, d, f)))
        if ok != len(fs) or b in (0, 15):
            print(f"[塊外遮罩 b{b}] 逐位元相同 {ok}／{len(fs)} {'✅' if ok == len(fs) else '⚠⚠'}", flush=True)
# 稠密起點（種子不同）
try:
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    for b in (1, 4):
        L = lambda p: (lambda v: np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64))(PlyData.read(p)["vertex"].data)
        A = L(os.path.join(S, "sfmfill_tri44_crop/dup1", f"block_{b}.ply")); B = L(os.path.join(T, "sfmfill_tri44_crop/dup1", f"block_{b}.ply"))
        d, _ = cKDTree(A).query(B)
        print(f"[稠密起點 b{b}（逐塊種子）] 點數 現行 {len(A):,}、新 {len(B):,}（{100 * (len(B) / len(A) - 1):+.2f}%）｜新點到現行最近距離 中位 {np.median(d) * 100:.2f} m、p90 {np.percentile(d, 90) * 100:.2f} m", flush=True)
except Exception as e:
    print(f"[稠密起點] 比不了：{e}")
print("DONE prep_compare")
