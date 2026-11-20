"""11-20 使用者「有水體識別結果的分佈圖嗎」：俯視圖＝SfM 可靠點（灰，陸地底圖）＋水體標記（淺藍）＋海面物件 v9 的海面格（深藍）＋4x4 分塊與塊號。"""
import sys, glob, subprocess, numpy as np, torch
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates
F = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK TC"], capture_output=True, text=True).stdout.strip(); fm.fontManager.addfont(F)
plt.rcParams.update({"font.sans-serif": [fm.FontProperties(fname=F).get_name()], "axes.unicode_minus": False, "figure.dpi": 150})
D = "data/matrix_city/aerial/train/block_all"; S = "outputs/lab/_xfer/sea"
w = np.load(f"{S}/water_bodies.npz")
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
xyz = np.array([p.xyz for p in pts.values() if len(p.image_ids) >= 3 and p.error < 1.5])
rng = np.random.default_rng(0); xyz = xyz[rng.choice(len(xyz), min(400000, len(xyz)), replace=False)]
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.).numpy() if hasattr(pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.), "numpy") else None
Rt = p["extra_data"]["rotation_transform"][:3, :3].double().numpy()
fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(xyz[:, 0], xyz[:, 1], s=0.05, c="#b8b7b0", rasterized=True, label="SfM 可靠點（陸地與建物）")
lab = w["labels"]; ii, jj = np.nonzero(lab > 0)
ax.scatter(w["x0"] + (ii + 0.5) * w["res"], w["y0"] + (jj + 0.5) * w["res"], s=0.08, c="#9cc3ef", rasterized=True, label="水體（全場景 1 個，水面 z＝−0.026，30,832 個 SfM 水面點）")
for f in sorted(glob.glob(f"{S}/v9_block_*.npz")):
    t = np.load(f)["tiles"]; ax.scatter(t[:, 0], t[:, 1], s=0.05, c="#2a78d6", rasterized=True)
ax.scatter([], [], s=1, c="#2a78d6", label="海面物件 v9 的海面格（錨定海面層的位置）")
bbx = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
mn, mx = bbx.min.numpy(), bbx.max.numpy()
lo = np.percentile(xyz[:, :2], 0.5, 0); hi = np.percentile(xyz[:, :2], 99.5, 0); pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
for k in range(len(mn)):
    a0 = np.clip(mn[k], -1e3, 1e3); a1 = np.clip(mx[k], -1e3, 1e3)
    cs = np.array([[a0[0], a0[1]], [a1[0], a0[1]], [a1[0], a1[1]], [a0[0], a1[1]], [a0[0], a0[1]]])
    wc = (np.c_[cs, np.zeros(len(cs))] @ Rt)[:, :2]        # 分區座標 => 世界（Rt 正交，轉置＝反轉）
    ax.plot(wc[:, 0], wc[:, 1], color="#52514e", lw=0.9)
    # 塊號：放在（盒 ∩ 顯示範圍）的中心
    inner = np.c_[np.clip(cs[:4], lo.min() - 50, hi.max() + 50), np.zeros(4)] @ Rt
    cc = np.clip(wc[:4].mean(0), lo, hi) if np.abs(mn[k]).max() > 50 or np.abs(mx[k]).max() > 50 else wc[:4].mean(0)
    xs = np.clip(wc[:4, 0], lo[0], hi[0]); ys = np.clip(wc[:4, 1], lo[1], hi[1])
    ax.text((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, f"b{k}", ha="center", va="center", fontsize=11, color="#0b0b0b", weight="bold")
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.set_aspect("equal"); ax.set_xlabel("x（場景單位）"); ax.set_ylabel("y（場景單位）")
ax.legend(loc="lower left", fontsize=8, markerscale=6, framealpha=0.9)
ax.set_title("水體識別結果（俯視）：每張影像的海面區 → 多視角投票的海面格 → 全場景連通水體", fontsize=11)
fig.tight_layout(); fig.savefig("紀錄/圖_水體識別_1120.png", bbox_inches="tight"); print("=> 紀錄/圖_水體識別_1120.png")
