"""看圖用的精簡 ckpt：拿掉 optimizer 狀態，只留某塊分區盒（外擴 m_c x 內側塊尺寸，同我方裁盒起點）內的顆粒。原檔不動。
state_dict 裡第一維＝N 的張量全部刪同一批（同 tools/probe/oracle_behind_1117.py），其餘照抄 => viewer／載入器照舊能讀。
用法：python tools/slim_view_ckpt.py --src <ckpt> --dst <ckpt> --block 1 [--m_c 0.15]（--block -1＝不裁，只拿掉 optimizer）
★ 11-21 --clip_far f：另外刪掉「離可靠 SfM（track>=3、誤差<1.5）平面範圍超過 f x 範圍最長邊」的顆粒（場景相對比例，不用公尺；
  使用者 viewer 看到邊緣塊遠處的背景擬合面元）。f=0.25：m39／base9c15 合併各刪 129／169 顆（tools/probe/far_floaters_1121.py）。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--block", type=int, default=-1); ap.add_argument("--m_c", type=float, default=0.15)
ap.add_argument("--clip_far", type=float, default=0.0, help="0＝不剪；>0＝刪掉離 SfM 平面範圍超過 f x 最長邊的顆粒")
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all", help="--clip_far 用的資料集（SfM）")
ap.add_argument("--part", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
a = ap.parse_args()
ck = torch.load(a.src, map_location="cpu", weights_only=False); sd = ck["state_dict"]
key = next(k for k in sd if k.endswith("gaussians.means"))
m = sd[key]; N = m.shape[0]
keep = torch.ones(N, dtype=torch.bool)
if a.block >= 0:
    p = torch.load(a.part, map_location="cpu", weights_only=False)
    Rp = p["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    ids = p["partition_coordinates"]["id"].numpy(); xys = p["partition_coordinates"]["xy"].double().numpy()
    size = p["scene_config"]["partition_size"].double().numpy()
    inner = np.median(np.minimum(size[:, 0], size[:, 1]))
    ref = np.median(size[(size[:, 0] < 10 * inner) & (size[:, 1] < 10 * inner)], axis=0); mc = a.m_c * ref.min()
    k = int(np.where((ids[:, 0] == a.block % 4) & (ids[:, 1] == a.block // 4))[0][0])
    lo, hi = xys[k] - mc, xys[k] + size[k] + mc
    xy = (m.double().numpy() @ Rp.T)[:, :2]
    keep = torch.from_numpy((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
if a.clip_far > 0:
    from internal.utils.colmap import read_points3D_binary
    p = torch.load(a.part, map_location="cpu", weights_only=False)
    Rp = p["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    pts = read_points3D_binary(os.path.join(a.D, "sparse/0/points3D.bin"))
    q = np.stack([x.xyz for x in pts.values()])[np.array([len(x.image_ids) >= 3 and x.error < 1.5 for x in pts.values()])] @ Rp.T
    lo, hi = np.percentile(q[:, :2], 0.5, axis=0), np.percentile(q[:, :2], 99.5, axis=0); mf = a.clip_far * float((hi - lo).max())
    xy = (m.double().numpy() @ Rp.T)[:, :2]
    near = (xy[:, 0] >= lo[0] - mf) & (xy[:, 0] <= hi[0] + mf) & (xy[:, 1] >= lo[1] - mf) & (xy[:, 1] <= hi[1] + mf)
    print(f"[clip_far {a.clip_far}] SfM 平面範圍 {lo.round(2)}～{hi.round(2)}，外擴 {mf:.2f} => 刪 {int((keep & ~torch.from_numpy(near)).sum()):,} 顆")
    keep = keep & torch.from_numpy(near)
for k_, v in list(sd.items()):
    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == N:
        sd[k_] = v[keep].clone()
ck["optimizer_states"] = []
os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
torch.save(ck, a.dst)
print(f"{a.src} => {a.dst}｜N {N:,} -> {int(keep.sum()):,}｜{os.path.getsize(a.dst) / 2 ** 20:.0f} MiB")
