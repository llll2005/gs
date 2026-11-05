"""11-03：塊內巨大面元（3σ 長軸 > 20 m）是什麼 —— 高度（相對 SfM 地面）、大小、opacity、躺平與否。"""
import sys, glob, numpy as np, torch
sys.path.insert(0, ".")
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.colmap import read_points3D_binary
from internal.density_controllers.merge_ops import _qmat
D = "data/matrix_city/aerial/train/block_all"; B = int(sys.argv[1])
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin"); Pz = np.array([x.xyz for x in pts.values()])
ground = np.percentile(Pz[:, 2], 5)
print(f"b{B}｜SfM 地面（z 5 分位）{ground * 100:.1f} m｜巨大＝3σ 長軸 > 20 m；高度＝中心 z − 地面")
for a in sys.argv[2:]:
    lab, pat = a.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    g = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m = g("means"); inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(m.double() @ Rt.T)[:, :2])[B]
    m = m[inb].numpy(); o = torch.sigmoid(g("opacities")).reshape(-1)[inb].numpy(); s = torch.exp(g("scales"))[inb].numpy() * 100
    nz = np.abs(_qmat(g("rotations")[inb])[:, 2, 2].numpy()); del sd
    la = 3 * s.max(1); big = la > 20
    h = (m[big, 2] - ground) * 100
    q = lambda x: "／".join(f"{v:.0f}" for v in np.percentile(x, [10, 50, 90]))
    print(f"  {lab:12} 巨大 {big.sum():>6,} 顆｜長軸 p10/50/90 {q(la[big])} m｜高度 p10/50/90 {q(h)} m｜地面下 {100 * np.mean(h < -5):.0f}%"
          f"｜o>0.5 {100 * np.mean(o[big] > 0.5):.0f}%｜躺平(|n_z|>0.8) {100 * np.mean(nz[big] > 0.8):.0f}%｜豎直(<0.3) {100 * np.mean(nz[big] < 0.3):.0f}%", flush=True)
