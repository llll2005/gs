"""粒子尺度 / 表面取樣間距。b7 的塊空間範圍較大 => 絕對尺度不可比，要正規化。純 CPU。"""
import glob, os, re, sys, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tools")
from eval_official_test import load_test_cameras
from internal.utils import colmap as C
D="data/matrix_city/aerial/train/block_all"; pat=re.compile(r"step=(\d+)")
names, cams = load_test_cameras(D, 1.2)
pts=C.read_points3D_binary(os.path.join(D,"sparse/0/points3D.bin"))
SFM=np.stack([p.xyz for p in pts.values()]).astype(np.float32)
rng=np.random.default_rng(0)
for run,blk,part in [("best_b7",7,"002_001"),("sched30_b7",7,"002_001"),("sched30_b12",12,"002_002")]:
    fs=sorted(glob.glob("outputs/%s/blocks/block_%d/checkpoints/*.ckpt"%(run,blk)),key=lambda p:int(pat.search(p).group(1)))
    sd=torch.load(fs[-1],map_location="cpu")["state_dict"]
    xyz=sd[[k for k in sd if k.endswith("means")][0]].numpy()
    sc=torch.exp(sd[[k for k in sd if k.endswith("scales")][0]]).numpy()
    want={l.strip() for l in open("%s/partition/partitions-dim_5_5_visibility_0.08/%s.txt"%(D,part)) if l.strip()}
    idx=[i for i,n in enumerate(names) if n in want][::10][:8]
    seen=np.zeros(len(SFM),bool)
    S=torch.tensor(SFM)
    for i in idx:
        c=cams[i]; p=torch.cat([S,torch.ones_like(S[:,:1])],1)@c.full_projection.cpu()
        vis=(p[:,3]>1e-6).numpy(); uv=np.zeros((len(SFM),2),np.float32)
        pv=p.numpy(); uv[vis]=pv[vis,:2]/pv[vis,3:4]
        seen |= vis&(np.abs(uv[:,0])<1)&(np.abs(uv[:,1])<1)
    P=SFM[seen]
    sub=P[rng.choice(len(P),min(50000,len(P)),replace=False)]
    nn,_=cKDTree(P).query(sub,k=2); pitch=float(np.median(nn[:,1]))     # 表面取樣間距
    gs=float(np.median(sc.max(1)))                                       # 粒子長軸尺度
    print("%-14s 表面取樣間距 %.5f   粒子長軸中位 %.5f   => **相對尺度 %.2f**"%(run,pitch,gs,gs/pitch))
print("\n  相對尺度大 = 一顆粒子橫跨好幾個表面取樣點的距離 => 在深度不連續處會架成半透明的片")
