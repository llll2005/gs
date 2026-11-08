# 貼地覆蓋拆成近樓格／遠離樓（ground_layer_1103 同定義：2 m 地面格、SfM 中位 < 3 m、非巨大、離格中心 ±0.5 m）
# 用法：python tools/probe/ground_split_1109.py <block> label=ckpt ...（11-08 scratchpad ground_split_1108.py 改吃 ckpt 路徑）
import sys, glob, numpy as np, torch
sys.path.insert(0, ".")
from scipy.ndimage import maximum_filter
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.colmap import read_points3D_binary
D="data/matrix_city/aerial/train/block_all"; B=int(sys.argv[1])
p=torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt",map_location="cpu",weights_only=False)
pc=PartitionCoordinates(id=p["partition_coordinates"]["id"],xy=p["partition_coordinates"]["xy"])
bb=pc.get_bounding_boxes(p["scene_config"]["partition_size"],enlarge=0.); Rt=p["extra_data"]["rotation_transform"][:3,:3].double()
pts=read_points3D_binary(f"{D}/sparse/0/points3D.bin")
Pt=np.stack([x.xyz for x in pts.values()]); trk=np.array([len(x.image_ids) for x in pts.values()]); err=np.array([x.error for x in pts.values()]); del pts
Pt=Pt[(trk>=3)&(err<1.5)]
C=0.02; lo=np.floor(Pt[:,:2].min(0)/C).astype(int)-2; ci=np.floor(Pt[:,:2]/C).astype(int)-lo; shp=tuple(ci.max(0)+3)
cid=ci[:,0]*shp[1]+ci[:,1]; o=np.lexsort((Pt[:,2],cid)); u_,f_,c_=np.unique(cid[o],return_index=True,return_counts=True)
med=np.full(shp[0]*shp[1],np.nan); med[u_]=Pt[o,2][f_+c_//2]; med=med.reshape(shp)
tall=np.nan_to_num(med,nan=-9)>0.2; near_b=maximum_filter(tall.astype(np.uint8),size=11)>0
for lab,f in [(a.split("=",1)[0],a.split("=",1)[1]) for a in sys.argv[2:]]:
    sd=torch.load(f,map_location="cpu",weights_only=False)["state_dict"]; g=lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m=g("means"); inb=CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb,coordinates=(m.double()@Rt.T)[:,:2])[B]
    m=m[inb].double().numpy(); op=torch.sigmoid(g("opacities")).reshape(-1)[inb].numpy(); s=torch.exp(g("scales"))[inb].numpy()*100; del sd
    c2=np.floor(m[:,:2]/C).astype(int)-lo; ok=(c2[:,0]>=0)&(c2[:,0]<shp[0])&(c2[:,1]>=0)&(c2[:,1]<shp[1])
    gz=np.full(len(m),np.nan); gz[ok]=med[c2[ok,0],c2[ok,1]]; on=np.isfinite(gz)&(gz<0.03)
    nb=np.zeros(len(m),bool); nb[ok]=near_b[c2[ok,0],c2[ok,1]]
    cells=np.unique(c2[on,0]*shp[1]+c2[on,1]); area=len(cells)*4.0
    cn=near_b.reshape(-1)[cells]; an=cn.sum()*4.0; af=(~cn).sum()*4.0
    dz=(m[:,2]-gz)*100; cov=op*2*np.pi*s[:,0]*s[:,1]; big=3*s.max(1)>20
    k=on&~big&(np.abs(dz)<0.5)
    print(f"{lab:14} 貼地覆蓋 全部 {cov[k].sum()/area:.3f}｜近樓格 {cov[k&nb].sum()/an:.3f}｜遠離樓 {cov[k&~nb].sum()/af:.3f}｜貼地顆數 {k.sum():,}｜o>0.9 {100*np.mean(op[k]>0.9):.1f}%｜面積中位 {np.median(2*np.pi*s[k,0]*s[k,1]):.3f} m²", flush=True)
print(f"近樓格比例（地面格）{100*np.mean(near_b.reshape(-1)[cells]):.0f}%")
