# 11-08 通用：CityGS 不透明立面顆粒（|n_z|<0.3、塊內、z>50 m）1.5 m 內最近的我方不透明顆粒——依高度帶：沒有／對齊（|n·n_ref|>0.9）／平躺（|n_z|>0.9）／歪
# 用法：python tools/probe/orient_band_1109.py <block> label=ckpt ...（CityGS 參考預設 outputs/official_release/blocks/block_<B>/checkpoints/*.ckpt，可用環境變數 CITYGS_CKPT 指定）
import glob, os, sys, numpy as np, torch
sys.path.insert(0, ".")
from scipy.spatial import cKDTree
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
D="data/matrix_city/aerial/train/block_all"; B=int(sys.argv[1])
p=torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt",map_location="cpu",weights_only=False)
pc=PartitionCoordinates(id=p["partition_coordinates"]["id"],xy=p["partition_coordinates"]["xy"])
bb=pc.get_bounding_boxes(p["scene_config"]["partition_size"],enlarge=0.); Rt=p["extra_data"]["rotation_transform"][:3,:3].double()
inblk=lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb,coordinates=(torch.from_numpy(x).double()@Rt.T)[:,:2])[B].numpy()
def nrm(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],1)
def ld(f):
    sd=torch.load(f,map_location="cpu",weights_only=False)["state_dict"]
    m=sd["gaussian_model.gaussians.means"].double().numpy(); o=torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy()
    return m,o,nrm(sd["gaussian_model.gaussians.rotations"].double().numpy())
cm,co,cn=ld(os.environ.get("CITYGS_CKPT") or sorted(glob.glob(f"outputs/official_release/blocks/block_{B}/checkpoints/*.ckpt"))[0])
ref=inblk(cm)&(co>0.5)&(np.abs(cn[:,2])<0.3)&(cm[:,2]>0.5); rm,rn=cm[ref],cn[ref]
BANDS=[0.5,1.5,2.5,9]; BN=["50~150m","150~250m",">250m"]
print(f"b{B} CityGS 立面參考顆粒：" + "、".join(f"{BN[i]} {int(((rm[:,2]>=BANDS[i])&(rm[:,2]<BANDS[i+1])).sum()):,}" for i in range(3)))
print("各高度帶：沒有（1.5 m 內無不透明顆粒）／對齊／平躺／歪（%）")
for lab,fam in [(a.split("=",1)[0],a.split("=",1)[1]) for a in sys.argv[2:]]:
    fs=sorted(glob.glob(fam))
    if not fs: print(f"{lab:16} 缺 ckpt"); continue
    m,o,n=ld(fs[0]); k=inblk(m)&(o>0.5); m2,n2=m[k],n[k]
    dd,ii=cKDTree(m2).query(rm,k=1,distance_upper_bound=0.015); has=np.isfinite(dd)
    al=np.zeros(len(rm),bool); fl=np.zeros(len(rm),bool)
    al[has]=np.abs((n2[ii[has]]*rn[has]).sum(1))>0.9; fl[has]=np.abs(n2[ii[has],2])>0.9
    s=f"{lab:16}"
    for bi in range(3):
        b=(rm[:,2]>=BANDS[bi])&(rm[:,2]<BANDS[bi+1])
        s+=f"｜{BN[bi]} {100*np.mean(~has[b]):4.1f}／{100*np.mean(al[b]):4.1f}／{100*np.mean(fl[b]&~al[b]):4.1f}／{100*np.mean(has[b]&~al[b]&~fl[b]):4.1f}"
    print(s, flush=True)
