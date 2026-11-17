"""11-17 使用者「只留一層、完整覆蓋」的前置量測（不訓練）：
  ① 省多少：把被藏住的層拿掉後，同一批訓練視角的 forward／backward GPU 時間（lean 路徑＝一般訓練步；renderer 設定照 ckpt，含 sbox／pbw）
  ② 縫長什麼樣：拿掉之後畫面有變的像素＝原本靠後層上色的像素＝前層的縫；量佔畫面比例、連通塊大小分布（px）、在地面還是立面
變體（同一個 ckpt 只刪顆粒、不重訓）：
  全部｜去地下＝陸地地面格（2 m 格、可靠 SfM 中位 z < 3 m 且 >= −2 m；同 ground_split_sea_1117）上中心低於 SfM 中位 0.5 m 以上的面元
  另外可給 --oracle <ckpt>（tools/probe/oracle_behind_1117.py 的輸出＝GT 刪牆後層）=> 再加「去牆後」「去牆後＋去地下」
縫的判定：拿掉後的 RGB 與原本的平均絕對差 > --dc（預設 0.04）或 alpha 掉超過 0.1。
⚠ 計時要 [solo]（同卡有別的行程會不準）。用法（lab）：python tools/probe/hidden_layer_1117.py --block 2 --ckpt <glob> [--oracle <glob>]
"""
import argparse, glob, os, sys
import numpy as np, torch
from scipy.ndimage import label
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_points3D_binary
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--oracle", default="")
ap.add_argument("--nview", type=int, default=24); ap.add_argument("--rep", type=int, default=3); ap.add_argument("--dc", type=float, default=0.04)
a = ap.parse_args()
pick = lambda pat: sorted(x for x in glob.glob(pat) if "xyz" not in x)[-1]
CK = pick(a.ckpt); OR = pick(a.oracle) if a.oracle else ""
D = "data/matrix_city/aerial/train/block_all"
dev = torch.device("cuda")

# 地面格（同 ground_split_sea_1117）
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
Pt = np.stack([x.xyz for x in pts.values()]); trk = np.array([len(x.image_ids) for x in pts.values()]); err = np.array([x.error for x in pts.values()]); del pts
Pt = Pt[(trk >= 3) & (err < 1.5)]
CELL = 0.02
key = lambda xy: (np.floor(xy[:, 0] / CELL).astype(np.int64) << 32) + np.floor(xy[:, 1] / CELL).astype(np.int64)
ks = key(Pt[:, :2]); o_ = np.argsort(ks); ks, zs = ks[o_], Pt[o_, 2]
uk, st = np.unique(ks, return_index=True); med = np.array([np.median(z) for z in np.split(zs, st[1:])])
gcell = dict(zip(uk[(med < 0.03) & (med >= -0.02)], med[(med < 0.03) & (med >= -0.02)]))   # 只要陸地格


def under_mask(means):
    m = means.detach().cpu().double().numpy()
    gz = np.array([gcell.get(int(x), np.nan) for x in key(m[:, :2])])
    return torch.from_numpy(np.isfinite(gz) & ((m[:, 2] - gz) * 100 <= -0.5))


def load(path, drop_under):
    model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(path, device=dev, eval_mode=False, pre_activate=False)
    dmh = ck["datamodule_hyper_parameters"]; del ck
    n0 = model.n_gaussians; nd = 0
    if drop_under:
        keep = ~under_mask(model.get_property("means")).to(dev); nd = int((~keep).sum())
        model.properties = {k: torch.nn.Parameter(model.get_property(k)[keep].detach().clone(), requires_grad=True) for k in model.property_names}
    return model, renderer, dmh, n0, nd


V = [("全部", CK, False), ("去地下", CK, True)] + ([("去牆後", OR, False), ("去牆後＋去地下", OR, True)] if OR else [])
model, renderer, dmh, _, _ = load(CK, False)
_pp = dmh["parser"]
if getattr(_pp, "points_from", None) == "ply" and not os.path.exists(os.path.join(dmh["path"], _pp.ply_file)):
    _pp.points_from = "random"
ts = _pp.instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(CK)), global_rank=0).get_outputs().train_set
idx = np.linspace(0, len(ts.cameras) - 1, a.nview).round().astype(int).tolist()
cams = [ts.cameras[i].to_device(dev) for i in idx]
del model, renderer; torch.cuda.empty_cache()
print(f"b{a.block}｜{CK}{'｜oracle ' + OR if OR else ''}｜視角 {len(cams)}｜sbox/pbw 照 ckpt 的 renderer", flush=True)
bg = torch.zeros(3, device=dev)
g = torch.Generator(device=dev).manual_seed(0)
W = [None] * len(cams)


def timed(model, renderer):
    params = [p for p in model.parameters() if p.requires_grad]
    tf, tb, out_imgs = [], [], []
    for k, cam in enumerate(cams):
        for p in params:
            p.grad = None
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        out = renderer(cam, model, bg_color=bg, _lean=True)
        e1.record()
        img = out["render"]
        if W[k] is None:
            W[k] = torch.rand(img.shape, device=dev, generator=g) - 0.5
        (img * W[k]).sum().backward()
        e2.record(); torch.cuda.synchronize()
        tf.append(e0.elapsed_time(e1)); tb.append(e1.elapsed_time(e2))
    return np.array(tf), np.array(tb)


@torch.no_grad()
def renders(model, renderer):
    R = []
    for cam in cams:
        out = renderer(cam, model, bg_color=bg)
        R.append((out["render"].clamp(0, 1).cpu(), out["rend_alpha"].cpu() if torch.is_tensor(out.get("rend_alpha")) else None,
                  out["rend_normal"].cpu() if torch.is_tensor(out.get("rend_normal")) else None))
    return R


res = {}; base_r = None
for name, path, du in V:
    model, renderer, _, n0, nd = load(path, du)
    timed(model, renderer)                                      # 暖機
    T = [timed(model, renderer) for _ in range(a.rep)]
    tf = np.median(np.stack([t[0] for t in T]), 0); tb = np.median(np.stack([t[1] for t in T]), 0)
    r = renders(model, renderer)
    if base_r is None:
        base_r = r
    res[name] = (model.n_gaussians, nd, tf, tb, r)
    del model, renderer; torch.cuda.empty_cache()

N0, _, tf0, tb0, _ = res["全部"]
print(f"{'變體':16}{'N':>12}{'刪掉':>10}{'forward ms':>12}{'backward ms':>13}{'合計 ms':>10}{'對全部':>9}", flush=True)
for name, (n, nd, tf, tb, _) in res.items():
    tot = tf.mean() + tb.mean(); tot0 = tf0.mean() + tb0.mean()
    print(f"{name:16}{n:>12,}{nd:>10,}{tf.mean():>12.2f}{tb.mean():>13.2f}{tot:>10.2f}{100 * (tot / tot0 - 1):>+8.1f}%", flush=True)

print("縫＝拿掉之後畫面有變的像素（RGB 平均絕對差 > %.2f 或 alpha 掉 > 0.1）；連通塊（8 連通）大小分布＝縫像素落在各大小區間的比例；地面／立面用渲染法向 |n_z| 分" % a.dc, flush=True)
BINS = [(1, 4), (5, 16), (17, 64), (65, 256), (257, 4096), (4097, 10 ** 9)]
for name, (n, nd, tf, tb, r) in res.items():
    if name == "全部":
        continue
    frac, sizes, flat, wall = [], [], 0, 0
    for (i0, a0, n0_), (i1, a1, _) in zip(base_r, r):
        d = (i1 - i0).abs().mean(0)
        gap = d > a.dc
        if a0 is not None and a1 is not None:
            gap |= (a0[0] - a1[0]) > 0.1
        gap = gap.numpy(); frac.append(gap.mean())
        lab, nl = label(gap, structure=np.ones((3, 3)))
        if nl:
            cnt = np.bincount(lab.ravel())[1:]; sizes.append(cnt)
        if n0_ is not None:
            nz = n0_[2].abs().numpy()
            flat += int((gap & (nz > 0.8)).sum()); wall += int((gap & (nz < 0.3)).sum())
    s = np.concatenate(sizes) if sizes else np.zeros(0, int)
    tot = max(int(s.sum()), 1)
    dist = "｜".join(f"{lo}～{hi if hi < 10 ** 9 else '':}px {100 * s[(s >= lo) & (s <= hi)].sum() / tot:.0f}%" for lo, hi in BINS)
    print(f"[{name}] 縫佔畫面 中位 {100 * np.median(frac):.2f}%、平均 {100 * np.mean(frac):.2f}%｜連通塊 {len(s):,} 個、大小中位 {np.median(s) if len(s) else 0:.0f} px｜"
          f"縫像素 {dist}｜縫在躺平處 {100 * flat / tot:.0f}%、立面 {100 * wall / tot:.0f}%", flush=True)
print("DONE hidden_layer")
