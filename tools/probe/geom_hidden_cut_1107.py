"""11-07 使用者「1、2 作看看，記得量測時間」之 2：用幾何判斷「躲在表面後面」的不透明顆粒，離線刪掉看傷不傷（不改光柵器）。GPU；只讀＋存 ckpt。
每顆中心投到本塊全部訓練相機：F＝在畫面內的台數；H＝在畫面內且被擋的台數（該像素渲染 α > 0.95、顆粒比表面深 > --margin_px 個像素足跡）。
刪法（只刪 o > 0.5 的）：全部被擋（F >= 3 且 H＝F）、95% 被擋（F >= 3 且 H >= 0.95F）。
⚠ 只看中心：中心被擋、但面元大到邊緣露出來的會被誤刪 => 正是要量的風險（held-out／F1／低角度會掉）。
時間：離線判斷本身的牆鐘；刪前／刪後在同一批訓練視角的平均渲染時間（ms／幀）。
用法：python tools/probe/geom_hidden_cut_1107.py --block 3 --ckpt <ckpt> --out <資料夾>
"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary, read_cameras_binary
from internal.cameras.cameras import Cameras

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--margin_px", type=float, default=5.0)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda")
names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}; cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = [n for n in names if n in ims]


def mk(nm):
    im = ims[nm]; cc = cams[im.camera_id]; W, H = round(cc.width / 1.2), round(cc.height / 1.2); s = W / cc.width
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(im.qvec2rotmat()), T=t(im.tvec), fx=t([cc.params[0] * s])[0], fy=t([cc.params[1] * s])[0], cx=t([cc.params[2] * s])[0],
                   cy=t([cc.params[3] * s])[0], width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1), distortion_params=None,
                   camera_type=torch.zeros(1, dtype=torch.int8))[0]


ck = sorted(glob.glob(a.ckpt))[-1]
model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
for fl in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
    if hasattr(rnd, fl):
        setattr(rnd, fl, False)
model.to(dev)
X = model.get_xyz.detach().float(); N = X.shape[0]; Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
F = torch.zeros(N, device=dev); Hc = torch.zeros(N, device=dev); bg = torch.zeros(3, device=dev)
CAMS = [mk(n).to_device(dev) for n in names]
torch.cuda.synchronize(); t0 = time.perf_counter()
for cd in CAMS:
    with torch.no_grad():
        o = rnd(cd, model, bg_color=bg)
        al = o["rend_alpha"].reshape(o["rend_alpha"].shape[-2:]); zs = o["surf_depth"].reshape(al.shape); Hh, Ww = al.shape
        p = Xh @ cd.world_to_camera.float(); zc = p[:, 2]
        uu = (float(cd.fx) * p[:, 0] / zc.clamp_min(1e-6) + float(cd.cx)).floor().long(); vv = (float(cd.fy) * p[:, 1] / zc.clamp_min(1e-6) + float(cd.cy)).floor().long()
        inn = (zc > 1e-3) & (uu >= 0) & (uu < Ww) & (vv >= 0) & (vv < Hh)
        uu, vv = uu.clamp(0, Ww - 1), vv.clamp(0, Hh - 1)
        F += inn.float()
        Hc += (inn & (al[vv, uu] > 0.95) & (zs[vv, uu] > 0) & (zc > zs[vv, uu] * (1 + a.margin_px / float(cd.fx)))).float()
torch.cuda.synchronize(); t_judge = time.perf_counter() - t0
op = model.get_opacities().detach().reshape(-1) > 0.5
cuts = {"全部被擋": (F >= 3) & (Hc == F) & op, "95%被擋": (F >= 3) & (Hc >= 0.95 * F) & op}


def render_ms(m, n=60):
    sel = CAMS[:: max(1, len(CAMS) // n)][:n]
    with torch.no_grad():
        for cd in sel[:5]:
            rnd(cd, m, bg_color=bg)
        torch.cuda.synchronize(); t = time.perf_counter()
        for cd in sel:
            rnd(cd, m, bg_color=bg)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / len(sel) * 1e3


ms0 = render_ms(model)
print(f"b{B}｜{len(CAMS)} 台訓練相機｜{N:,} 顆（o > 0.5：{int(op.sum()):,}）｜離線判斷 {t_judge:.1f} s（{t_judge / len(CAMS) * 1e3:.0f} ms／台）｜原模型渲染 {ms0:.2f} ms／幀", flush=True)
raw = torch.load(ck, map_location="cpu"); sd = raw["state_dict"]
assert sd["gaussian_model.gaussians.means"].shape[0] == N
summary = {"N": N, "judge_s": t_judge, "render_ms": ms0}
for cn, cut in cuts.items():
    keep = ~cut
    props = {k: v[keep] for k, v in model.properties.items()}
    m2 = model; old = model.properties; m2.properties = props
    ms1 = render_ms(m2); m2.properties = old
    kc = keep.cpu()
    sd2 = {k: (v[kc] if hasattr(v, "shape") and v.dim() and v.shape[0] == N else v) for k, v in sd.items()}
    ck2 = {k: v for k, v in raw.items() if k not in ("state_dict", "optimizer_states")}; ck2["state_dict"] = sd2; ck2["optimizer_states"] = []
    od = f"{a.out}/{cn}/blocks/block_{B}/checkpoints"; os.makedirs(od, exist_ok=True)
    torch.save(ck2, f"{od}/{os.path.basename(ck)}")
    src = os.path.join(os.path.dirname(os.path.dirname(ck)), "cameras.json")
    if os.path.exists(src):
        shutil.copy(src, f"{a.out}/{cn}/blocks/block_{B}/cameras.json")
    print(f"{cn}：刪 {int(cut.sum()):,} 顆（全部是 o > 0.5）｜剩 {int(keep.sum()):,}｜渲染 {ms1:.2f} ms／幀（原 {ms0:.2f}）", flush=True)
    summary[cn] = {"cut": int(cut.sum()), "render_ms": ms1}
json.dump(summary, open(f"{a.out}/summary.json", "w"), ensure_ascii=False)
print("DONE")
