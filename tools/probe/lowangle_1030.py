"""10-30 使用者「離開空拍視角就扭曲」：合成低角度視角的透空率（不需要 GT；lab GPU）。
每個指定塊：取塊盒內 SfM 點的 2～98 百分位當內容範圍，在範圍外圈放 12 台相機（高度 --height，空拍是 1.5～5.0），
朝中心看、俯角 --pitch 度，1600x900、焦距同 MatrixCity test 相機。每個模型（變體）渲染 RGB／alpha／深度：
  存 <out>/<變體>/b<B>_cam<k>.png（RGB）與 b<B>.npz（alpha、depth，float16）
比較（--compare <參考變體目錄> <變體目錄...>，CPU）：在參考（CityGS release）alpha>0.95 的像素上，
  透空率＝變體 alpha<0.5 或 深度 > 1.2×參考深度 的比例；浮在前面＝深度 < 0.8×參考 的比例；並列平均 alpha。
模型來源同 render_frames_1030.py：--base <合併 ckpt>（可加 --swap_block／--cand_runs）或 --base_blocks <跑次目錄>（只取塊內重組）。
用法（cwd＝gs）：
  python ../labrun/lowangle_1030.py --base_blocks ../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim --blocks 3 2 6 12 --out outputs/lab/lowangle_1030 --name citygs_release
  python ../labrun/lowangle_1030.py --base outputs/lab/<A+dist4>/deploy/merged_fp16.ckpt --blocks 3 2 6 12 --out outputs/lab/lowangle_1030 --name ad4 [--swap_block 3 --cand_runs ...]
  python ../labrun/lowangle_1030.py --compare outputs/lab/lowangle_1030/citygs_release outputs/lab/lowangle_1030/ad4 ...
"""
import argparse, glob, os, re, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "."); sys.path.insert(0, "tools")
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=None)
ap.add_argument("--base_blocks", default=None)
ap.add_argument("--swap_block", type=int, default=None)
ap.add_argument("--cand_runs", nargs="*", default=[])
ap.add_argument("--blocks", type=int, nargs="*", default=[3])
ap.add_argument("--name", default="model")
ap.add_argument("--out", default="outputs/lab/lowangle_1030")
ap.add_argument("--height", type=float, default=0.6)
ap.add_argument("--pitch", type=float, default=12.0)
ap.add_argument("--ncam", type=int, default=12)
ap.add_argument("--compare", nargs="*", default=None)
ap.add_argument("--part44", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
a = ap.parse_args()

if a.compare:
    ref = a.compare[0]
    print(f"參考：{ref}（只在參考 alpha>0.95 的像素上算）")
    for vd in a.compare[1:]:
        for f in sorted(glob.glob(os.path.join(ref, "b*.npz"))):
            g = os.path.join(vd, os.path.basename(f))
            if not os.path.exists(g):
                print(f"  {os.path.basename(vd)} {os.path.basename(f)}：缺"); continue
            R, V = np.load(f), np.load(g)
            ra, rd = R["alpha"].astype(np.float32), R["depth"].astype(np.float32)
            va, vd_ = V["alpha"].astype(np.float32), V["depth"].astype(np.float32)
            m = ra > 0.95
            see = ((va < 0.5) | (vd_ > 1.2 * rd)) & m
            front = (vd_ < 0.8 * rd) & (va > 0.5) & m
            print(f"  {os.path.basename(vd):<40} {os.path.basename(f)[:-4]}  參考實心像素 {m.mean() * 100:5.1f}%  "
                  f"透空率 {see.sum() / max(m.sum(), 1) * 100:5.1f}%  前方浮物 {front.sum() / max(m.sum(), 1) * 100:5.1f}%  "
                  f"平均 alpha 參考 {ra[m].mean():.3f}／變體 {va[m].mean():.3f}", flush=True)
    sys.exit(0)

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_points3D_binary, read_cameras_binary
from internal.cameras.cameras import Cameras
P_ = "gaussian_model.gaussians."
dev = torch.device("cuda")


def last_ckpt(d):
    cs = [c for c in glob.glob(os.path.join(d, "checkpoints", "*.ckpt")) if re.search(r"step=(\d+)", c) and "_inbox" not in c]
    return max(cs, key=lambda c: int(re.search(r"step=(\d+)", c).group(1))) if cs else None


def inbox(means, part_pt, block_id):
    from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
    P = torch.load(part_pt, map_location="cpu")
    coords = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
    boxes = coords.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.0)
    x = means.float() @ P["extra_data"]["rotation_transform"][:3, :3].T
    if P["scene_config"]["contract"]:
        x = CityGSPartitioning.contract_to_unisphere(x[:, :2], P["scene_config"]["aabb"], ord=torch.inf)
    return CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=x[:, :2])[block_id]


def part_of(ck, run_root):
    il = ck["datamodule_hyper_parameters"]["parser"].image_list
    for c in (il, os.path.join(run_root, il)):
        p = os.path.join(os.path.dirname(c), "partitions.pt")
        if os.path.exists(p):
            return p
    raise SystemExit(f"⛔ 找不到 partitions.pt（image_list={il}）")


def act_props(c):
    """⚠ 10-31 修：載入器預設 pre_activate（scales 取 exp、opacity 取 sigmoid、SH 併成 shs），基準的 model.properties 是激活後的值；
    直接拿 state_dict 原始參數去拼會混用兩種表示（scale 錯 => 光柵器配 14.8 GB／越界）。候選與重組一律走載入器（同 tools/swap_block_eval.py）。"""
    m, _, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(c, device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in m.properties.items()}
    del m
    return pr, ck


def props_of(ck):
    pr = {k[len(P_):]: v for k, v in ck["state_dict"].items() if k.startswith(P_)}
    if "shs_dc" in pr:
        pr["shs"] = torch.cat([pr.pop("shs_dc"), pr.pop("shs_rest")], 1)
    return pr


# ── 模型 ──
if a.base:
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.base, device=dev, eval_mode=True)
    base = {k: v.detach().float().cpu() for k, v in model.properties.items()}
else:
    rd = a.base_blocks.rstrip("/"); root = rd.split("/outputs/")[0]
    parts, part_pt, first = [], None, None
    for d in sorted(glob.glob(os.path.join(rd, "blocks", "block_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        c = last_ckpt(d)
        if c is None:
            continue
        pr, ck = act_props(c)
        if part_pt is None:
            part_pt = part_of(ck, root); first = c
        m = inbox(pr["means"], part_pt, ck["datamodule_hyper_parameters"]["parser"].block_id)
        parts.append({k: v[m].float() for k, v in pr.items()}); del ck, pr
    base = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(first, device=dev, eval_mode=True)
print(f"[基準] N={base['means'].shape[0]:,}", flush=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
    setattr(renderer, f, False)
variants = [(a.name, None)]          # 10-31：換塊變體改成用到才組（原本一次組好全部，十幾個 27M 顆的副本把 RAM 吃爆、rc=137）
keep = None
if a.swap_block is not None:
    keep = ~inbox(base["means"], a.part44, a.swap_block)
    for r in a.cand_runs:
        rdir = r if os.path.isdir(r) else os.path.join("outputs/lab", r)
        c = last_ckpt(os.path.join(rdir, "blocks", f"block_{a.swap_block}"))
        if c is None:
            print(f"  ⚠ {r} 沒有 block_{a.swap_block} ckpt => 跳過"); continue
        variants.append((f"{a.name}_swap{a.swap_block}_{os.path.basename(r.rstrip('/'))}", c))


def build(src):
    if src is None:
        return base
    pr, ck = act_props(src)
    m = inbox(pr["means"], part_of(ck, "."), a.swap_block)
    return {k: torch.cat([base[k][keep], pr[k][m].float()]) for k in base}


# ── 合成相機 ──
P = torch.load(a.part44, map_location="cpu")
Rr = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = P["partition_coordinates"]["id"].numpy()
pts = read_points3D_binary("data/matrix_city/aerial/train/block_all/sparse/0/points3D.bin")
X = np.array([p.xyz for p in pts.values()]); Q = X @ Rr.T
tc = read_cameras_binary("data/matrix_city/aerial/test/block_all_test_official2/sparse/0/cameras.bin")
c0 = tc[list(tc)[0]]; W, H = round(c0.width / 1.2), round(c0.height / 1.2); fx = c0.params[0] * W / c0.width


def cams_for(b):
    k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0])
    lo = P["partition_coordinates"]["xy"][k].double().numpy(); hi = lo + P["scene_config"]["partition_size"][k].double().numpy()
    q = Q[np.all((Q[:, :2] >= lo) & (Q[:, :2] < hi), 1)]
    q = q[(q[:, 2] > np.percentile(q[:, 2], 2)) & (q[:, 2] < np.percentile(q[:, 2], 98))]
    clo, chi = np.percentile(q[:, :2], 2, 0), np.percentile(q[:, :2], 98, 0)
    ctr = np.r_[(clo + chi) / 2, a.height]; rad = 0.6 * np.linalg.norm(chi - clo) / 2
    Rs, Ts = [], []
    for j in range(a.ncam):
        th = 2 * np.pi * j / a.ncam
        C_r = ctr + np.r_[rad * np.cos(th), rad * np.sin(th), 0.0]                        # 分區座標
        fwd = np.r_[(ctr - C_r)[:2], 0.0]; fwd /= np.linalg.norm(fwd)
        fwd = np.cos(np.radians(a.pitch)) * fwd + np.r_[0, 0, -np.sin(np.radians(a.pitch))]
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); down = np.cross(fwd, right)
        Rc_r = np.stack([right, down, fwd])                                             # 列＝相機軸（分區座標）
        Rw = Rc_r @ Rr                                                                  # world->cam（世界座標）
        Cw = Rr.T @ C_r
        Rs.append(Rw); Ts.append(-Rw @ Cw)
    t = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    n = len(Rs)
    return Cameras(R=t(Rs), T=t(Ts), fx=t([fx] * n), fy=t([fx] * n), cx=t([W / 2] * n), cy=t([H / 2] * n),
                   width=torch.full((n,), W, dtype=torch.int16), height=torch.full((n,), H, dtype=torch.int16),
                   appearance_id=torch.zeros(n, dtype=torch.int), normalized_appearance_id=torch.zeros(n),
                   distortion_params=None, camera_type=torch.zeros(n, dtype=torch.int8))


bg = torch.zeros(3, device=dev)
camsets = {b: cams_for(b) for b in a.blocks}
for vn, src in variants:
    props = build(src)
    model.properties = {k: t.to(dev) for k, t in props.items()}
    od = os.path.join(a.out, vn); os.makedirs(od, exist_ok=True)
    for b, cams in camsets.items():
        if vn != a.name and b != a.swap_block:
            continue                      # 換塊變體只渲被換的那一塊
        al, de = [], []
        for i in range(len(cams)):
            with torch.no_grad():
                o = renderer(cams[i].to_device(dev), model, bg_color=bg)
            img = (o["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(od, f"b{b}_cam{i:02d}.png"))
            al.append(o["rend_alpha"][0].cpu().numpy().astype(np.float16)); de.append(o["surf_depth"][0].cpu().numpy().astype(np.float16))
        np.savez_compressed(os.path.join(od, f"b{b}.npz"), alpha=np.stack(al), depth=np.stack(de))
        print(f"[{vn}] b{b}：{len(cams)} 台合成相機（高度 {a.height}、俯角 {a.pitch}°）平均 alpha {np.mean([x.mean() for x in al]):.3f} => {od}", flush=True)
    model.properties = {k: t[:1].to(dev) for k, t in props.items()}
    if src is not None:
        del props
    torch.cuda.empty_cache()
print("DONE lowangle", flush=True)
