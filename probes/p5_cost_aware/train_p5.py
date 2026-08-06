# -*- coding: utf-8 -*-
"""P5 probe: MCMC-2DGS on gsplat 1.4 — baseline / cost-aware arms + block_12 acid test.

Spec & gate: 紀錄/README.md §3. Data loading reuses the repo's
EstimatedDepthBlockColmap parser so block selection and the train/val split match
the CityGS-side runs exactly. Numbers are NOT comparable across rasterizer bases
(no depth/normal reg here; different lr schedule) — deliverables are mechanism
evidence: quality-vs-peak-VRAM, it/s, and whether block_12 survives natively.

Arms:
  --cost_aware off : stock gsplat MCMCStrategy (relocation prices value only)
  --cost_aware on  : dead-mask extended by value-per-cost criterion under an
                     intersection budget with dual-ascent shadow price lambda.
"""
import argparse
import dataclasses
import math
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from gsplat import rasterization_2dgs
from gsplat.strategy import MCMCStrategy
from internal.utils.ssim import ssim
from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthBlockColmap


# ---------------------------------------------------------------- data

def build_sets(cfg_path, data_path, block, out_dir):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ia = dict(cfg["data"]["parser"].get("init_args", {}))
    ia["block_id"] = block
    valid = {f.name for f in dataclasses.fields(EstimatedDepthBlockColmap)}
    ia = {k: v for k, v in ia.items() if k in valid}
    out = EstimatedDepthBlockColmap(**ia).instantiate(path=data_path, output_path=out_dir, global_rank=0).get_outputs()
    return out.train_set, out.val_set


def load_inv_depth(image_set, i, W, H, device):
    """CityGS-side depth supervision: estimated_depths/*.npy are inverse depths
    (Depth Anything disparity), aligned to SfM by per-image scale/offset stored in
    extra_data by the parser. Returns [H,W] inverse depth or None."""
    info = image_set.extra_data[i] if image_set.extra_data else None
    if info is None:
        return None
    d = image_set.extra_data_processor(info)  # np.load * scale + offset -> tensor
    d = d.to(device)
    if d.shape[-2:] != (H, W):
        d = F.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return d


def cam_to_gsplat(cam, device):
    viewmat = cam.world_to_camera.T.to(device)  # internal stores row-vector form; gsplat wants standard W2C
    K = torch.tensor([[float(cam.fx), 0, float(cam.cx)],
                      [0, float(cam.fy), float(cam.cy)],
                      [0, 0, 1]], device=device)
    return viewmat[None], K[None], int(cam.width), int(cam.height)


def load_gt(path, W, H, device):
    img = Image.open(path).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img)).float().to(device).permute(2, 0, 1) / 255.0


def tiled_forward_backward(params, viewmat, fx, fy, cx, cy, W, H, gt, sh_now, K_strips,
                           opacity_reg, scale_reg, immune_op,
                           need_geo=False, gt_inv=None, depth_w=0.0, lambda_normal=0.0):
    """K-strip camera-crop forward + per-strip backward (bounds peak VRAM to ~1/K on the
    intersection buffer; verified in verify_tiling.py). Loss = 0.8*L1 + 0.2*SSIM per strip
    (SSIM boundary is approximate across strips — acceptable) + opacity/scale reg added once.
    Returns (loss_scalar_for_log, aggregated_info_for_cost_aware). Grads accumulate into leaves;
    caller must have zero_grad'd before and step() after."""
    device = params["means"].device
    tiles_sum = torch.zeros(params["means"].shape[0], device=device)
    loss_log = 0.0
    for k in range(K_strips):
        h = H // K_strips
        v0 = k * h
        if k == K_strips - 1:
            h = H - v0
        Kmat = torch.tensor([[fx, 0, cx], [0, fy, cy - v0], [0, 0, 1]], device=device)[None]
        colors = torch.cat([params["sh0"], params["shN"]], 1)

        render_colors, render_alphas, render_normals, surf_normals, _, render_median, info = rasterization_2dgs(
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, viewmat, Kmat, W, h,
            sh_degree=sh_now, packed=True, render_mode="RGB+ED" if need_geo else "RGB")

        render = render_colors[0, ..., :3].permute(2, 0, 1).clamp(0, 1)      # [3,h,W]
        gt_strip = gt[:, v0:v0 + h, :]
        l1 = F.l1_loss(render, gt_strip)
        ssim_loss = 1.0 - ssim(render[None], gt_strip[None])
        strip_loss = (0.8 * l1 + 0.2 * ssim_loss) * (h / H)          # weight by strip fraction

        if need_geo:
            if gt_inv is not None:
                gt_inv_strip = gt_inv[v0:v0 + h, :]
                pred_inv = 1.0 / (render_median[0, ..., 0].clamp_min(0.0) + 1e-8)
                d_loss = 1.0 - ssim(pred_inv[None, None], gt_inv_strip[None, None])
                strip_loss = strip_loss + (depth_w * d_loss) * (h / H)
            if lambda_normal > 0:
                n_r, n_s = render_normals[0], surf_normals[0]           # [h,W,3]
                valid = (n_s.norm(dim=-1) > 0.5).float()
                n_loss = ((1.0 - (n_r * n_s).sum(-1)) * valid).sum() / valid.sum().clamp(min=1)
                strip_loss = strip_loss + (lambda_normal * n_loss) * (h / H)

        if k == 0:                                                    # per-gaussian regs added once
            op = torch.sigmoid(params["opacities"])
            strip_loss = strip_loss + opacity_reg * (op * (op < immune_op)).abs().mean() \
                                    + scale_reg * torch.exp(params["scales"]).abs().mean()
        strip_loss.backward()
        loss_log += float(strip_loss)
        # aggregate per-gaussian tile cost across strips (for cost-aware λ)
        tpg = info.get("tiles_per_gauss", None)
        if tpg is not None:
            with torch.no_grad():
                if tpg.dim() == 1 and tpg.shape[0] == tiles_sum.shape[0]:
                    tiles_sum += tpg.float()
                elif "gaussian_ids" in info:
                    tiles_sum.index_add_(0, info["gaussian_ids"].long(), tpg.float())
    return loss_log, {"tiles_per_gauss": tiles_sum}


# ---------------------------------------------------------------- init

def init_params(ply_path, device, sh_degree=3):
    """The depth-init PLYs are full 3DGS-convention gaussian PLYs (RTG-style):
    f_dc_* = SH DC coefficients, opacity = pre-activation logit, scale_0/1 = log
    2D tangential scales, rot_0..3 = wxyz quaternion. Reuse them directly so the
    init matches the CityGS-side runs; pad the missing 3rd (normal) scale small."""
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    names = {p.name for p in v.properties}
    assert {"f_dc_0", "opacity", "scale_0", "rot_0"} <= names, f"unexpected PLY fields: {sorted(names)}"
    xyz = np.stack([np.asarray(v[k]) for k in "xyz"], 1).astype(np.float32)
    N = xyz.shape[0]
    K = (sh_degree + 1) ** 2

    sh0 = torch.from_numpy(np.stack([np.asarray(v[f"f_dc_{i}"]) for i in range(3)], 1)).float()[:, None, :]
    shN = torch.zeros(N, K - 1, 3)
    op = torch.from_numpy(np.asarray(v["opacity"]).astype(np.float32))
    if float(op.min()) >= 0.0 and float(op.max()) <= 1.0:   # stored as activation → to logits
        op = torch.logit(op.clamp(1e-4, 1 - 1e-4))
    sc2 = torch.from_numpy(np.stack([np.asarray(v[f"scale_{i}"]) for i in range(2)], 1).astype(np.float32))
    if float(sc2.max()) > 0 and float(sc2.min()) >= 0:      # stored as linear → to log
        sc2 = torch.log(sc2.clamp(min=1e-6))
    sc3 = torch.cat([sc2, torch.full((N, 1), float(sc2.min(dim=1).values.mean()) - 4.0)], 1)
    quats = torch.from_numpy(np.stack([np.asarray(v[f"rot_{i}"]) for i in range(4)], 1).astype(np.float32))

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz)),
        "quats": torch.nn.Parameter(quats),
        "scales": torch.nn.Parameter(sc3),
        "opacities": torch.nn.Parameter(op),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)
    print(f"[init] opacity mean {torch.sigmoid(op).mean():.3f}, scale2d median {sc2.exp().median():.4f}")
    return params


def make_optimizers(params, scene_scale, max_steps):
    lrs = {"means": 1.6e-4 * scene_scale, "quats": 1e-3, "scales": 5e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": [params[k]], "lr": lr, "name": k}], eps=1e-15)
            for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / max_steps))
    return opts, means_sched


# ---------------------------------------------------------------- cost-aware strategy

class CostAwareMCMCStrategy(MCMCStrategy):
    """Stock MCMC + value-per-cost dead criterion (see 紀錄/README.md §3).

    cost_i  = EMA of screen tiles touched (∝ intersections ∝ rasterizer VRAM)
    value_i = opacity (what stock MCMC implicitly prices)
    dead_i  = opacity<=min_opacity OR value/cost < lambda ; lambda by dual ascent
    Implemented by temporarily masking `params` opacity? No — we mirror
    step_post_backward and pass an extended mask via the relocate op directly.
    """

    def __init__(self, *args, intersection_budget=8e6, dual_eta=0.05, cost_ema=0.8,
                 max_condemn_frac=0.10, cost_ceiling_tiles=2000.0, storage_cost_tiles=10.0,
                 lambda_mode="budget", vram_target_gb=5.4, eta_down=0.15, K_strips=1, **kw):
        super().__init__(*args, **kw)
        self.intersection_budget = intersection_budget
        self.dual_eta = dual_eta
        # Path (b): drive λ from measured VRAM instead of a hand-set tile budget.
        # ratio = VRAM_current / VRAM_target -> λ auto-scales to any GPU, no calibration.
        self.lambda_mode = lambda_mode           # "vram" (Path b) | "budget" (legacy proxy)
        self.vram_target_gb = vram_target_gb
        # Asymmetric dual: linear rise when over, EXPONENTIAL decay when under, so λ doesn't
        # stay stuck high after overshoot (the "50k frozen" pathology). λ <- λ*(1-eta_down).
        self.eta_down = eta_down
        self.last_ratio = 0.0
        self.cost_ema = cost_ema
        self.max_condemn_frac = max_condemn_frac
        # Hard per-point ceiling = infinite marginal price above this cost. This is the
        # spike killer: dual ascent reacts to AVERAGE usage, but the b12 death mode is a
        # single monster view (near-camera splat covering the frame) — those points must
        # be condemned on sight regardless of lambda. 2000 tiles ~ a 700px-radius splat.
        self.cost_ceiling_tiles = cost_ceiling_tiles
        # Every gaussian also costs params+Adam memory even when never rendered — the
        # CityGS-stack trim's 71% visibility cut is exactly the removal of such points.
        # Pricing storage as a constant makes never-visible points sellable assets
        # (value/storage ~ 0.1) instead of protected bystanders.
        self.storage_cost_tiles = storage_cost_tiles
        self.lmbda = 0.0
        self._cost = None
        self._usage_ema = None
        self.last_usage = 0.0
        self.max_point_cost = 0.0
        self.K_strips = K_strips
        self.window_max_tau = 0.0

    @torch.no_grad()
    def update_cost(self, info, n):
        # exact per-gaussian tile counts from the rasterizer (∝ intersections ∝ VRAM)
        tpg = info.get("tiles_per_gauss", None)
        if tpg is None:
            return
        tpg = tpg.float()
        frame = torch.zeros(n, device=tpg.device)
        if tpg.shape[0] == n:                       # non-packed [N] (C=1)
            frame = tpg
        elif "gaussian_ids" in info:                # packed [nnz] + ids
            frame.index_add_(0, info["gaussian_ids"].long(), tpg)
        else:
            return
        if self._cost is None or self._cost.shape[0] != n:
            self._cost = frame
        else:
            self._cost = self.cost_ema * self._cost + (1 - self.cost_ema) * frame
        usage = float(frame.sum())
        self.last_usage = usage
        self.max_point_cost = max(self.max_point_cost, float(frame.max()))

        current_tau = usage / n if n > 0 else 0
        self.window_max_tau = max(getattr(self, 'window_max_tau', 0.0), current_tau)

        self._usage_ema = usage if self._usage_ema is None else 0.9 * self._usage_ema + 0.1 * usage
        # load ratio: measured VRAM (Path b) or intersection-budget proxy (legacy)
        if self.lambda_mode == "vram":
            # current allocation at update_cost time (post-backward) — stable, tracks count,
            # and unlike max_memory_allocated it falls when the population shrinks (so λ can decay)
            ratio = (torch.cuda.memory_allocated() / 2 ** 30) / self.vram_target_gb
        else:
            ratio = self._usage_ema / self.intersection_budget
        self.last_ratio = ratio
        # asymmetric dual ascent: rise linearly when over, decay geometrically when under
        if ratio >= 1.0:
            self.lmbda = self.lmbda + self.dual_eta * (ratio - 1.0)
        else:
            self.lmbda = self.lmbda * (1.0 - self.eta_down)
        self.lmbda = max(0.0, self.lmbda)

    @torch.no_grad()
    def extra_dead_mask(self, params):
        n = params["opacities"].shape[0]
        if self._cost is None or self._cost.shape[0] != n:
            return None
        # (1) hard ceiling: monsters are condemned regardless of lambda / condemn cap
        ceiling_mask = self._cost > self.cost_ceiling_tiles
        # (2) budget term: value per TOTAL cost (render tiles + constant storage price)
        mask = ceiling_mask
        if self.lmbda > 0:
            value = torch.sigmoid(params["opacities"])
            total_cost = self._cost + self.storage_cost_tiles
            vpc = value / total_cost
            med = vpc.median().clamp(min=1e-12)
            budget_mask = (vpc / med) < self.lmbda
            k_max = max(1, int(self.max_condemn_frac * n))
            if int(budget_mask.sum()) > k_max:       # condemn worst offenders only
                idx = torch.topk(vpc, k_max, largest=False).indices
                budget_mask = torch.zeros_like(budget_mask)
                budget_mask[idx] = True
            mask = mask | budget_mask
        return mask if mask.any() else None

    sell_lambda = 1.0   # above this shadow price, condemned points are REMOVED (N shrinks)

    def step_post_backward(self, params, optimizers, state, step, info, lr):
        """Full three-mode economic controller (mirrors the parent's body):
        - BUY  (add_new): only while the budget is slack (lambda == 0)
        - MOVE (relocate): always — condemned points recycled like opacity-dead ones
        - SELL (remove): when the shadow price is high, condemned points are truly
          removed so the population (and its VRAM floor) actually shrinks. Stock MCMC
          has no shrink operator at all — N grows monotonically to cap, which is why
          the v2 arm still died: relocation conserves N while adds kept buying.
        """
        from gsplat.strategy.ops import inject_noise_to_position, remove

        state["binoms"] = state["binoms"].to(params["means"].device)
        binoms = state["binoms"]

        if (step < self.refine_stop_iter and step > self.refine_start_iter
                and step % self.refine_every == 0):

            # --- DYNAMIC CEILING (Risk-Adjusted N_max) ---
            if not hasattr(self, 'orig_cap_max'):
                self.orig_cap_max = getattr(self, 'cap_max', 2500000)

            V_os_gb = 0.718
            gamma_peak = 36.0 # Radix sort peak bytes/isect
            F_floats = 59 # sh3
            M = 4 # params+grad+2Adam
            omega = 2.0 # Spatial concentration risk premium (怪物集中度懲罰)

            denom = (M * F_floats * 4) + gamma_peak * getattr(self, 'window_max_tau', 0.0) * (omega / self.K_strips)
            V_target_bytes = (self.vram_target_gb * 0.9 - V_os_gb) * (1024**3)
            dynamic_n_max = int(V_target_bytes / max(denom, 1))

            # 更新 MCMC 的動態天花板 (能屈能伸，但不超過初始設定的硬上限)
            self.cap_max = min(dynamic_n_max, self.orig_cap_max)
            self.window_max_tau = 0.0 # 重置追蹤視窗

            # 如果 τ 暴增導致新天花板反向下壓並低於當前顆數，立刻觸發第三層 λ_budget 強制賣出
            if params["means"].shape[0] > self.cap_max:
                self.lmbda = max(self.lmbda, self.sell_lambda + 0.1)
            # ---------------------------------------------

            extra = self.extra_dead_mask(params)
            if extra is not None and extra.any():
                n_extra = int(extra.sum())
                if self.lmbda > self.sell_lambda:
                    n0 = params["means"].shape[0]
                    remove(params, optimizers, {}, extra)
                    print(f"[cost-aware] step {step}: SOLD {n0 - params['means'].shape[0]} "
                          f"(lambda={self.lmbda:.2f}, usage_ema={self._usage_ema/1e6:.1f}M)")
                else:
                    with torch.no_grad():
                        params["opacities"].data[extra] = math.log(1e-3 / (1 - 1e-3))
                    print(f"[cost-aware] step {step}: condemned {n_extra} "
                          f"(lambda={self.lmbda:.2f}, usage_ema={self._usage_ema/1e6:.1f}M)")
            self._cost = None

            _diag = os.environ.get("DIAG_N") == "1"
            n_a = params["means"].shape[0]                       # after condemn
            self._relocate_gs(params, optimizers, binoms)
            n_b = params["means"].shape[0]                       # after relocate
            added = False
            # BUY only with a safety margin below the budget (EMA lags real pressure)
            if self.lmbda <= 0 and (self._usage_ema or 0) < 0.7 * self.intersection_budget:
                self._add_new_gs(params, optimizers, binoms)
                added = True
            n_c = params["means"].shape[0]                       # after add
            if _diag:
                print(f"[N-diag] step {step}: after_condemn {n_a:,} -> after_reloc {n_b:,} "
                      f"-> after_add {n_c:,} (add={'Y' if added else 'gated'}, "
                      f"n_dead={int((torch.sigmoid(params['opacities'])<=0.005).sum())})")
            torch.cuda.empty_cache()

        inject_noise_to_position(params=params, optimizers=optimizers, state={},
                                 scaler=lr * self.noise_lr)


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(params, val_set, device, sh_degree, max_views=None):
    psnrs = []
    idxs = range(len(val_set.cameras)) if max_views is None else np.linspace(0, len(val_set.cameras) - 1, max_views).astype(int)
    for i in idxs:
        cam = val_set.cameras[i]
        viewmat, K, W, H = cam_to_gsplat(cam, device)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out = rasterization_2dgs(params["means"], params["quats"], torch.exp(params["scales"]),
                                 torch.sigmoid(params["opacities"]), colors, viewmat, K, W, H,
                                 sh_degree=sh_degree, packed=True)
        render = out[0][0].permute(2, 0, 1).clamp(0, 1)
        gt = load_gt(val_set.image_paths[i], W, H, device)
        psnrs.append(float(-10 * torch.log10(F.mse_loss(render, gt))))
    return float(np.mean(psnrs))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_id", type=int, required=True)
    ap.add_argument("--init_ply", required=True)
    ap.add_argument("--config", default="configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml")
    ap.add_argument("--data_path", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--cap_max", type=int, default=1_700_000)
    ap.add_argument("--max_steps", type=int, default=30_000)
    ap.add_argument("--refine_start", type=int, default=1000)
    ap.add_argument("--refine_stop_frac", type=float, default=0.7)
    ap.add_argument("--refine_every", type=int, default=150)
    ap.add_argument("--cost_aware", choices=["on", "off"], default="off")
    ap.add_argument("--intersection_budget", type=float, default=6e7)
    ap.add_argument("--lambda_mode", choices=["budget", "vram"], default="budget",
                    help="vram = Path(b): drive λ from measured VRAM (VRAM_current/VRAM_target), "
                         "auto-scales to any GPU, no budget calibration")
    ap.add_argument("--vram_target_gb", type=float, default=5.4)
    ap.add_argument("--eta_down", type=float, default=0.15,
                    help="asymmetric λ decay rate when under budget (fixes the frozen-high-λ pathology)")
    ap.add_argument("--opacity_reg", type=float, default=0.007)
    ap.add_argument("--scale_reg", type=float, default=0.007)
    ap.add_argument("--immune_op", type=float, default=0.9)
    ap.add_argument("--depth_reg", choices=["on", "off"], default="off",
                    help="inverse-depth SSIM vs estimated depths, weight 0.5*0.05^(t/T) (aggr17 recipe)")
    ap.add_argument("--depth_w_init", type=float, default=0.5)
    ap.add_argument("--depth_w_final_factor", type=float, default=0.05)
    ap.add_argument("--lambda_normal", type=float, default=0.0)
    ap.add_argument("--normal_from_iter", type=int, default=0)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--K_strips", type=int, default=1,
                    help=">1 tiles each view into K horizontal camera-crop strips with per-strip "
                         "backward -> bounds peak VRAM to ~1/K on the intersection buffer (the K "
                         "lever of the unified budget framework). depth/normal reg skipped when >1.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    train_set, val_set = build_sets(args.config, args.data_path, args.block_id, args.out)
    print(f"[data] block {args.block_id}: train {len(train_set.cameras)}, val {len(val_set.cameras)}")

    params = init_params(args.init_ply, device, args.sh_degree)
    n0 = params["means"].shape[0]
    scene_scale = float((params["means"].max(0).values - params["means"].min(0).values).norm() / 2)
    opts, means_sched = make_optimizers(params, scene_scale, args.max_steps)
    print(f"[init] {n0:,} gaussians, scene_scale {scene_scale:.2f}")

    kw = dict(cap_max=args.cap_max, refine_start_iter=args.refine_start,
              refine_stop_iter=int(args.max_steps * args.refine_stop_frac),
              refine_every=args.refine_every, min_opacity=0.005, verbose=False)
    strategy = (CostAwareMCMCStrategy(intersection_budget=args.intersection_budget,
                                      lambda_mode=args.lambda_mode, vram_target_gb=args.vram_target_gb,
                                      eta_down=args.eta_down, K_strips=args.K_strips, **kw)
                if args.cost_aware == "on" else MCMCStrategy(**kw))
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state()

    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(42)
    t0, tick = time.time(), time.time()
    log_path = os.path.join(args.out, "train_log.txt")
    logf = open(log_path, "a")

    for step in range(args.max_steps):
        i = int(rng.integers(len(train_set.cameras)))
        cam = train_set.cameras[i]
        viewmat, K, W, H = cam_to_gsplat(cam, device)
        gt = load_gt(train_set.image_paths[i], W, H, device)

        sh_now = min(step // 1000, args.sh_degree)

        if args.K_strips > 1:
            # K-strip tiled forward+backward (bounds peak VRAM); zero_grad first, backward inside
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            fx, fy, cx, cy = float(K[0, 0, 0]), float(K[0, 1, 1]), float(K[0, 0, 2]), float(K[0, 1, 2])

            need_geo = args.depth_reg == "on" or args.lambda_normal > 0
            gt_inv, depth_w = None, 0.0
            if args.depth_reg == "on":
                gt_inv = load_inv_depth(train_set, i, W, H, device)
                depth_w = args.depth_w_init * (args.depth_w_final_factor ** min(step / args.max_steps, 1.0))
            lnorm = args.lambda_normal if (args.lambda_normal > 0 and step >= args.normal_from_iter) else 0.0

            loss_v, info = tiled_forward_backward(
                params, viewmat, fx, fy, cx, cy, W, H, gt, sh_now, args.K_strips,
                args.opacity_reg, args.scale_reg, args.immune_op,
                need_geo, gt_inv, depth_w, lnorm)
            loss = torch.tensor(loss_v)
        else:
            colors = torch.cat([params["sh0"], params["shN"]], 1)
            need_geo = args.depth_reg == "on" or args.lambda_normal > 0
            render_colors, render_alphas, render_normals, surf_normals, _, render_median, info = rasterization_2dgs(
                params["means"], params["quats"], torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), colors, viewmat, K, W, H,
                sh_degree=sh_now, packed=True,
                render_mode="RGB+ED" if need_geo else "RGB")  # +ED enables normals_from_depth
            render = render_colors[0, ..., :3].permute(2, 0, 1).clamp(0, 1)

            l1 = F.l1_loss(render, gt)
            ssim_loss = 1.0 - ssim(render[None], gt[None])
            op = torch.sigmoid(params["opacities"])
            op_reg = (op * (op < args.immune_op)).abs().mean()
            sc_reg = torch.exp(params["scales"]).abs().mean()
            loss = 0.8 * l1 + 0.2 * ssim_loss + args.opacity_reg * op_reg + args.scale_reg * sc_reg

            if args.depth_reg == "on":
                gt_inv = load_inv_depth(train_set, i, W, H, device)
                if gt_inv is not None:
                    pred_inv = 1.0 / (render_median[0, ..., 0].clamp_min(0.0) + 1e-8)
                    d_w = args.depth_w_init * (args.depth_w_final_factor ** min(step / args.max_steps, 1.0))
                    d_loss = 1.0 - ssim(pred_inv[None, None], gt_inv[None, None])  # aggr17: pure depth-SSIM
                    loss = loss + d_w * d_loss
            if args.lambda_normal > 0 and step >= args.normal_from_iter:
                n_r, n_s = render_normals[0], surf_normals[0]           # [H,W,3]
                valid = (n_s.norm(dim=-1) > 0.5).float()
                n_loss = ((1.0 - (n_r * n_s).sum(-1)) * valid).sum() / valid.sum().clamp(min=1)
                loss = loss + args.lambda_normal * n_loss

            for o in opts.values():
                o.zero_grad(set_to_none=True)
            loss.backward()

        if args.cost_aware == "on":
            strategy.update_cost(info, params["means"].shape[0])
        _dn = os.environ.get("DIAG_N") == "1"
        _n0 = params["means"].shape[0] if _dn else 0
        strategy.step_post_backward(params, opts, state, step, info, lr=opts["means"].param_groups[0]["lr"])
        if _dn and params["means"].shape[0] != _n0:
            print(f"[N-step] step {step}: step_post_backward changed N {_n0:,} -> {params['means'].shape[0]:,}")
        _n1 = params["means"].shape[0] if _dn else 0
        for o in opts.values():
            o.step()
        if _dn and params["means"].shape[0] != _n1:
            print(f"[N-step] step {step}: optimizer.step changed N {_n1:,} -> {params['means'].shape[0]:,}")
        means_sched.step()

        if step % 500 == 0 or step == args.max_steps - 1:
            peak = torch.cuda.max_memory_allocated() / 2**30
            extra = ""
            if args.cost_aware == "on":
                extra = (f" usage {strategy.last_usage/1e6:.1f}M lambda {strategy.lmbda:.3f}"
                         f" maxcost {strategy.max_point_cost:.0f}")
                strategy.max_point_cost = 0.0
            msg = (f"step {step} loss {float(loss):.4f} n {params['means'].shape[0]:,} "
                   f"peakVRAM {peak:.2f}G {(500 / (time.time() - tick)):.2f}it/s{extra}")
            print(msg); logf.write(msg + "\n"); logf.flush()
            tick = time.time()
        if (step + 1) % args.val_every == 0 or step == args.max_steps - 1:
            p = evaluate(params, val_set, device, args.sh_degree, max_views=12)
            msg = f"VAL step {step+1}: psnr {p:.3f}"
            print(msg); logf.write(msg + "\n"); logf.flush()

    final_psnr = evaluate(params, val_set, device, args.sh_degree)
    peak = torch.cuda.max_memory_allocated() / 2**30
    summary = (f"FINAL block {args.block_id} arm={args.cost_aware}: psnr {final_psnr:.3f} "
               f"n {params['means'].shape[0]:,} peakVRAM {peak:.2f}G "
               f"wall {(time.time()-t0)/3600:.2f}h")
    print(summary); logf.write(summary + "\n"); logf.close()
    torch.save({k: v.detach().cpu() for k, v in params.items()}, os.path.join(args.out, "params.pt"))


if __name__ == "__main__":
    main()