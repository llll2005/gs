# -*- coding: utf-8 -*-
"""Per-block cap_max calibration for the closed-form budget formula:

    N_max = (V_target * frag_safety - V_os) / (M*F*4 + gamma * tau_max * tau_growth / K)

Constants are kernel-specific; tau is (mostly) geometry-specific. Workflow:

1. `taus`       — measure per-block worst-view tile coverage tau with a cheap
                  gsplat forward pass (no training, no grad). tau ranks block
                  overdraw pathology (b12 ~73 vs benign blocks ~10-30).
2. `probe-trim` — measure the trim kernel's actual bytes/point: render+backward
                  at N and N/2 on real block geometry, linear-fit slope and
                  intercept -> gamma_trim (render bytes per tile-intersection)
                  and V_os_trim (non-per-point overhead incl. loss activations).
3. `caps`       — combine 1+2 into a per-block cap_max table (CSV + stdout).

tau_growth (default 1.65) corrects init-tau -> trained-tau drift, calibrated on
b12 (init tau_MAX 73 -> ~120 observed in the K2 freeze test, 2026-07-13).

⚠ STALE FOR MCMC CONFIGS SINCE 2026-08-06. `gamma_trim` and `vos_trim` were measured while
`gaussian_splatting.py` still ran the SPLIT backward with `retain_graph=True` on every densify
step, so the autograd graph stayed alive across two backwards. MCMC never read the viewspace
gradient that split existed to produce, and it is now skipped
(`_split_backward_needed`), which frees the graph after a single backward and lowers the peak.

The constants are therefore CONSERVATIVE for MCMC: `block_caps.csv` under-estimates N_max and we
have been capping below what the card can hold. Re-running `probe-trim` on the current code should
raise the ceiling -- which is the thing "b12@2M hits a ~1.5M wall" has been stuck on.
Until that is redone, treat the caps as a lower bound.

Usage:
  python tools/calibrate_block_caps.py taus --blocks 0-24
  python tools/calibrate_block_caps.py probe-trim --block 12 [--sb]
  python tools/calibrate_block_caps.py caps --gamma_trim <G> --vos_trim <V> [--taus_csv ...]
"""
import argparse
import csv
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "probes", "p5_cost_aware"))

import numpy as np
import torch

GB = 2 ** 30
DEFAULT_CONFIG = "configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml"
DEFAULT_DATA = "data/matrix_city/aerial/train/block_all"
DEFAULT_PLY_DIR = "data/matrix_city/aerial/train/block_all/depth_init"


def parse_blocks(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_ply_xyz_rgb(path):
    from internal.utils.gaussian_utils import GaussianPlyUtils
    try:
        g = GaussianPlyUtils.load_from_ply(path)
        xyz = torch.tensor(np.asarray(g.xyz)).float()
        # depth-init plys carry color in sh dc
        from internal.utils.sh_utils import SH2RGB
        rgb = torch.clamp(SH2RGB(torch.tensor(np.asarray(g.features_dc)).float().squeeze()), 0., 1.)
        if rgb.dim() == 1:
            rgb = rgb[None, :].repeat(xyz.shape[0], 1)
        return xyz, rgb
    except Exception:
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"]
        xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], -1)).float()
        if "red" in v.data.dtype.names:
            rgb = torch.tensor(np.stack([v["red"], v["green"], v["blue"]], -1)).float() / 255.0
        else:
            rgb = torch.full_like(xyz, 0.5)
        return xyz, rgb


# ---------------------------------------------------------------- taus

def cmd_taus(args):
    """Per-block worst-view tau via gsplat forward (no grad)."""
    from gsplat import rasterization_2dgs
    from train_p5 import build_sets, init_params, cam_to_gsplat

    device = "cuda"
    rows = []
    for b in parse_blocks(args.blocks):
        ply = os.path.join(args.ply_dir, "block_%d.ply" % b)
        if not os.path.exists(ply):
            print("block %2d: no init ply, skip" % b)
            continue
        try:
            train_set, _ = build_sets(args.config, args.data_path, b, "/tmp/calib_caps")
        except Exception as e:
            print("block %2d: dataparser failed (%s), skip" % (b, e))
            continue
        params = init_params(ply, device, 0)  # sh degree irrelevant for tau
        N = params["means"].shape[0]
        taus = []
        idxs = np.linspace(0, len(train_set.cameras) - 1, min(args.n_views, len(train_set.cameras))).astype(int)
        with torch.no_grad():
            for i in idxs:
                cam = train_set.cameras[int(i)].to_device(device)
                viewmat, K3, W, H = cam_to_gsplat(cam, device)
                colors = torch.cat([params["sh0"], params["shN"]], 1)
                _, _, _, _, _, _, info = rasterization_2dgs(
                    params["means"], params["quats"], torch.exp(params["scales"]),
                    torch.sigmoid(params["opacities"]), colors, viewmat, K3, W, H,
                    sh_degree=0, packed=True, render_mode="RGB")
                tpg = info.get("tiles_per_gauss")
                taus.append(float(tpg.float().sum()) / N if tpg is not None else float("nan"))
        del params
        torch.cuda.empty_cache()
        tau_mean, tau_max = float(np.mean(taus)), float(np.max(taus))
        rows.append({"block": b, "N_init": N, "tau_mean": round(tau_mean, 1), "tau_max": round(tau_max, 1)})
        print("block %2d: N_init %8d  tau mean %6.1f  MAX %6.1f" % (b, N, tau_mean, tau_max))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "N_init", "tau_mean", "tau_max"])
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s (%d blocks)" % (args.out, len(rows)))


# ---------------------------------------------------------------- probe-trim

def _trim_step_peak(model, renderer, cameras, device, n_views, optimizer=None):
    """Peak allocated bytes over n_views of render fwd+bwd (approximating the
    training-loss activation footprint: rgb + normal + dist terms)."""
    bg = torch.zeros(3, device=device)
    peaks = []
    idxs = np.linspace(0, len(cameras) - 1, n_views).astype(int)
    for i in idxs:
        cam = cameras[int(i)].to_device(device)
        for p in model.gaussians.values():
            if p.grad is not None:
                p.grad = None
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        out = renderer(cam, model, bg_color=bg)
        loss = out["render"].mean() \
            + (out["rend_normal"] * out["surf_normal"]).sum(dim=0).mean() \
            + out["rend_dist"].mean()
        loss.backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)  # keep grads+state resident like training
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    return max(peaks)


def _build_trim_model(xyz, rgb, device, sb, keep=None):
    if sb:
        from internal.models.gaussian_2d_sb import Gaussian2DSB
        cfg = Gaussian2DSB(sh_degree=0, sb_number=2)
    else:
        from internal.models.gaussian_2d import Gaussian2D
        cfg = Gaussian2D(sh_degree=3)
    model = cfg.instantiate()
    if keep is not None:
        xyz, rgb = xyz[keep], rgb[keep]
    model.setup_from_pcd(xyz.to(device), rgb.to(device))
    model.to(device)
    for p in model.gaussians.values():
        p.requires_grad_(True)
    return model


def cmd_probe_trim(args):
    from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer
    from train_p5 import build_sets

    device = "cuda"
    train_set, _ = build_sets(args.config, args.data_path, args.block, "/tmp/calib_caps")
    ply = os.path.join(args.ply_dir, "block_%d.ply" % args.block)
    xyz, rgb = load_ply_xyz_rgb(ply)
    N_full = xyz.shape[0]
    renderer = SepDepthTrim2DGSRenderer(depth_ratio=1.0)
    if args.sb:
        from internal.renderers.sep_depth_trim_2dgs_sb_renderer import SepDepthTrim2DGSSBRenderer
        renderer = SepDepthTrim2DGSSBRenderer(depth_ratio=1.0)

    results = {}
    for frac in (0.5, 1.0):
        keep = torch.randperm(N_full)[: int(N_full * frac)]
        model = _build_trim_model(xyz, rgb, device, args.sb, keep)
        # materialize Adam state so M=4 accounting is real, then measure
        opt = torch.optim.Adam([p for p in model.gaussians.values() if p.numel() > 0], lr=0.0, eps=1e-15)
        peak = _trim_step_peak(model, renderer, train_set.cameras, device, args.n_views, optimizer=opt)
        n = int(N_full * frac)
        results[frac] = (n, peak)
        print("frac %.1f: N=%d  peak=%.3f GB" % (frac, n, peak / GB))
        del model, opt
        torch.cuda.empty_cache()

    (n1, p1), (n2, p2) = results[0.5], results[1.0]
    slope = (p2 - p1) / (n2 - n1)          # bytes per point, all-in (model M=4 + render)
    intercept = p1 - slope * n1            # non-per-point overhead
    F = 25 if args.sb else 59
    model_bpp = 4 * F * 4                  # M=4 (params+grad+2 Adam)
    render_bpp = max(0.0, slope - model_bpp)
    print("\n===== TRIM PROBE (block %d, %s) =====" % (args.block, "SB" if args.sb else "SH3"))
    print("slope      = %.0f B/pt  (model %d + render %.0f)" % (slope, model_bpp, render_bpp))
    print("V_os_trim  = %.3f GB   (incl. loss activations)" % (intercept / GB))
    print("NOTE: render B/pt here is at THIS block's init tau; divide by tau_max")
    print("      from `taus` to get gamma_trim, or pass render_bpp directly to `caps`.")


# ---------------------------------------------------------------- caps

def cmd_caps(args):
    taus = {}
    with open(args.taus_csv) as f:
        for row in csv.DictReader(f):
            taus[int(row["block"])] = (int(row["N_init"]), float(row["tau_max"]))

    Vt = args.vram_target_gb * args.frag_safety
    kernels = [
        # (name, F, gamma bytes/isect, K)
        ("trim-sh3", 59, args.gamma_trim, 1),
        ("trim-sb2", 25, args.gamma_trim, 1),
        ("gsplat-sh3-K8", 59, args.gamma_gsplat, 8),
    ]
    print("N_max = (%.2f - V_os) / (4*F*4 + gamma*tau_max*%.2f/K)   [Vt=%.2fGB]" % (
        Vt, args.tau_growth, Vt))
    header = "block  tau_max | " + " | ".join("%-14s" % k[0] for k in kernels)
    print(header); print("-" * len(header))
    rows = []
    for b in sorted(taus):
        n_init, tau = taus[b]
        row = {"block": b, "N_init": n_init, "tau_max": tau}
        cells = []
        for name, F, gamma, K in kernels:
            vos = args.vos_trim if name.startswith("trim") else args.vos_gsplat
            denom = 4 * F * 4 + gamma * tau * args.tau_growth / K
            n_max = max(0.0, (Vt - vos) * GB / denom)
            row["cap_" + name] = int(n_max)
            cells.append("%-14s" % ("%.2fM" % (n_max / 1e6)))
        rows.append(row)
        print("%5d  %7.1f | %s" % (b, tau, " | ".join(cells)))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG)
    common.add_argument("--data_path", default=DEFAULT_DATA)
    common.add_argument("--ply_dir", default=DEFAULT_PLY_DIR)

    p = sub.add_parser("taus", parents=[common], help="per-block worst-view tau (gsplat fwd, no training)")
    p.add_argument("--blocks", default="0-24")
    p.add_argument("--n_views", type=int, default=8)
    p.add_argument("--out", default="紀錄/block_taus.csv")
    p.set_defaults(func=cmd_taus)

    p = sub.add_parser("probe-trim", parents=[common], help="trim kernel bytes/pt + V_os via N vs N/2 fit")
    p.add_argument("--block", type=int, default=12)
    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--sb", action="store_true", help="probe the SB-color variant (F=25)")
    p.set_defaults(func=cmd_probe_trim)

    p = sub.add_parser("caps", help="emit per-block cap_max table from taus csv + calibrated constants")
    p.add_argument("--taus_csv", default="紀錄/block_taus.csv")
    p.add_argument("--gamma_trim", type=float, required=True, help="trim render bytes per tile-intersection")
    p.add_argument("--vos_trim", type=float, required=True, help="trim V_os in GB (from probe-trim)")
    p.add_argument("--gamma_gsplat", type=float, default=36.0, help="gsplat peak bytes/isect (CUB sort incl.)")
    p.add_argument("--vos_gsplat", type=float, default=0.718)
    p.add_argument("--vram_target_gb", type=float, default=5.4)
    p.add_argument("--frag_safety", type=float, default=0.9)
    p.add_argument("--tau_growth", type=float, default=1.65, help="init->trained tau drift (b12: 73->~120)")
    p.add_argument("--out", default="紀錄/block_caps.csv")
    p.set_defaults(func=cmd_caps)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
