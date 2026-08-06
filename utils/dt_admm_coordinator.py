"""
C5: DT-ADMM Outer Loop Coordinator (Plan A — offline graph)

Drives the alternating ADMM optimization across blocks:

  Per outer iteration t:
    1. Load latest per-block checkpoints
    2. Extract 12-dim boundary features via boundary_graph.load_block_features()
    3. Build boundary graph via BoundaryGraph (C3)
    4. Run GAT z-update for n_steps (C4 BoundaryGATZUpdater)
    5. Dual update: y_v += rho * (x_v - z_v)  for each boundary node
    6. Save per-block admm_state.pt (dual_y, z_consensus, is_boundary)
    7. Launch K_inner primal steps via train_citygs_partitions.py

Usage:
    python utils/dt_admm_coordinator.py \
        --name RTG_s3_mc_aerial_sh2_trim \
        --config configs/dt_admm_gat_mc_aerial.yaml \    # MUST be a DtAdmmDensityController config
        --partitions_pt data/matrix_city/aerial/train/block_all/partitions.pt \
        --blocks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24 \
        --init_mode depth \
        --depth_init_dir data/matrix_city/aerial/train/block_all/depth_init \
        --n_outer 10 \
        --k_inner 500 \
        --gat_n_steps 20 \
        --gat_lr 1e-3 \
        --rho 0.1 \
        --admm_state_dir outputs/RTG_s3_mc_aerial_sh2_trim/admm_state \
        --admm_start_outer 0

The --blocks argument accepts the same format as train_citygs_partitions.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Allow importing project modules from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from internal.utils.boundary_graph import (
    BoundaryGraph,
    BoundaryGraphConfig,
    load_block_features,
    load_partition_aabbs,
)
from internal.renderers.dt_admm_gat_renderer import BoundaryGATZUpdater, GATZConfig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DT-ADMM outer loop coordinator")
    p.add_argument("--name",           required=True, help="Experiment name (outputs/$NAME)")
    p.add_argument("--config",         required=True, help="Block training config YAML")
    p.add_argument("--partitions_pt",  required=True, help="Path to partitions.pt")
    p.add_argument("--blocks",         default="",    help="Comma-separated block IDs (empty = all)")
    p.add_argument("--init_mode",      default="depth", choices=["depth", "coarse"],
                   help="Initialisation mode forwarded to train_citygs_partitions.py")
    p.add_argument("--depth_init_dir", default="",   help="Depth-init PLY directory (init_mode=depth)")
    p.add_argument("--coarse_ckpt",    default="",   help="Coarse checkpoint (init_mode=coarse)")

    p.add_argument("--n_outer",        type=int,   default=10,   help="ADMM outer iterations")
    p.add_argument("--k_inner",        type=int,   default=500,  help="Primal steps per iteration")
    p.add_argument("--gat_n_steps",    type=int,   default=20,   help="GAT gradient steps per z-update")
    p.add_argument("--gat_lr",         type=float, default=1e-3, help="GAT Adam learning rate")
    p.add_argument("--rho",            type=float, default=0.1,  help="ADMM penalty rho")
    p.add_argument("--d_boundary",     type=float, default=0.1,  help="Boundary band width (COLMAP units)")
    p.add_argument("--grid_dim",       default="5,5",            help="Partition grid dimensions rows,cols")
    p.add_argument("--admm_state_dir", required=True,            help="Directory for per-block admm_state.pt")
    p.add_argument("--admm_start_outer", type=int, default=0,    help="Skip ADMM AL injection before this outer iter")
    p.add_argument("--device",         default="cpu",            help="Device for GAT z-update")
    return p.parse_args()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_latest_ckpt(block_dir: Path) -> Optional[Path]:
    """Return the checkpoint with the highest step count in block_dir."""
    ckpts = list(block_dir.glob("checkpoints/epoch=*-step=*.ckpt"))
    if not ckpts:
        return None
    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("-step=")[1])
        except (IndexError, ValueError):
            return 0
    return max(ckpts, key=_step)


def ckpt_step(ckpt: Path) -> int:
    """Parse the global_step from an epoch=X-step=Y.ckpt filename."""
    try:
        return int(ckpt.stem.split("-step=")[1])
    except (IndexError, ValueError):
        return 0


def load_admm_state(
    ckpt: Path,
    state_path: Path,
    n: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load per-block ADMM state.

    dual_y: read from the CHECKPOINT's density_controller.dual_y buffer.
      The checkpoint's buffer is topology-consistent (clone/split/prune are propagated
      by DtAdmmDensityControllerModule during the K_inner primal steps). Reading from
      admm_state.pt would give the PRE-primal shape (before densification), causing a
      shape mismatch that resets y to zeros every outer iteration.

    z_consensus, is_boundary: read from admm_state.pt (set by coordinator's z-update).
      These are rebuilt from scratch each outer iteration via the boundary graph + GAT,
      so the checkpoint value is not used.
    """
    # Load y from checkpoint (topology-consistent, same shape as current Gaussian count)
    y = torch.zeros(n, 12, device=device, dtype=torch.float32)
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False).get("state_dict", {})
        y_ckpt = sd.get("density_controller.dual_y", None)
        if y_ckpt is not None and y_ckpt.shape[0] == n:
            y = y_ckpt.to(device)
        elif y_ckpt is not None:
            print(f"  [ADMM] y shape mismatch ({y_ckpt.shape[0]} vs {n}) — initializing to 0")
    except Exception as e:
        print(f"  [ADMM] could not load dual_y from checkpoint: {e}")

    # Load z and is_boundary from coordinator-saved state
    z  = torch.zeros(n, 12, device=device, dtype=torch.float32)
    ib = torch.zeros(n,     device=device, dtype=torch.bool)
    if state_path.exists():
        try:
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            z_saved  = st.get("z_consensus", None)
            ib_saved = st.get("is_boundary", None)
            if z_saved is not None and z_saved.shape[0] == n:
                z = z_saved.to(device)
            if ib_saved is not None and ib_saved.shape[0] == n:
                ib = ib_saved.to(device)
        except Exception as e:
            print(f"  [ADMM] could not load z/ib from admm_state: {e}")

    return y, z, ib


def save_admm_state(
    state_path: Path,
    dual_y: torch.Tensor,
    z_consensus: torch.Tensor,
    is_boundary: torch.Tensor,
    feat_std: Optional[torch.Tensor] = None,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dual_y": dual_y.cpu(),
        "z_consensus": z_consensus.cpu(),
        "is_boundary": is_boundary.cpu(),
    }
    # (12,) diagonal preconditioner scale — read by DtAdmmDensityController's AL penalty
    # so the primal weighting matches the dual/z-update. Defaults to ones (unweighted).
    payload["feat_std"] = (torch.ones(12) if feat_std is None else feat_std.cpu())
    torch.save(payload, state_path)


# ── z-update (C4) ────────────────────────────────────────────────────────────

def compute_feat_std(node_feats: torch.Tensor, cond_cap_frac: float = 0.1) -> torch.Tensor:
    """
    Per-dimension scale for the diagonal ADMM preconditioner (weighted/normalized-space).

    Returns a (12,) per-dim scale s, used as the diagonal ADMM preconditioner (w=1/s^2),
    the GAT input normalizer, and the y_norm scale — all three share this same s.

    Two safeguards:
    1. Condition cap: clamp std to floor = cond_cap_frac * max(std), so the inter-property
       weight ratio (1/std^2) is capped at 1/cond_cap_frac^2 (~100 at 0.1). A FROZEN
       property (e.g. opacity pinned at 0.99 by B1 freeze_opacity) has ~zero variance ->
       naive 1/std^2 would explode and let that one dim dominate; the floor prevents it.
    2. Geomean normalization: divide by geomean(std) so geomean(1/s^2) = 1. The
       preconditioner then only REDISTRIBUTES consensus strength across properties
       (volume-preserving) and leaves rho as the single overall-strength knob. Without
       this, w=1/std^2 << 1 would silently weaken the whole consensus by ~std^2 and
       confound the rho sweep.
    """
    std = node_feats.std(dim=0)                       # (12,)
    floor = max(1e-6, cond_cap_frac * float(std.max()))
    std = std.clamp(min=floor)
    return std / std.log().mean().exp()               # geomean(std) == 1


def run_gat_z_update(
    graph: Dict,
    block_dual_y: Dict[int, torch.Tensor],   # {bid: (N_bnd_i, 12)}
    rho: float,
    n_steps: int,
    lr: float,
    device: str,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Run BoundaryGATZUpdater for n_steps.

    Returns (z_per_block, feat_std):
      z_per_block : {bid: z_bnd (N_bnd_i, 12)} in raw (denormalized) feature space.
      feat_std    : (12,) per-dim preconditioner scale (cond-number capped). The primal
                    AL penalty and dual update must use this SAME std for weighted ADMM:
                      penalty w_d = 1/std_d^2 ;  dual y += rho * w * (x - z).
                    GAT y_norm uses y_raw * std (derived: denormalized weighted-ADMM z*
                    = x + std^2 * y/rho requires y_norm = y_raw * std).
    """
    node_feats   = graph["node_features"].to(device)      # (N_bnd, 12)
    edge_index   = graph["edge_index"].to(device)         # (2, E)
    node_bids    = graph["node_block_id"].to(device)      # (N_bnd,)
    block_offsets = graph["block_offsets"]                # {bid: start_idx}

    # Assemble full dual_y vector ordered by node ordering in the graph
    y_raw = torch.zeros_like(node_feats)
    for bid, y_bnd in block_dual_y.items():
        if bid not in block_offsets:
            continue
        start = block_offsets[bid]
        mask  = node_bids == bid
        y_raw[mask] = y_bnd.to(device)

    cfg     = GATZConfig(in_dim=12, rho=rho, lr=lr, n_steps=n_steps)
    updater = BoundaryGATZUpdater(cfg).to(device)
    opt     = torch.optim.Adam(updater.parameters(), lr=lr)

    # Diagonal preconditioner: whiten with cond-capped std so all 12 properties are
    # conditioned comparably (fixes single-scalar-rho ill-conditioning, Boyd diag-P).
    mean      = node_feats.mean(dim=0)
    feat_std  = compute_feat_std(node_feats)              # (12,) cond-capped
    h_norm    = (node_feats - mean) / feat_std
    y_norm    = y_raw * feat_std                          # weighted-ADMM convention (see docstring)
    stats     = (mean, feat_std)

    for step in range(n_steps):
        opt.zero_grad()
        z_v, loss = updater.forward_loss(h_norm, edge_index, y_norm)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            gap = updater.residual_gap(h_norm, edge_index, y_norm)
            print(f"  [GAT] step {step:3d}  loss={loss.item():.4f}  gap={gap:.4f}")

    with torch.no_grad():
        z_norm = updater.forward(h_norm, edge_index)
        z_raw  = updater.denormalize(z_norm.detach(), stats)  # (N_bnd, 12)

    # Split back per block
    z_per_block: Dict[int, torch.Tensor] = {}
    for bid in block_offsets:
        mask = node_bids == bid
        z_per_block[bid] = z_raw[mask].cpu()

    return z_per_block, feat_std.detach().cpu()


# ── Dual update ───────────────────────────────────────────────────────────────

def dual_update(
    x_bnd: torch.Tensor,
    z_bnd: torch.Tensor,
    y_bnd: torch.Tensor,
    rho: float,
    feat_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weighted ADMM dual step:  y_v += rho * w * (x_v - z_v),  w = 1/std^2  (diagonal P).

    feat_std=None falls back to the original unweighted update (w=1), used for warm-up
    outers before the first z-update.
    """
    if feat_std is None:
        return y_bnd + rho * (x_bnd - z_bnd)
    w = 1.0 / (feat_std.to(x_bnd.device) ** 2)        # (12,) broadcasts over (M, 12)
    return y_bnd + rho * w * (x_bnd - z_bnd)


# ── Primal step launcher ──────────────────────────────────────────────────────

def build_block_train_cmd(
    args: argparse.Namespace,
    bid: int,
    next_max_steps: int,
    admm_state_path: str,
) -> List[str]:
    """
    Build the main.py fit command for one block's K_inner primal steps.

    Passes --ckpt_path last so the CLI auto-selects the latest checkpoint in
    outputs/{name}/blocks/block_{bid}/checkpoints/. Per-block overrides:
      --max_steps              last_step + k_inner  (exact K_inner more steps)
      --model.density...       per-block admm_state_path
    """
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"), "fit",
        "--config", args.config,
        f"-n={args.name}",
        "--data.parser.block_id", str(bid),
        "--max_steps", str(next_max_steps),   # shorthand for trainer.max_steps
        "--ckpt_path", "last",                # CLI auto-finds latest ckpt in output dir
        "--model.density.init_args.admm_state_path", admm_state_path,
        "--model.density.init_args.rho", str(args.rho),  # keep primal penalty rho == z/dual rho
        "--logger", "tensorboard",
    ]


def launch_primal_steps(
    args: argparse.Namespace,
    active_blocks: List[int],
    block_ckpts: Dict[int, Path],
    block_n:     Dict[int, int],
    admm_dir:    Path,
    outer_iter:  int,
) -> None:
    """Launch K_inner primal steps per block sequentially (single-GPU)."""
    for bid in active_blocks:
        ckpt = block_ckpts.get(bid)
        if ckpt is None:
            print(f"  [ADMM] block {bid}: no checkpoint — skipping primal")
            continue

        last_step      = ckpt_step(ckpt)
        next_max_steps = last_step + args.k_inner
        state_path     = str(admm_dir / f"block_{bid}_state.pt")

        cmd = build_block_train_cmd(args, bid, next_max_steps, state_path)
        print(f"\n[ADMM outer {outer_iter}] block {bid}: step {last_step}→{next_max_steps}")
        print(f"  {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if ret.returncode != 0:
            # A non-zero primal means NO consensus signal was applied this outer — the
            # whole run is meaningless if it persists. Make it loud and abort rather than
            # quietly grinding through 30 outers that all no-op.
            print(f"\n  {'!'*60}")
            print(f"  [ADMM] FATAL: block {bid} primal exited code {ret.returncode}")
            print(f"  [ADMM] The main.py fit subprocess failed. Common causes:")
            print(f"  [ADMM]   - exit 2 + 'Option ... is not accepted' → wrong --config "
                  f"(needs DtAdmmDensityController; pre-flight should have caught this)")
            print(f"  [ADMM]   - CUDA OOM / missing ckpt / bad block_id → see the dump above")
            print(f"  [ADMM] Aborting: a no-op primal makes the rest of the ADMM run invalid.")
            print(f"  {'!'*60}\n")
            sys.exit(ret.returncode)


# ── Config pre-flight ─────────────────────────────────────────────────────────

def validate_config(config_path: str) -> None:
    """Fail fast (before any training) if --config can't accept ADMM injection.

    The coordinator passes --model.density.init_args.admm_state_path / .rho to each
    primal step. Only DtAdmmDensityController accepts those. Passing a plain RTG /
    vanilla config makes main.py die with a 100-line jsonargparse usage dump per
    block ("Option 'admm_state_path' is not accepted"). Catch it here with one line.
    """
    p = Path(config_path)
    if not p.is_file():
        sys.exit(f"[ADMM] FATAL: --config not found: {config_path}")
    text = p.read_text()
    if "DtAdmmDensityController" not in text:
        sys.exit(
            "[ADMM] FATAL: --config density controller does not accept admm_state_path.\n"
            f"        config: {config_path}\n"
            "        need:   model.density.class_path = "
            "internal.density_controllers.dt_admm_density_controller.DtAdmmDensityController\n"
            "        fix:    use configs/dt_admm_gat_mc_aerial.yaml (depth-init) — NOT the plain "
            "RTG_* base config.\n"
            "        (The RTG/vanilla density controllers reject --model.density.init_args.admm_state_path,\n"
            "         which is why main.py prints a long usage dump and exits code 2 per block.)"
        )


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_config(args.config)

    rows, cols = map(int, args.grid_dim.split(","))
    outputs_dir = REPO_ROOT / "outputs" / args.name
    admm_dir    = Path(args.admm_state_dir)
    admm_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Parse block list
    if args.blocks:
        block_ids = [int(b) for b in args.blocks.replace(" ", "").split(",")]
    else:
        block_ids = list(range(rows * cols))

    # Load partition AABBs
    aabbs = load_partition_aabbs(args.partitions_pt)

    for outer in range(args.n_outer):
        print(f"\n{'='*60}")
        print(f"[ADMM] outer iteration {outer + 1}/{args.n_outer}")
        print(f"{'='*60}")

        # ── 1. Load per-block features from latest checkpoints ──────────────
        block_features: Dict[int, torch.Tensor] = {}
        block_ckpts:    Dict[int, Path]          = {}
        block_n:        Dict[int, int]           = {}
        missing = []

        for bid in block_ids:
            block_dir = outputs_dir / "blocks" / f"block_{bid}"
            ckpt = find_latest_ckpt(block_dir)
            if ckpt is None:
                print(f"  [ADMM] block {bid}: no checkpoint found, skipping")
                missing.append(bid)
                continue
            try:
                feats = load_block_features(str(ckpt), device=device)
                block_features[bid] = feats
                block_ckpts[bid]    = ckpt
                block_n[bid]        = feats.shape[0]
                print(f"  [ADMM] block {bid}: {feats.shape[0]:,} Gaussians  step={ckpt_step(ckpt)}  ({ckpt.name})")
            except Exception as e:
                print(f"  [ADMM] block {bid}: feature load failed ({e}), skipping")
                missing.append(bid)

        active_blocks = [b for b in block_ids if b not in missing]
        if len(active_blocks) < 2:
            print("[ADMM] fewer than 2 active blocks — skipping z-update, running primal")
            launch_primal_steps(args, active_blocks, block_ckpts, block_n, admm_dir, outer)
            continue

        # ── 2. Build boundary graph ──────────────────────────────────────────
        aabb_dilation = 3 * args.d_boundary if outer > 0 else 0.0
        bg_cfg = BoundaryGraphConfig(
            grid_dim=(rows, cols),
            d_boundary=args.d_boundary,
            aabb_dilation=aabb_dilation,
        )
        bg = BoundaryGraph(bg_cfg, block_aabbs=aabbs)
        graph = bg.build({bid: block_features[bid] for bid in active_blocks})

        n_bnd_nodes = graph["node_features"].shape[0]
        n_edges     = graph["edge_index"].shape[1]
        print(f"  [ADMM] boundary graph: {n_bnd_nodes:,} nodes, {n_edges:,} edges")

        if n_bnd_nodes == 0 or n_edges == 0:
            print("  [ADMM] empty graph — skipping z-update")
            launch_primal_steps(args, active_blocks, block_ckpts, block_n, admm_dir, outer)
            continue

        # ── 3. Load / initialise per-block ADMM state ────────────────────────
        block_dual_y:    Dict[int, torch.Tensor] = {}
        block_z_cons:    Dict[int, torch.Tensor] = {}
        block_is_bnd:    Dict[int, torch.Tensor] = {}

        for bid in active_blocks:
            state_path = admm_dir / f"block_{bid}_state.pt"
            n = block_n[bid]
            y, z, ib = load_admm_state(block_ckpts[bid], state_path, n, device)

            # Slice to boundary nodes using the graph's boundary mask
            bnd_mask_np = graph["boundary_masks"].get(bid)
            if bnd_mask_np is None or not bnd_mask_np.any():
                continue
            bnd_mask = torch.from_numpy(bnd_mask_np).to(device)

            if y.shape[0] != n:
                y  = torch.zeros(n, 12, device=device, dtype=torch.float32)
                z  = torch.zeros(n, 12, device=device, dtype=torch.float32)
                ib = torch.zeros(n,     device=device, dtype=torch.bool)

            block_dual_y[bid] = y[bnd_mask]
            block_z_cons[bid] = z[bnd_mask]
            block_is_bnd[bid] = bnd_mask

        # ── 4. GAT z-update ──────────────────────────────────────────────────
        # feat_std (12,) = diagonal preconditioner scale; None on warm-up outers
        # (no z-update yet) → dual/penalty fall back to unweighted (w=1).
        feat_std = None
        if outer >= args.admm_start_outer:
            print(f"  [ADMM] running GAT z-update ({args.gat_n_steps} steps) …")
            z_per_block, feat_std = run_gat_z_update(
                graph,
                block_dual_y={bid: block_dual_y.get(bid, torch.zeros(0, 12)) for bid in active_blocks},
                rho=args.rho,
                n_steps=args.gat_n_steps,
                lr=args.gat_lr,
                device=device,
            )
            print(f"  [ADMM] preconditioner feat_std (12-dim): "
                  f"{[round(float(s), 4) for s in feat_std]}")
        else:
            print(f"  [ADMM] outer {outer} < admm_start_outer {args.admm_start_outer}, skipping z-update")
            z_per_block = {bid: block_z_cons.get(bid, torch.zeros(0, 12)) for bid in active_blocks}

        # ── 5. Dual update + save admm_state.pt ─────────────────────────────
        for bid in active_blocks:
            state_path = admm_dir / f"block_{bid}_state.pt"
            n = block_n[bid]
            y_full, z_full, ib_full = load_admm_state(block_ckpts[bid], state_path, n, device)

            bnd_mask_np = graph["boundary_masks"].get(bid)
            if bnd_mask_np is None:
                continue
            bnd_mask = torch.from_numpy(bnd_mask_np).to(device)
            if not bnd_mask.any():
                save_admm_state(state_path, y_full, z_full, bnd_mask, feat_std=feat_std)
                continue

            y_bnd = block_dual_y.get(bid, torch.zeros(bnd_mask.sum(), 12, device=device))
            z_bnd = z_per_block.get(bid, torch.zeros(bnd_mask.sum(), 12, device=device)).to(device)
            x_bnd = block_features[bid][bnd_mask, :12].to(device)

            # Dual update (weighted ADMM; feat_std=None on warm-up outers → unweighted)
            y_bnd_new = dual_update(x_bnd, z_bnd, y_bnd, rho=args.rho, feat_std=feat_std)

            # Write back into full-N tensors
            y_full = y_full.to(device)
            z_full = z_full.to(device)
            if y_full.shape[0] != n:
                y_full = torch.zeros(n, 12, device=device, dtype=torch.float32)
                z_full = torch.zeros(n, 12, device=device, dtype=torch.float32)

            y_full[bnd_mask] = y_bnd_new.detach()
            z_full[bnd_mask] = z_bnd.detach()
            ib_full = bnd_mask.clone()

            save_admm_state(state_path, y_full, z_full, ib_full, feat_std=feat_std)
            # Per-property residual breakdown — diagnoses whether a single scalar rho
            # is actually moving opacity/color consensus or only position (xyz).
            # 12-dim layout: [0:3]means [3:6]normal [6:8]scale [8:9]opacity [9:12]sh_dc
            res = (x_bnd - z_bnd)
            _groups = {"xyz": (0, 3), "nrm": (3, 6), "scl": (6, 8), "opa": (8, 9), "sh": (9, 12)}
            _per = "  ".join(
                f"{k}={res[:, a:b].pow(2).sum(-1).sqrt().mean():.4f}" for k, (a, b) in _groups.items()
            )
            print(
                f"  [ADMM] block {bid}: saved state "
                f"||y||={y_bnd_new.norm(dim=-1).mean():.4f}  "
                f"||x-z||={(res ** 2).sum(dim=-1).sqrt().mean():.4f}\n"
                f"            per-prop ||x-z||:  {_per}"
            )

        # ── 6. Launch primal (K_inner steps) ────────────────────────────────
        launch_primal_steps(args, active_blocks, block_ckpts, block_n, admm_dir, outer)

    print("\n[ADMM] coordinator done.")


if __name__ == "__main__":
    main()
