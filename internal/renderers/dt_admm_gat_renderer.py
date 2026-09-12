"""
C4: GAT z-update for DT-ADMM-GAT

Implements the ADMM consensus step:
  z_v = x_v + GATLayer(h_v, edge_index)   [residual form]
  L_z = (rho/2)||x_v - z_v||^2 - y_v^T * z_v

where:
  x_v = boundary Gaussian features (current primal, 12-dim 2DGS)
  y_v = dual variable (ADMM multiplier, same dim, initialized to 0)
  z_v = consensus variable (cross-block agreement target)

Confirmed math (Gemini Round 2):
  z*_v = x_v + y_v/rho  (closed-form optimal, dynamic supervision target)
  dL_z/dz_v = rho*(z_v - x_v) - y_v  (gradient driving GAT optimization)

Design decisions:
  - 1-layer GAT: sufficient for local boundary consensus (Plan A offline graph)
  - Spectral norm on W: Lipschitz ≤ 1, z_v stays in convex hull of neighbor features
  - Residual output: delta=0 at init → z_v=x_v at first iteration (ADMM warm start)
  - Per-dim z-score normalization: handles scale heterogeneity across 12 feature dims
    (positions ~[-100,100], normals ~[-1,1], scales ~[0,0.1], opacity [0,1], SH_dc ~[-0.5,0.5])
  - hidden_dim=64: small for CPU/host-offload compatibility, adequate for 12-dim features
  - Zero-init proj weight: ensures delta=0 at init without needing explicit warm-up

Plan A integration (C5):
  Per outer ADMM iteration:
    1. Train K_inner=500 steps (per-block primal, z and y detached)
    2. Save checkpoints → load features → build BoundaryGraph (C3)
    3. Run BoundaryGATZUpdater for n_steps gradient steps → get z_v
    4. Dual update: y_v += rho * (x_v - z_v)
    5. Next primal step uses updated z_v and y_v
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GATZConfig:
    in_dim: int = 12
    """Input/output feature dimension (fixed: 12-dim 2DGS node features)."""
    hidden_dim: int = 64
    """GAT hidden dimension for W projection."""
    negative_slope: float = 0.2
    """LeakyReLU slope for attention score computation."""
    rho: float = 0.1
    """ADMM penalty parameter. Controls consensus strength vs. rendering fidelity."""
    beta_cross: float = 0.1
    """Cross-block consistency weight. Adds (beta/2)||z_v - x_u||^2 for each cross-block
    edge (v,u), pulling z_v toward x_u even when y=0. Without this term, the z-update
    is stuck at z=x when y=0 (trivial fixed point). Set equal to rho for equal weighting
    between self-consistency and cross-block consensus (z* = (x_v + x_u)/2)."""
    lr: float = 1e-3
    """Learning rate for GAT parameters (Adam)."""
    n_steps: int = 20
    """Gradient steps on GAT params per outer ADMM iteration."""


# ── GAT Layer ─────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single GAT layer with spectral-normalized W.

    alpha_vu = Softmax_u(LeakyReLU(a^T [W*h_v || W*h_u]))  (v=dst, u=src)
    agg_v    = sum_u  alpha_vu * W * h_u
    delta_v  = proj(agg_v)    ← projected back to in_dim

    W is spectral-normalized to keep z_v in the convex hull of neighbor features,
    satisfying the ADMM consensus stability requirement.
    """

    def __init__(self, in_dim: int, hidden_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.W    = nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim, bias=False))
        self.a    = nn.Parameter(torch.empty(2 * hidden_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.proj  = nn.Linear(hidden_dim, in_dim, bias=False)
        nn.init.zeros_(self.proj.weight)  # delta=0 at init → z_v=x_v on first pass

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            h:          (N, in_dim) node features (normalized)
            edge_index: (2, E) directed edges, row-0=src (u), row-1=dst (v)
        Returns:
            delta: (N, in_dim) residual update
        """
        if edge_index.shape[1] == 0:
            return torch.zeros_like(h)

        src, dst = edge_index[0], edge_index[1]
        N  = h.shape[0]
        Wh = self.W(h)  # (N, hidden_dim)

        # Attention: e_vu = a^T [W*h_v || W*h_u]  (v=dst receives from u=src)
        e = self.leaky(
            (torch.cat([Wh[dst], Wh[src]], dim=-1) * self.a).sum(dim=-1)
        )  # (E,)

        # Softmax over in-edges per dst node.
        # Subtract global max for numerical stability (out-of-place, gradient-safe).
        # Per-segment max would be tighter but requires inplace scatter_reduce which
        # conflicts with autograd when chained with clamp.
        e_shifted = e - e.max().detach()
        exp_e   = e_shifted.exp()
        exp_sum = torch.zeros(N, device=h.device, dtype=h.dtype).scatter_add(
            0, dst, exp_e
        )  # (N,) — out-of-place, gradient-safe
        alpha = exp_e / (exp_sum[dst] + 1e-16)  # (E,)

        # Aggregate: agg_v = sum_u alpha_vu * W*h_u  (out-of-place scatter_add)
        agg = torch.zeros(N, Wh.shape[1], device=h.device, dtype=h.dtype).scatter_add(
            0,
            dst.unsqueeze(1).expand(-1, Wh.shape[1]),
            alpha.unsqueeze(1) * Wh[src],
        )  # (N, hidden_dim)

        return self.proj(agg)  # (N, in_dim)


# ── Z-Updater ─────────────────────────────────────────────────────────────────

class BoundaryGATZUpdater(nn.Module):
    """
    GAT-based z-update for DT-ADMM boundary consensus.

    Typical usage per outer ADMM iteration:

        updater  = BoundaryGATZUpdater(GATZConfig())
        opt      = torch.optim.Adam(updater.parameters(), lr=cfg.lr)

        # features from BoundaryGraph.build()['node_features']  (N, 12)
        h_norm, stats = updater.normalize(graph['node_features'].to(device))
        y_norm = y_raw / stats[1]   # scale y by same std (see normalize docstring)

        for _ in range(cfg.n_steps):
            opt.zero_grad()
            z_v, loss = updater.forward_loss(h_norm, graph['edge_index'].to(device), y_norm)
            loss.backward()
            opt.step()

        z_raw = updater.denormalize(z_v.detach(), stats)  # consensus target in raw space
    """

    def __init__(self, config: GATZConfig):
        super().__init__()
        self.config = config
        self.gat    = GATLayer(config.in_dim, config.hidden_dim, config.negative_slope)

    # ── Feature normalization ─────────────────────────────────────────────────

    @staticmethod
    def normalize(h: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Z-score normalize per feature dimension.

        y_v must be scaled by the same std before passing to forward_loss:
            y_norm = y_raw / std   (mean-shift not applied to y since y is a delta variable)

        Returns (h_norm, (mean, std)).
        """
        mean = h.mean(dim=0)
        std  = h.std(dim=0).clamp(min=1e-6)
        return (h - mean) / std, (mean, std)

    @staticmethod
    def denormalize(h_norm: Tensor, stats: Tuple[Tensor, Tensor]) -> Tensor:
        mean, std = stats
        return h_norm * std + mean

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, h_norm: Tensor, edge_index: Tensor) -> Tensor:
        """z_v = x_v + delta (all in normalized space)."""
        return h_norm + self.gat(h_norm, edge_index)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        z_v: Tensor,
        x_v: Tensor,
        y_v: Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        L_z = (rho/2)||x_v - z_v||^2 - y_v^T*z_v + (beta/2)||z_dst - x_src||^2

        Terms:
          primal : (rho/2)||x_v - z_v||^2    — consistency of z with current primal x
          dual   : -y_v^T * z_v              — ADMM dual correction
          cross  : (beta/2)||z_v[dst]-x_v[src]||^2  — cross-block consistency

        The cross term is essential for breaking the trivial fixed point when y=0.
        Without it, z*=x and the dual never accumulates.
        With it, z* = (rho*x_v + beta*mean(x_u)) / (rho + beta*|N_v|) — a weighted
        average pulled toward cross-block neighbors even at y=0.

        All tensors in normalized feature space.
        """
        rho         = self.config.rho
        primal_term = (rho / 2) * ((x_v - z_v) ** 2).sum(dim=-1).mean()
        dual_term   = (y_v * z_v).sum(dim=-1).mean()
        loss        = primal_term - dual_term

        beta = self.config.beta_cross
        if edge_index is not None and beta > 0 and edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            # z_dst should be close to x_src (cross-block feature consistency)
            cross_term = (beta / 2) * ((z_v[dst] - x_v[src]) ** 2).sum(dim=-1).mean()
            loss = loss + cross_term

        return loss

    def forward_loss(
        self,
        h_norm: Tensor,
        edge_index: Tensor,
        y_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass + L_z computation (including cross-block consistency).

        x_v = h_norm (current primal = boundary node features).
        y_v must be in normalized space (divide raw y by feature std).

        Returns (z_v, loss). z_v is in normalized space — denormalize before use.
        """
        z_v  = self.forward(h_norm, edge_index)
        loss = self.compute_loss(z_v, x_v=h_norm, y_v=y_v, edge_index=edge_index)
        return z_v, loss

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def z_optimal(self, x_v: Tensor, y_v: Tensor) -> Tensor:
        """z*_v = x_v + y_v/rho — closed-form ADMM optimum (normalized space)."""
        return x_v + y_v / self.config.rho

    @torch.no_grad()
    def residual_gap(
        self,
        h_norm: Tensor,
        edge_index: Tensor,
        y_v: Tensor,
    ) -> float:
        """
        ||z_v - z*_v||^2 / N — how far GAT output is from ADMM optimum.
        Useful for monitoring convergence of the n_steps inner optimization.
        """
        z_v    = self.forward(h_norm, edge_index)
        z_star = self.z_optimal(h_norm, y_v)
        return float(((z_v - z_star) ** 2).sum(dim=-1).mean())
