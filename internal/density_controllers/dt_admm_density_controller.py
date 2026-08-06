"""
C5: DT-ADMM-GAT Density Controller

Extends RTGStableDensityControllerModule with ADMM dual-variable (y_v) and consensus
target (z_v) management. The Augmented Lagrangian (AL) penalty injected in before_backward
drives boundary Gaussians toward inter-block consensus:

  L_admm = (rho/2) * ||h(x_bnd) - (z_bnd - y_bnd/rho)||^2

where h(x_bnd) is the 12-dim feature of each boundary Gaussian (differentiable w.r.t.
the Gaussian parameters), z_bnd is the detached consensus target from the last z-update
(C4), and y_bnd is the detached dual variable.

Dynamic topology inheritance (CLAUDE.md):
  Clone  → child inherits parent dual_y and z_consensus
  Split  → children get dual_y=0, z_consensus=0  (re-proven from boundary zero)
  Prune  → dual_y, z_consensus, is_boundary entries removed

ADMM outer loop is driven by utils/dt_admm_coordinator.py (Plan A offline graph):
  Per K_inner primal steps: coordinator loads checkpoints → builds BoundaryGraph →
  runs GAT z-update → computes dual update → saves admm_state.pt → relaunches training
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from .rtg_stable_density_controller import RTGStableDensityController, RTGStableDensityControllerModule
from .vanilla_density_controller import VanillaGaussianModel
from internal.utils.general_utils import build_rotation


@dataclass
class DtAdmmDensityController(RTGStableDensityController):
    rho: float = 0.1
    """ADMM penalty parameter rho (same value used in C4 BoundaryGATZUpdater)."""
    admm_start_iter: int = 5000
    """Global step before which the AL penalty is not injected (let primal burn in first)."""
    admm_state_path: str = ""
    """Path to admm_state.pt written by dt_admm_coordinator.py for this block.
    When non-empty and the file exists, dual_y / z_consensus / is_boundary are loaded
    from this file (injected into the checkpoint state_dict before PL applies it)."""
    k_inner: int = 500
    """Primal steps per ADMM outer iteration (for coordinator reference)."""
    gat_n_steps: int = 20
    """GAT gradient steps per outer iteration (for coordinator reference)."""
    gat_lr: float = 1e-3
    """GAT learning rate (for coordinator reference)."""

    def instantiate(self, *args, **kwargs) -> "DtAdmmDensityControllerModule":
        return DtAdmmDensityControllerModule(self)


class DtAdmmDensityControllerModule(RTGStableDensityControllerModule):

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self, stage: str, pl_module) -> None:
        super().setup(stage, pl_module)
        if stage == "fit" and not hasattr(self, "dual_y"):
            n = pl_module.gaussian_model.n_gaussians
            device = pl_module.device
            self._register_admm_buffers(n, device)

    def _register_admm_buffers(self, n: int, device) -> None:
        self.register_buffer(
            "dual_y", torch.zeros(n, 12, device=device, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "z_consensus", torch.zeros(n, 12, device=device, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "is_boundary", torch.zeros(n, device=device, dtype=torch.bool), persistent=True
        )
        # feat_std is per-dimension (12,), topology-invariant → register ONCE and never
        # reset on densification (else each clone/split would wipe the loaded preconditioner).
        if not hasattr(self, "feat_std"):
            self.register_buffer(
                "feat_std", torch.ones(12, device=device, dtype=torch.float32), persistent=True
            )

    def on_load_checkpoint(self, module, checkpoint) -> None:
        super().on_load_checkpoint(module, checkpoint)
        n = checkpoint["state_dict"]["density_controller.max_radii2D"].shape[0]
        device = module.device

        # Initialise checkpoint entries so PL can fill them via load_state_dict
        for key, shape, dtype in [
            ("density_controller.dual_y",      (n, 12), torch.float32),
            ("density_controller.z_consensus", (n, 12), torch.float32),
            ("density_controller.is_boundary", (n,),    torch.bool),
        ]:
            if key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = torch.zeros(shape, dtype=dtype)
        # feat_std (12,) defaults to ones (unweighted) when the checkpoint lacks it
        if "density_controller.feat_std" not in checkpoint["state_dict"]:
            checkpoint["state_dict"]["density_controller.feat_std"] = torch.ones(12, dtype=torch.float32)

        # Override with coordinator state if provided (injected before PL applies state_dict)
        state_path = self.config.admm_state_path
        if state_path and os.path.exists(state_path):
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            checkpoint["state_dict"]["density_controller.dual_y"]      = st["dual_y"].cpu()
            checkpoint["state_dict"]["density_controller.z_consensus"] = st["z_consensus"].cpu()
            checkpoint["state_dict"]["density_controller.is_boundary"] = st["is_boundary"].cpu()
            if "feat_std" in st:  # weighted-ADMM preconditioner from the coordinator
                checkpoint["state_dict"]["density_controller.feat_std"] = st["feat_std"].cpu()
            print(f"[DtAdmm] loaded ADMM state override from {state_path}")

        # Register buffers so PL can copy checkpoint values into them
        self._register_admm_buffers(n, device)

    def after_density_changed(self, gaussian_model, optimizers, pl_module) -> None:
        super().after_density_changed(gaussian_model, optimizers, pl_module)
        n = gaussian_model.n_gaussians
        device = pl_module.device
        if not hasattr(self, "dual_y") or self.dual_y.shape[0] != n:
            self._register_admm_buffers(n, device)

    # ── AL penalty injection ─────────────────────────────────────────────────

    def before_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module) -> None:
        super().before_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)

        if global_step < self.config.admm_start_iter:
            return
        if not self.is_boundary.any():
            return

        al_loss = self._compute_al_penalty(gaussian_model)
        if al_loss is not None and al_loss.requires_grad:
            pl_module.manual_backward(al_loss)

    def _extract_boundary_features(self, gaussian_model: VanillaGaussianModel) -> Tensor:
        """
        Extract 12-dim h(x_bnd) from the live model for all boundary Gaussians.

        Feature layout (same as load_block_features in boundary_graph.py):
          [0:3]  means     — world-space position (raw)
          [3:6]  normal    — surfel normal = R[:,2] from quaternion (differentiable)
          [6:8]  scale_2d  — activated 2D scales (exp of log-scale)
          [8:9]  opacity   — activated opacity (sigmoid)
          [9:12] SH_dc     — DC spherical harmonic (shape (N,3))

        All operations are differentiable so AL penalty gradients flow back to
        means, rotations, scales, opacities, and shs_dc.
        """
        bnd = self.is_boundary  # (N,) bool

        means   = gaussian_model.get_means()[bnd]      # (M, 3)
        quats   = gaussian_model.get_rotations()[bnd]  # (M, 4) normalized quaternions
        scales  = gaussian_model.get_scales()[bnd]     # (M, 2) activated
        opacs   = gaussian_model.get_opacities()[bnd]  # (M, 1)
        sh_dc   = gaussian_model.get_shs_dc()[bnd, 0, :]  # (M, 3)

        # Normal = third column of rotation matrix (differentiable through quats)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        nx = 2 * (x * z + w * y)
        ny = 2 * (y * z - w * x)
        nz = 1 - 2 * (x * x + y * y)
        normals = torch.stack([nx, ny, nz], dim=1)  # (M, 3)

        return torch.cat([means, normals, scales, opacs, sh_dc], dim=1)  # (M, 12)

    def _compute_al_penalty(self, gaussian_model: VanillaGaussianModel) -> Optional[Tensor]:
        """
        Weighted ADMM AL penalty (diagonal preconditioner P = diag(w), w = 1/std^2):

          L_admm = (rho/2) * sum_d w_d * (h(x)_d - target_d)^2     mean over boundary nodes
          target = z_bnd - y_bnd / (rho * w)   (= z - std^2 * y / rho)

        std (feat_std, 12-dim) is the cond-capped scale set by the coordinator's z-update,
        equalising the consensus pull across the 6 heterogeneous 2DGS properties so a single
        scalar rho is no longer ill-conditioned (Boyd diagonal-P). The dual update in the
        coordinator (y += rho*w*(x-z)) and the GAT y_norm use the SAME std → consistent.
        feat_std=ones reduces this exactly to the original unweighted penalty (legacy).
        """
        bnd = self.is_boundary
        if not bnd.any():
            return None

        h_x = self._extract_boundary_features(gaussian_model)  # (M, 12), grad-enabled

        z_bnd = self.z_consensus[bnd].detach()  # (M, 12)
        y_bnd = self.dual_y[bnd].detach()       # (M, 12)

        std = self.feat_std.to(h_x.device).clamp(min=1e-6)  # (12,)
        w   = 1.0 / (std ** 2)                              # (12,)
        target = z_bnd - y_bnd / (self.config.rho * w)      # = z - std^2 * y / rho

        residual = h_x - target                             # (M, 12)
        return (self.config.rho / 2) * (w * residual ** 2).sum(dim=-1).mean()

    # ── Topology: clone inherits ADMM state ──────────────────────────────────

    def _densify_and_clone(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List):
        """Clone children inherit parent dual_y, z_consensus, is_boundary."""
        # Compute clone mask (same criteria as RTGStable/CityGSV2)
        grad_threshold = self.config.densify_grad_threshold
        percent_dense  = self.config.percent_dense
        scene_extent   = self.cameras_extent

        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = selected & (
            gaussian_model.get_scales().max(dim=1).values <= percent_dense * scene_extent
        )
        axis_ratio = (
            gaussian_model.get_scales().min(dim=1).values /
            gaussian_model.get_scales().max(dim=1).values
        )
        selected = selected & (axis_ratio > self.config.axis_ratio_threshold)

        # Save parent ADMM state before Gaussian array is extended
        n_before         = gaussian_model.n_gaussians
        parent_dual_y    = self.dual_y[selected].clone()
        parent_z_cons    = self.z_consensus[selected].clone()
        parent_is_bnd    = self.is_boundary[selected].clone()

        # RTGStable handles Gaussian extension + eta inheritance
        super()._densify_and_clone(grads, gaussian_model, optimizers)

        n_after = gaussian_model.n_gaussians
        n_new   = n_after - n_before
        if n_new <= 0:
            return

        dev = self.dual_y.device

        new_y = torch.zeros(n_after, 12, device=dev, dtype=torch.float32)
        new_y[:n_before] = self.dual_y
        new_y[n_before:] = parent_dual_y
        self.dual_y = new_y

        new_z = torch.zeros(n_after, 12, device=dev, dtype=torch.float32)
        new_z[:n_before] = self.z_consensus
        new_z[n_before:] = parent_z_cons
        self.z_consensus = new_z

        new_ib = torch.zeros(n_after, device=dev, dtype=torch.bool)
        new_ib[:n_before] = self.is_boundary
        new_ib[n_before:] = parent_is_bnd
        self.is_boundary = new_ib

    # ── Topology: split / new → ADMM state = 0 ───────────────────────────────

    def _densification_postfix(self, new_properties: dict, gaussian_model: VanillaGaussianModel, optimizers: List):
        """Called for split children (clone bypasses this via grandparent call in RTGStable)."""
        n_existing = gaussian_model.n_gaussians
        n_new      = next(iter(new_properties.values())).shape[0]

        old_y  = self.dual_y.clone()
        old_z  = self.z_consensus.clone()
        old_ib = self.is_boundary.clone()

        super()._densification_postfix(new_properties, gaussian_model, optimizers)

        dev = old_y.device

        new_y = torch.zeros(n_existing + n_new, 12, device=dev, dtype=torch.float32)
        new_y[:n_existing] = old_y
        self.dual_y = new_y

        new_z = torch.zeros(n_existing + n_new, 12, device=dev, dtype=torch.float32)
        new_z[:n_existing] = old_z
        self.z_consensus = new_z

        new_ib = torch.zeros(n_existing + n_new, device=dev, dtype=torch.bool)
        new_ib[:n_existing] = old_ib
        self.is_boundary = new_ib

    # ── Topology: prune removes ADMM state ───────────────────────────────────

    def _prune_points(self, mask: Tensor, gaussian_model: VanillaGaussianModel, optimizers: List):
        valid = ~mask
        super()._prune_points(mask, gaussian_model, optimizers)  # also filters eta
        self.dual_y      = self.dual_y[valid]
        self.z_consensus = self.z_consensus[valid]
        self.is_boundary = self.is_boundary[valid]
