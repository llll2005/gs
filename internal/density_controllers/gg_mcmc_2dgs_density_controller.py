"""
Gradient-Guided Surfel MCMC (step-3 ablation over base MCMC-2DGS).

Base MCMC samples relocation targets / new Gaussians ∝ opacity ("rich get richer": recycled
budget piles onto already-good high-opacity regions; high-error holes with low opacity starve).
This subclass redirects sampling by the per-Gaussian view-space positional gradient
(xyz_gradient_accum ≈ local reconstruction error), so recycled budget goes where error is high.

See 紀錄/gradient_guided_mcmc_spec.md. Gate (budget sweep) passed: quality is count-limited
(~0.4 dB/100k, still rising at 414k), NOT a flat capacity ceiling, so reallocation has room.

Two pieces:
  1. Re-introduce per-Gaussian gradient accumulation (MCMC removed it). Maintained across the
     frequent topology changes from relocate / add_new / Trim-renderer prune, with a defensive
     self-heal (re-init if the buffer size ever desyncs from the live Gaussian count).
  2. Sampling weight (fixes Gemini's `opacity*grad`, which still zeros low-opacity high-error
     spots): grad-dominant with an opacity floor. Modes are switchable for the ablation table.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from lightning import LightningModule

from .density_controller import Utils
from .mcmc_2dgs_density_controller import (
    MCMC2DGSDensityController,
    MCMC2DGSDensityControllerImpl,
)


@dataclass
class GGMCMC2DGSDensityController(MCMC2DGSDensityController):
    relocation_weight_mode: Literal["opacity", "grad", "grad_op", "powered"] = "grad_op"
    """opacity = base MCMC (control); grad = pure error; grad_op = grad*(opacity+floor) (main);
    powered = grad**alpha * opacity**beta."""
    grad_floor: float = 0.1
    """opacity floor in grad_op so low-opacity high-error regions are not starved (fixes the
    `opacity*grad` flaw)."""
    grad_alpha: float = 1.0
    grad_beta: float = 1.0

    def instantiate(self, *args, **kwargs):
        assert self.cap_max > 0, "cap_max must > 0"
        return GGMCMC2DGSDensityControllerImpl(self)


class GGMCMC2DGSDensityControllerImpl(MCMC2DGSDensityControllerImpl):
    config: GGMCMC2DGSDensityController

    # ── gradient-accumulation state ──────────────────────────────────────────
    def _init_grad_state(self, n: int, device) -> None:
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)
        if stage == "fit":
            self._init_grad_state(pl_module.gaussian_model.n_gaussians, pl_module.device)

    def before_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module) -> None:
        super().before_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)
        if global_step < self.config.densify_until_iter:
            outputs["viewspace_points"].retain_grad()  # idempotent; renderer also retains

    def update_states(self, outputs) -> None:
        visibility = outputs["visibility_filter"]
        # self-heal: rebuild if the buffer ever desyncs in COUNT or DEVICE (it is created at
        # setup() while the model may still be on CPU, and plain attributes don't follow .to()).
        if (not hasattr(self, "xyz_gradient_accum")) \
                or self.xyz_gradient_accum.shape[0] != visibility.shape[0] \
                or self.xyz_gradient_accum.device != visibility.device:
            self._init_grad_state(visibility.shape[0], visibility.device)
        grad = outputs["viewspace_points"].grad
        if grad is None:
            return
        grad_norm = torch.norm(grad[visibility, :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[visibility] += grad_norm
        self.denom[visibility] += 1

    def _mean_grads(self) -> torch.Tensor:
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1.0)
        grads[grads.isnan()] = 0.0
        return grads.squeeze(-1)  # [N]

    # ── after_backward: accumulate every step, relocate/add_new at densify steps ──
    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module) -> None:
        if global_step >= self.config.densify_until_iter:
            return
        with torch.no_grad():
            self.update_states(outputs)
        # parent runs relocate_gs + add_new_gs at densify steps (using our grad-aware probs)
        super().after_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)
        # reset accumulation after a densify step (count may have changed)
        if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
            self._init_grad_state(gaussian_model.n_gaussians, gaussian_model.get_means().device)

    # ── keep grad state aligned under topology changes ───────────────────────
    def _prune_points(self, mask, gaussian_model, optimizers) -> None:
        super()._prune_points(mask, gaussian_model, optimizers)  # prunes model properties
        # Only prune the accum if it is in sync (count + device); otherwise leave it stale and
        # let update_states' self-heal rebuild it to the live count next step.
        if hasattr(self, "xyz_gradient_accum") \
                and self.xyz_gradient_accum.shape[0] == mask.shape[0] \
                and self.xyz_gradient_accum.device == mask.device:
            valid = ~mask
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
            self.denom = self.denom[valid]

    def after_density_changed(self, gaussian_model, optimizers, pl_module) -> None:
        super().after_density_changed(gaussian_model, optimizers, pl_module)
        self._init_grad_state(gaussian_model.n_gaussians, pl_module.device)

    # ── gradient-aware sampling weight ───────────────────────────────────────
    def _sampling_weight(self, opacity: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        # opacity, grad: [M], both >= 0 (opacity in (0,1), grad is a norm)
        mode = self.config.relocation_weight_mode
        if mode == "opacity":
            return opacity + 1e-12
        g = grad / (grad.mean() + 1e-8)  # relative error (scale-free)
        if mode == "grad":
            return g + 1e-12
        if mode == "grad_op":
            return g * (opacity + self.config.grad_floor) + 1e-12
        if mode == "powered":
            return torch.pow(g + 1e-8, self.config.grad_alpha) * torch.pow(opacity + 1e-8, self.config.grad_beta) + 1e-12
        raise ValueError(f"unknown relocation_weight_mode: {mode}")

    # ── override relocate_gs / add_new_gs only to swap the sampling probs ─────
    def relocate_gs(self, gaussian_model, optimizers, dead_mask) -> None:
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return

        opac = gaussian_model.get_opacities()[alive_indices, 0]
        grad = self._mean_grads()[alive_indices]
        probs = self._sampling_weight(opac, grad)  # gradient-aware (vs base: opac only)

        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        new_params = self._get_new_params(gaussian_model, reinit_idx, ratio=ratio)
        for attr_name in new_params:
            gaussian_model.get_property(attr_name)[dead_indices] = new_params[attr_name]
        gaussian_model.opacities[reinit_idx] = gaussian_model.opacities[dead_indices]
        gaussian_model.scales[reinit_idx] = gaussian_model.scales[dead_indices]
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=reinit_idx)

    def add_new_gs(self, gaussian_model, optimizers) -> int:
        cap_max = self.config.cap_max
        current_num_points = gaussian_model.n_gaussians
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0

        opac = gaussian_model.get_opacities().squeeze(-1)
        grad = self._mean_grads()
        probs = self._sampling_weight(opac, grad)  # gradient-aware (vs base: opac only)

        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        new_params = self._get_new_params(gaussian_model, add_idx, ratio=ratio)
        gaussian_model.opacities[add_idx] = new_params["opacities"]
        gaussian_model.scales[add_idx] = new_params["scales"]
        gaussian_model.properties = Utils.cat_tensors_to_properties(new_params, gaussian_model, optimizers)
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=add_idx)
        return num_gs
