"""
Edge-Aware Surfel MCMC (grafts the EdgeAware densification signal onto the WINNING MCMC base).

Background: the standalone edge-aware grad-densify controller (edge_aware_density_controller.py,
"Result B") beat plain grad-densify (+0.65 PSNR) but still lost to the MCMC depth-init baseline
(21.70 vs 22.18) because grad-densify is architecturally weaker than MCMC's relocate-dead +
Langevin placement at equal budget. The edge signal is a good *selection* signal that was bolted
onto the losing mechanism. This controller moves it onto the winner.

MCMC samples relocation targets / new Gaussians ∝ opacity. GGMCMC (parent) already redirects that
by per-Gaussian view-space gradient (reconstruction error). Here we add a second, complementary
guidance signal -- the GT-image Laplacian edge score sampled at each Gaussian's projected center,
accumulated across views (same buffer machinery as the gradient accum) -- and multiply it into the
sampling weight:

    probs = base_weight(opacity, grad) * (1 + edge_beta * edge_score)

So with relocation_weight_mode="opacity" this is the originally proposed probs ∝ opacity*edge;
with "grad_op" it combines all three (grad * opacity * edge). edge_beta=0 recovers the parent GG.

Edge score = mean over views of the per-Gaussian sampled Laplacian magnitude (98th-pct normalized
to [0,1]); a Gaussian never sitting on GT structure keeps factor ~1 (neutral). No CUDA changes.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lightning import LightningModule

from .gg_mcmc_2dgs_density_controller import (
    GGMCMC2DGSDensityController,
    GGMCMC2DGSDensityControllerImpl,
)


@dataclass
class EdgeMCMC2DGSDensityController(GGMCMC2DGSDensityController):
    edge_beta: float = 2.0
    """edge boost strength: probs *= (1 + edge_beta * edge_score). 0 = disable (== parent GG)."""
    edge_norm_percentile: float = 0.98
    """robust [0,1] normalization of the Laplacian edge map (avoids one hot pixel dominating)."""

    def instantiate(self, *args, **kwargs):
        assert self.cap_max > 0, "cap_max must > 0"
        return EdgeMCMC2DGSDensityControllerImpl(self)


class EdgeMCMC2DGSDensityControllerImpl(GGMCMC2DGSDensityControllerImpl):
    config: EdgeMCMC2DGSDensityController

    # ── buffers: edge accum shares the grad accum lifecycle ──────────────────
    def _init_grad_state(self, n: int, device) -> None:
        super()._init_grad_state(n, device)  # xyz_gradient_accum / denom
        self.edge_accum = torch.zeros((n, 1), device=device)
        self.edge_denom = torch.zeros((n, 1), device=device)
        self._laplacian_kernel = torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device
        ).view(1, 1, 3, 3)

    def before_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module) -> None:
        super().before_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)
        # stash batch + model so update_states (which only receives `outputs`) can reach the
        # current camera, GT image, and world-space Gaussian means.
        self._cur_batch = batch
        self._cur_gaussian_model = gaussian_model

    def update_states(self, outputs) -> None:
        super().update_states(outputs)  # grad accum (+ self-heal that rebuilds BOTH buffers)
        if self.config.edge_beta > 0:
            self._accumulate_edge(outputs)

    # ── per-view edge sampling, accumulated like the gradient ────────────────
    @torch.no_grad()
    def _accumulate_edge(self, outputs) -> None:
        batch = getattr(self, "_cur_batch", None)
        if batch is None:
            return
        camera = batch[0]
        gt = batch[1][1]  # (name, gt_image, mask) -> [3, H, W]
        visibility = outputs["visibility_filter"]

        means = self._cur_gaussian_model.get_xyz  # [N, 3]
        device = self.edge_accum.device
        dtype = means.dtype

        # GT edge map (Laplacian magnitude), robustly normalized to ~[0, 1].
        gt = gt.to(device=device, dtype=dtype)
        gray = (0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2])[None, None]  # [1,1,H,W]
        edge = F.conv2d(gray, self._laplacian_kernel.to(dtype), padding=1).abs()[0, 0]  # [H, W]
        H, W = edge.shape
        denom = torch.quantile(
            edge.flatten().float(), self.config.edge_norm_percentile
        ).to(dtype).clamp_min(1e-8)
        edge_norm = (edge / denom).clamp(0., 1.)

        # project Gaussian centers -> pixels for this view
        proj = camera.get_full_perspective_projection().to(device=device, dtype=dtype)  # [4,4]
        ph = torch.cat([means, torch.ones_like(means[:, :1])], dim=-1)  # [N,4]
        p = ph @ proj
        z = p[:, 2]
        u = (p[:, 0] / z.clamp_min(1e-6)).round().long()
        v = (p[:, 1] / z.clamp_min(1e-6)).round().long()
        in_view = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        sampled = torch.zeros(means.shape[0], device=device, dtype=dtype)
        if in_view.any():
            sampled[in_view] = edge_norm[v[in_view], u[in_view]]

        # accumulate over the visible set (consistent with grad accumulation)
        self.edge_accum[visibility] += sampled[visibility].unsqueeze(-1)
        self.edge_denom[visibility] += 1

    def _mean_edges(self) -> torch.Tensor:
        e = self.edge_accum / self.edge_denom.clamp(min=1.0)
        e[e.isnan()] = 0.0
        return e.squeeze(-1)  # [N] in [0, 1]

    def _edge_factor(self, indices) -> torch.Tensor:
        # (1 + beta * edge_score); never-on-structure Gaussians keep ~1 (neutral boost).
        e = self._mean_edges()[indices]
        return 1.0 + self.config.edge_beta * e

    # ── keep edge buffers aligned under prune (grad buffers handled by super) ─
    def _prune_points(self, mask, gaussian_model, optimizers) -> None:
        super()._prune_points(mask, gaussian_model, optimizers)
        if hasattr(self, "edge_accum") \
                and self.edge_accum.shape[0] == mask.shape[0] \
                and self.edge_accum.device == mask.device:
            valid = ~mask
            self.edge_accum = self.edge_accum[valid]
            self.edge_denom = self.edge_denom[valid]

    # ── inject the edge factor into the parent's sampling probabilities ──────
    def relocate_gs(self, gaussian_model, optimizers, dead_mask) -> None:
        if self.config.edge_beta <= 0:
            return super().relocate_gs(gaussian_model, optimizers, dead_mask)

        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return

        opac = gaussian_model.get_opacities()[alive_indices, 0]
        grad = self._mean_grads()[alive_indices]
        probs = self._sampling_weight(opac, grad) * self._edge_factor(alive_indices)

        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        new_params = self._get_new_params(gaussian_model, reinit_idx, ratio=ratio)
        for attr_name in new_params:
            gaussian_model.get_property(attr_name)[dead_indices] = new_params[attr_name]
        gaussian_model.opacities[reinit_idx] = gaussian_model.opacities[dead_indices]
        gaussian_model.scales[reinit_idx] = gaussian_model.scales[dead_indices]
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=reinit_idx)

    def add_new_gs(self, gaussian_model, optimizers) -> int:
        if self.config.edge_beta <= 0:
            return super().add_new_gs(gaussian_model, optimizers)

        from .density_controller import Utils
        cap_max = self.config.cap_max
        current_num_points = gaussian_model.n_gaussians
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0

        opac = gaussian_model.get_opacities().squeeze(-1)
        grad = self._mean_grads()
        probs = self._sampling_weight(opac, grad) * self._edge_factor(torch.arange(opac.shape[0], device=opac.device))

        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        new_params = self._get_new_params(gaussian_model, add_idx, ratio=ratio)
        gaussian_model.opacities[add_idx] = new_params["opacities"]
        gaussian_model.scales[add_idx] = new_params["scales"]
        gaussian_model.properties = Utils.cat_tensors_to_properties(new_params, gaussian_model, optimizers)
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=add_idx)
        return num_gs
