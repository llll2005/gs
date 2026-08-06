"""Edge-aware densification controller (Result B of merge_math_new_algo).

Unifies the densification-score family (3DGS / AbsGS / Pixel-GS / 2508 EAS) which all
fit the polynomial  score_i = sum_pixel w(p) * f(grad_L_i, p).  We approximate the
per-pixel-per-Gaussian quantities (which our Trim2DGS rasterizer does NOT expose, so
absgrad/EAS are blocked at the CUDA level) with cheap available quantities:

    score_i ~= edge_at_projected_center_i * ||screenspace_grad_i||

The edge term is a free GT-image Laplacian edge map sampled at each Gaussian's projected
center for the current view.  We boost (not suppress) edge regions:

    weight_i = 1 + beta * edge_norm_i          (edge_norm in [0,1], out-of-view -> 1)

so the baseline densify volume is preserved while split candidates are biased toward
object boundaries / high-frequency structure -- precisely where large blurry surfels
straddle and produce large artifacts.  No CUDA changes; works on 2DGS.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .citygsv2_density_controller import (
    CityGSV2DensityController,
    CityGSV2DensityControllerModule,
)


@dataclass
class EdgeAwareDensityController(CityGSV2DensityController):
    # Strength of the edge boost applied to the densification gradient.
    edge_weight_beta: float = 2.0

    # Robust normalization: divide edge map by this quantile (avoids one hot pixel
    # dominating the [0,1] range).
    edge_norm_percentile: float = 0.98

    def instantiate(self, *args, **kwargs) -> "EdgeAwareDensityControllerModule":
        return EdgeAwareDensityControllerModule(self)


class EdgeAwareDensityControllerModule(CityGSV2DensityControllerModule):
    config: EdgeAwareDensityController

    def _init_state(self, n_gaussians: int, device):
        super()._init_state(n_gaussians, device)
        # Per-Gaussian edge weight for the current view; defaults to neutral (1.0) so
        # any accumulation before the first compute behaves like the baseline.
        self._edge_weight = None
        # Laplacian kernel buffer (lazily moved to device).
        self.register_buffer(
            "_laplacian_kernel",
            torch.tensor(
                [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
            ).view(1, 1, 3, 3),
            persistent=False,
        )

    @torch.no_grad()
    def _compute_edge_weight(self, batch, gaussian_model) -> torch.Tensor:
        camera = batch[0]
        gt = batch[1][1]  # (name, gt_image, mask) -> gt_image [3, H, W]

        means = gaussian_model.get_xyz  # [N, 3]
        n = means.shape[0]
        device = means.device
        kernel = self._laplacian_kernel.to(device=device, dtype=means.dtype)

        # --- GT edge map (Laplacian magnitude), robustly normalized to ~[0, 1] ---
        gt = gt.to(device=device, dtype=means.dtype)
        gray = (0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2])[None, None]  # [1,1,H,W]
        edge = F.conv2d(gray, kernel, padding=1).abs()[0, 0]  # [H, W]
        H, W = edge.shape
        denom = torch.quantile(
            edge.flatten().float(), self.config.edge_norm_percentile
        ).to(edge.dtype).clamp_min(1e-8)
        edge_norm = (edge / denom).clamp(0., 1.)

        # --- project Gaussian centers to pixels for this view ---
        proj = camera.get_full_perspective_projection().to(device=device, dtype=means.dtype)  # [4,4]
        ph = torch.cat([means, torch.ones_like(means[:, :1])], dim=-1)  # [N,4]
        p = ph @ proj  # [N,4]
        z = p[:, 2]
        u = (p[:, 0] / z.clamp_min(1e-6)).round().long()
        v = (p[:, 1] / z.clamp_min(1e-6)).round().long()
        in_view = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        sampled = torch.zeros(n, device=device, dtype=means.dtype)
        if in_view.any():
            sampled[in_view] = edge_norm[v[in_view], u[in_view]]

        # Out-of-view Gaussians keep weight 1 (neutral); edges get boosted.
        weight = 1.0 + self.config.edge_weight_beta * sampled
        return weight.unsqueeze(-1)  # [N,1]

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module) -> None:
        # Refresh the edge weight for THIS view before super().update_states accumulates
        # gradients (which happens first inside super().after_backward, before N changes).
        if global_step < self.config.densify_until_iter:
            self._edge_weight = self._compute_edge_weight(batch, gaussian_model)
        super().after_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)

    def _add_densification_stats(self, grad, update_filter, scale):
        scaled_grad = grad[update_filter, :2]
        if scale is not None:
            scaled_grad = scaled_grad * scale
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)

        ew = self._edge_weight
        if ew is not None and ew.shape[0] == update_filter.shape[0]:
            grad_norm = grad_norm * ew[update_filter]

        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1
