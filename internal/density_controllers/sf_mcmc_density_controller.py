"""
SF-MCMC: Split-Funded MCMC for 2D Gaussian Surfels.

Motivation (see 紀錄/方向重定調_2026-07-03.md §6): base MCMC's growth operator is
dynamically DEGENERATE for high-opacity parents — a new/relocated Gaussian is written at
the parent's exact position (`_get_new_params` copies `means`), the exploration noise is
gated to opacity <~ 0.005 (`op_sigmoid(1-o, k=100, x0=0.995)`), and
`replace_tensors_to_optimizers` zeroes both twins' Adam moments. Identical params +
identical gradients + identical moments -> the pair separates only through float
non-determinism. MCMC therefore recycles budget into redundant stacks instead of new
spatial structure, and it entirely lacks DGD's detail-refinement SPLIT (the one mechanism
our experiment matrix flags as untested; gg-MCMC showed reweighting the SAME operator is
net-zero).

SF-MCMC routes a fraction of both budgets (dead-slot relocation + add-to-cap growth) into
a symmetry-breaking, structure-targeted split:
  * candidates  = top-q of alive surfels by accumulated view-space gradient x world major
                  scale ("big and structurally wrong"; `random` mode is the ablation arm
                  isolating the signal from the operator),
  * opacity     = MCMC 1->2 renormalization (1-(1-o)^(1/2); 0.99 -> 0.9) — NOT SteepGS's
                  1/2 weighting, which would destroy the alpha=0.99 opaque-surfel prior,
  * geometry    = children offset +/- sigma * s_major along the parent's world major
                  tangent axis, major-axis scale shrunk (vanilla-split style) -> local
                  frequency capacity actually increases, twins are born separated.

Cap semantics are inherited unchanged: splits consume existing budget (dead slots or the
5%-to-cap growth), so the population never exceeds cap_max.

Defaults keep every non-split path byte-identical to base MCMC (`relocation_weight_mode
= "opacity"`), so arm1-vs-arm0 is a single-variable comparison.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from lightning import LightningModule

from internal.utils.gaussian_projection import build_rotation_matrix
from internal.utils.train_console import console_event
from .density_controller import Utils
from .gg_mcmc_2dgs_density_controller import (
    GGMCMC2DGSDensityController,
    GGMCMC2DGSDensityControllerImpl,
)


@dataclass
class SFMCMC2DGSDensityController(GGMCMC2DGSDensityController):
    relocation_weight_mode: Literal["opacity", "grad", "grad_op", "powered"] = "opacity"
    """override gg's default back to base-MCMC sampling: the split op must be the ONLY
    change vs arm0."""

    split_fraction: float = 0.5
    """fraction of each budget (dead slots at relocation; growth quota at add_new) routed
    to splits instead of opacity-sampled cloning."""

    split_top_q: float = 0.05
    """candidate pool = top-q fraction of eligible (alive & opaque-enough) surfels by
    score; splits are taken from the pool head (highest score first)."""

    split_candidate_mode: Literal["grad_scale", "random"] = "grad_scale"
    """grad_scale = accumulated view-space grad x world major scale (arm1);
    random = uniform over eligible (arm2: isolates the operator from the signal)."""

    split_min_opacity: float = 0.1
    """never split a dying surfel: renorm would push the pair toward the dead threshold
    and the budget is better spent by plain relocation."""

    split_offset_sigma: float = 0.5
    """children are placed at +/- sigma * s_major along the world major tangent axis
    (of the pre-shrink scale), mirroring vanilla split's sampling radius."""

    split_scale_shrink: float = 1.6
    """divisor applied to the major-axis scale of both children (vanilla 3DGS split uses
    1.6 for all axes; we shrink only the split axis to preserve the surfel's minor
    extent)."""

    def instantiate(self, *args, **kwargs):
        assert self.cap_max > 0, "cap_max must > 0"
        assert 0.0 <= self.split_fraction <= 1.0
        assert 0.0 < self.split_top_q <= 1.0
        return SFMCMC2DGSDensityControllerImpl(self)


class SFMCMC2DGSDensityControllerImpl(GGMCMC2DGSDensityControllerImpl):
    config: SFMCMC2DGSDensityController

    _split_events: int = 0
    _split_total: int = 0

    # ── candidate selection ───────────────────────────────────────────────────
    def _select_split_parents(self, gaussian_model, budget: int, dead_mask: Optional[torch.Tensor]) -> torch.Tensor:
        opac = gaussian_model.get_opacities().squeeze(-1)  # [N]
        eligible = opac >= self.config.split_min_opacity
        if dead_mask is not None:
            eligible &= ~dead_mask
        eligible_idx = eligible.nonzero(as_tuple=True)[0]
        if eligible_idx.numel() == 0:
            return eligible_idx

        pool_size = max(1, int(self.config.split_top_q * eligible_idx.numel()))
        n_split = min(budget, pool_size, eligible_idx.numel())
        if self.config.split_candidate_mode == "random":
            perm = torch.randperm(eligible_idx.numel(), device=eligible_idx.device)
            return eligible_idx[perm[:n_split]]

        major_scale = gaussian_model.get_scales()[eligible_idx].max(dim=-1).values
        grads = self._mean_grads()  # [N], self-healed by gg's update_states every step
        g = grads[eligible_idx]
        g = g / (g.mean() + 1e-8)
        score = (g + 1e-8) * major_scale
        top = torch.topk(score, k=n_split)
        return eligible_idx[top.indices]

    # ── split math: 1->2 renorm + symmetry-breaking geometry ─────────────────
    def _make_split(self, gaussian_model, parents: torch.Tensor) -> Tuple[dict, dict]:
        """Returns (child_params, parent_params): raw-parameter-space dicts covering ALL
        model properties for the children, and the in-place updates for the parents."""
        n = parents.shape[0]
        device = parents.device
        opac_old = gaussian_model.get_opacities()[parents, 0]        # [n], activated
        scale_old = gaussian_model.get_scales()[parents]             # [n, 2], activated

        # MCMC 1->N renormalization with N=2 (opacity: 1-(1-o)^(1/2); scale: opacity-driven
        # coeff, exact for the two real components thanks to the parent class's zero-pad).
        # int64 to match the proven base path (bincount+1 -> compute_relocation)
        two = torch.full((n,), 2, dtype=torch.long, device=device)
        new_opacity, new_scaling = self.compute_relocation(
            opacity_old=opac_old, scale_old=scale_old, N=two,
        )
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps, min=0.005,
        )

        # Symmetry breaking along the world major tangent axis.
        major_dim = scale_old.argmax(dim=-1)                         # [n] in {0, 1}
        rot = build_rotation_matrix(gaussian_model.get_rotations()[parents])  # [n, 3, 3]
        axis = rot.gather(2, major_dim.view(n, 1, 1).expand(n, 3, 1)).squeeze(-1)  # [n, 3]
        offset = axis * (scale_old.gather(1, major_dim.unsqueeze(-1)) * self.config.split_offset_sigma)

        new_scaling = new_scaling.clone()
        shrunk = new_scaling.gather(1, major_dim.unsqueeze(-1)) / self.config.split_scale_shrink
        new_scaling.scatter_(1, major_dim.unsqueeze(-1), shrunk)

        raw_opacity = gaussian_model.opacity_inverse_activation(new_opacity)
        raw_scaling = gaussian_model.scale_inverse_activation(new_scaling)
        means_old = gaussian_model.get_means()[parents]

        child_params = {
            "means": means_old + offset,
            "opacities": raw_opacity,
            "scales": raw_scaling,
        }
        for attr_name, value in gaussian_model.properties.items():
            if attr_name not in child_params:
                child_params[attr_name] = value[parents]

        parent_params = {
            "means": means_old - offset,
            "opacities": raw_opacity,
            "scales": raw_scaling,
        }
        return child_params, parent_params

    def _apply_parent_updates(self, gaussian_model, parents: torch.Tensor, parent_params: dict) -> None:
        for attr_name, value in parent_params.items():
            gaussian_model.get_property(attr_name)[parents] = value

    def _log_split(self, n_split: int, path: str, gaussian_model) -> None:
        self._split_events += 1
        self._split_total += n_split
        if self._split_events <= 3 or self._split_events % 10 == 0:
            console_event(
                f"[SF-MCMC] split#{self._split_events} ({path}): {n_split} parents -> pairs, "
                f"total {self._split_total}, N={gaussian_model.n_gaussians}"
            )

    # ── relocation path: dead slots become children of split parents ─────────
    def relocate_gs(self, gaussian_model, optimizers, dead_mask) -> None:
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return
        budget = int(self.config.split_fraction * n_dead)
        if budget > 0:
            parents = self._select_split_parents(gaussian_model, budget, dead_mask)
            n_split = parents.shape[0]
            if n_split > 0:
                dead_idx = dead_mask.nonzero(as_tuple=True)[0][:n_split]
                child_params, parent_params = self._make_split(gaussian_model, parents)
                for attr_name in child_params:
                    gaussian_model.get_property(attr_name)[dead_idx] = child_params[attr_name]
                self._apply_parent_updates(gaussian_model, parents, parent_params)
                # mirror base: reset Adam moments of every modified live slot
                touched = torch.cat([parents, dead_idx])
                self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=touched)
                self._log_split(n_split, "relocate", gaussian_model)
                # consumed slots are no longer dead (child opacity >= renorm(0.1) ~ 0.051)
                dead_mask = dead_mask.clone()
                dead_mask[dead_idx] = False
                if int(dead_mask.sum()) == 0:
                    return
        super().relocate_gs(gaussian_model, optimizers, dead_mask)

    # ── growth path: part of the 5%-to-cap quota appended as split children ──
    def add_new_gs(self, gaussian_model, optimizers) -> int:
        cap_max = self.config.cap_max
        current_num_points = gaussian_model.n_gaussians
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0

        budget = int(self.config.split_fraction * num_gs)
        n_split = 0
        if budget > 0:
            parents = self._select_split_parents(gaussian_model, budget, dead_mask=None)
            n_split = parents.shape[0]
            if n_split > 0:
                child_params, parent_params = self._make_split(gaussian_model, parents)
                self._apply_parent_updates(gaussian_model, parents, parent_params)
                gaussian_model.properties = Utils.cat_tensors_to_properties(child_params, gaussian_model, optimizers)
                self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=parents)
                self._log_split(n_split, "add", gaussian_model)

        # keep the gg grad buffer usable for the base sampling below (count changed)
        if n_split > 0:
            self._init_grad_state(gaussian_model.n_gaussians, gaussian_model.get_means().device)

        remaining = num_gs - n_split
        if remaining > 0:
            opac = gaussian_model.get_opacities().squeeze(-1)
            grad = self._mean_grads()
            probs = self._sampling_weight(opac, grad)  # mode defaults to "opacity" = base MCMC
            add_idx, ratio = self._sample_alives(probs=probs, num=remaining)
            new_params = self._get_new_params(gaussian_model, add_idx, ratio=ratio)
            gaussian_model.opacities[add_idx] = new_params["opacities"]
            gaussian_model.scales[add_idx] = new_params["scales"]
            gaussian_model.properties = Utils.cat_tensors_to_properties(new_params, gaussian_model, optimizers)
            self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=add_idx)
        return num_gs
