"""
DBP — DGD-Budget-Prune density controller (new algorithm, merges 3 components).

Reverse-engineered from CityGSV2 (this session): the quality engine is DGD — densification driven
by the SSIM-loss viewspace gradient (Tab 7: "gradient source is the most critical for sub-optimal
results"). In this codebase DGD is realized in gaussian_splatting.py:426-433 (densify gradient is
reset to `org_grad` = the grad of metrics["loss"], which citygsv2_metrics.py sets to the SSIM +
regularizer term, NOT the L1 term). It flows into clone/split via xyz_gradient_accum. Our MCMC
pipeline never used it (MCMC places points by opacity) — a likely cause of the ~3.8 dB intercept
gap to CityGSV2.

DBP keeps DGD (inherited from CityGSV2DensityController: clone/split + elongation filter) and adds:
  - Budget cap (MCMC insight): a hard ceiling on gaussian count.
  - Importance-prune recycling (our micro-novelty, LightGaussian-style): whenever a DGD densify
    pushes the count over the cap, prune the lowest-importance surfels (opacity * area^v) back to
    the cap. = ONLINE explore-then-prune — DGD explores where structural error is high, the cap
    keeps only the highest-value points it produced.
  - Optional reset-free (our MCMC insight): `disable_opacity_reset`.

Why it can beat CityGSV2: CityGSV2 grows count via crude opacity-reset churn and uses ALL ~4M
points (low point-efficiency). DBP keeps the same structural targeting (DGD) but bounds the budget
by importance -> same quality at fewer points. Predicted (scaling law a≈16.7 inherited from DGD,
+0.46 efficiency from prune vs same count): ~28-29 PSNR at ~2-3M points.

NOTE: the online cap uses a CHEAP importance proxy (opacity * area^v, no rendering). For a final
lossless harvest, apply the full transmittance-based `importance_prune_2dgs` once at the end.

⚠ STATUS (2026-06-27) — this grad-densify-based DBP has a fatal limitation on 6GB, kept for record:
  - E2-B showed grad-densify does NOT grow the count on 6GB from our feasible inits:
    cropped-coarse collapses (opacity death, 258k->42k), depth-init stalls (well-fit -> low SSIM
    gradient -> clone/split rarely fires -> ~280k, NOT growing). So this DBP also caps out ~280k.
  - The reason CityGSV2 reaches 4M is the FULL coarse init (4.39M, high gradient) which OOMs on 6GB.
  - Redesign "MCMC base + SSIM-gradient sampling" was checked and is == gg_mcmc (its
    MCMCCityGSV2Metrics produces extra_loss -> the DGD block sets viewspace.grad to the SSIM
    gradient -> gg_mcmc already accumulated & sampled by it) -> net-zero (already tested).
  - => the SSIM signal helps in grad-densify's CLONE/SPLIT, not in MCMC's relocate/add. The
    genuinely-untested lever is MCMC + the SPLIT mechanism (refine large high-SSIM-gradient surfels
    into smaller ones — the one thing MCMC lacks). The stronger lever is CPU OFFLOADING (CLM /
    GS-Scale): MCMC grows fine on 6GB, just VRAM-capped at ~1.7M(sh3)/~4M(sh2); offloading lifts
    the cap so MCMC can reach 4M+ and quality follows. See 紀錄/DGD反推與DBP_2026-06-27.md.
"""
from dataclasses import dataclass

import torch

from .citygsv2_density_controller import CityGSV2DensityController, CityGSV2DensityControllerModule


@dataclass
class DBPDensityController(CityGSV2DensityController):
    cap_max: int = 2_000_000              # hard budget; importance-prune enforces it online
    cap_v_pow: float = 0.1                # area exponent in the cap-importance proxy
    disable_opacity_reset: bool = False   # reset-free option (our MCMC insight)

    def instantiate(self, *args, **kwargs) -> "DBPDensityControllerModule":
        return DBPDensityControllerModule(self)


class DBPDensityControllerModule(CityGSV2DensityControllerModule):
    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers):
        # DGD densify (clone/split on the SSIM gradient) + elongation filter + opacity/size prune
        super()._densify_and_prune(max_screen_size, gaussian_model, optimizers)
        # online budget enforcement via importance-prune (keep the best DGD produced)
        self._enforce_cap(gaussian_model, optimizers)

    @torch.no_grad()
    def _enforce_cap(self, gaussian_model, optimizers):
        cap = self.config.cap_max
        n = gaussian_model.n_gaussians
        if cap <= 0 or n <= cap:
            return
        opac = gaussian_model.get_opacities().squeeze(-1)            # [N], (0,1)
        area = torch.prod(gaussian_model.get_scales(), dim=1)        # [N], surfel area
        imp = opac * torch.pow(area + 1e-12, self.config.cap_v_pow)  # cheap importance proxy
        n_prune = int(n - cap)
        low_idx = torch.topk(imp, n_prune, largest=False).indices    # lowest-importance
        prune_mask = torch.zeros(n, dtype=torch.bool, device=imp.device)
        prune_mask[low_idx] = True                                   # True = prune
        self._prune_points(prune_mask, gaussian_model, optimizers)
        print(f"[DBP] cap {cap:,}: pruned {n_prune:,} lowest-importance -> {gaussian_model.n_gaussians:,}")

    def _reset_opacities(self, gaussian_model, optimizers):
        if self.config.disable_opacity_reset:
            return
        super()._reset_opacities(gaussian_model, optimizers)
