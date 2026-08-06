"""
Native importance pruning for 2D Gaussian surfels (LightGaussian-style, 2DGS backend).

LightGaussian's score (count_and_score) goes through gsplat's 3D rasterizer, which needs 3D
scales — incompatible with 2DGS surfels (2D scale). Rather than pad to 3D (numerically unstable
degenerate Gaussians), we keep LightGaussian's ALGORITHM (global importance = contribution *
size^v_pow, prune the bottom k%) but compute it on the NATIVE 2DGS renderer:

  - contribution: the SepDepthTrim2DGS renderer already returns per-surfel transmittance via
    `record_transmittance=True`; we accumulate the mean of each surfel's top-K transmittance
    across all train views (identical to the renderer's own Trim contribution).
  - size: torch.prod(scales, dim=1) over the 2 surfel scales = surfel AREA (the 2D analogue of
    LightGaussian's 3D volume).

See 紀錄/主線_Gaussian效率.md. Used to measure how much redundancy remains after MCMC.
"""

import torch
from internal.utils.topk_contribution import contribution_accumulator

from internal.density_controllers.density_controller import Utils


@torch.no_grad()
def native_2dgs_importance_prune(module, prune_percent: float, v_pow: float = 0.1) -> None:
    model = module.gaussian_model
    renderer = module.renderer
    cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
    device = model.get_scales().device
    K = getattr(renderer, "K", 5)

    # 1. harvest per-surfel contribution = mean of top-K transmittance over all train views
    #    (replicates the Trim renderer's own contribution; the renderer returns per-surfel `trans`)
    # Shared with the trim renderer. The hand-rolled loop this replaces only inserted values that
    # beat the running maximum, so a view that belonged in the top-K but was not a new record was
    # dropped -- and the seed `[trans]*K` copied the first camera into every slot. It did shift
    # correctly (unlike the renderer's copy, which also overwrote slot 0 mid-shift), but it was
    # still not the mean of the K largest. See `tests/topk_contribution_test.py`.
    bg = module._fixed_background_color().to(device)
    push, gather = contribution_accumulator(K)
    for i in range(len(cameras)):
        camera = cameras[i].to_device(device)
        push(renderer(camera, model, bg_color=bg, record_transmittance=True))  # [N]
    contribution = gather()  # [N]

    # 2. surfel area (= 2D analogue of volume, since a surfel's 3rd axis is identically 0 so
    #    LightGaussian's V = 4/3*pi*abc would be 0 for every point)
    # 3. importance V_imp = contribution * gamma(area), gamma = (min(area/area_p90, 1))^v_pow
    #
    # The min(.,1) clip is LightGaussian Eq.4 and was MISSING here until 2026-07-28. It is the
    # half of Eq.4 that actually matters: dividing by the p90 is a constant across points and
    # cannot change the ranking, but the clip caps the volume bonus for the largest 10%, which
    # is precisely the anti-floater guard the paper introduces it for -- "using Gaussian volume
    # alone tends to overemphasize background Gaussians, leading to excessive pruning of
    # Gaussians modeling fine geometry ... clipping the range between 0 and 1, to avoid
    # excessive floating Gaussians derived from vanilla 3D-GS" (Sec. 3.2, Eq. 4).
    # Without it a floater 1000x the median area collects a 10^0.3 ~ 2x importance bonus, so it
    # outranks a real surface with half its contribution -- our exact monster pathology.
    area = torch.prod(model.get_scales(), dim=1)            # [N]
    area_p90 = torch.quantile(area, 0.9).clamp_min(1e-12)
    v_imp = contribution * torch.pow((area / area_p90).clamp(max=1.0), v_pow)

    # 4. prune the bottom `prune_percent` by importance
    n = v_imp.shape[0]
    k = int(prune_percent * n)
    if k <= 0:
        return
    threshold = torch.kthvalue(v_imp, k).values
    keep_mask = v_imp > threshold  # True = keep
    # NB strictly-greater with ties: if MORE than k points share the threshold value -- which
    # happens when many have importance exactly 0, i.e. invisible in every view -- this removes
    # all of them, not k. Asking for 20% can remove considerably more. Left as is: those points
    # are worthless by construction and keeping them to honour a percentage would be worse. The
    # print below reports the ACTUAL count next to the requested percent, so the gap is visible.
    print(f"[2DGS-imp-prune] N={n:,} prune={int((~keep_mask).sum()):,} ({prune_percent:.0%}) "
          f"v_pow={v_pow} area_p90={float(area_p90):.3e} clipped={100*float((area>=area_p90).float().mean()):.1f}%")

    model.properties = Utils.prune_properties(keep_mask, model, module.gaussian_optimizers)
    module.density_updated_by_renderer()
    print(f"[2DGS-imp-prune] remaining={model.get_scales().shape[0]:,}")
