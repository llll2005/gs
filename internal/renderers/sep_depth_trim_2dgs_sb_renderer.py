"""Trim 2DGS renderer with Spherical Beta color (DBS port).

Same rasterization pipeline as SepDepthTrim2DGSRenderer; only the color path
changes: instead of letting the CUDA rasterizer evaluate SH, colors are
precomputed in Python as

    base = clamp(SH_eval(shs, dir) + 0.5, 0)      # diffuse (deg 0) or low-deg SH
    color = clamp(base + sum_i c_i * max(0, <mu_i, dir>)^(4*exp(b_i)), 0)

and passed via colors_precomp. Requires the gaussian model to be
Gaussian2DSB (provides sb_params).

Perf notes (2026-07-18, math-identical fast paths):
- record_transmittance passes (trimming, every 500 steps x all cameras) are
  color-independent -> zeros, no SB eval.
- raw-params SB eval: softplus inline on the rgb slice, no activation cat.
- dirs normalized once, shared by SH base and SB lobes.
- sh_degree==0 shortcut: base = C0 * dc + 0.5 without the generic eval_sh path.
"""
import torch

from .sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer
from internal.utils.sh_utils import eval_sh, C0
from internal.utils.sb_utils import eval_spherical_beta_raw


class SepDepthTrim2DGSSBRenderer(SepDepthTrim2DGSRenderer):
    def _get_color_inputs(self, pc, viewpoint_camera, record_transmittance=False):
        means = pc.get_xyz
        if record_transmittance:
            # transmittance recording ignores colors entirely — skip the SB eval
            return None, means.new_zeros((means.shape[0], 3))

        dirs = means - viewpoint_camera.camera_center  # [N, 3]; grads flow to means (matches CUDA SH path)
        dirs_norm = torch.nn.functional.normalize(dirs, dim=-1)

        if pc.active_sh_degree == 0:
            # `shs_dc` during training, but GaussianModelLoader hands tools a model whose
            # properties have been merged into a single `shs` — reading only the first key
            # made every loader-based tool (cull_dust, render_traj, ...) crash on SB models.
            g = pc.gaussians
            dc = g["shs_dc"] if "shs_dc" in g else g["shs"][:, :1]
            base = C0 * dc.squeeze(1)  # [N, 3]
        else:
            # get_features is [N, K, 3], eval_sh wants [N, 3, K]
            base = eval_sh(pc.active_sh_degree, pc.get_features.transpose(1, 2), dirs_norm)
        base = torch.clamp_min(base + 0.5, 0.0)

        colors = eval_spherical_beta_raw(pc.gaussians["sb_params"], dirs_norm, base)
        return None, colors
