"""Spherical Beta (SB) color evaluation, ported from Deformable Beta Splatting.

Math (see beta-splatting submodules/gsplat/cuda/csrc/spherical_beta.cuh):

    C(d) = c0 + sum_i c_i * max(0, dot(mu_i, d))^(4 * exp(b_i))

where each lobe i has 6 raw parameters [r, g, b, theta, phi, beta]:
    mu_i = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
and the rgb amplitudes are activated with softplus(x, beta=10*ln2) upstream.

Only the forward is ported; backward is handled by autograd (the CUDA
backward in the reference implementation matches the analytic gradient of
this expression exactly).
"""
import math

import torch
import torch.nn.functional as F


def sb_params_activation(sb_params: torch.Tensor) -> torch.Tensor:
    """Activate raw SB parameters: softplus on the rgb amplitudes, rest as-is.

    Args:
        sb_params: [..., L, 6] raw parameters [r, g, b, theta, phi, beta]
    """
    rgb = F.softplus(sb_params[..., :3], beta=math.log(2) * 10)
    return torch.cat([rgb, sb_params[..., 3:]], dim=-1)


def eval_spherical_beta(sb_params: torch.Tensor, dirs: torch.Tensor, base_color: torch.Tensor) -> torch.Tensor:
    """Add SB lobe contributions on top of a base color.

    Args:
        sb_params: [N, L, 6] activated parameters (rgb already softplus-ed)
        dirs: [N, 3] view directions (gaussian center - camera center), unnormalized
        base_color: [N, 3] base (diffuse) color, e.g. clamp(SH_eval + 0.5, min=0)

    Returns:
        [N, 3] colors, clamped to >= 0
    """
    return _sb_lobes(sb_params[..., :3], sb_params[..., 3:], F.normalize(dirs, dim=-1), base_color)


def eval_spherical_beta_raw(raw_sb_params: torch.Tensor, unit_dirs: torch.Tensor, base_color: torch.Tensor) -> torch.Tensor:
    """Math-identical fast path: takes RAW sb_params (softplus applied inline on
    the rgb slice, no activation cat/materialization) and pre-normalized dirs
    (skips the second normalize). Used by the training renderer.
    """
    rgb = F.softplus(raw_sb_params[..., :3], beta=math.log(2) * 10)
    return _sb_lobes(rgb, raw_sb_params[..., 3:], unit_dirs, base_color)


def _sb_lobes(rgb: torch.Tensor, angles_beta: torch.Tensor, d: torch.Tensor, base_color: torch.Tensor) -> torch.Tensor:
    """Core lobe math. rgb [N,L,3] (activated), angles_beta [N,L,3] =
    [theta, phi, beta], d [N,3] unit view dirs, base_color [N,3]."""
    theta = angles_beta[..., 0]  # [N, L]
    phi = angles_beta[..., 1]
    beta = angles_beta[..., 2]

    sin_theta = torch.sin(theta)
    mu = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1)  # [N, L, 3]

    dot = (mu * d.unsqueeze(1)).sum(dim=-1)  # [N, L]
    # betaTerm = dot^(4*exp(beta)) for dot > 0 else 0. The clamp inside pow keeps
    # the backward finite at dot == 0 (the where() zeroes the false branch's grad).
    exponent = 4.0 * torch.exp(beta)
    beta_term = torch.where(
        dot > 0,
        dot.clamp_min(1e-8).pow(exponent),
        torch.zeros_like(dot),
    )  # [N, L]

    colors = base_color + (beta_term.unsqueeze(-1) * rgb).sum(dim=1)  # [N, 3]
    return colors.clamp_min(0.0)
