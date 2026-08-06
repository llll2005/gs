"""AtomGS Edge-Aware Normal Loss (arXiv 2405.12369, eq. 1-2).

Penalizes geometric roughness (curvature of the depth-derived normal map) weighted DOWN at
image edges, so flat regions (roofs / roads -- where our large floaters live) are smoothed while
building edges stay sharp:

    L = mean( |∇N| * ω(|∇I|) ),   ω(x) = (x - 1)^q,  q even,  x = |∇I| ∈ [0,1]

ω is 1 on flat GT regions (|∇I|≈0 -> strong smoothing) and 0 on edges (|∇I|≈1 -> no smoothing).
N (the depth-derived normal map, AtomGS eq.1) is already produced by the 2DGS renderer as
`surf_normal`, so we reuse it instead of recomputing from depth. Pure loss, no CUDA changes;
orthogonal to edge-aware densification (that places points; this shapes geometry).
"""
import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def _channel_grad_mag(img: torch.Tensor) -> torch.Tensor:
    """Per-pixel gradient magnitude of a [C,H,W] map, summed over channels -> [H,W]."""
    c = img.shape[0]
    kx = _SOBEL_X.to(img.device, img.dtype).repeat(c, 1, 1, 1)  # [C,1,3,3]
    ky = _SOBEL_Y.to(img.device, img.dtype).repeat(c, 1, 1, 1)
    x = img.unsqueeze(0)  # [1,C,H,W]
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12).sum(dim=1).squeeze(0)  # [H,W]
    return mag


def edge_aware_normal_loss(
    surf_normal: torch.Tensor,
    gt_image: torch.Tensor,
    q: int = 2,
    edge_scale: float = 3.0,
) -> torch.Tensor:
    """surf_normal,[3,H,W] depth-derived normal (renderer output); gt_image,[3,H,W]."""
    curvature = _channel_grad_mag(surf_normal)  # |∇N|, raw (the term we minimize)

    # |∇I| mapped to [0,1] mean-relative: edge_scale*mean(|∇I|) -> 1 (a "strong" edge). Mean-
    # relative is robust to a few boundary outliers (a high percentile gets dominated by them,
    # compressing every real edge toward 0). gt has no grad so no detach needed.
    edge = _channel_grad_mag(gt_image.to(surf_normal.dtype))
    den = (edge_scale * edge.mean()).clamp_min(1e-8)
    edge = (edge / den).clamp(0., 1.)

    omega = (edge - 1.0).pow(q)  # q even -> [0,1]; 1 on flat, 0 on edges
    return (curvature * omega).mean()
