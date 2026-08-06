"""
FreGS progressive frequency regularization loss (CVPR 2024, arXiv 2403.06908).

Regularizes amplitude + phase discrepancies between the rendered image and GT in Fourier
space, with frequency annealing (low-to-high cutoff growing over training) to drive
coarse-to-fine densification and suppress over-reconstruction (a few large Gaussians covering
high-variance regions -> large artifacts). Image-space loss -> works with ANY density
mechanism (MCMC or grad-densify).

Paper eq.9-12 (low/high amplitude/phase discrepancy) + eq.13 (annealed high-freq band).
"""
import torch


def frequency_loss(pred: torch.Tensor, gt: torch.Tensor, step: int, total_steps: int,
                   d0_frac: float = 0.2, anneal_until_frac: float = 0.5) -> torch.Tensor:
    """pred, gt: [C,H,W] in [0,1]. Returns scalar FreGS loss (unweighted; caller scales)."""
    C, H, W = pred.shape
    # ortho-normalized FFT so amplitudes are O(image values), then center the spectrum
    Fp = torch.fft.fftshift(torch.fft.fft2(pred, norm="ortho"), dim=(-2, -1))
    Fg = torch.fft.fftshift(torch.fft.fft2(gt, norm="ortho"), dim=(-2, -1))
    amp_diff = (Fp.abs() - Fg.abs()).abs()                 # [C,H,W]
    # wrapped phase difference in (-pi, pi]
    dphi = Fp.angle() - Fg.angle()
    pha_diff = torch.atan2(torch.sin(dphi), torch.cos(dphi)).abs()

    # radial frequency (normalized 0..1), center = (H/2, W/2)
    yy, xx = torch.meshgrid(torch.arange(H, device=pred.device), torch.arange(W, device=pred.device), indexing="ij")
    r = torch.sqrt(((yy - H / 2.0) ** 2 + (xx - W / 2.0) ** 2))
    r = r / r.max().clamp_min(1e-6)                          # [H,W] in [0,1]

    # annealing: low-pass always on; high band grows from d0 to 1.0 over [0, anneal_until]
    prog = min(max(step / max(total_steps * anneal_until_frac, 1.0), 0.0), 1.0)
    dt = d0_frac + prog * (1.0 - d0_frac)
    low = (r <= d0_frac).unsqueeze(0)                       # [1,H,W]
    high = ((r > d0_frac) & (r <= dt)).unsqueeze(0)

    eps = 1e-8
    d_la = (amp_diff * low).sum() / (low.sum() * C + eps)
    d_lp = (pha_diff * low).sum() / (low.sum() * C + eps)
    d_ha = (amp_diff * high).sum() / (high.sum() * C + eps)
    d_hp = (pha_diff * high).sum() / (high.sum() * C + eps)
    return d_la + d_lp + d_ha + d_hp
