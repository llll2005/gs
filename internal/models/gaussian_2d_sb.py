"""2D Gaussian (surfel) model with Spherical Beta color, ported from
Deformable Beta Splatting (DBS).

Color = clamp(SH_eval(shs, dir) + 0.5, 0) + SB lobes(dir). Typically used with
sh_degree=0 so the SH part reduces to a diffuse DC color and all view
dependence comes from the SB lobes. This cuts the color parameter count from
48 floats (SH3) to 3 + 6*sb_number floats per point — the "F lever" in the
budget formula N_max = (V_target - V_os) / (4MF + gamma*tau/K).

Only the color representation changes; geometry (2D surfel scales/rotations)
and all density-control behavior are inherited unchanged. `sb_params` rides
along relocation/add/prune automatically because the MCMC density controller
iterates `gaussian_model.properties` generically.
"""
from dataclasses import dataclass
from typing import Dict

import torch

from .gaussian_2d import Gaussian2D, Gaussian2DModel
from internal.utils.sb_utils import sb_params_activation


@dataclass
class Gaussian2DSB(Gaussian2D):
    sb_number: int = 2
    """number of SB lobes per gaussian; 6 raw params [r,g,b,theta,phi,beta] each"""

    sb_params_lr: float = 0.0025
    """learning rate for sb_params (DBS default)"""

    sb_lobe_init: str = "uniform_angle"
    """How lobe directions are seeded: "uniform_angle" or "uniform_sphere".

    "uniform_angle" draws `theta ~ U(0, pi)`, which is what has been shipped and what every result
    in the table was trained with. It is NOT uniform on the sphere: the area element carries
    sin(theta), so a polar cap of theta < pi/6 gets 1/6 = 16.7% of the lobes where the sphere
    measure would give (1 - cos(pi/6))/2 = 6.7% -- the poles are oversampled about 2.5x.

    "uniform_sphere" draws `theta = arccos(1 - 2u)` instead. Left opt-in rather than made the
    default because changing it changes every initialisation, and SB already sits 0.46 dB under SH3
    on this content -- if the skew is part of that, it deserves a controlled A/B rather than a
    silent switch. The DBS paper specifies zero-init for `b` (which we do) but says nothing about
    lobe directions, so "DBS init" was never an accurate attribution for this line."""

    def instantiate(self, *args, **kwargs) -> "Gaussian2DSBModel":
        return Gaussian2DSBModel(self)


class Gaussian2DSBModel(Gaussian2DModel):
    config: Gaussian2DSB

    def get_extra_property_names(self):
        return super().get_extra_property_names() + ["sb_params"]

    def _init_sb_params(self, n: int) -> torch.Tensor:
        """[r,g,b, theta, phi, beta]; raw rgb 0 -> small positive amplitude after softplus,
        beta 0 -> Gaussian-like lobe (DBS §4: "we set b to zero at initialization")."""
        L = self.config.sb_number
        sb_params = torch.zeros((n, L, 6))
        if self.config.sb_lobe_init == "uniform_sphere":
            sb_params[..., 3] = torch.acos(1 - 2 * torch.rand((n, L)))
        elif self.config.sb_lobe_init == "uniform_angle":
            sb_params[..., 3] = torch.pi * torch.rand((n, L))
        else:
            raise ValueError(f"unknown sb_lobe_init: {self.config.sb_lobe_init}")
        sb_params[..., 4] = 2 * torch.pi * torch.rand((n, L))
        return sb_params

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_pcd(xyz=xyz, rgb=rgb, property_dict=property_dict, *args, **kwargs)
        property_dict["sb_params"] = self._init_sb_params(property_dict["means"].shape[0])

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_number(n=n, property_dict=property_dict, *args, **kwargs)
        property_dict["sb_params"] = self._init_sb_params(n)

    def get_sb_params(self) -> torch.Tensor:
        """activated SB parameters [N, L, 6] (rgb softplus-ed)"""
        return sb_params_activation(self.gaussians["sb_params"])

    def training_setup(self, module):
        optimizers, schedulers = super().training_setup(module)
        # Ride on the CONSTANT-LR optimizer; the group name must match the property name so the
        # density controller's generic per-group handling picks it up. The parent builds
        # [means_optimizer (scheduled), constant_lr_optimizer], so index 1 is the right one --
        # asserted rather than assumed, because attaching sb_params to the scheduled optimizer
        # would silently decay it to lr_final alongside `means` and nothing would raise.
        assert len(optimizers) == 2 and any(
            g.get("name") == "opacities" for g in optimizers[1].param_groups), \
            "parent optimizer layout changed; sb_params would attach to the wrong optimizer"
        optimizers[1].add_param_group({
            "params": [self.gaussians["sb_params"]],
            "lr": self.config.sb_params_lr,
            "name": "sb_params",
        })
        print("  sb_params={} (sb_number={})".format(self.config.sb_params_lr, self.config.sb_number))
        return optimizers, schedulers
