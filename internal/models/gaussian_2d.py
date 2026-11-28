from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel


@dataclass
class Gaussian2D(VanillaGaussian):
    surf_tex: bool = False
    """★ 2026-11-28 面內亮度漸變（使用者「面元記主要顏色＋漸變範式＋透明度」）：每顆多 2 個參數 g（property `tex`），
    像素顏色＝顏色 x max(0, 1 + g·s)，s＝光線與面元交點的局部座標（以尺度為單位）。起點全 0＝與原本渲染完全相同。
    需要支援 surf_tex 的光柵器（diff_trim_surfel_rasterization.SURF_TEX_SUPPORTED）。False＝原行為。"""
    surf_tex_lr: float = 0.0025

    def instantiate(self, *args, **kwargs) -> "Gaussian2DModel":
        return Gaussian2DModel(self)


class Gaussian2DModelMixin:
    def get_extra_property_names(self):
        return ["tex"] if bool(getattr(self.config, "surf_tex", False)) else []

    def _add_surf_tex(self, property_dict, n):
        if bool(getattr(self.config, "surf_tex", False)):
            property_dict["tex"] = nn.Parameter(torch.zeros((n, 2), dtype=torch.float32).requires_grad_(True))

    def training_setup(self, module):
        optimizers, schedulers = super().training_setup(module)
        if bool(getattr(self.config, "surf_tex", False)):
            lr = float(getattr(self.config, "surf_tex_lr", 0.0025))
            optimizers[-1].add_param_group({"params": [self.gaussians["tex"]], "lr": lr, "name": "tex"})
            print(f"  tex={lr}（★ 11-28 面內亮度漸變）")
        return optimizers, schedulers

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_pcd(
            xyz=xyz,
            rgb=rgb,
            property_dict=property_dict,
            *args,
            **kwargs,
        )
        with torch.no_grad():
            property_dict["scales"] = property_dict["scales"][..., :2]
            # key to a quality comparable to hbb1/2d-gaussian-splatting
            property_dict["rotations"].copy_(torch.rand_like(property_dict["rotations"]))
        self._add_surf_tex(property_dict, xyz.shape[0])

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_number(
            n=n,
            property_dict=property_dict,
            *args,
            **kwargs,
        )
        property_dict["scales"] = property_dict["scales"][..., :2]
        self._add_surf_tex(property_dict, n)


class Gaussian2DModel(Gaussian2DModelMixin, VanillaGaussianModel):
    pass
