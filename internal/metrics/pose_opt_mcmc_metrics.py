"""
3D Gaussian Splatting as Markov Chain Monte Carlo
https://ubc-vision.github.io/3dgs-mcmc/

Most codes are copied from https://github.com/ubc-vision/3dgs-mcmc
"""

import os
import torch
import kornia
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass, field

from .metric import MetricImpl
from .mcmc_metrics import MCMCMetrics, MCMCMetricsImpl
from ..renderers.gsplat_camera_opt import GSplatCameraOptRendererModule
from ..configs.instantiate_config import InstantiatableConfig
from internal.utils.gradient_loss import compute_gradient_loss

@dataclass
class PoseOptMCMCMetrics(MCMCMetrics):

    corr_path: str = None

    epi_loss_max_iter: int = 30_000

    epi_loss_weight: float = 1e-3

    magnitude_loss_alpha: float = 0.1

    magnitude_loss_weight: float = 0.0

    gradient_loss_only: bool = False

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return PoseOptMCMCMetricsImpl(self)

class PoseOptMCMCMetricsImpl(MCMCMetricsImpl):

    def __init__(self, config: InstantiatableConfig, *args, **kwargs) -> None:

        super(PoseOptMCMCMetricsImpl, self).__init__(config, *args, **kwargs)

        self.corr_points_i_normalized = None
        self.corr_points_j_normalized = None
        self.corr_points_i = None
        self.corr_points_j = None
        self.image_names_i = None
        self.image_names_j = None
        self.appearance_ids_i = None
        self.appearance_ids_j = None
        self.initialized = False

        self.c2ws_i = None
        self.c2ws_j = None
        self.Ks_i = None
        self.Ks_j = None

        assert config.corr_path is not None, "Please provide a valid path to the correspondence file."
        print(f"Loading tracking data from {config.corr_path}")
        correspondence = torch.load(config.corr_path)
        self.corr_points_i_normalized = correspondence["corr_points_i"].clone().detach()
        self.corr_points_j_normalized = correspondence["corr_points_j"].clone().detach()
        self.corr_points_i_normalized[..., 0] /= correspondence["original_width"]
        self.corr_points_i_normalized[..., 1] /= correspondence["original_height"]
        self.corr_points_j_normalized[..., 0] /= correspondence["original_width"]
        self.corr_points_j_normalized[..., 1] /= correspondence["original_height"]
        self.corr_weights = correspondence["corr_weights"].clone().detach()
        self.image_names_i = correspondence["image_names_i"]
        self.image_names_j = correspondence["image_names_j"]
    
    def initialize_corr(self, trainset, device):

        in_mask_i = np.isin(self.image_names_i, trainset.image_names)
        in_mask_j = np.isin(self.image_names_j, trainset.image_names)
        in_mask = in_mask_i & in_mask_j

        self.corr_points_i = torch.tensor(self.corr_points_i_normalized[in_mask], device=trainset.cameras.R.device, dtype=trainset.cameras.R.dtype)
        self.corr_points_j = torch.tensor(self.corr_points_j_normalized[in_mask], device=trainset.cameras.R.device, dtype=trainset.cameras.R.dtype)
        self.corr_weights = torch.tensor(self.corr_weights[in_mask], device=trainset.cameras.R.device, dtype=trainset.cameras.R.dtype)
        self.image_names_i = self.image_names_i[in_mask]
        self.image_names_j = self.image_names_j[in_mask]

        c2ws = torch.linalg.inv(torch.transpose(trainset.cameras.world_to_camera, -2, -1))
        Ks = torch.stack([camera.get_K() for camera in trainset.cameras], dim=0)
        indexes_i = [trainset.image_names.index(img_name) for img_name in self.image_names_i]
        indexes_j = [trainset.image_names.index(img_name) for img_name in self.image_names_j]
        self.c2ws_i = c2ws[indexes_i]
        self.c2ws_j = c2ws[indexes_j]
        self.Ks_i = Ks[indexes_i]
        self.Ks_j = Ks[indexes_j]

        self.corr_points_i[..., 0] *= trainset.cameras.width[indexes_i][:, None]
        self.corr_points_i[..., 1] *= trainset.cameras.height[indexes_i][:, None]
        self.corr_points_j[..., 0] *= trainset.cameras.width[indexes_j][:, None]
        self.corr_points_j[..., 1] *= trainset.cameras.height[indexes_j][:, None]

        self.appearance_ids_i = trainset.cameras.appearance_id[indexes_i].to(device)
        self.appearance_ids_j = trainset.cameras.appearance_id[indexes_j].to(device)

        self.corr_points_i = self.corr_points_i.to(device)
        self.corr_points_j = self.corr_points_j.to(device)
        self.corr_weights = self.corr_weights.to(device)
        self.c2ws_i = self.c2ws_i.to(device)
        self.c2ws_j = self.c2ws_j.to(device)
        self.Ks_i = self.Ks_i.to(device)
        self.Ks_j = self.Ks_j.to(device)

        self.initialized = True
        print(f"[INFO] Initialized {len(self.corr_points_i)} correspondences from {self.config.corr_path}.")
    
    def epipolar_loss(self, pl_module, basic_metrics, step, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        if not self.initialized and self.config.corr_path is not None:
            train_set = pl_module._trainer.datamodule.dataparser_outputs.train_set
            self.initialize_corr(train_set, device=basic_metrics[0]["loss"].device)
        
        if self.initialized and step < self.config.epi_loss_max_iter and \
              isinstance(pl_module.renderer, GSplatCameraOptRendererModule):
            c2ws_corrected_i = pl_module.renderer.model(self.c2ws_i, self.appearance_ids_i)
            c2ws_corrected_j = pl_module.renderer.model(self.c2ws_j, self.appearance_ids_j)
            w2cs_corrected_i = torch.linalg.inv(c2ws_corrected_i)
            w2cs_corrected_j = torch.linalg.inv(c2ws_corrected_j)

            P_i = self.Ks_i @ w2cs_corrected_i
            P_j = self.Ks_j @ w2cs_corrected_j
            Fm = kornia.geometry.epipolar.fundamental_from_projections(P_i[:, :3], P_j[:, :3])
            err = kornia.geometry.symmetrical_epipolar_distance(self.corr_points_i, self.corr_points_j, Fm, squared=False, eps=1e-08)
            lepipolar = (err * self.corr_weights.squeeze(-1)).mean()
        
        else:
            self.corr_points_i_normalized = None
            self.corr_points_j_normalized = None
            self.corr_points_i = None
            self.corr_points_j = None
            self.image_names_i = None
            self.image_names_j = None
            self.appearance_ids_i = None
            self.appearance_ids_j = None

            self.c2ws_i = None
            self.c2ws_j = None
            self.Ks_i = None
            self.Ks_j = None

            lepipolar = torch.tensor(0.0, device=basic_metrics[0]["loss"].device)

        if pl_module.renderer.config.model.pose_opt_type == "sfm":
            norm_embed = torch.norm(pl_module.renderer.model.embeds(batch[0].appearance_id), dim=-1)
            target_mag = self.config.magnitude_loss_alpha * basic_metrics[0]["loss"]
            l_magnitude = max(0, target_mag - norm_embed) * self.config.magnitude_loss_weight
            basic_metrics[0]["loss"] = basic_metrics[0]["loss"] + l_magnitude
            basic_metrics[0]["err_magnitude"] = l_magnitude
            basic_metrics[1]["err_magnitude"] = True

        basic_metrics[0]["loss"] = basic_metrics[0]["loss"] + lepipolar * self.config.epi_loss_weight
        basic_metrics[0]["err_epipolar"] = lepipolar
        basic_metrics[1]["err_epipolar"] = True

        return basic_metrics

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        
        if not self.config.gradient_loss_only:
            basic_metrics = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
            return self.epipolar_loss(pl_module, basic_metrics, step, batch, outputs)
        else:
            camera, image_info, _ = batch
            image_name, gt_image, masked_pixels = image_info
            image = outputs["render"]

            # calculate loss
            if masked_pixels is not None:
                gt_image = gt_image.clone()
                gt_image[masked_pixels] = image.detach()[masked_pixels]  # copy masked pixels from prediction to G.T.
            rgb_diff_loss = self.rgb_diff_loss_fn(outputs["render"], gt_image)
            ssim_metric = self.ssim(outputs["render"], gt_image)
            loss = compute_gradient_loss(gt_image.permute(1, 2, 0)[None], outputs["render"].permute(1, 2, 0)[None])

            return {
                "loss": loss,
                "rgb_diff": rgb_diff_loss,
                "ssim": ssim_metric,
            }, {
                "loss": True,
                "rgb_diff": True,
                "ssim": True,
            }
