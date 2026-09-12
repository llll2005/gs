"""
Scaffold-GS (anchor + MLP) adapted to 2D Gaussian Surfels, integrated into the
gaussian-splatting-lightning framework. Port of ../Scaffold-GS/scene/gaussian_model.py +
gaussian_renderer/generate_neural_gaussians, with the generated primitives made 2D (surfel)
so we keep CityGSV2's geometry/mesh + depth/normal regularization.

Representation:
  - anchors (= the framework property "means")  [N,3]   voxelized from depth-init pcd
  - per-anchor: offset [N, K*3], anchor_feat [N, feat_dim], anchor_scaling [N,6]  (extra properties)
  - shared MLPs: mlp_opacity (Tanh), mlp_cov (linear, ->scale+rot), mlp_color (Sigmoid)
Every render, for visible anchors, the MLPs take (feat, view_dir, dist) -> K neural surfels.
View-dependent color via MLP (the core Scaffold appearance mechanism) directly addresses the
2D-surfel "one rigid color per pixel" distortion (see 紀錄/SOGS移植藍圖.md).

This version uses FIXED anchors (StaticDensityController). Anchor growing/pruning is the next
layer (scaffold_density_controller.py) — see blueprint. feature_bank / appearance disabled for v1.
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
from torch import nn

from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel
from .gaussian_2d import Gaussian2DModelMixin


@dataclass
class ScaffoldGaussian2D(VanillaGaussian):
    feat_dim: int = 32
    n_offsets: int = 10
    voxel_size: float = 0.001        # in scene units; <=0 -> auto (median knn dist)
    max_anchors: int = 100000        # cap anchor count (anchors*n_offsets surfels/frame must fit 6GB)
    init_ply_path: str = ""          # voxelize THIS dense pcd into anchors (run with initialize_from=null)
    update_init_factor: int = 16
    # learning rates (Scaffold defaults, scaled by spatial_lr_scale where noted)
    anchor_lr: float = 0.0
    offset_lr: float = 0.01
    feature_lr: float = 0.0075
    anchor_scaling_lr: float = 0.007
    mlp_lr: float = 0.004

    def instantiate(self, *args, **kwargs) -> "ScaffoldGaussian2DModel":
        return ScaffoldGaussian2DModel(self)


class ScaffoldGaussian2DModel(Gaussian2DModelMixin, VanillaGaussianModel):
    config: ScaffoldGaussian2D

    # ---- extra (anchor) properties carried by the framework property system ----
    def get_extra_property_names(self):
        return ["offset", "anchor_feat", "anchor_scaling"]

    def _build_mlps(self, device):
        fd = self.config.feat_dim
        k = self.config.n_offsets
        # +3 view dir +1 dist
        self.mlp_opacity = nn.Sequential(
            nn.Linear(fd + 3 + 1, fd), nn.ReLU(True), nn.Linear(fd, k), nn.Tanh()).to(device)
        self.mlp_cov = nn.Sequential(
            nn.Linear(fd + 3 + 1, fd), nn.ReLU(True), nn.Linear(fd, 7 * k)).to(device)
        self.mlp_color = nn.Sequential(
            nn.Linear(fd + 3 + 1, fd), nn.ReLU(True), nn.Linear(fd, 3 * k), nn.Sigmoid()).to(device)

    @staticmethod
    def _voxelize(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
        # keep one point per occupied voxel (centroid of the quantized grid cell)
        q = torch.round(xyz / voxel_size)
        _, inv = torch.unique(q, dim=0, return_inverse=True)
        n = int(inv.max().item()) + 1
        out = torch.zeros((n, 3), dtype=xyz.dtype, device=xyz.device)
        cnt = torch.zeros((n, 1), dtype=xyz.dtype, device=xyz.device)
        out.index_add_(0, inv, xyz)
        cnt.index_add_(0, inv, torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device))
        return out / cnt.clamp_min(1.0)

    def setup_from_pcd(self, xyz, rgb, *args, **kwargs):
        from internal.utils.general_utils import inverse_sigmoid
        if self.config.init_ply_path:
            # voxelize a dense depth-init pcd into anchors (ignore the passed COLMAP sparse pcd)
            from internal.utils.gaussian_utils import GaussianPlyUtils
            xyz = GaussianPlyUtils.load_from_ply(self.config.init_ply_path).to_parameter_structure().xyz
        if isinstance(xyz, np.ndarray):
            xyz = torch.tensor(xyz)
        xyz = xyz.float()
        device = xyz.device

        vs = self.config.voxel_size
        if vs <= 0:
            from simple_knn._C import distCUDA2
            d = torch.clamp_min(distCUDA2(xyz.cuda()), 1e-7).to(device)
            vs = float(torch.sqrt(d).median().item())
        anchors = self._voxelize(xyz, vs)
        if anchors.shape[0] > self.config.max_anchors:
            sel = torch.randperm(anchors.shape[0], device=anchors.device)[:self.config.max_anchors]
            anchors = anchors[sel]
        n = anchors.shape[0]
        self._voxel_size = vs
        print(f"[Scaffold2D] voxel_size={vs:.5f} -> {n:,} anchors (from {xyz.shape[0]:,} pts), "
              f"{n * self.config.n_offsets:,} surfels/frame")

        fd, k = self.config.feat_dim, self.config.n_offsets
        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(anchors.cuda()), 1e-7).to(device)
        log_scale = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)   # anchor_scaling [N,6]

        # vestigial 2DGS properties (kept for the framework contract; renderer uses generate())
        shs = torch.zeros((n, 1, 3))
        scales2d = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((n, 4))
        opac = inverse_sigmoid(0.1 * torch.ones((n, 1)))

        property_dict = {
            "means": nn.Parameter(anchors.requires_grad_(True)),
            "shs_dc": nn.Parameter(shs.requires_grad_(True)),
            "shs_rest": nn.Parameter(torch.zeros((n, (self.config.sh_degree + 1) ** 2 - 1, 3)).requires_grad_(True)),
            "scales": nn.Parameter(scales2d.requires_grad_(True)),
            "rotations": nn.Parameter(rots.requires_grad_(True)),
            "opacities": nn.Parameter(opac.requires_grad_(True)),
            # anchor extras
            "offset": nn.Parameter(torch.zeros((n, k * 3)).requires_grad_(True)),
            "anchor_feat": nn.Parameter(torch.zeros((n, fd)).requires_grad_(True)),
            "anchor_scaling": nn.Parameter(log_scale.requires_grad_(True)),
        }
        self._build_mlps(device)
        self.set_properties(property_dict)
        self.active_sh_degree = 0

    def setup_from_number(self, n: int, *args, **kwargs):
        # minimal path (e.g. checkpoint restore builds params then loads weights)
        from internal.utils.general_utils import inverse_sigmoid
        fd, k = self.config.feat_dim, self.config.n_offsets
        property_dict = {
            "means": nn.Parameter(torch.zeros((n, 3)).requires_grad_(True)),
            "shs_dc": nn.Parameter(torch.zeros((n, 1, 3)).requires_grad_(True)),
            "shs_rest": nn.Parameter(torch.zeros((n, (self.config.sh_degree + 1) ** 2 - 1, 3)).requires_grad_(True)),
            "scales": nn.Parameter(torch.zeros((n, 2)).requires_grad_(True)),
            "rotations": nn.Parameter(torch.zeros((n, 4)).requires_grad_(True)),
            "opacities": nn.Parameter(inverse_sigmoid(0.1 * torch.ones((n, 1))).requires_grad_(True)),
            "offset": nn.Parameter(torch.zeros((n, k * 3)).requires_grad_(True)),
            "anchor_feat": nn.Parameter(torch.zeros((n, fd)).requires_grad_(True)),
            "anchor_scaling": nn.Parameter(torch.zeros((n, 6)).requires_grad_(True)),
        }
        self._build_mlps(torch.device("cpu"))
        self.set_properties(property_dict)
        self.active_sh_degree = 0

    # ---- the heart: MLP -> K neural 2D surfels per visible anchor ----
    def generate_neural_surfels(self, camera):
        anchor = self.get_property("means")                              # [N,3]
        feat = self.get_property("anchor_feat")                          # [N,fd]
        k = self.config.n_offsets
        offsets = self.get_property("offset").view(-1, k, 3)             # [N,K,3]
        grid_scaling = torch.exp(self.get_property("anchor_scaling"))    # [N,6]

        ob_view = anchor - camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist.clamp_min(1e-8)
        cat = torch.cat([feat, ob_view, ob_dist], dim=1)                 # [N, fd+3+1]

        neural_opacity = self.mlp_opacity(cat).reshape(-1, 1)            # [N*K,1] in (-1,1)
        mask = (neural_opacity > 0.0).view(-1)
        opacity = neural_opacity[mask]                                   # [M,1]
        color = self.mlp_color(cat).reshape(-1, 3)                      # [N*K,3] in (0,1)
        scale_rot = self.mlp_cov(cat).reshape(-1, 7)                    # [N*K,7]

        concat = torch.cat([grid_scaling, anchor], dim=-1)             # [N,9]
        concat = concat.unsqueeze(1).repeat(1, k, 1).reshape(-1, 9)    # [N*K,9]
        allp = torch.cat([concat, color, scale_rot, offsets.reshape(-1, 3)], dim=-1)
        masked = allp[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, off = masked.split([6, 3, 3, 7, 3], dim=-1)

        # ★ 2D adaptation: surfel scale = 2 dims (drop the 3rd); rot = quaternion
        scaling = scaling_repeat[:, 3:5] * torch.sigmoid(scale_rot[:, :2])  # [M,2]
        rot = torch.nn.functional.normalize(scale_rot[:, 3:7])             # [M,4]
        xyz = repeat_anchor + off * scaling_repeat[:, :3]                  # [M,3]
        # mask / neural_opacity returned for anchor-growing stats (ScaffoldDensityController)
        return xyz, color, opacity.clamp(0.0, 1.0), scaling, rot, mask, neural_opacity

    @torch.no_grad()
    def build_grown_anchor_props(self, grads, threshold, offset_mask, update_depth=3,
                                 update_hierachy_factor=4, update_init_factor=16):
        """Scaffold anchor_growing: multi-resolution candidate anchors from high-offset-gradient
        positions, deduped against existing anchors. Returns a dict of new property rows to cat
        (via Utils.cat_tensors_to_properties), or None. (ported from ../Scaffold-GS)"""
        from internal.utils.general_utils import inverse_sigmoid
        from torch_scatter import scatter_max
        from functools import reduce
        k = self.config.n_offsets
        fd = self.config.feat_dim
        vs = self._voxel_size
        anchor = self.get_property("means")
        feat_all = self.get_property("anchor_feat")
        scaling3 = torch.exp(self.get_property("anchor_scaling"))[:, :3]
        offset = self.get_property("offset").view(-1, k, 3)
        device = anchor.device

        n0 = anchor.shape[0] * k
        new_chunks = []
        for i in range(update_depth):
            cur_thr = threshold * ((update_hierachy_factor // 2) ** i)
            cand = (grads >= cur_thr) & offset_mask
            cand = cand & (torch.rand_like(cand.float()) > (0.5 ** (i + 1)))
            inc = anchor.shape[0] * k - n0
            if inc != 0:
                cand = torch.cat([cand, torch.zeros(inc, dtype=torch.bool, device=device)], dim=0)
            all_xyz = (anchor.unsqueeze(1) + offset * scaling3.unsqueeze(1)).view(-1, 3)
            size_factor = update_init_factor // (update_hierachy_factor ** i)
            cur_size = vs * max(size_factor, 1)
            grid = torch.round(anchor / cur_size).int()
            sel_xyz = all_xyz[cand]
            if sel_xyz.shape[0] == 0:
                continue
            sel_grid = torch.round(sel_xyz / cur_size).int()
            uniq, inv = torch.unique(sel_grid, return_inverse=True, dim=0)
            # dedup against existing anchors (chunked)
            cs = 4096
            dup = []
            for j in range(grid.shape[0] // cs + (1 if grid.shape[0] % cs else 0)):
                dup.append((uniq.unsqueeze(1) == grid[j * cs:(j + 1) * cs]).all(-1).any(-1).view(-1))
            keep = ~reduce(torch.logical_or, dup) if dup else torch.ones(uniq.shape[0], dtype=torch.bool, device=device)
            cand_anchor = uniq[keep] * cur_size
            if cand_anchor.shape[0] == 0:
                continue
            new_scaling = torch.log(torch.ones_like(cand_anchor).repeat(1, 2) * cur_size)  # [.,6]
            new_feat0 = feat_all.unsqueeze(1).repeat(1, k, 1).view(-1, fd)[cand]
            new_feat = scatter_max(new_feat0, inv.unsqueeze(1).expand(-1, fd), dim=0)[0][keep]
            m = cand_anchor.shape[0]
            new_chunks.append({
                "means": cand_anchor,
                "anchor_feat": new_feat,
                "anchor_scaling": new_scaling,
                "offset": torch.zeros((m, k * 3), device=device),
                "scales": torch.log(torch.ones((m, 2), device=device) * cur_size),
                "rotations": torch.cat([torch.ones((m, 1), device=device), torch.zeros((m, 3), device=device)], 1),
                "opacities": inverse_sigmoid(0.1 * torch.ones((m, 1), device=device)),
                "shs_dc": torch.zeros((m, 1, 3), device=device),
                "shs_rest": torch.zeros((m, (self.config.sh_degree + 1) ** 2 - 1, 3), device=device),
            })
        if not new_chunks:
            return None
        return {kk: torch.cat([c[kk] for c in new_chunks], dim=0) for kk in new_chunks[0]}

    def training_setup(self, module):
        from internal.optimizers import Adam  # noqa
        ss = self.config.optimization.spatial_lr_scale
        if ss <= 0:
            ss = module.trainer.datamodule.dataparser_outputs.camera_extent
        # ★ optimizer[0] = per-property single-param groups (named by property) so anchor
        #   growing/pruning can use Utils.cat_tensors_to_properties / prune_properties (which
        #   assert one param per group). optimizer[1] = MLPs (multi-param, NOT touched by surgery).
        prop_groups = [
            {"params": [self.gaussians["means"]], "lr": self.config.anchor_lr * ss, "name": "means"},
            {"params": [self.gaussians["offset"]], "lr": self.config.offset_lr * ss, "name": "offset"},
            {"params": [self.gaussians["anchor_feat"]], "lr": self.config.feature_lr, "name": "anchor_feat"},
            {"params": [self.gaussians["anchor_scaling"]], "lr": self.config.anchor_scaling_lr, "name": "anchor_scaling"},
            {"params": [self.gaussians["shs_dc"]], "lr": 0.0, "name": "shs_dc"},
            {"params": [self.gaussians["shs_rest"]], "lr": 0.0, "name": "shs_rest"},
            {"params": [self.gaussians["scales"]], "lr": 0.0, "name": "scales"},
            {"params": [self.gaussians["rotations"]], "lr": 0.0, "name": "rotations"},
            {"params": [self.gaussians["opacities"]], "lr": 0.0, "name": "opacities"},
        ]
        mlp_groups = [
            {"params": list(self.mlp_opacity.parameters()), "lr": self.config.mlp_lr, "name": "mlp_opacity"},
            {"params": list(self.mlp_cov.parameters()), "lr": self.config.mlp_lr, "name": "mlp_cov"},
            {"params": list(self.mlp_color.parameters()), "lr": self.config.mlp_lr, "name": "mlp_color"},
        ]
        prop_opt = torch.optim.Adam(prop_groups, lr=0.0, eps=1e-15)
        mlp_opt = torch.optim.Adam(mlp_groups, lr=0.0, eps=1e-15)
        return [prop_opt, mlp_opt], []
