"""
Anchor growing / pruning for Scaffold-2DGS (port of ../Scaffold-GS adjust_anchor).
Accumulates per-(anchor,offset) viewspace-gradient stats each step; periodically grows new
anchors where offset gradients are high (model.build_grown_anchor_props) and prunes anchors with
low accumulated opacity. Optimizer-state surgery uses the framework's Utils (same as MCMC), so the
per-property optimizer groups (named in ScaffoldGaussian2DModel.training_setup) stay consistent.
"""
from dataclasses import dataclass
from typing import List

import torch
from lightning import LightningModule

from .density_controller import DensityController, DensityControllerImpl, Utils


@dataclass
class ScaffoldDensityController(DensityController):
    start_stat_iter: int = 500          # accumulate offset/opacity stats from here (Scaffold start_stat)
    densify_from_iter: int = 1500       # grow from here (Scaffold update_from)
    densify_until_iter: int = 15_000
    update_interval: int = 100
    densify_grad_threshold: float = 0.0002
    min_opacity: float = 0.005
    success_threshold: float = 0.8
    max_total_anchors: int = 150000   # safety cap (anchors*n_offsets surfels must fit 6GB)

    def instantiate(self, *args, **kwargs) -> "ScaffoldDensityControllerImpl":
        return ScaffoldDensityControllerImpl(self)


class ScaffoldDensityControllerImpl(DensityControllerImpl):
    config: ScaffoldDensityController

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)
        if stage != "fit":
            return
        n = pl_module.gaussian_model.n_gaussians
        k = pl_module.gaussian_model.config.n_offsets
        dev = pl_module.device
        self.opacity_accum = torch.zeros((n, 1), device=dev)
        self.anchor_demon = torch.zeros((n, 1), device=dev)
        self.offset_grad_accum = torch.zeros((n * k, 1), device=dev)
        self.offset_denom = torch.zeros((n * k, 1), device=dev)

    def _sync_device(self, dev):
        self.opacity_accum = self.opacity_accum.to(dev)
        self.anchor_demon = self.anchor_demon.to(dev)
        self.offset_grad_accum = self.offset_grad_accum.to(dev)
        self.offset_denom = self.offset_denom.to(dev)

    @torch.no_grad()
    def _training_statis(self, outputs, model):
        k = model.config.n_offsets
        neural_opacity = outputs["neural_opacity"]          # [N*K,1]
        self._sync_device(neural_opacity.device)
        sel_mask = outputs["selection_mask"]                # [N*K] bool
        vis = outputs["visibility_filter"]                  # [M] bool
        grad = outputs["viewspace_points"].grad             # [M,3]
        n = model.n_gaussians
        anchor_visible = torch.ones(n, dtype=torch.bool, device=neural_opacity.device)

        temp_op = neural_opacity.clone().view(-1)
        temp_op[temp_op < 0] = 0
        temp_op = temp_op.view(-1, k)                       # [N,K]
        self.opacity_accum[anchor_visible] += temp_op.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible] += 1

        av = anchor_visible.unsqueeze(1).repeat(1, k).view(-1)      # [N*K]
        combined = torch.zeros_like(self.offset_grad_accum, dtype=torch.bool).squeeze(1)
        combined[av] = sel_mask
        temp = combined.clone()
        combined[temp] = vis
        if vis.any():
            grad_norm = torch.norm(grad[vis, :2], dim=-1, keepdim=True)
            self.offset_grad_accum[combined] += grad_norm
            self.offset_denom[combined] += 1

    @torch.no_grad()
    def _adjust_anchor(self, model, optimizers):
        cfg = self.config
        # optimizers[0] = property optimizer (single-param groups); optimizers[1] = MLPs (skip)
        prop_opt = [optimizers[0]]
        k = model.config.n_offsets
        # ---- grow ----
        denom = self.offset_denom
        grads = self.offset_grad_accum / denom.clamp_min(1.0)
        grads[denom.squeeze(1) == 0] = 0.0
        grads = grads.squeeze(1)
        offset_mask = (denom > cfg.update_interval * cfg.success_threshold * 0.5).squeeze(1)

        dev = self.opacity_accum.device
        m = 0
        if model.n_gaussians < cfg.max_total_anchors:
            new_props = model.build_grown_anchor_props(grads, cfg.densify_grad_threshold, offset_mask)
            if new_props is not None:
                m = new_props["means"].shape[0]
                print(f"[grow] {model.n_gaussians}+{m} anchors  offset_mask={int(offset_mask.sum())}  grads.max={grads.max():.2e}", flush=True)
                model.properties = Utils.cat_tensors_to_properties(new_props, model, prop_opt)
                self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros((m, 1), device=dev)], 0)
                self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros((m, 1), device=dev)], 0)

        # ★ reset ONLY qualifying offsets (Scaffold), pad for new anchors — so rarely-visible
        #   offsets keep accumulating across intervals (key for aerial: anchors seen in few views).
        self.offset_grad_accum[offset_mask] = 0
        self.offset_denom[offset_mask] = 0
        if m > 0:
            pad = torch.zeros((m * k, 1), device=dev)
            self.offset_grad_accum = torch.cat([self.offset_grad_accum, pad], 0)
            self.offset_denom = torch.cat([self.offset_denom, pad.clone()], 0)

        # ---- prune anchors with low accumulated opacity (only well-visited anchors) ----
        anchors_mask = (self.anchor_demon > cfg.update_interval * cfg.success_threshold).squeeze(1)
        prune = (self.opacity_accum < cfg.min_opacity * self.anchor_demon).squeeze(1) & anchors_mask
        # reset opacity stats for well-visited anchors (Scaffold)
        if bool(anchors_mask.any()):
            self.opacity_accum[anchors_mask] = 0
            self.anchor_demon[anchors_mask] = 0
        if bool(prune.any()):
            keep = ~prune
            model.properties = Utils.prune_properties(keep, model, prop_opt)
            self.opacity_accum = self.opacity_accum[keep]
            self.anchor_demon = self.anchor_demon[keep]
            self.offset_grad_accum = self.offset_grad_accum.view(-1, k)[keep].reshape(-1, 1)
            self.offset_denom = self.offset_denom.view(-1, k)[keep].reshape(-1, 1)

    def after_backward(self, outputs, batch, gaussian_model, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter or global_step <= self.config.start_stat_iter:
            return
        self._training_statis(outputs, gaussian_model)   # accumulate from start_stat
        if global_step > self.config.densify_from_iter and global_step % self.config.update_interval == 0:
            self._adjust_anchor(gaussian_model, optimizers)
