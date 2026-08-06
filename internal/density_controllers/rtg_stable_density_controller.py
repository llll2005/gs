"""
RTG-SLAM Stable/Unstable Gaussian Management (RTG-SLAM Sec. 3.2)

Implements four mechanisms from the original paper:
  B1 — α ∈ {0.99, 0.1} + lr_α = 0  (Sec 3.1)
       depth_init PLY already uses 0.99 (INIT_ALPHA in depth_init_blocks.py).
       freeze_opacity=True sets opacities lr=0 and disables opacity-based pruning,
       relying on B4+B5 for Gaussian lifecycle instead.

  B3 — M_c mask + transparent Gaussian adding  (Sec 3.2, Eq. 6)
       At each densification interval: back-project high-color-error pixels that
       already have geometry (M_c) and add α=0.1 transparent Gaussians to correct
       residual color errors.

  B4 — e_i error count → stable→unstable reversion  (Sec 3.2 State management)
       Project stable Gaussian centres to the current camera; if rendered color error
       at that pixel exceeds stable_revert_threshold for stable_revert_count consecutive
       densification intervals, revert the Gaussian to unstable (eta=0).

  B5 — Long-term unstable pruning  (Sec 3.2 State management)
       Track birth_step for every Gaussian.  Gaussians that remain unstable for more
       than max_unstable_intervals densification intervals are pruned as outliers.

Note: B2 (first-opaque depth rendering via ray-disc intersection) requires modifying
      the CUDA rasteriser kernel and is not implemented here.  depth_ratio=1.0 (median
      depth, already in config) is the closest available approximation.

Topology rules (matching ADMM dual-variable conventions in CLAUDE.md):
  Clone  → children inherit parent's eta AND birth_step; stable_error_count resets to 0
  Split  → children start at eta=0, birth_step=current_step, stable_error_count=0
  Prune  → buffers filtered by valid mask
"""
from dataclasses import dataclass, field
import torch
from .citygsv2_density_controller import CityGSV2DensityController, CityGSV2DensityControllerModule
from .vanilla_density_controller import Utils, VanillaGaussianModel, List
from internal.utils.train_console import console_event


@dataclass
class RTGStableDensityController(CityGSV2DensityController):

    # ── Original RTGStable params ─────────────────────────────────────────────

    stable_threshold: int = 10
    """Densification intervals a Gaussian must survive before becoming stable.
    Default 10 × densification_interval (500) = 5 000 steps."""

    depth_init_immune: bool = False
    """When True, initial Gaussians (from depth-init PLY) start at eta=stable_threshold,
    making them immediately immune to opacity_reset and opacity-based pruning."""

    # ── B1: opacity freezing ──────────────────────────────────────────────────

    freeze_opacity: bool = False
    """RTG-SLAM B1: freeze opacity learning rate to 0 (lr_α = 0).
    Disables opacity-based pruning; lifecycle managed by B4 + B5 instead."""

    # ── B3: M_c transparent Gaussian adding ───────────────────────────────────

    mc_enabled: bool = False
    """RTG-SLAM B3: enable transparent Gaussian adding at M_c pixels."""

    mc_color_threshold: float = 0.1
    """δ_c: per-pixel mean L1 color error above which a pixel is in M_c."""

    mc_sample_ratio: float = 0.05
    """Fraction of M_c pixels to add Gaussians for (RTG-SLAM uses 5%)."""

    mc_transparent_alpha: float = 0.1
    """α for newly added transparent Gaussians (RTG-SLAM: 0.1)."""

    mc_scale_factor: float = 0.01
    """Transparent Gaussian scale = depth/focal * mc_scale_factor.
    RTG-SLAM says radius < 0.01m; this factor controls the proportion."""

    # ── B4: stable→unstable reversion via error count ─────────────────────────

    stable_revert_enabled: bool = False
    """RTG-SLAM B4: enable stable→unstable reversion based on per-Gaussian error count."""

    stable_revert_threshold: float = 0.1
    """δ_c for B4: pixel L1 error above which a stable Gaussian's e_i is incremented."""

    stable_revert_count: int = 5
    """δ_e: e_i must exceed this to revert a stable Gaussian to unstable."""

    # ── B5: long-term unstable pruning ───────────────────────────────────────

    max_unstable_intervals: int = 0
    """δ_t: max densification intervals a Gaussian can remain unstable before pruning.
    0 = disabled."""

    # ── Hybrid soft-cap population control（硬天花板 cap + 軟門檻節流帶）──────────
    cap_max: int = 0
    """硬天花板：N 不得超過此值（0=關）。防 OOM。B3 加點限於剩餘預算、grad-densify 達 cap 停長，
    並在每次 densify 後 importance-prune（opacity×面積）保底到 cap。永遠生效（不受 soft_*_iter 限制）。"""

    soft_cap_enabled: bool = False
    """開啟 cap_max 之下的軟節流帶（需 cap_max>0）。控穩態/重分配，不影響硬 cap 的防 OOM 保證。"""

    soft_cap_band: int = 200000
    """軟帶寬度：軟區 = [cap_max - soft_cap_band, cap_max]。N 進軟區開始節流，越接近 cap 越強。"""

    soft_densify_scale: float = 0.3
    """軟區內增生保留比例下限（0..1）。keep 從軟區入口 1.0 線性降到 cap 的此值。
    grad-densify 以「grad_threshold /= keep」在源頭少加；B3 以 n_add*=keep。不製造 add-then-prune churn。"""

    soft_prune_scale: float = 1.5
    """軟區內剪枝加強倍率（>=1）。等效 B5 的 max_unstable_intervals /= 此值 → 更快剪掉 unstable（未證明）點。"""

    soft_from_iter: int = -1
    """軟節流生效起始步（-1 = 用 densify_from_iter）。"""

    soft_until_iter: int = -1
    """軟節流生效終止步（-1 = 用 densify_until_iter）。關進 densify 窗口 → 精修期族群固定 → 不傷收斂。"""

    def instantiate(self, *args, **kwargs) -> "RTGStableDensityControllerModule":
        return RTGStableDensityControllerModule(self)


class RTGStableDensityControllerModule(CityGSV2DensityControllerModule):

    # ── Internal state ────────────────────────────────────────────────────────
    _current_step: int = 0
    _opacity_lr_frozen: bool = False

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self, stage: str, pl_module) -> None:
        super().setup(stage, pl_module)
        if stage == "fit":
            n = pl_module.gaussian_model.n_gaussians
            dev = pl_module.device
            init_eta = self.config.stable_threshold if self.config.depth_init_immune else 0
            self.register_buffer(
                "eta",
                torch.full((n,), init_eta, dtype=torch.long, device=dev),
                persistent=True,
            )
            if self.config.depth_init_immune:
                console_event(f"[RTGStable] depth_init_immune=True: {n:,} initial Gaussians start stable (eta={init_eta})")
            # B5: birth_step — record when each Gaussian was created
            if self.config.max_unstable_intervals > 0:
                self.register_buffer(
                    "birth_step",
                    torch.zeros(n, dtype=torch.long, device=dev),
                    persistent=True,
                )
            # B4: stable_error_count — track consecutive high-error intervals
            if self.config.stable_revert_enabled:
                self.register_buffer(
                    "stable_error_count",
                    torch.zeros(n, dtype=torch.long, device=dev),
                    persistent=True,
                )

    def on_load_checkpoint(self, module, checkpoint):
        super().on_load_checkpoint(module, checkpoint)
        n = checkpoint["state_dict"]["density_controller.max_radii2D"].shape[0]
        dev = module.device
        if "density_controller.eta" not in checkpoint["state_dict"]:
            self.register_buffer(
                "eta",
                torch.full((n,), self.config.stable_threshold, dtype=torch.long, device=dev),
                persistent=True,
            )
        else:
            self.register_buffer("eta", torch.zeros(n, dtype=torch.long, device=dev), persistent=True)
        # B5
        if self.config.max_unstable_intervals > 0:
            self.register_buffer("birth_step", torch.zeros(n, dtype=torch.long, device=dev), persistent=True)
        # B4
        if self.config.stable_revert_enabled:
            self.register_buffer("stable_error_count", torch.zeros(n, dtype=torch.long, device=dev), persistent=True)

    def after_density_changed(self, gaussian_model, optimizers, pl_module) -> None:
        super().after_density_changed(gaussian_model, optimizers, pl_module)
        n = gaussian_model.n_gaussians
        dev = pl_module.device
        if not hasattr(self, "eta") or self.eta.shape[0] != n:
            self.register_buffer("eta", torch.zeros(n, dtype=torch.long, device=dev), persistent=True)
        if self.config.max_unstable_intervals > 0:
            if not hasattr(self, "birth_step") or self.birth_step.shape[0] != n:
                self.register_buffer("birth_step", torch.zeros(n, dtype=torch.long, device=dev), persistent=True)
        if self.config.stable_revert_enabled:
            if not hasattr(self, "stable_error_count") or self.stable_error_count.shape[0] != n:
                self.register_buffer("stable_error_count", torch.zeros(n, dtype=torch.long, device=dev), persistent=True)

    # ── B1: freeze opacity learning rate ─────────────────────────────────────

    def _maybe_freeze_opacity_lr(self, pl_module) -> None:
        """Set opacities param-group lr to 0 on first call if freeze_opacity=True."""
        if not self.config.freeze_opacity or self._opacity_lr_frozen:
            return
        opts = pl_module.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for opt in opts:
            actual = opt.optimizer if hasattr(opt, "optimizer") else opt
            for pg in actual.param_groups:
                if pg.get("name") == "opacities":
                    pg["lr"] = 0.0
                    console_event("[RTGStable] B1: opacity lr frozen to 0.0")
                    self._opacity_lr_frozen = True
                    return

    # ── Confidence increment + B3/B4 dispatch ────────────────────────────────

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module):
        self._current_step = global_step
        self._maybe_freeze_opacity_lr(pl_module)

        is_densify_step = (
            self.config.densify_from_iter < global_step < self.config.densify_until_iter
            and global_step % self.config.densification_interval == 0
        )
        if is_densify_step:
            self.eta += 1
            # B4: error-count based stable→unstable reversion (no Gaussian count change, safe here)
            if self.config.stable_revert_enabled and hasattr(self, "stable_error_count"):
                self._update_stable_error_count(outputs, batch, gaussian_model)

        # super() calls update_states(outputs) which uses visibility_filter from the current render.
        # B3 must come AFTER this call — adding Gaussians here would expand max_radii2D before
        # update_states runs, causing a shape mismatch on visibility_filter.
        super().after_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)

        # B3: add transparent Gaussians at M_c pixels — after update_states + densification
        if is_densify_step and self.config.mc_enabled:
            self._add_mc_gaussians(outputs, batch, gaussian_model, optimizers, pl_module)

        # Hybrid hard cap: 所有加點後保證 N<=cap_max（防 OOM 的最後保底）
        if is_densify_step and self.config.cap_max > 0:
            self._enforce_hard_cap(gaussian_model, optimizers)

    # ── Stable mask ──────────────────────────────────────────────────────────

    @property
    def _stable_mask(self) -> torch.Tensor:
        return self.eta >= self.config.stable_threshold

    # ── B1: opacity reset (stable Gaussians skip) ─────────────────────────────

    def _reset_opacities(self, gaussian_model: VanillaGaussianModel, optimizers: List):
        opacities_new = gaussian_model.opacity_inverse_activation(
            torch.min(
                gaussian_model.get_opacities(),
                torch.ones_like(gaussian_model.get_opacities()) * 0.01,
            )
        )
        stable = self._stable_mask
        if stable.any():
            opacities_new[stable] = gaussian_model.get_property("opacities")[stable]
        new_parameters = Utils.replace_tensors_to_properties(
            tensors={"opacities": opacities_new}, optimizers=optimizers
        )
        gaussian_model.update_properties(new_parameters)
        n_s, n_t = stable.sum().item(), stable.shape[0]
        console_event(f"[RTGStable] opacity_reset — reset: {n_t - n_s:,}  protected: {n_s:,} stable")

    # ── Hybrid soft-cap helpers ──────────────────────────────────────────────

    def _soft_active(self, step: int) -> bool:
        """軟節流是否在此步生效（開關 + 階段窗口）。硬 cap 不走這裡、永遠生效。"""
        if not self.config.soft_cap_enabled or self.config.cap_max <= 0:
            return False
        lo = self.config.soft_from_iter if self.config.soft_from_iter >= 0 else self.config.densify_from_iter
        hi = self.config.soft_until_iter if self.config.soft_until_iter >= 0 else self.config.densify_until_iter
        return lo <= step < hi

    def _soft_keep(self, n: int) -> float:
        """軟區內的增生保留比例 keep∈[soft_densify_scale, 1]；軟區外=1.0。"""
        cap = self.config.cap_max
        soft_lo = cap - self.config.soft_cap_band
        if n <= soft_lo:
            return 1.0
        t = min(max((n - soft_lo) / max(cap - soft_lo, 1), 0.0), 1.0)
        return 1.0 - t * (1.0 - self.config.soft_densify_scale)

    def _importance_score(self, gaussian_model: VanillaGaussianModel) -> torch.Tensor:
        """便宜的每顆重要度 = opacity × surfel 面積（2 個 scale 相乘）。
        freeze 下 unstable/B3 透明點(α=0.1、面積小)天然最低 → 硬 cap 會先剪它們、保護 stable。"""
        opac = gaussian_model.get_opacities().squeeze(-1)
        area = gaussian_model.get_scales().abs().prod(dim=1)
        return opac * area

    def _enforce_hard_cap(self, gaussian_model: VanillaGaussianModel, optimizers: List) -> None:
        """硬 cap 保底：若 N>cap_max，剪掉最低重要度的溢出顆數，保證 N<=cap_max。"""
        cap = self.config.cap_max
        n = gaussian_model.n_gaussians
        if cap <= 0 or n <= cap:
            return
        n_remove = n - cap
        score = self._importance_score(gaussian_model)
        remove_idx = torch.topk(score, n_remove, largest=False).indices
        prune_mask = torch.zeros(n, dtype=torch.bool, device=score.device)
        prune_mask[remove_idx] = True
        self._prune_points(prune_mask, gaussian_model, optimizers)
        console_event(f"[RTGStable] hard cap: pruned {n_remove:,} lowest-importance -> {gaussian_model.n_gaussians:,} (cap {cap:,})")

    # ── Prune: B1 (skip opacity prune when frozen), B5 (lifetime pruning) ────

    def _densify_and_prune(self, max_screen_size, gaussian_model: VanillaGaussianModel, optimizers: List):
        prune_extent = self.prune_extent
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Hybrid: 硬 cap 達上限就停 grad 成長；軟帶則臨時提高 grad_threshold（源頭少加）
        cap = self.config.cap_max
        n_now = gaussian_model.n_gaussians
        grow = not (cap > 0 and n_now >= cap)
        _orig_thr = self.config.densify_grad_threshold
        if grow and self._soft_active(self._current_step):
            keep = self._soft_keep(n_now)
            self.config.densify_grad_threshold = _orig_thr / max(keep, 1e-3)
        if grow:
            self._densify_and_clone(grads, gaussian_model, optimizers)
            self._densify_and_split(grads, gaussian_model, optimizers)
        self.config.densify_grad_threshold = _orig_thr

        # B1: skip opacity-based pruning when freeze_opacity=True
        if not self.config.freeze_opacity:
            min_opacity = self.config.cull_opacity_threshold
            prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
            prune_mask = prune_mask & ~self._stable_mask
        else:
            prune_mask = torch.zeros(gaussian_model.n_gaussians, dtype=torch.bool,
                                     device=gaussian_model.get_means().device)

        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_vs), big_ws)

        # B5: prune long-term unstable Gaussians（軟區內以 soft_prune_scale 提早剪）
        if self.config.max_unstable_intervals > 0 and hasattr(self, "birth_step"):
            eff_max_unstable = self.config.max_unstable_intervals
            if self._soft_active(self._current_step):
                eff_max_unstable = max(1, int(eff_max_unstable / max(self.config.soft_prune_scale, 1e-3)))
            max_lifetime = eff_max_unstable * self.config.densification_interval
            long_lived = (self._current_step - self.birth_step) > max_lifetime
            long_lived_unstable = long_lived & ~self._stable_mask
            if long_lived_unstable.any():
                n_b5 = long_lived_unstable.sum().item()
                console_event(f"[RTGStable] B5: pruning {n_b5:,} long-term unstable Gaussians (> {self.config.max_unstable_intervals} intervals)")
                prune_mask = prune_mask | long_lived_unstable

        self._prune_points(prune_mask, gaussian_model, optimizers)
        torch.cuda.empty_cache()

    # ── Topology: maintain buffers across clone / split / prune ──────────────

    def _prune_points(self, mask: torch.Tensor, gaussian_model: VanillaGaussianModel, optimizers: List):
        valid = ~mask
        super()._prune_points(mask, gaussian_model, optimizers)
        self.eta = self.eta[valid]
        if hasattr(self, "birth_step"):
            self.birth_step = self.birth_step[valid]
        if hasattr(self, "stable_error_count"):
            self.stable_error_count = self.stable_error_count[valid]

    def _densification_postfix(self, new_properties: dict, gaussian_model: VanillaGaussianModel, optimizers: List):
        """Called after split and M_c adding. New Gaussians start unstable with current birth_step."""
        n_existing = gaussian_model.n_gaussians
        n_new = next(iter(new_properties.values())).shape[0]
        existing_eta = self.eta.clone()
        existing_birth = self.birth_step.clone() if hasattr(self, "birth_step") else None
        existing_err = self.stable_error_count.clone() if hasattr(self, "stable_error_count") else None

        super()._densification_postfix(new_properties, gaussian_model, optimizers)

        # eta: new Gaussians start at 0 (unstable)
        new_eta = torch.zeros(n_existing + n_new, dtype=torch.long, device=existing_eta.device)
        new_eta[:n_existing] = existing_eta
        self.eta = new_eta

        # B5: birth_step = current step for new Gaussians
        if existing_birth is not None:
            new_birth = existing_birth.new_full((n_new,), self._current_step)
            self.birth_step = torch.cat([existing_birth, new_birth])

        # B4: stable_error_count = 0 for new Gaussians
        if existing_err is not None:
            self.stable_error_count = torch.cat([existing_err, existing_err.new_zeros(n_new)])

    def _densify_and_clone(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List):
        """Clone: children inherit parent's eta and birth_step; stable_error_count resets."""
        from .vanilla_density_controller import VanillaDensityControllerImpl
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )
        axis_ratio = gaussian_model.get_scales().min(dim=1).values / gaussian_model.get_scales().max(dim=1).values
        selected_pts_mask = torch.logical_and(selected_pts_mask, axis_ratio > self.config.axis_ratio_threshold)

        if not selected_pts_mask.any():
            return

        # Save parent buffers before topology changes
        parent_eta = self.eta[selected_pts_mask].clone()
        parent_birth = self.birth_step[selected_pts_mask].clone() if hasattr(self, "birth_step") else None

        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        n_existing = gaussian_model.n_gaussians
        existing_eta = self.eta.clone()
        existing_birth = self.birth_step.clone() if hasattr(self, "birth_step") else None
        existing_err = self.stable_error_count.clone() if hasattr(self, "stable_error_count") else None

        # Bypass our _densification_postfix override to avoid double-management
        VanillaDensityControllerImpl._densification_postfix(self, new_properties, gaussian_model, optimizers)

        n_new = selected_pts_mask.sum().item()

        # eta: clone children inherit parent's eta
        new_eta = torch.zeros(n_existing + n_new, dtype=torch.long, device=existing_eta.device)
        new_eta[:n_existing] = existing_eta
        new_eta[n_existing:] = parent_eta
        self.eta = new_eta

        # B5: clone children inherit parent's birth_step
        if existing_birth is not None and parent_birth is not None:
            new_birth = torch.zeros(n_existing + n_new, dtype=torch.long, device=existing_birth.device)
            new_birth[:n_existing] = existing_birth
            new_birth[n_existing:] = parent_birth
            self.birth_step = new_birth

        # B4: clone children start with 0 error count
        if existing_err is not None:
            new_err = torch.zeros(n_existing + n_new, dtype=torch.long, device=existing_err.device)
            new_err[:n_existing] = existing_err
            self.stable_error_count = new_err

    # ── B4: error-count based stable→unstable reversion ─────────────────────

    def _update_stable_error_count(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel) -> None:
        """Project stable Gaussian centres to the current camera; increment e_i if
        rendered color error at that pixel exceeds stable_revert_threshold."""
        stable_mask = self._stable_mask
        if not stable_mask.any():
            return

        rendered = outputs.get("render")
        if rendered is None:
            return

        camera, image_info, _ = batch
        _, gt_image, _ = image_info
        color_error = (rendered.detach() - gt_image).abs().mean(0)  # (H, W)
        H, W = color_error.shape

        # Project stable Gaussian centres to image plane
        means3D = gaussian_model.get_means()[stable_mask]  # (N_s, 3)
        w2c = camera.world_to_camera  # (4, 4)
        pts_h = torch.cat([means3D, means3D.new_ones(len(means3D), 1)], dim=1)  # (N_s, 4)
        pts_cam = pts_h @ w2c.T  # (N_s, 4)
        z = pts_cam[:, 2]  # (N_s,)

        fx, fy = float(camera.fx), float(camera.fy)
        cx, cy = float(camera.cx), float(camera.cy)
        u = (pts_cam[:, 0] / z.clamp(min=1e-3) * fx + cx).long()
        v = (pts_cam[:, 1] / z.clamp(min=1e-3) * fy + cy).long()

        in_image = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not in_image.any():
            return

        stable_indices = torch.where(stable_mask)[0]
        in_image_global = stable_indices[in_image]
        u_v = u[in_image].clamp(0, W - 1)
        v_v = v[in_image].clamp(0, H - 1)
        err = color_error[v_v, u_v]  # (N_valid,)

        high_err = err > self.config.stable_revert_threshold
        # Increment for high-error Gaussians; decrement (floor 0) for recovering ones
        self.stable_error_count[in_image_global[high_err]] += 1
        self.stable_error_count[in_image_global[~high_err]] = (
            self.stable_error_count[in_image_global[~high_err]] - 1
        ).clamp(min=0)

        # Revert Gaussians whose error count exceeds δ_e
        revert = (self.stable_error_count >= self.config.stable_revert_count) & stable_mask
        if revert.any():
            self.eta[revert] = 0
            self.stable_error_count[revert] = 0
            console_event(f"[RTGStable] B4: {revert.sum().item():,} stable→unstable (e_i >= {self.config.stable_revert_count})")

    # ── B3: M_c transparent Gaussian adding ──────────────────────────────────

    @staticmethod
    def _normal_to_quats(normals: torch.Tensor) -> torch.Tensor:
        """[w,x,y,z] quaternions rotating [0,0,1] → each normal. normals: (N,3) normalised."""
        eps = 1e-7
        z_ax = normals.new_tensor([0., 0., 1.]).expand_as(normals)
        dot = (normals * z_ax).sum(-1).clamp(-1.0, 1.0)          # (N,)
        cross = torch.cross(z_ax, normals, dim=-1)                # (N, 3)
        cross_unit = cross / cross.norm(dim=-1, keepdim=True).clamp(min=eps)

        half = torch.acos(dot.clamp(-1.0, 1.0)) / 2.0
        w = torch.cos(half).unsqueeze(-1)                         # (N, 1)
        xyz = torch.sin(half).unsqueeze(-1) * cross_unit          # (N, 3)
        quats = torch.cat([w, xyz], dim=-1)                       # (N, 4)

        # Degenerate cases
        identity = normals.new_tensor([1., 0., 0., 0.])
        flip_x = normals.new_tensor([0., 1., 0., 0.])
        quats[dot > 1.0 - eps] = identity
        quats[dot < -1.0 + eps] = flip_x
        return quats

    def _add_mc_gaussians(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel,
                          optimizers: List, pl_module) -> None:
        """RTG-SLAM B3: back-project M_c pixels and add α=0.1 transparent Gaussians."""
        if "surf_depth" not in outputs or "render" not in outputs:
            return

        camera, image_info, _ = batch
        _, gt_image, _ = image_info

        rendered = outputs["render"].detach()           # (3, H, W)
        surf_depth = outputs["surf_depth"].detach()     # (1, H, W)
        device = surf_depth.device

        # M_c: geometrically covered pixels with large color error
        color_error = (rendered - gt_image).abs().mean(0)  # (H, W)
        d = surf_depth.squeeze()                            # (H, W)
        mc_mask = (color_error > self.config.mc_color_threshold) & (d > 0.001)

        if not mc_mask.any():
            return

        H, W = d.shape
        fx, fy = float(camera.fx), float(camera.fy)
        cx, cy = float(camera.cx), float(camera.cy)

        # Build pixel grids
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )

        # Camera-space positions at M_c pixels
        x_cam = (u_grid - cx) / fx * d   # (H, W)
        y_cam = (v_grid - cy) / fy * d   # (H, W)
        pts_cam_mc = torch.stack([x_cam, y_cam, d], dim=-1)[mc_mask]  # (N_mc, 3)

        # Camera-to-world transform
        R_c2w = camera.world_to_camera[:3, :3].T  # (3, 3)
        pts_world = pts_cam_mc @ R_c2w.T + camera.camera_center  # (N_mc, 3)

        # Sample mc_sample_ratio of M_c pixels (RTG-SLAM: 5%)
        n_mc = pts_world.shape[0]
        n_add = max(1, int(self.config.mc_sample_ratio * n_mc))
        # Hybrid: 軟帶節流 + 硬 cap 預算限制（B3 是爆量主源，這裡必須擋）
        if self._soft_active(self._current_step):
            n_add = int(n_add * self._soft_keep(gaussian_model.n_gaussians))
        if self.config.cap_max > 0:
            n_add = min(n_add, max(0, self.config.cap_max - gaussian_model.n_gaussians))
        if n_add <= 0:
            return
        idx = torch.randperm(n_mc, device=device)[:n_add]
        pts_world = pts_world[idx]

        # GT color at sampled M_c pixels
        mc_v = v_grid[mc_mask][idx].long().clamp(0, H - 1)
        mc_u = u_grid[mc_mask][idx].long().clamp(0, W - 1)
        rgb = gt_image[:, mc_v, mc_u].T                     # (n_add, 3)
        C0 = 0.28209479177387814
        shs_dc = ((rgb - 0.5) / C0).unsqueeze(1).float()    # (n_add, 1, 3)

        # Depth-adaptive scale: depth/focal * mc_scale_factor (small transparent disc)
        d_sampled = d[mc_mask][idx]
        focal_mean = (fx + fy) / 2.0
        scale_val = (d_sampled / focal_mean * self.config.mc_scale_factor).clamp(min=1e-5)
        log_scale = torch.log(scale_val).unsqueeze(1).expand(-1, 2)  # (n_add, 2)

        # Rotation: disc faces toward camera (camera-space [0,0,-1] in world)
        cam_forward_world = R_c2w[:, 2]   # z-axis of camera in world
        normals = (-cam_forward_world).unsqueeze(0).expand(n_add, -1).float()
        quats = self._normal_to_quats(normals)  # (n_add, 4)

        # Opacity: logit(mc_transparent_alpha)
        alpha = self.config.mc_transparent_alpha
        logit_alpha = float(torch.log(torch.tensor(alpha / (1.0 - alpha))))
        opacities = torch.full((n_add, 1), logit_alpha, device=device, dtype=torch.float32)

        # SH rest: zeros (only DC term carries color)
        rest_dim = gaussian_model.gaussians["shs_rest"].shape[1]
        shs_rest = torch.zeros(n_add, rest_dim, 3, device=device, dtype=torch.float32)

        new_properties = {
            "means":     pts_world.float(),
            "scales":    log_scale.float(),
            "rotations": quats.float(),
            "opacities": opacities,
            "shs_dc":    shs_dc,
            "shs_rest":  shs_rest,
        }

        self._densification_postfix(new_properties, gaussian_model, optimizers)
        console_event(f"[RTGStable] B3: +{n_add:,} transparent Gaussians (M_c={n_mc:,}, err>{self.config.mc_color_threshold:.2f})")
