from typing import Dict, Tuple, Union, Callable, Optional, List

import lightning
import torch
from internal.utils.topk_contribution import contribution_accumulator
import math
from .renderer import Renderer
from .renderer import RendererOutputTypes, RendererOutputInfo, Renderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel

from diff_trim_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer




class SepDepthTrim2DGSRenderer(Renderer):
    def __init__(
            self,
            depth_ratio: float = 0.,
            K: int = 5,
            v_pow: float = 0.1,
            prune_ratio: float = 0.1,
            contribution_prune_from_iter : int = 1000,
            contribution_prune_interval: int = 500,
            contribution_prune_until_iter: int = -1,
            start_prune_ratio: float = 0.0,
            diable_start_trimming: bool = False,
            diable_trimming: bool = False,
            skip_surf_normal: bool = False,
            trim_subsample_probe: int = 0,
    ):
        """`skip_surf_normal`（2026-08-26）：True 時不計算 `surf_normal`。

        `depth_to_normal` 把深度圖反投影成 3D 點再取鄰域叉積，**全幅逐像素且可微**，
        而它唯一的訓練期消費者是 `gs2d_metrics` 的 normal loss。現行配方 `lambda_normal = 0`
        ⇒ 純浪費。（metric 端的閘門已省下 9%，見 研究總覽 §11.35；這是剩下的那塊。）

        ⚠ 失敗模式是**大聲的**：`lambda_normal > 0` 卻設了本旗標 ⇒ gs2d_metrics 取
        `outputs['surf_normal']` 直接 KeyError，不會安靜地算錯。
        ⚠ viewer 的 `surf_normal` 輸出也會消失（只影響互動檢視，不影響訓練/評測）。
        ⚠ 不影響 `surf_depth` —— 那是兩個 allmap 通道的線性混合，很便宜，
          且深度 loss 與 `rtg_stable_density_controller` 都要用。
        """
        super().__init__()

        # hyper-parameters for trimming
        self.depth_ratio = depth_ratio

        self.K = K
        self.v_pow = v_pow
        self.prune_ratio = prune_ratio
        self.contribution_prune_from_iter = contribution_prune_from_iter
        self.contribution_prune_interval = contribution_prune_interval
        # -1 = follow the density controller's densify_until_iter (the shipped behaviour). Set it
        # explicitly to keep harvesting after densify stops -- trimming and densify are separate
        # mechanisms and tying them together makes a "densify off, keep shrinking" schedule
        # impossible to express. See 紀錄/研究總覽.md §4.
        self.contribution_prune_until_iter = contribution_prune_until_iter
        self.start_prune_ratio = start_prune_ratio
        self.diable_start_trimming = diable_start_trimming
        self.diable_trimming = diable_trimming
        self.skip_surf_normal = skip_surf_normal
        # >0 時，每次 trim 事件**額外**用「每 N 台取一台」再算一次 contribution，
        # 報告與全視角版的 Spearman 與底部 10% 遮罩重疊。只讀，不改變剪枝結果。
        # 目的（§11.48）：trim pass 佔 profile 視窗 80%、約 9 小時跑次的 10%，
        # 而 `_measure_multiview_contribution` 早就在用 24 台取樣。若排序一致就能省 90%。
        self.trim_subsample_probe = trim_subsample_probe

    def _depth_ssim_loss(self, a, b):
        pass

    def _get_color_inputs(self, pc, viewpoint_camera, record_transmittance=False):
        """Return (shs, colors_precomp) for the rasterizer; exactly one is non-None.
        Subclasses override this to precompute colors in Python (e.g. SB color).
        record_transmittance passes are color-independent — subclasses may skip
        the color computation there."""
        return pc.get_features, None

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            record_transmittance=False,
            record_coverage=False,
            **kwargs,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            record_transmittance=record_transmittance,
            debug=False,
            # principal-point y shift for K-strip off-center crops (strip_cameras.py);
            # 0.0 for ordinary full-frame cameras -> bit-identical to the old behavior.
            # Guarded for rasterizer builds without the pp_shifty patch.
            **({"pp_shifty": getattr(viewpoint_camera, "strip_pp_shifty", 0.0)}
               if "pp_shifty" in GaussianRasterizationSettings._fields else {}),
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        cov3D_precomp = None
        scales = pc.get_scaling[..., :2]
        rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs, colors_precomp = self._get_color_inputs(pc, viewpoint_camera, record_transmittance)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        output = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        if record_transmittance:
            transmittance_sum, num_covered_pixels, radii = output
            # Per-COVERED-PIXEL mean, not the sum: size-normalised, so a large primitive and a
            # small one with the same per-pixel weight score the same.
            transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
            if record_coverage:
                # What `num_covered_pixels` actually counts (forward.cu:406/452/455/459, read
                # 2026-08-07): the atomicAdd sits BEFORE both the `alpha < 1/255` skip and the
                # `T < 1e-4` early-out, but the whole loop is gated by `!done`. So it counts the
                # (primitive, pixel) evaluations the kernel ACTUALLY PERFORMED -- including
                # partially occluded ones (small T) and negligible-alpha ones, excluding anything
                # behind an already-saturated pixel, which is never reached.
                #
                # ⚠ An earlier comment here called it "frustum membership, including fully
                # occluded" and that is wrong: a primitive hidden behind an opaque surface reads 0,
                # exactly like one outside the frustum. `reduce="frustum_topk"` inherits that
                # limitation. What it IS, is a per-primitive render COST: sum_i c_i is the total
                # blend work, and that is the quantity that predicts VRAM (see
                # tools/measure_overdraw_waste.py and tools/fit_vram_model.py).
                return transmittance, num_covered_pixels
            return transmittance
        else:
            rendered_image, radii, allmap = output

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_to_camera[:3, :3].T)).permute(2, 0, 1)

        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1;
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        surf_depth = render_depth_expected * (1 - self.depth_ratio) + (self.depth_ratio) * render_depth_median

        rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'view_normal': -allmap[2:5],
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
        })
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        # 2026-08-26：`depth_to_normal` 是全幅逐像素且可微（反投影 + 鄰域叉積），
        # 唯一的訓練期消費者是 lambda_normal=0 的 normal loss ⇒ 可跳過。見建構子 docstring。
        # ⚠ 2026-08-28：用 `getattr` 不是 `self.skip_surf_normal` —— ckpt 當 init 時 renderer 是
        # 從 checkpoint **反序列化**的（_ctx 陷阱 1），舊 ckpt 沒有這個屬性 => AttributeError。
        # `opprofile` 就是這樣死的。新增 renderer 屬性一律要對舊 ckpt 保持相容。
        if not getattr(self, "skip_surf_normal", False):
            surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
            surf_normal = surf_normal.permute(2, 0, 1)
            # remember to multiply with accum_alpha since render_normal is unnormalized.
            surf_normal = surf_normal * (render_alpha).detach()
            rets['surf_normal'] = surf_normal

        return rets
    
    def before_training_step(
            self,
            step: int,
            module,
    ):
        if step != 1 or self.diable_trimming or self.diable_start_trimming:
            return
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        device =  module.gaussian_model.get_xyz.device
        push, gather = contribution_accumulator(self.K)
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                push(self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True
                ))

            contribution = gather()
            probe_n = getattr(self, "trim_subsample_probe", 0)
            if probe_n > 0:
                push2, gather2 = contribution_accumulator(self.K)
                used = 0
                for i in range(0, len(cameras), probe_n):
                    m2, _ = self(cameras[i].to_device(device), module.gaussian_model,
                                 bg_color=module._fixed_background_color().to(device),
                                 record_transmittance=True, record_coverage=True)
                    push2(m2); used += 1
                    del m2
                c2 = gather2()

                def _rank(v):
                    r = torch.empty_like(v)
                    r[v.argsort()] = torch.arange(v.numel(), dtype=v.dtype, device=v.device)
                    return r
                ra, rb = _rank(contribution), _rank(c2)
                n = ra.numel()
                rho = float(((ra - ra.mean()) * (rb - rb.mean())).sum()
                            / (ra.std(unbiased=False) * rb.std(unbiased=False) * n).clamp_min(1e-12))
                k = max(1, int(n * self.prune_ratio))
                ma = torch.zeros(n, dtype=torch.bool, device=contribution.device)
                ma[contribution.topk(k, largest=False).indices] = True
                mb = torch.zeros(n, dtype=torch.bool, device=contribution.device)
                mb[c2.topk(k, largest=False).indices] = True
                print(f"[trim-subsample] step={step} 全視角 {len(cameras)} 台 vs 取樣 {used} 台："
                      f"Spearman={rho:.4f} 底部{100*self.prune_ratio:.0f}%遮罩重疊="
                      f"{100*float((ma & mb).sum())/k:.1f}%  "
                      f"（判準：rho>0.99 且 重疊>95% => 可取樣，省約 9%）")
                del c2
            tile = torch.quantile(contribution, self.start_prune_ratio)
            prune_mask = contribution <= tile
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            print("Trimming done.")
        torch.cuda.empty_cache()

    def after_training_step(
            self,
            step: int,
            module,
    ):
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        until = self.contribution_prune_until_iter
        if until < 0:
            until = module.density_controller.config.densify_until_iter
        if self.diable_trimming or (step > until) \
           or (step < self.contribution_prune_from_iter) \
           or (step % self.contribution_prune_interval != 0):
           return
        
        device =  module.gaussian_model.get_xyz.device

        push, gather = contribution_accumulator(self.K)
        blur = None
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                mean, covered = self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True,
                    record_coverage=True,
                )
                push(mean)
                # Mini-Splatting's blur criterion (arXiv 2403.14166 Eq. 2) wants S_i = #pixels where
                # i is the argmax-weight Gaussian. The soft equivalent sum_p T_i*a_i is already
                # here, and both sum over i to the covered-pixel count, so the paper's threshold
                # transfers. Measured on cap4m (紀錄 §11.9.1): 0.01% of primitives exceed it while
                # carrying 12.6% of the blending weight, and 74.6% of that weight sits on textured
                # content -- genuine under-reconstruction, not efficient flat regions.
                raw = mean * covered.float()
                blur = raw if blur is None else blur + raw
                del mean, covered, raw

            contribution = gather()
            if blur is not None:
                module.density_controller.blur_score = blur / max(len(cameras), 1)
            tile = torch.quantile(contribution, self.prune_ratio)
            prune_mask = (contribution <= tile)
            # 診斷（2026-08-25）：`<=` 在有並列時會超剪。零貢獻的來源是 EXACT_SUPPORT ——
            # opacity <= 1/255 的粒子根本不進 binning，貢獻恆為 0。若零貢獻佔比 > prune_ratio，
            # 這一刀就會把它們**全部**剪掉，遠超過名目比例，破平衡公式的 0.9x 假設隨之失效。
            # 只在 trim 事件時算（每 100~500 步一次），成本可忽略。
            n_tot = contribution.numel()
            n_zero = int((contribution <= 0).sum())
            n_cut = int(prune_mask.sum())
            # ⚠⚠ 2026-08-25：我加上面那段診斷時，Edit 把這一行連同 print 一起換掉了，
            # 等於「trim 完全不生效」跑了兩個 probe（probe_t500 / probe_t250，結果已作廢）。
            # 診斷只能「加」，絕對不能碰到會改變狀態的呼叫。
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            print(f"Trimming done. step={step} N={n_tot} 剪={n_cut} ({100.0*n_cut/max(n_tot,1):.2f}%, "
                  f"名目 {100.0*self.prune_ratio:.0f}%) 零貢獻={n_zero} ({100.0*n_zero/max(n_tot,1):.2f}%) "
                  f"門檻={float(tile):.3e}")
        torch.cuda.empty_cache()

    @staticmethod
    def depths_to_points(view, depthmap):
        device = view.world_to_camera.device
        c2w = (view.world_to_camera.T).inverse()
        W, H = view.width, view.height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, W / 2],
            [0, H / 2, 0, H / 2],
            [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ view.full_projection
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T

        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
        return points

    @classmethod
    def depth_to_normal(cls, view, depth):
        """
            view: view camera
            depth: depthmap
        """
        points = cls.depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            'render_alpha': RendererOutputInfo("rend_alpha", type=RendererOutputTypes.GRAY),
            'render_normal': RendererOutputInfo("rend_normal", type=RendererOutputTypes.NORMAL_MAP),
            'view_normal': RendererOutputInfo("view_normal", type=RendererOutputTypes.NORMAL_MAP),
            'render_dist': RendererOutputInfo("rend_dist", type=RendererOutputTypes.GRAY),
            'surf_depth': RendererOutputInfo("surf_depth", type=RendererOutputTypes.GRAY),
            'surf_normal': RendererOutputInfo("surf_normal", type=RendererOutputTypes.NORMAL_MAP),
        }
