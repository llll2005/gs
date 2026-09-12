"""
Renderer for Scaffold-2DGS: every frame, ask the model to MLP-generate K neural 2D surfels
per visible anchor (view-dependent color), then rasterize them with the Trim2DGS rasterizer
to produce render + depth/normal/dist maps (so CityGSV2 depth/normal regularization still works).

Subclasses SepDepthTrim2DGSRenderer to reuse the allmap geometry post-processing. Color comes
from the MLP (colors_precomp), NOT spherical harmonics. Trimming is disabled (generated surfels
are not persistent model properties). See internal/models/scaffold_gaussian_2d.py.
"""
import math
import torch
from diff_trim_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer


class Scaffold2DGSRenderer(SepDepthTrim2DGSRenderer):
    def forward(self, viewpoint_camera, pc, bg_color, scaling_modifier=1.0, record_transmittance=False, **kwargs):
        # MLP-generate neural 2D surfels for this view
        xyz, color, opacity, scales, rotations, sel_mask, neural_opacity = pc.generate_neural_surfels(viewpoint_camera)

        screenspace_points = torch.zeros_like(xyz, requires_grad=True, device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

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
            sh_degree=0,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            record_transmittance=record_transmittance,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        output = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        if record_transmittance:
            transmittance_sum, num_covered_pixels, radii = output
            return transmittance_sum / (num_covered_pixels + 1e-6)

        rendered_image, radii, allmap = output
        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            # for anchor growing (ScaffoldDensityController)
            "selection_mask": sel_mask,
            "neural_opacity": neural_opacity,
        }

        # ---- geometry maps (same as SepDepthTrim2DGS) ----
        render_alpha = allmap[1:2]
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_to_camera[:3, :3].T)).permute(2, 0, 1)
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        render_dist = allmap[6:7]
        surf_depth = render_depth_expected * (1 - self.depth_ratio) + self.depth_ratio * render_depth_median
        surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1) * (render_alpha).detach()

        rets.update({
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "view_normal": -allmap[2:5],
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
        })
        return rets
