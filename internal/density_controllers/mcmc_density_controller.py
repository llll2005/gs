"""
3D Gaussian Splatting as Markov Chain Monte Carlo
https://ubc-vision.github.io/3dgs-mcmc/

Most codes are copied from https://github.com/ubc-vision/3dgs-mcmc
"""

from typing import Literal, Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
import math
from lightning import LightningModule
import torch
from gsplat.relocation import compute_relocation
from internal.utils.general_utils import inverse_sigmoid, build_rotation
from internal.utils.gaussian_projection import compute_cov_3d

from .density_controller import DensityController, DensityControllerImpl, Utils


@dataclass
class MCMCDensityController(DensityController):
    cap_max: int
    """
    the maximum number of Gaussians
    """

    noise_lr: float = 5e5

    densify_from_iter: int = 500

    densify_until_iter: int = 25_000

    densification_interval: int = 100

    min_opacity: float = 0.005

    N_max: int = 51
    """
    https://github.com/ubc-vision/3dgs-mcmc/blob/main/utils/reloc_utils.py
    """

    blur_split_budget: float = 0.0
    """
    範圍 [0, 1)。0 = 關（出廠的純 opacity 取樣）。建議 0.3。

    Share of each densification step's budget given to primitives that alone explain a large patch,
    i.e. that are under-reconstructing it. 0 = off (opacity-only, the shipped MCMC behaviour);
    0.3 = 30% of new Gaussians are cloned from over-threshold hosts.
    A SHARE rather than a multiplier because candidates are ~0.01% of the population, so any
    plausible multiplier is swallowed by the rest of the distribution (紀錄 §11.9.1).
    Mini-Splatting arXiv 2403.14166 Eq. 2; see 紀錄/研究總覽.md §11.9.
    """

    blur_split_threshold: float = 288.0
    """
    單位是像素，預設 288 = 2e-4 x 900 x 1600（論文設定套到我方解析度）。
    改解析度時要跟著改；只有「分數/門檻」的比值被使用。

    theta_blur * H * W in pixels. 288 = 2e-4 * 900 * 1600, the paper's setting at our resolution.
    Only the ratio score/threshold is used, so this just sets what "one threshold over" means.
    """

    blur_split_weight: float = 0.0
    """
    DEPRECATED, ignored. Kept only so configs saved by runs from 2026-08-08 still load -- removing
    the field made `main.py test` fail on blursplit_b12 with "Option not accepted". It was a
    multiplier on the sampling probability, which cannot move a population that is 0.01% of the
    total; `blur_split_budget` replaces it. See 紀錄/研究總覽.md §11.9.1.
    """

    long_axis_spread: float = 0.0
    """
    ⛔ 保持 0。已實測證偽，見下方。

    Offset each new Gaussian from its host by up to this fraction of the host's LONGEST axis,
    along that axis. 0 = off (position copied verbatim, the shipped MCMC behaviour).

    ⛔ MEASURED 2026-08-10 AND REFUTED -- keep at 0. `blurlas_b12` (blur budget 0.3 + spread 0.5)
    scored 24.224 / LPIPS 0.4678 / texratio 0.3762 against `blurbudget_b12`'s 24.353 / 0.4674 /
    0.3783: PSNR down 0.129 = 2.4x the measured noise floor, everything else flat.

    Two reasons, both visible in hindsight:
      1. The premise was that stacked children can only separate via SGLD noise, whose scale is
         op_sigmoid(1-o) ~ 0 above opacity 0.03. True, but irrelevant -- only 0.01% of primitives
         in a finished model are bit-identical in position, so they do separate anyway (most likely
         via non-deterministic atomicAdd ordering in the rasteriser, amplified chaotically).
      2. It actively breaks MCMC's Eq. 9. `compute_relocation` derives the corrected opacity and
         scale on the assumption that the N copies are CO-LOCATED, which is what makes their
         combined contribution equal the original's. Displacing them invalidates that, so every
         densification step injects a small image perturbation.

    Kept rather than deleted because this is a real mechanism from the literature (Long-Axis Split,
    ImprovingDensification arXiv 2508.12313 §Methods) and we appear to be the only ones with a
    measured noise floor able to say it does not transfer to MCMC's relocation framework. Useful
    for related work; do not enable. See 紀錄/研究總覽.md §11.9.3.
    """

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        assert self.cap_max > 0, "cap_max must > 0"
        return MCMCDensityControllerImpl(self)


class MCMCDensityControllerImpl(DensityControllerImpl):
    # MCMC densifies via relocate/add_new, never via the viewspace gradient. Declaring it lets
    # gaussian_splatting skip the split backward (see `_split_backward_needed`), which saves an
    # extra backward and a retained graph every densify step.
    READS_VIEWSPACE_GRAD = False

    config: MCMCDensityController

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        # initialize binoms
        N_max = self.config.N_max
        binoms = torch.zeros((N_max, N_max), dtype=torch.float, device=pl_module.device)
        for n in range(N_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)

        self.register_buffer("binoms", binoms, persistent=False)

        # initialize opacities and scales
        if stage == "fit" and pl_module.hparams["initialize_from"] is None:
            self._opacities_and_scales_initialization(pl_module.gaussian_model)
            pl_module.on_train_batch_end_hooks.append(self._add_xyz_noise)

    @staticmethod
    def _opacities_and_scales_initialization(gaussian_model) -> None:
        # looks like it does not affect the final result
        with torch.no_grad():
            gaussian_model.scales.copy_(gaussian_model.scales + math.log(0.1))
            gaussian_model.opacities.copy_(inverse_sigmoid(torch.ones_like(gaussian_model.opacities) * 0.5))

    def after_backward(self, outputs: dict, batch, gaussian_model, optimizers, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return
        if global_step <= self.config.densify_from_iter:
            return
        if global_step % self.config.densification_interval != 0:
            return

        with torch.no_grad():
            dead_mask = (gaussian_model.get_opacities() <= self.config.min_opacity).squeeze(-1)
            # ⚠ 2026-08-25：我曾在這裡加 densify 診斷輸出，但**這個方法被
            # `MCMC2DGSDensityControllerImpl.after_backward` 覆寫**（我方所有跑次都走子類），
            # 所以它從來沒執行過 —— 沉默地量到「什麼都沒有」。要加診斷請加在子類。
            # 該問題最後是用 `tools/trim_audit.py`（零 GPU，從族群曲線反推）解掉的，見 §11.21。
            # replace based on alive Gaussians
            self.relocate_gs(gaussian_model, optimizers, dead_mask)
            self.add_new_gs(gaussian_model, optimizers)

    @staticmethod
    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    def _add_xyz_noise(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        # TODO: really need to add noise till the end of training?

        # prevent adding noise to checkpoint
        if pl_module.is_final_step is True:
            return

        with torch.no_grad():
            cov_3d = compute_cov_3d(
                scales=gaussian_model.get_scales(),
                scale_modifier=1.,
                quaternions=gaussian_model.get_rotations(),
            )

            # TODO: get lr more efficiently
            xyz_lr = -1
            for opt in pl_module.gaussian_optimizers:
                for param_group in opt.param_groups:
                    if param_group["name"] == "means":
                        xyz_lr = param_group["lr"]
                if xyz_lr >= 0:
                    break

            assert xyz_lr >= 0

            noise = torch.randn_like(gaussian_model.means) * (self.op_sigmoid(1 - gaussian_model.get_opacities())) * self.config.noise_lr * xyz_lr
            noise = torch.bmm(cov_3d, noise.unsqueeze(-1)).squeeze(-1)
            gaussian_model.means.add_(noise)

    def compute_relocation(self, opacity_old, scale_old, N) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert torch.all(N <= self.config.N_max)  # whether such a check is necessary?
        return compute_relocation(
            opacity_old,
            scale_old,
            N,
            self.binoms,
        )

    def _get_new_params(self, gaussian_model, idxs, ratio) -> Dict[str, torch.Tensor]:
        # `idxs`: indices of the alive Gaussians
        # `ratio`: sample frequencies of those indices, `ratio[index]=frequency`
        # compute new opacities and scales
        new_opacity, new_scaling = self.compute_relocation(
            opacity_old=gaussian_model.get_opacities()[idxs, 0],  # [N_sample]
            scale_old=gaussian_model.get_scales()[idxs],  # [N_sample]
            N=ratio[idxs, 0] + 1,  # pick the frequencies of those Gaussian to be sampled, [N_sample]
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = gaussian_model.opacity_inverse_activation(new_opacity)
        new_scaling = gaussian_model.scale_inverse_activation(new_scaling.reshape(-1, 3))

        new_params = {
            "opacities": new_opacity,
            "scales": new_scaling,
        }

        # get other properties from model, then add to `new_params`
        for attr_name, value in gaussian_model.properties.items():
            if attr_name not in new_params:
                new_params[attr_name] = value[idxs]

        # Long-axis spread. Every child is copied to the host's exact position above, so N copies
        # start stacked and only SGLD noise separates them -- a random walk that the MCMC paper
        # itself calls irrecoverable once a Gaussian leaves its support region. MCMC's Eq. 9 fixes
        # opacity and scale so the stack renders like the original, but says nothing about
        # position. ImprovingDensification (arXiv 2508.12313) makes exactly this point and places
        # children along the long axis instead.
        # It matters most for the blur-split hosts: those are the primitives that alone cover a
        # large patch, and stacking their children at the centre cannot cover it.
        # 0 = off (shipped behaviour, position copied verbatim).
        spread = getattr(self.config, "long_axis_spread", 0.0)
        means_name = getattr(gaussian_model, "_mean_name", "means")
        if spread > 0 and means_name in new_params:
            scales = gaussian_model.get_scales()[idxs]                       # [M, K]
            long_len, long_ax = scales.max(dim=-1)                           # extent + which axis
            R = build_rotation(gaussian_model.get_rotations()[idxs])         # [M, 3, 3]
            direction = torch.gather(
                R, 2, long_ax.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)   # that axis in world
            # Uniform in [-1, 1] rather than a fixed +-: children are sampled WITH replacement, so
            # there is no copy index to alternate on, and a deterministic offset would move every
            # child of a host to the same place -- back to a stack, just displaced.
            t = torch.rand_like(long_len) * 2.0 - 1.0
            new_params[means_name] = new_params[means_name] + \
                direction * (spread * long_len * t).unsqueeze(-1)

        return new_params

    @staticmethod
    def _sample_alives(probs, num, alive_indices=None):
        # `probs` are opacities of alive Gaussians
        # `alive_indices` are the indices of those alive Gaussians
        # `num` is the number of other Gaussians
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)  # pdf
        # Sample `num` Gaussians from alive part. Higher the opacity, higher the sample times
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        # Count the frequency of each value, `ratio[index]=frequency`
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    @staticmethod
    def replace_tensors_to_optimizers(gaussian_model, optimizers, inds=None):
        # get current parameters
        properties = gaussian_model.properties

        # replace
        new_parameters = Utils.replace_tensors_to_properties(properties, optimizers=optimizers, selector=inds)

        # update
        gaussian_model.properties = new_parameters

    def relocate_gs(self, gaussian_model, optimizers, dead_mask):
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (gaussian_model.get_opacities()[alive_indices, 0])
        # `reinit_idx` are the sampled alive indices; `ratio` are the values of sample frequency, `ratio[index]=frequency`
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        new_params = self._get_new_params(gaussian_model, reinit_idx, ratio=ratio)
        for attr_name in new_params:
            gaussian_model.get_property(attr_name)[dead_indices] = new_params[attr_name]

        # update the opacities and scales of the sampled Gaussians too
        gaussian_model.opacities[reinit_idx] = gaussian_model.opacities[dead_indices]
        gaussian_model.scales[reinit_idx] = gaussian_model.scales[dead_indices]

        # post-processing: update states of optimizer based on `reinit_idx`, and recreate nn.Parameter for the updated tensors
        # NOT a TODO — `reinit_idx` only is CORRECT, verified against the paper (3DGS-MCMC,
        # NeurIPS 2024, §3.4 Implementation): "we reset the moment statistics for the target
        # Gaussian (the original one that is cloned) so that it is biased to stay stationary,
        # while for the new ones (source) we retain the moment statistics to encourage
        # exploration. This is because 'dead' (source) Gaussians are dominated by the noise
        # term in (7)". `reinit_idx` = the sampled live hosts = paper's "target" -> reset.
        # `dead_indices` = the Gaussians that were moved = paper's "source" -> deliberately
        # left alone. Applying it to dead_indices too would delete the exploration mechanism.
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=reinit_idx)

    def add_new_gs(self, gaussian_model, optimizers):
        cap_max = self.config.cap_max

        """
        gradually increase the number of live Gaussians by 5% until the maximum desired number of Gaussians is met
        """
        current_num_points = gaussian_model.n_gaussians
        # 每次 densify 事件的成長倍率。原本寫死 1.05（3dgs-mcmc 原版）；改成可調是為了
        # 「總步數砍半」那類壓縮排程 —— 步數少一半、事件也少一半，若每次變動量不提高，
        # 族群就到不了 cap（研究總覽 §11.11）。預設 1.05 = 原行為。
        target_num = min(cap_max, int(getattr(self.config, "add_ratio", 1.05) * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = gaussian_model.get_opacities().squeeze(-1)

        # ── 成本感知／AbsGS 導向的增生取樣（2026-08-26，§11.34 的結論）─────────────────
        # `vpc_prune_frac 0.05` 慘敗（-2.34 dB）的根因是**接線與診斷不符**：診斷（§11.28）
        # 量的是「**移除**」低 v/c 的成本-價值前緣，但那個旗標把 mask 併進 `dead_mask`
        # ⇒ 實際走 relocation，把粒子丟到依 opacity 取樣的隨機宿主、毀掉它原本在做的事。
        # ⇒ 正確的用法是**決定往哪裡增生（改變集合）**，而不是把既有粒子搬走（破壞集合）。
        # 這裡就是那個插槽 —— `blur_split_budget` 也是用同一個位置。
        #
        # 兩個獨立旗標，預設 0 = 現行行為：
        #   absgrad_densify   probs *= (1 + w * |g|/mean|g|)   AbsGS 訊號，天花板實測 16x（§11.30）
        #   cost_aware_densify probs /= (c/median(c))^w        成本感知，c = 螢幕半徑^2
        # 保留 opacity 當基底：MCMC 的分裂公式 o_new = 1-(1-o)^(1/N) 會把父代 opacity 分給
        # 子代，抽到低 opacity 的父代只會生出更淡的子代 ⇒ 乘法式只改「質量往哪去」。
        _n = probs.shape[0]
        _w = getattr(self.config, "absgrad_densify", 0.0)
        if _w > 0:
            _g = getattr(self, "_absgrad_accum", None)
            if _g is not None and _g.shape[0] == _n and float(_g.sum()) > 0:
                probs = probs * (1.0 + _w * _g / _g.mean().clamp_min(1e-30))
        # w > 0：往**便宜**的地方增生 —— 我方成本感知命題。
        # w < 0：往**貴**的地方增生 —— Taming 3DGS 的方向（2026-08-29 讀原文查出）。
        #   它的 score 有一項 `c^i_g` =「g 在該視角覆蓋的像素數」，權重 **+0.1（正的）**，
        #   理由 "large projections ... lead to a blurry appearance" ⇒ 大足跡該多分裂。
        #   ⚠ 但 0.1 相對 `∇g` 的 50、`D^i_g` 的 50 只佔約 0.2% ⇒ **他們並沒有真的檢驗過
        #     這個方向**，只是給了個近乎 no-op 的小權重加一句敘述。
        #   ⇒ 兩個方向都值得量，負值路徑因此保持可達（原本 `> 0` 會靜默忽略負值）。
        _cw = getattr(self.config, "cost_aware_densify", 0.0)
        if _cw != 0:
            _r = getattr(self, "_max_radii2D", None)
            if _r is not None and _r.shape[0] == _n and float(_r.max()) > 0:
                _c = _r.float().clamp_min(1.0) ** 2
                probs = probs / (_c / _c.median().clamp_min(1e-12)).pow(_cw)
        # Blur split. MCMC picks hosts by opacity, which is blind to WHERE detail is missing: a
        # primitive that alone explains a large patch is under-reconstructing it, and cloning THAT
        # one is what adds detail. Mini-Splatting (arXiv 2403.14166 Eq. 2) splits on exactly that
        # statistic; here it arrives as `blur_score` (sum_p T*alpha per primitive per view) from the
        # trim pass in sep_depth_trim_2dgs_renderer, which already visits every camera, so it costs
        # nothing extra. weight 0 = shipped behaviour, so no existing config changes.
        # ⚠ Premise measured (紀錄 §11.9.1), NOT yet validated on a training run.
        #
        # The knob is a BUDGET SHARE, not a multiplier. A multiplier cannot work here: the
        # candidates are ~0.01% of the population (303 of 3.6M on cap4m), so a 3x boost moves them
        # from 0.0084% to 0.0253% of the sampling mass -- 15 of 60,000 additions, which is what the
        # first attempt measured (blursplit_b12 tracked its control within noise). Mini-Splatting
        # splits ALL over-threshold Gaussians deterministically; a budget share is the closest
        # equivalent that still fits MCMC's sample-with-replacement scheme, and it stays meaningful
        # however few candidates there are.
        share = getattr(self.config, "blur_split_budget", 0.0)
        score = getattr(self, "blur_score", None)
        thr = getattr(self.config, "blur_split_threshold", 288.0)
        if share > 0 and score is not None and score.shape[0] == probs.shape[0]:
            over = score > thr
            m_over, m_rest = probs[over].sum(), probs[~over].sum()
            if m_over > 0 and m_rest > 0:
                # rescale so that mass(over) / total == share
                probs = probs.clone()
                probs[over] *= (share / (1.0 - share)) * m_rest / m_over
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        new_params = self._get_new_params(gaussian_model, add_idx, ratio=ratio)

        gaussian_model.opacities[add_idx] = new_params["opacities"]
        gaussian_model.scales[add_idx] = new_params["scales"]

        # densification postfix for new part
        gaussian_model.properties = Utils.cat_tensors_to_properties(new_params, gaussian_model, optimizers)

        # postfix for selected part
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=add_idx)

        return num_gs
