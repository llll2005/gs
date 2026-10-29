from typing import Dict, Tuple, Union, Callable, Optional, List

import os

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
            exact_conic_aabb: bool = False,
            trim_by_value_per_cost: bool = False,
            trim_value_per_cost_alpha: float = 1.0,
            trim_by_tile_topk: bool = False,
            trim_tile_k_soft: float = 32.0,
            lean_train: bool = False,
            record_reduce: bool = False,
            tile_cull: bool = False,
            absgrad_gate: bool = False,
            bwd_sat_skip: bool = False,
            bwd_reduce: bool = False,
            bwd_pair_bit: bool = False,
            trim_running_vc: bool = False,
            trim_soft_rho: float = 0.0,
            trim_schedule: str = "",
            trim_by_mean_contribution: bool = False,
            trim_cost_beta_px: float = 0.0,
            sh_split: bool = False,
            sh_adam_in_backward: bool = False,
            reseed_views: int = 0,
            reseed_frac: float = 1.0,
            reseed_by_error: bool = True,
            reseed_opacity: float = 0.5,
            reseed_scale_px: float = 1.5,
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

        # ★ 2026-10-02 `lean_train`：剃除訓練與 trim pass 裡「算了但沒人用」的光柵器計算
        #   （光柵器 lean_render，見 cuda_rasterizer/forward.cu 的 MODE、backward.cu 的 GEOM）。
        #   訓練 forward 只算顏色＋alpha（不算深度／法線／中位深度／distortion 6 個通道），
        #   backward 跳過幾何梯度（含每個 (像素,顆粒) 配對 3 個加 0 的 atomicAdd），
        #   Python 端也不做法線轉換／nan_to_num 等全幅後處理；trim pass（record）只累積 T*alpha 與覆蓋數。
        #   驗證／測試／檢視器走一般 forward，輸出不變。
        #   ⚠ 失敗模式是大聲的：normal／dist／depth 任一權重 > 0 時，取幾何輸出會 KeyError，
        #     或在 backward 被光柵器擋下（RuntimeError），不會安靜地算錯。
        self.lean_train = lean_train
        self._lean_announced = False
        # ★ 2026-10-03 速度旗標（光柵器，見 forward.cu recordReduceCUDA、rasterizer_impl.cu tileMayContribute）：
        #   record_reduce  trim pass 的 record 在 block 內先加總，每個 (tile,顆粒) 只做一次 global atomic
        #                  （覆蓋數逐位元相同；T*alpha 只差浮點加總順序 => trim 判準邊界上可能差幾顆）
        #   tile_cull      非 record 的呼叫只綁「可能 alpha >= 1/255」的 tile => 渲染逐位元相同、少三~四成配對
        #                  ⚠ `tiles`（逐顆 tile 數）會變成剔除後的數 => exact_tile_cost 類機制的成本語意會變
        self.record_reduce = record_reduce
        self.tile_cull = tile_cull
        # ★ 2026-10-09 `absgrad_gate`：光柵器 backward 每個有效 (像素,顆粒) 配對多做一次 atomicAdd 累加 |dL/ds|（AbsGS），
        #   只有 density controller 在增生期（absgrad_densify > 0，或 harvest_relocate）才讀它；收割期與 absgrad_densify=0 時
        #   照算沒人用。開啟後只在需要的步驟才算（光柵器 absgrad 欄位）；其他梯度逐位元不受影響（.z 從沒被讀）。
        self.absgrad_gate = absgrad_gate
        # ★ 2026-10-09 backward 最佳化（光柵器 backward.cu renderCUDA_opt；只在 lean_train 開時生效）：
        #   bwd_sat_skip  從 tile 內最大的最後貢獻者開始，飽和後的配對不再整批載入（處理的配對與順序完全相同）
        #   bwd_reduce    每個 (像素,顆粒) 的梯度先在 warp／block 內加總，每批每顆只做一次全域 atomic（浮點加總順序級）
        self.bwd_sat_skip = bwd_sat_skip
        self.bwd_reduce = bwd_reduce
        # ★ 2026-10-16 配對貢獻 bit（使用者「試試」）：forward（lean 訓練）記每個 (tile,顆粒) 配對有沒有被任一像素混合，
        #   backward 只處理被混合過的（其他配對對每個像素都是 valid=false => 梯度不變，只省掉重算交點與指數的工作）
        self.bwd_pair_bit = bwd_pair_bit
        # ★ 2026-10-16 取代 trim（使用者「直接實作」）：
        #   trim_running_vc  lean 訓練 forward 順便累加逐顆 Σ T·α（V）與覆蓋數（C），週期 trim 改用這段期間的累積值算 V/C，
        #                    不再每 500 步掃全部訓練相機（增生期每步約 30% 的時間）。這段期間沒被任何訓練視角看到的（C=0）這輪不剪。
        #                    需要 lean_train 與 trim_by_mean_contribution（判準 V/C）。
        #   trim_soft_rho    >0 => GaussianSpa 式軟投影取代硬剪：每次 trim 把 V/C 前 (1-prune_ratio) 的顆粒當支撐集，
        #                    支撐外的在 loss 加 rho/2·mean((o+λ)²)（λ 每次投影累加 o）把 opacity 拉向 0、由 MCMC 回收；
        #                    增生期最後一次 trim 把仍在支撐外的硬剪掉。出處：GaussianSpa（ADMM 稀疏化）＋判準換成我方 V/C。
        self.trim_running_vc = trim_running_vc
        # ★ 2026-10-24 SH 分開存放／SH 的 Adam 融進 backward（紀錄/加速論文對照.md A5、B2；光柵器 cuda_rasterizer/sh_adam.h）
        #   sh_split             訓練步把 shs_dc (N,1,3) 與 shs_rest (N,15,3) 分開交給光柵器，不再每步 torch.cat 成 (N,16,3)
        #                        （本機 4050、1.29M 顆：cat 的 forward＋backward 9.1 ms／步）。渲染逐位元相同。
        #   sh_adam_in_backward  （需 sh_split＋FusedAdam）SH 兩組的 Adam 在光柵器 backward 裡就地做完，不寫出 SH 梯度；
        #                        更新式與 FusedAdam 逐條相同。差別：densify 那一步（每 150 步）SH 照樣更新（原流程那步所有參數都不更新）。
        self.sh_split = sh_split
        self.sh_adam_in_backward = sh_adam_in_backward
        # ★ 2026-10-24 trim 後從渲染深度撒點（使用者 10-24「每次 trim 之後用 Mini-Splatting2 的方法撒點」；紀錄/加速論文對照.md C6）
        #   reseed_views > 0：累加式 trim 每剪一次（增生期），挑這麼多個訓練視角，用渲染的中位深度（depth_ratio=1 時的 surf_depth；
        #   Mini-Splatting2 用的是射線上權重最大那顆的深度）把像素反投影成新顆粒，數量＝本次剪掉的顆數 × reseed_frac（不超過 cap）。
        #   像素只取 alpha > 0.5 的；reseed_by_error=True 依 |render − GT|（三通道平均）加權抽，否則均勻抽；只留本塊分區盒內的點。
        #   新顆粒：顏色＝GT 像素（SH dc，高階 0）、opacity＝reseed_opacity、兩軸大小＝深度/焦距 × reseed_scale_px（約這麼多像素寬）、
        #   面朝渲染法線（rend_normal，世界座標）；Adam 狀態補 0（Utils.cat_tensors_to_properties）。
        #   與 MCMC 的關係：剪掉的容量原本由 add_new_gs（複製 opacity x absgrad 抽到的母點）補回；撒點先補，MCMC 照常補到 cap。
        self.reseed_views = reseed_views
        self.reseed_frac = reseed_frac
        self.reseed_by_error = reseed_by_error
        self.reseed_opacity = reseed_opacity
        self.reseed_scale_px = reseed_scale_px
        self.trim_soft_rho = trim_soft_rho
        self._acc_stop = False
        # ★ 2026-10-09 `trim_schedule`＝"步數:門檻,..."（使用者的三次門檻剪）：只在這幾步 trim，剪掉 v <= 門檻的（v＝現行判準：
        #   最佳視角的每像素平均 T·α），不再固定比例、也不再每 500 步。空字串＝原本的週期 trim。
        self.trim_schedule = trim_schedule
        self._absgrad_announced = False

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
        # impossible to express. See 紀錄/new_archived/研究總覽_v1_至2026-09-13.md §4.
        self.contribution_prune_until_iter = contribution_prune_until_iter
        self.start_prune_ratio = start_prune_ratio
        self.diable_start_trimming = diable_start_trimming
        self.diable_trimming = diable_trimming
        self.skip_surf_normal = skip_surf_normal
        # ★★ 2026-09-13：週期 trim 的剪枝判準從**價值** v 換成**每單位成本的價值** v/c。
        #   依據（全部在修正後的資料上量的）：
        #     ① rho(v,c)≈0（三個模型：-0.006/+0.030/-0.050）=> v 與 c **解耦**
        #        => 按 v/c 排序選出與按 v **不同**的一批（top10% 重疊只有 54~67%）
        #     ② 按 v/c 貪婪比按 v 貪婪，在同成本預算下多拿到的總價值：
        #        預算 10% => +48~57%／25% => +21~27%／50% => +5~9%／75% => +1~2%
        #     ③ 按 contribution 排序剪枝只比**隨機**剪枝好 **1.27 倍**（MSE@50%）
        #        => 現行判準本身就很弱
        #   ⇒ c_i 本來就在同一個 pass 裡算好（`cost_acc`）=> **零額外成本**的判準替換。
        #   ⚠ 這是「破壞集合」那一側；記憶記著 vpc_prune_frac 的 -2.34 dB 是把低 v/c 的粒子
        #     **搬走**（relocate）——移除與搬移不是同一件事，但先驗要謹慎。
        # ★ 2026-09-21：精確圓錐不對稱外接盒。上游的 `truncated_R * extent(1)` 是線性化，
        #   在 88.7% 的（顆粒,相機）組合上高估（白付 tile）、11.3% 上低估（漏覆蓋，最大 73,491 px）。
        #   離線 Σtile -41.2%、本機 forward -24.5%。⚠ **會改變渲染輸出**（把漏掉的覆蓋補回來）
        #   => 不是免費的，要 A/B 驗品質；預設關。
        self.exact_conic_aabb = exact_conic_aabb
        self.trim_by_value_per_cost = trim_by_value_per_cost
        # alpha：把「對成本計價的強度」連續化。score = v / c^alpha
        #   alpha=0 即現行判準（等同關閉）、alpha=1 即純 v/c。
        # 2026-09-13 離線掃描（gate15000@15k，`tools/binning_and_trim_probe.py`）：
        #   「每犧牲一單位 Σv 回收的 Σc」在 **alpha=0.5 有峰**（41.09，現行判準的 2.3 倍），
        #   alpha=1 已越過峰（27.59），而 alpha=1 端到端實測輸基準 **-0.57 dB**
        #   且是「紋理比↑ 光度全↓」的虛假細節簽名 => 病因像是計價過猛，不是方向錯。
        # ⚠ 保留布林旗標而不是換成純 alpha：舊 ckpt 存了 renderer 的 init args，
        #   移除參數會讓那些 ckpt 載入時炸掉（本檔踩過「ckpt 覆蓋 renderer 旗標」）。
        self.trim_value_per_cost_alpha = trim_value_per_cost_alpha
        self._vpc_announced = False
        # ★ 2026-10-01：週期 trim 判準改成「逐 tile 名次」（使用者提的 tile 局部背包）。
        #   每顆記下它在所有 (視角, tile) 裡拿到的**最好名次**（internal/utils/tile_topk.py），
        #   剪掉最好名次最差的 prune_ratio（並列時再依貢獻）。事後剪枝（未重訓）量到：同 N 下 Load 最低、
        #   同成本下品質最好（紀錄/實驗分析/06_事後剪枝與fine-tune.md）。與 trim_by_value_per_cost 同時開時以本旗標為準。
        #   ⚠ 用 getattr 讀：ckpt 反序列化的舊 renderer 沒有這個屬性。
        self.trim_by_tile_topk = trim_by_tile_topk
        # 2026-10-01：與 trim_by_value_per_cost **同時開**時合成一個判別式（背包觀點：v/c＝全場總預算、
        #   tile 名次＝每個 tile 的容量上限）：score = (v / c^α) / (1 + 最好名次 / K_soft)
        self.trim_tile_k_soft = trim_tile_k_soft
        self._tk_announced = False
        # ★ 2026-10-12：trim 判準改成「總貢獻 ÷ (β·視角數 + 覆蓋像素)」（`紀錄/new_archived/第五批_廢案_2026-10-15/13_舊版_trim判準與opacity熵的負交互.md` §4 提案 A＋B；結果 `實驗分析/13`）。
        #   V_i = Σ_視角 Σ_像素 T·α（迴圈裡的 `blur`，本來就在算）；C_i = Σ_視角 覆蓋像素（`cost_acc`）。
        #   β=0 時 V/C ＝「各視角每像素貢獻、以覆蓋像素加權的平均」；現行 v 是「最好那個視角」的同一個量。
        #   ⇒ 與 v/c 不同，它**沒有**對大小懲罰兩次（v/c ∝ 1/大小，讓微小顆粒永遠不被剪，舊版 13 §3）。
        #   β（像素單位，每顆每視角的固定成本）讓「幾乎不畫東西的顆粒」不再免費。
        #   離線比賽（tools/discriminant_race.py，b12 單次剪枝、held-out 本塊區域、同成本預算）：
        #   V/C 在保留 90/70/50% 成本時都第一，較現行 v +0.07/+0.24/+0.10 dB，v/c −0.26/−1.83/−5.98（2026-10-12）。
        #   與 trim_by_value_per_cost／trim_by_tile_topk 同時開時以本旗標為準。
        self.trim_by_mean_contribution = trim_by_mean_contribution
        self.trim_cost_beta_px = trim_cost_beta_px
        self._mc_announced = False
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

        # ⚠ 舊 ckpt 還原出來的 renderer 是**直接反序列化**的物件，沒有後來新增的屬性
        #   => 一律用 getattr 取（稽核清單「ckpt 覆蓋 renderer 旗標」同一類）。
        _conic = getattr(self, "exact_conic_aabb", False)
        # lean：訓練步（training_forward 傳 _lean=True）與 record（trim pass）才用；驗證／測試走完整輸出
        _lean = bool(getattr(self, "lean_train", False)) and (bool(kwargs.get("_lean", False)) or record_transmittance)
        # ★ 2026-10-29 這一步要幾何通道（normal loss）=> 光柵器不走 lean（其餘照舊：kwargs["_lean"] 仍標示這是訓練步，SH 分開存放／Adam 進 backward 不受影響）
        if kwargs.get("_geo", False) and not record_transmittance:
            _lean = False
        # ★ 2026-10-19 dist 版 lean（使用者 10-19：dist 臂也要有 lean 的最佳化，時間才可比）：training_forward 在 lambda_dist > 0 時設 _lean_dist
        _lean_dist = _lean and (not record_transmittance) and bool(kwargs.get("_lean_dist", False))
        if _lean_dist and "lean_dist" not in GaussianRasterizationSettings._fields:
            raise RuntimeError("lambda_dist > 0 且開了 lean_train，但光柵器不認得 lean_dist 欄位 —— 沒重編（或關掉 lean）")
        if _lean and "lean_render" not in GaussianRasterizationSettings._fields:
            raise RuntimeError(
                "設了 lean_train 但光柵器不認得 lean_render 欄位 —— 沒重編。\n"
                "    rm -rf submodules/diff-surfel-rasterization-trim-pp/build 後重裝（lab 要強制 CUDA_HOME）")
        if _conic and "exact_conic_aabb" not in GaussianRasterizationSettings._fields:
            raise RuntimeError(
                "設了 exact_conic_aabb 但光柵器不認得這個欄位 —— 沒重編。\n"
                "    rm -rf submodules/diff-surfel-rasterization-trim-pp/build 後重裝\n"
                "    （CLAUDE.md『Rebuilding the trim rasterizer』；lab 要強制 CUDA_HOME）")
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
            # ★ 2026-09-21 精確圓錐不對稱外接盒。同樣做欄位存在性守衛：光柵器沒重編時
            #   **不會靜默失效**，而是下面的 assert 會當場擋住（設了卻拿不到 = 量錯的來源）。
            **({"exact_conic_aabb": _conic}
               if "exact_conic_aabb" in GaussianRasterizationSettings._fields else {}),
            # 2026-09-22：tiles 改成可選（保護共用環境裡其他 checkout 的 3 元組介面）
            **({"return_tiles": True}
               if "return_tiles" in GaussianRasterizationSettings._fields else {}),
            **({"lean_render": True} if _lean else {}),
            **({"lean_dist": True} if _lean_dist else {}),
            **({"bwd_sat_skip": True} if (_lean and not record_transmittance and getattr(self, "bwd_sat_skip", False)) else {}),
            **({"bwd_reduce": True} if (_lean and not record_transmittance and getattr(self, "bwd_reduce", False)) else {}),
            **({"bwd_pair_bit": True} if (_lean and not record_transmittance and getattr(self, "bwd_pair_bit", False)) else {}),
            **({"train_record": True} if (_lean and not record_transmittance and getattr(self, "trim_running_vc", False)
                                          and not getattr(self, "_acc_stop", False)) else {}),
            **({"absgrad": bool(kwargs.get("_absgrad", True))}
               if (getattr(self, "absgrad_gate", False) and "absgrad" in GaussianRasterizationSettings._fields) else {}),
            # 純量測（tools/tile_budget_audit.py）：每個 (tile, 顆粒) 配對的幾何有效／遮擋可達／逐像素工作量
            **({"audit_tiles": True} if kwargs.get("_audit", False) else {}),
            **({"record_reduce": True} if (record_transmittance and getattr(self, "record_reduce", False)) else {}),
            **({"tile_cull": True} if (not record_transmittance and not kwargs.get("_audit", False)
                                       and getattr(self, "tile_cull", False)) else {}),
        )
        if getattr(self, "trim_running_vc", False) and "train_record" not in GaussianRasterizationSettings._fields:
            raise RuntimeError("設了 trim_running_vc 但光柵器不認得 train_record 欄位 —— 沒重編")
        for _f in ("bwd_sat_skip", "bwd_reduce", "bwd_pair_bit"):
            if getattr(self, _f, False) and _f not in GaussianRasterizationSettings._fields:
                raise RuntimeError(f"設了 {_f} 但光柵器不認得這個欄位 —— 沒重編")
        if getattr(self, "absgrad_gate", False) and "absgrad" not in GaussianRasterizationSettings._fields:
            raise RuntimeError("設了 absgrad_gate 但光柵器不認得 absgrad 欄位 —— 沒重編")
        for _f in ("record_reduce", "tile_cull"):
            if getattr(self, _f, False) and _f not in GaussianRasterizationSettings._fields:
                raise RuntimeError(f"設了 {_f} 但光柵器不認得這個欄位 —— 沒重編")
        if kwargs.get("_audit", False) and "audit_tiles" not in GaussianRasterizationSettings._fields:
            raise RuntimeError("_audit 需要有 audit_tiles 欄位的光柵器（lean 驗證時另外裝的那份，用 PYTHONPATH 指過去）")

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
        # ★ 2026-10-24 SH 分開存放（只在訓練步、基底的 SH 顏色路徑、沒有塊外池時）
        _sh_adam = kwargs.get("_sh_adam", None)
        shs_rest = None
        if (getattr(self, "sh_split", False) or _sh_adam is not None) and bool(kwargs.get("_lean", False)) \
                and not record_transmittance and colors_precomp is None \
                and type(self)._get_color_inputs is SepDepthTrim2DGSRenderer._get_color_inputs \
                and getattr(pc, "_outside_pool", None) is None and hasattr(pc, "shs_dc"):
            import diff_trim_surfel_rasterization as _R
            if not getattr(_R, "SH_SPLIT_SUPPORTED", False):
                raise RuntimeError("設了 sh_split／sh_adam_in_backward 但光柵器不支援 —— 沒重編（rm -rf build 後重裝；lab 用 lab_pull.sh）")
            shs, shs_rest = pc.shs_dc, pc.shs_rest
            if not getattr(self, "_sh_split_announced", False):
                self._sh_split_announced = True
                print(f"[sh-split] ✅ 首次觸發：SH 分開交給光柵器 dc {tuple(shs.shape)}＋rest {tuple(shs_rest.shape)}"
                      + ("；SH 的 Adam 在 backward 裡做" if _sh_adam is not None else ""), flush=True)
        elif _sh_adam is not None:
            raise RuntimeError("sh_adam_in_backward：這一步走不了 SH 分開存放（非訓練步／顏色預先算好／塊外池）=> SH 會漏掉更新")

        # ★ 2026-10-10 塊外池（internal/utils/outside_pool.py）：可訓練集合之後串接池內顆粒（SH0、不受 MCMC 管）。
        #   means2D 的池那段是不帶梯度的 0 => screenspace_points 只收到可訓練那 N_in 列的梯度；逐顆輸出下面切回前 N_in 筆。
        _pool = getattr(pc, "_outside_pool", None)
        n_in = None
        if _pool is not None and _pool.n > 0:
            if shs is None:
                raise NotImplementedError("塊外池只支援 SH 顏色路徑（colors_precomp 的 renderer 未實作）")
            _pool.to(means3D.device)
            n_in = means3D.shape[0]
            pxyz, pop, psc, prot, pfeat = _pool.render_tensors(shs.shape[1])
            means3D = torch.cat([means3D, pxyz], 0)
            means2D = torch.cat([means2D, torch.zeros_like(pxyz)], 0)
            opacity = torch.cat([opacity, pop], 0)
            scales = torch.cat([scales, psc[..., :scales.shape[-1]]], 0)
            rotations = torch.cat([rotations, prot], 0)
            shs = torch.cat([shs, pfeat.to(shs.dtype)], 0)
            if not getattr(self, "_pool_announced", False):
                self._pool_announced = True
                print(f"[outside-pool] ✅ 首次串接：可訓練 {n_in:,} 顆＋塊外池 {_pool.n:,} 顆"
                      f"（{'可訓練 SH0' if _pool.trainable else '凍結 SH0'}，{_pool.bytes_per_point()} B／顆）", flush=True)

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
            **({"shs_rest": shs_rest, "sh_adam": _sh_adam} if shs_rest is not None else {}),
        )

        if kwargs.get("_audit", False):
            return output          # (reached_tiles, useful_tiles, radii, 逐像素 [迴圈次數, tile 配對數, 混合次數], tiles)
        _tiles_out = None
        if n_in is not None and not kwargs.get("_audit", False):
            # 逐顆輸出切回可訓練集合（trim／density controller 只認得它）；影像類輸出（render、allmap）不動
            if not record_transmittance and len(output) == 4:
                _tiles_out = output[3][n_in:]          # 池的 binning 成本另外交出去（離線 Load 要加上，否則低估）
            if record_transmittance:
                output = tuple(o[:n_in] for o in output)
            else:
                output = (output[0], output[1][:n_in], output[2]) + tuple(o[:n_in] for o in output[3:])
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
                if kwargs.get("record_radii", False):      # 2026-10-01：trim_by_tile_topk 要每顆的外接盒半徑
                    return transmittance, num_covered_pixels, radii
                return transmittance, num_covered_pixels
            return transmittance
        else:
            # 光柵器未重編（沒有 return_tiles 欄位）時回傳 3 個 => 相容解包
            if len(output) == 6:
                rendered_image, radii, allmap, tiles, _ta, _cn = output
                self._acc_add(pc, _ta, _cn)
            elif len(output) == 4:
                rendered_image, radii, allmap, tiles = output
            else:
                rendered_image, radii, allmap = output; tiles = None

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            # ★ 2026-09-21：逐顆 tile 數＝**精確的 binning 成本**（這個視角）。
            #   取代 `radii^2` 代理：後者假設正方形、忽略 tile 量化、忽略螢幕裁切，
            #   而且在 EXACT_CONIC_AABB 開啟後 radii 已與真實盒子脫鉤（刻意凍結）。
            #   ⚠ 它是**逐視角**的量；要當逐顆成本用必須自己決定窗口（max / mean / 單視角）。
            **({"tiles": tiles} if tiles is not None else {}),
            **({"tiles_outside": _tiles_out} if _tiles_out is not None else {}),
        }

        # additional regularizations
        render_alpha = allmap[1:2]

        if _lean:
            # 只有顏色與 alpha 有算；其餘幾何通道是 0，不能交出去
            if not self._lean_announced:
                self._lean_announced = True
                print("[lean] ✅ 首次觸發：訓練 forward 只算顏色＋alpha、backward 跳過幾何梯度、trim pass 只算 T*alpha 與覆蓋數", flush=True)
            rets['rend_alpha'] = render_alpha
            if _lean_dist:
                rets['rend_dist'] = allmap[6:7]   # MODE 5 有算 distortion（算式同完整路徑）
                if not getattr(self, "_lean_dist_announced", False):
                    self._lean_dist_announced = True
                    print("[lean-dist] ✅ 首次觸發：lean forward 多算 distortion、backward 加它的梯度（飽和跳過／先加總照用）", flush=True)
            return rets

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
        # 從 checkpoint **反序列化**的（紀錄/new_archived/_ctx.md 陷阱 1），舊 ckpt 沒有這個屬性 => AttributeError。
        # `opprofile` 就是這樣死的。新增 renderer 屬性一律要對舊 ckpt 保持相容。
        if not getattr(self, "skip_surf_normal", False) or kwargs.get("_geo", False):
            surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
            surf_normal = surf_normal.permute(2, 0, 1)
            # remember to multiply with accum_alpha since render_normal is unnormalized.
            surf_normal = surf_normal * (render_alpha).detach()
            rets['surf_normal'] = surf_normal

        return rets
    
    def training_forward(self, step, module, viewpoint_camera, pc, bg_color, render_types=None, **kwargs):
        if getattr(self, "absgrad_gate", False) and "_absgrad" not in kwargs:   # 呼叫端明確指定時（驗證工具）不覆寫
            # 誰會讀 |dL/ds|：MCMC 的 add_new_gs（增生期）、harvest_relocate（收割期續搬）、absgrad_report（只印）
            cfg = getattr(getattr(module, "density_controller", None), "config", None)
            w = float(getattr(cfg, "absgrad_densify", 0.0) or 0.0)
            rep = int(getattr(cfg, "absgrad_report", 0) or 0)
            until = int(getattr(cfg, "densify_until_iter", 0) or 0)
            need = rep > 0 or (w > 0 and (step < until or bool(getattr(cfg, "harvest_relocate", False))))
            kwargs["_absgrad"] = need
            if not self._absgrad_announced and not need:
                self._absgrad_announced = True
                print(f"[absgrad-gate] ✅ 首次關閉 absgrad 累加：step {step}（absgrad_densify={w}、densify_until={until}）")
        if "_lean_dist" not in kwargs:
            # ★ 2026-10-19：深度失真正則有作用的步 => dist 版 lean（其他幾何 loss 若 > 0，光柵器 backward 會當場報錯）
            mcfg = getattr(getattr(module, "metric", None), "config", None)
            lam = float(getattr(mcfg, "lambda_dist", 0.0) or 0.0)
            frm = int(getattr(mcfg, "dist_regularization_from_iter", 0) or 0)
            # ⚠ metric 收到的是 trainer.global_step + 1（gaussian_splatting.training_step），renderer 這裡是 trainer.global_step
            #   => 用 step + 1 判斷，且寧可早一步開（多算一步 distortion 無害；少算一步 metric 會讀不到 rend_dist）
            _di = int(getattr(mcfg, "lambda_dist_interval", 1) or 1)       # ★ 10-24 dist 每 k 步（與 metric 同一規則：metric 的 step＝這裡的 step + 1）
            kwargs["_lean_dist"] = lam > 0 and (step + 1) >= frm and (_di <= 1 or (step + 1) % _di == 0)
        if "_geo" not in kwargs:
            # ★ 2026-10-29 normal loss 有作用的步 => 要幾何通道（rend_normal、surf_normal）：該步不走 lean（與 metric 同一規則：metric 的 step＝這裡的 step + 1）
            mcfg = getattr(getattr(module, "metric", None), "config", None)
            ln = float(getattr(mcfg, "lambda_normal", 0.0) or 0.0)
            nfr = int(getattr(mcfg, "normal_regularization_from_iter", 0) or 0)
            _ni = int(getattr(mcfg, "lambda_normal_interval", 1) or 1)
            kwargs["_geo"] = ln > 0 and (step + 1) > nfr and (_ni <= 1 or (step + 1) % _ni == 0)
            if kwargs["_geo"] and not getattr(self, "_geo_announced", False):
                self._geo_announced = True
                print(f"[normal-geo] ✅ 首次觸發 step={step}：這一步光柵器不走 lean、算 surf_normal（normal 每 {_ni} 步）", flush=True)
        if getattr(self, "sh_adam_in_backward", False) and "_sh_adam" not in kwargs:
            # ★ 2026-10-24 SH 的 Adam 融進 backward：向持有 shs_dc／shs_rest 的 FusedAdam 拿本步的 m／v 與超參數（同一步只拿一次）
            #   10-24 使用者要求預設開（live config）=> 條件不符時退回「只分開存放」並印一次 ⚠（不靜默、不中斷）
            opts = [o for o in (getattr(module, "gaussian_optimizers", None) or []) if hasattr(o, "sh_adam_args")
                    and any(g.get("name") == "shs_rest" for g in o.param_groups)]
            _pool = getattr(pc, "_outside_pool", None)
            why = ("optimizer 不是 FusedAdam（或 shs_dc／shs_rest 不在同一個 optimizer）" if len(opts) != 1
                   else "FusedAdam 是稀疏模式" if getattr(opts[0], "sparse", False)
                   else "有塊外池（SH 要串接池內顆粒）" if (_pool is not None and _pool.n > 0) else "")
            if why:
                if not getattr(self, "_sh_adam_fallback_warned", False):
                    self._sh_adam_fallback_warned = True
                    print(f"⚠ [sh-adam-bwd] 不能在 backward 裡做 SH 的 Adam：{why} => 退回一般 optimizer.step（sh_split 照舊）", flush=True)
            else:
                kwargs["_sh_adam"] = opts[0].sh_adam_args(step_key=step)
        return self(viewpoint_camera=viewpoint_camera, pc=pc, bg_color=bg_color, render_types=render_types,
                    _lean=True, **kwargs)

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
            # ⚠ 環境變數優先於實例屬性：`--model.initialize_from <某.ckpt>` 會把 renderer
            # **整個從 ckpt 反序列化回來**，CLI 上的 `--model.renderer.init_args.*` 因此
            # 完全無效（config.yaml 仍會記下「要求值」，所以光看 config 查不出來）。
            # 2026-08-29 的 trimsub 就是這樣白跑 370 秒：`Trimming...` 有印、探針沒印。
            # 同一個陷阱先前讓 coarseft 實際上「完全不 trim」而歸因錯誤。
            probe_n = int(os.environ.get("TRIM_SUBSAMPLE_PROBE", "0") or 0) \
                or getattr(self, "trim_subsample_probe", 0)
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

    # ── 2026-10-16 trim 替代：逐顆狀態掛在模型上（gaussian_model._trim_state），density controller 剪枝／搬家時同步 ──
    @staticmethod
    def _trim_state(pc, keys=("V", "C")):
        n = pc.get_xyz.shape[0]
        st = getattr(pc, "_trim_state", None)
        if st is None:
            st = {}
            pc._trim_state = st
        for k in keys:
            t = st.get(k)
            dt = torch.bool if k == "off" else torch.float32
            if t is None:
                st[k] = torch.zeros(n, dtype=dt, device=pc.get_xyz.device)
            elif t.shape[0] < n:          # MCMC 新增的顆粒接在尾端 => 補 0
                st[k] = torch.cat([t, torch.zeros(n - t.shape[0], dtype=dt, device=t.device)])
            elif t.shape[0] > n:          # 不該發生（剪枝已同步）；保險起見整個重設
                st[k] = torch.zeros(n, dtype=dt, device=pc.get_xyz.device)
        return st

    def _acc_add(self, pc, ta, cn):
        with torch.no_grad():
            st = self._trim_state(pc)
            st["V"] += ta.float()
            st["C"] += cn.float()
            st["views"] = int(st.get("views", 0)) + 1
        if not getattr(self, "_acc_announced", False):
            self._acc_announced = True
            print(f"[trim-running] ✅ 首次觸發：訓練 forward 累加 V／C（本步有覆蓋的顆粒 {int((cn > 0).sum()):,}／{cn.numel():,}）", flush=True)

    def _running_trim(self, step, module, until):
        """取代週期 trim：用訓練中累加的 V／C（不掃相機）。回傳 True＝已處理。"""
        pc = module.gaussian_model
        st = self._trim_state(pc, ("V", "C"))
        n_views = int(st.get("views", 0))
        V, C = st["V"], st["C"]
        seen = C > 0
        score = torch.where(seen, V / torch.clamp_min(C, 1.0), torch.full_like(V, float("inf")))   # 沒被看到 => 這輪不剪
        n_seen = int(seen.sum())
        k_cut = int(self.prune_ratio * pc.get_xyz.shape[0])
        thr = torch.kthvalue(score.float().cpu(), max(1, min(k_cut, n_seen))).values.to(score.device) if n_seen > 0 and k_cut > 0 else None
        low = (score <= thr) & seen if thr is not None else torch.zeros_like(seen)
        rho = float(getattr(self, "trim_soft_rho", 0.0))
        tag = "trim-soft" if rho != 0 else "trim-running"
        if rho != 0 and step < until:   # ★ 10-18 rho<0＝反向（把支撐外的 opacity 往 1 推；使用者要的對照）
            # 軟投影（ADMM z／λ 更新）：支撐外＝low；λ ← λ + o（支撐外，z=0）；支撐內 λ=0
            ss = self._trim_state(pc, ("V", "C", "lam", "off"))
            o = pc.get_opacities().squeeze(-1).detach()
            # 上一輪支撐外的去向：被 MCMC 搬家的槽位 off 已被歸零（mcmc_density_controller.relocate_gs）=> 視為已回收
            prev_n = int(ss.get("off_n", 0)); still = ss["off"]
            prev_msg = ""
            if prev_n > 0:
                n_still = int(still.sum())
                prev_msg = (f"｜上一輪支撐外 {prev_n:,}：已回收 {max(prev_n - n_still, 0):,}、仍存活 {n_still:,}"
                            f"（o 中位 {float(o[still].median()) if n_still > 0 else 0:.3f}）")
            ss["lam"] = torch.where(low, ss["lam"] + o, torch.zeros_like(o))
            ss["off"] = low.clone()
            ss["off_n"] = int(low.sum())
            msg = f"支撐外 {int(low.sum()):,} 顆（λ 中位 {float(ss['lam'][low].median()) if bool(low.any()) else 0:.3f}）" + prev_msg
        else:
            if rho != 0:   # 增生期最後一次：把仍在支撐外的硬剪掉
                ss = self._trim_state(pc, ("V", "C", "lam", "off"))
                low = low | ss["off"]
            module.density_controller._prune_points(low, pc, module.gaussian_optimizers)
            msg = f"剪 {int(low.sum()):,} 顆"
            if rho != 0:
                pc._trim_state.pop("lam", None); pc._trim_state.pop("off", None); pc._trim_state.pop("off_n", None)
        print(f"[{tag}] step {step}：{n_views} 個訓練視角累加、看到 {n_seen:,}／{seen.numel():,} 顆、未觀測 {seen.numel() - n_seen:,} 顆不剪；"
              f"門檻 V/C {float(thr) if thr is not None else float('nan'):.3e}｜{msg}", flush=True)
        # 重設累加器（剪枝後 N 變了 => 以新 N 重建）
        pc._trim_state["V"] = torch.zeros(pc.get_xyz.shape[0], device=pc.get_xyz.device)
        pc._trim_state["C"] = torch.zeros(pc.get_xyz.shape[0], device=pc.get_xyz.device)
        pc._trim_state["views"] = 0
        if step >= until:
            self._acc_stop = True          # 增生期結束 => 不再累加（省 forward 的那點開銷）
        elif int(getattr(self, "reseed_views", 0) or 0) > 0 and rho == 0:
            self._reseed(step, module, int(float(getattr(self, "reseed_frac", 1.0)) * int(low.sum())))
        return True

    @torch.no_grad()
    def _reseed(self, step, module, n_target):
        """★ 2026-10-24 trim 後從渲染深度撒點（說明見 __init__ 的 reseed_*）。"""
        from internal.density_controllers.density_controller import Utils
        from internal.utils.outside_pool import block_inside_mask
        pc = module.gaussian_model
        cap = int(getattr(module.density_controller.config, "cap_max", 0) or 0)
        if cap > 0:
            n_target = min(n_target, cap - pc.n_gaussians)
        if n_target <= 0:
            return
        ds = module.trainer.train_dataloader.dataset
        nv = max(1, min(int(self.reseed_views), len(ds)))
        vids = torch.randperm(len(ds), generator=torch.Generator().manual_seed(int(step)))[:nv].tolist()
        per = [n_target // nv + (1 if i < n_target % nv else 0) for i in range(nv)]
        pcfg = module.trainer.datamodule.hparams["parser"]
        dev = pc.get_xyz.device
        bg = torch.zeros(3, device=dev)
        P, Ccol, Nrm, Sc = [], [], [], []
        chk = None
        for vi, k in zip(vids, per):
            if k <= 0:
                continue
            cam = ds.image_cameras[vi].to_device(dev)
            out = self(cam, pc, bg_color=bg)                      # 完整輸出（非 lean）：surf_depth／rend_alpha／rend_normal
            d, a = out["surf_depth"][0], out["rend_alpha"][0]
            H, W = d.shape
            _, gt, _ = ds.get_image(vi)
            gt = gt.to(dev)
            gt = gt.float() / 255.0 if gt.dtype == torch.uint8 else gt.float()
            if gt.shape[-2:] != (H, W):
                gt = torch.nn.functional.interpolate(gt[None], size=(H, W), mode="area")[0]
            valid = (a > 0.5) & torch.isfinite(d) & (d > 0.2)
            w = ((out["render"] - gt).abs().mean(0) if bool(getattr(self, "reseed_by_error", True)) else torch.ones_like(d)) * valid
            nval = int((w > 0).sum())
            if nval == 0:
                continue
            idx = torch.multinomial(w.flatten(), min(3 * k, nval), replacement=False)
            v, u = idx // W, idx % W
            z = d.flatten()[idx]
            tx, ty = math.tan(float(cam.fov_x) * 0.5), math.tan(float(cam.fov_y) * 0.5)
            fx, fy = W / (2 * tx), H / (2 * ty)                   # 與光柵器相同：主點在影像中心、像素中心＝索引 + 0.5
            pcam = torch.stack([(u.float() + 0.5 - W / 2) / fx * z, (v.float() + 0.5 - H / 2) / fy * z, z, torch.ones_like(z)], -1)
            pw = (pcam @ torch.linalg.inv(cam.world_to_camera))[:, :3]
            keep = block_inside_mask(pw, pcfg, 0.0)
            sel = keep.nonzero(as_tuple=True)[0][:k]
            if sel.numel() == 0:
                continue
            if chk is None:                                       # 自檢：用光柵器的投影矩陣把新點投回去，應落在原像素
                ph = torch.cat([pw[sel], torch.ones_like(pw[sel, :1])], -1) @ cam.full_projection
                ndc = ph[:, :2] / ph[:, 3:4]
                pix = ((ndc + 1) * torch.tensor([W, H], device=dev) - 1) * 0.5
                chk = float((pix - torch.stack([u[sel], v[sel]], -1).float()).abs().max())
            P.append(pw[sel])
            Ccol.append(gt[:, v[sel], u[sel]].T)
            Nrm.append(out["rend_normal"][:, v[sel], u[sel]].T)
            Sc.append(z[sel] / fx * float(self.reseed_scale_px))
        if not P:
            return
        P, Ccol, Nrm, Sc = torch.cat(P), torch.cat(Ccol), torch.cat(Nrm), torch.cat(Sc)
        n = P.shape[0]
        # 面朝法線的四元數（w, x, y, z）：把局部 z 軸轉到 n；n 幾乎是 -z 時繞 x 軸轉 180 度
        nz = torch.nn.functional.normalize(Nrm, dim=-1)
        bad = ~torch.isfinite(nz).all(-1) | (Nrm.norm(dim=-1) < 1e-6)
        nz[bad] = torch.tensor([0., 0., 1.], device=dev)
        axis = torch.stack([-nz[:, 1], nz[:, 0], torch.zeros_like(nz[:, 0])], -1)        # e_z x n
        cosv = nz[:, 2].clamp(-1, 1)
        q = torch.cat([(1 + cosv)[:, None], axis], -1)                                     # 未正規化的半角四元數
        anti = (1 + cosv) < 1e-6
        q[anti] = torch.tensor([0., 1., 0., 0.], device=dev)
        q = torch.nn.functional.normalize(q, dim=-1)
        props = pc.properties
        C0 = 0.28209479177387814
        new = {
            "means": P,
            "shs_dc": ((Ccol - 0.5) / C0)[:, None, :],
            "shs_rest": torch.zeros((n,) + tuple(props["shs_rest"].shape[1:]), device=dev),
            "opacities": torch.logit(torch.full((n, 1), float(self.reseed_opacity), device=dev)),
            "scales": torch.log(Sc.clamp_min(1e-7))[:, None].expand(n, props["scales"].shape[1]).contiguous(),
            "rotations": q,
        }
        missing = [k for k in props if k not in new]
        if missing:
            raise RuntimeError(f"reseed：模型有沒處理到的屬性 {missing}")
        n0 = pc.n_gaussians
        pc.properties = Utils.cat_tensors_to_properties(new, pc, module.gaussian_optimizers)
        if not getattr(self, "_reseed_announced", False) or step % 5000 == 0:
            self._reseed_announced = True
            print(f"[reseed] {'✅ 首次觸發 ' if chk is not None and not getattr(self, '_reseed_chk_done', False) else ''}step {step}：{nv} 個視角撒 {n:,} 顆"
                  f"（目標 {n_target:,}；N {n0:,} -> {pc.n_gaussians:,}）｜深度中位 {float(P.new_tensor(0) + Sc.median() * fx / float(self.reseed_scale_px)):.3f}"
                  f"｜大小中位 {float(Sc.median()):.4f}" + (f"｜自檢：投回像素最大誤差 {chk:.3f} px {'✅' if chk < 0.05 else '⚠⚠ 慣例可能不對'}" if chk is not None and not getattr(self, '_reseed_chk_done', False) else ""), flush=True)
            self._reseed_chk_done = True

    def after_training_step(
            self,
            step: int,
            module,
    ):
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        until = self.contribution_prune_until_iter
        if until < 0:
            until = module.density_controller.config.densify_until_iter
        _sch = str(getattr(self, "trim_schedule", "") or "").strip()
        _tau = None
        # ★ 2026-10-19 累加式 trim：超過 until 就停止累加（原本只在「until 那一步剛好觸發 trim」時才停；
        #   until 不是 interval 的倍數、或計時工具用 densify_until 0 模擬收割期時，會一路白累加，forward 多 ~3.5 ms）。
        #   真實 60k 訓練的 until（30k）是 500 的倍數，已在最後一次 trim 停 => 訓練行為不變。
        if (not _sch) and getattr(self, "trim_running_vc", False) and step > until and not getattr(self, "_acc_stop", False):
            self._acc_stop = True
            print(f"[trim-running] step {step} > until {until} => 停止累加 V/C", flush=True)
        if _sch:
            _tau = dict((int(a), float(b)) for a, b in (x.split(":") for x in _sch.split(",") if x.strip())).get(int(step))
            if self.diable_trimming or _tau is None:
                return
        elif self.diable_trimming or (step > until) \
           or (step < self.contribution_prune_from_iter) \
           or (step % self.contribution_prune_interval != 0):
           return
        
        device =  module.gaussian_model.get_xyz.device

        # ★ 2026-10-16 取代 trim：用訓練中累加的 V／C（不掃相機）；只支援 V/C 判準、週期 trim（不支援門檻剪／tile 名次）
        if bool(getattr(self, "trim_running_vc", False)) and _tau is None:
            if not bool(getattr(self, "trim_by_mean_contribution", False)) or not bool(getattr(self, "lean_train", False)):
                raise RuntimeError("trim_running_vc 需要 trim_by_mean_contribution=true（判準 V/C）與 lean_train=true")
            with torch.no_grad():
                self._running_trim(step, module, until)
            torch.cuda.empty_cache()
            return

        push, gather = contribution_accumulator(self.K)
        blur = None
        cost_acc = None                      # ★ 精確渲染成本 c_i 的累積（見迴圈內註解）
        with torch.no_grad():
            # `TRIM_SUBSAMPLE=n` => 週期 trim 只用每 n 台相機算 contribution（n=1 = 原行為）。
            #
            # ⛔⛔ 2026-09-12 定案：**這個旋鈕不要開。** 已在訓練後的幾何上量過
            #   （`tools/trim_subsample_probe.py`），判準（底部遮罩重疊 > 95%）**沒過**：
            #       同樣 71 台（n=4）的底部 10% 遮罩重疊
            #         step 1（下面那個舊探針）  99.8%   <- 未訓練的均勻殼，視角可互換
            #         step 14,999              85.56%  （誤剪 18,952 顆）
            #         step 29,999              81.41%  （誤剪 48,329 顆）
            #       n=2 也不過：94.78% / 92.32%
            #   ⇒ **舊的 99.8% 是在「最容易通過的那一刻」量的，不可引用。**
            #   根因是結構性的：`contribution_accumulator` 預設 `reduce="max"`（跨視角取最大）
            #   ⇒ 少看相機只會讓值變小 ⇒ 最佳視角被抽掉的粒子往底部掉、被誤剪。
            #   step 1 的殼粒子尺度全同、視角可互換；訓練後粒子對視角特化 ⇒ 那個時點測不到。
            # ⚠ 下面 `before_training_step` 裡的 `TRIM_SUBSAMPLE_PROBE` 只量 step==1
            #   （那裡第一行就是 `if step != 1: return`）⇒ 它**永遠**只會給出樂觀的數字。
            # ⚠ 只作用在**週期** trim。起始 trim（step==1，另一處）維持全視角：
            #   它只跑一次、不佔時間，但會砍掉 73~90% 的初始化，是被記錄的關鍵機制。
            # ⚠ 用環境變數而非 config：ckpt-init 會把 renderer 整個換掉（同 §11.52 的坑）。
            stride = max(1, int(os.environ.get("TRIM_SUBSAMPLE", "1") or 1))
            n_used = len(range(0, len(cameras), stride))
            print(f"Trimming...（相機 {n_used}/{len(cameras)}，stride={stride}）")
            _tk = bool(getattr(self, "trim_by_tile_topk", False))
            best_rank = None
            if _tk:
                from internal.utils.tile_topk import update_best_rank, NEVER
                best_rank = torch.full((module.gaussian_model.get_xyz.shape[0],), NEVER,
                                       dtype=torch.int64, device=device)
                _tk_pairs = 0
            for i in range(0, len(cameras), stride):
                camera = cameras[i].to_device(device)
                if _tk:
                    mean, covered, _radii = self(
                        camera,
                        module.gaussian_model,
                        bg_color=module._fixed_background_color().to(device),
                        record_transmittance=True,
                        record_coverage=True,
                        record_radii=True,
                    )
                    _tk_pairs += update_best_rank(best_rank, mean * covered.float(), _radii,
                                                  module.gaussian_model.get_xyz, camera)
                    del _radii
                else:
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
                # ★ 2026-09-07：精確渲染成本 c_i（`num_covered_pixels` 跨視角求和）。
                #   原本算完就 `del`，只有 blur-split（現行配方為 0）用到 mean*covered。
                #   而 `cost_aware_densify` 卻在用 `_max_radii2D²` 這個代理 ——
                #   實測代理把離散度**誇大 66%**（ceiling 15.26x vs 精確 9.19x，§11.109）。
                #   累積它是**零成本**（covered 本來就在手上）。
                cost_acc = covered.float() if cost_acc is None else cost_acc + covered.float()
                del mean, covered, raw

            contribution = gather()
            if blur is not None:
                module.density_controller.blur_score = blur / max(len(cameras), 1)
            score = contribution
            if self.trim_by_value_per_cost:
                if cost_acc is None:
                    print("⚠⚠ [trim-vpc] **設了 trim_by_value_per_cost 但拿不到 c_i，本機制不會生效**")
                else:
                    _a = float(self.trim_value_per_cost_alpha)
                    _cc = torch.clamp_min(cost_acc, 1.0)
                    score = contribution / (_cc if _a == 1.0 else _cc.pow(_a))
                    if not self._vpc_announced:
                        self._vpc_announced = True
                        _v, _c, _s = contribution, cost_acc, score
                        def _ceil(x):
                            k = max(1, int(x.numel() * 0.05))
                            return float(torch.topk(x, k).values.mean() / torch.clamp_min(x.mean(), 1e-30))
                        # ★ 決定性的數字：它**剪掉不同的一批嗎**。
                        #   ceiling 不同只證明「判準的分布變了」，不證明「被剪的集合變了」；
                        #   若重疊 ~100%，這機制就是昂貴的 no-op（第 8 個「看起來正常但沒作用」）。
                        #   兩個 quantile + 一次 & ⇒ 一次性、零額外 pass。
                        _mv = _v <= torch.quantile(_v, self.prune_ratio)
                        _ms = _s <= torch.quantile(_s, self.prune_ratio)
                        _ov = float((_mv & _ms).sum()) / max(float(_mv.sum()), 1.0)
                        print(f"[trim-vpc] ✅ 首次觸發：判準 v -> v/c^{_a:g}   "
                              f"ceiling(top5%)/mean  v={_ceil(_v):.2f}x  c={_ceil(_c):.2f}x  v/c={_ceil(_s):.2f}x")
                        print(f"[trim-vpc]    底部 {100*self.prune_ratio:.0f}% 遮罩重疊 = "
                              f"**{100*_ov:.2f}%**（~100% = 剪同一批 = no-op；越低代表換掉越多）")
            if best_rank is not None:
                # 分數＝-最好名次（越前面越好）；並列時以貢獻決勝（正規化到 [0, 0.5)，不跨名次）
                _c = contribution / torch.clamp_min(contribution.max(), 1e-30)
                _tk_score = (-best_rank.to(torch.float64) + 0.5 * _c.to(torch.float64))
                if not getattr(self, "_tk_announced", False):
                    self._tk_announced = True
                    _mv = contribution <= torch.quantile(contribution, self.prune_ratio)
                    _ms = _tk_score <= torch.quantile(_tk_score, self.prune_ratio)
                    _ov = float((_mv & _ms).sum()) / max(float(_mv.sum()), 1.0)
                    _ok = best_rank < NEVER
                    _never = 1.0 - float(_ok.float().mean())
                    _med = float(best_rank[_ok].float().median()) if bool(_ok.any()) else -1.0
                    print(f"[trim-tilek] ✅ 首次觸發：判準＝逐 tile 最好名次（{_tk_pairs:,} 個 (顆,tile) 對，{n_used} 視角）"
                          f"  從未上榜 {100*_never:.2f}%  名次中位 {_med:.0f}")
                    print(f"[trim-tilek]    與貢獻判準的底部 {100*self.prune_ratio:.0f}% 遮罩重疊 = **{100*_ov:.2f}%**"
                          f"（~100% = 剪同一批 = no-op）")
                if bool(getattr(self, "trim_by_value_per_cost", False)) and cost_acc is not None:
                    # 合成判別式：v/c（上面已算進 score）除以 tile 名次的軟懲罰；從未上榜 => 名次 NEVER => 分數趨近 0
                    _ks = float(getattr(self, "trim_tile_k_soft", 32.0))
                    _comb = score.to(torch.float64) / (1.0 + best_rank.to(torch.float64) / _ks)
                    if not getattr(self, "_comb_announced", False):
                        self._comb_announced = True
                        _r = self.prune_ratio
                        _mvc = score <= torch.quantile(score, _r)
                        _mrk = _tk_score <= torch.quantile(_tk_score, _r)
                        _mc = _comb <= torch.quantile(_comb, _r)
                        _ov = lambda a, b: 100.0 * float((a & b).sum()) / max(float(a.sum()), 1.0)
                        print(f"[trim-vpc+tilek] ✅ 合成判別式 (v/c^α)/(1+名次/{_ks:g})：底部 {100*_r:.0f}% 與純 v/c 重疊 "
                              f"{_ov(_mvc, _mc):.2f}%、與純名次重疊 {_ov(_mrk, _mc):.2f}%（都 ~100% = 合成無作用）", flush=True)
                    score = _comb
                else:
                    score = _tk_score
            if bool(getattr(self, "trim_by_mean_contribution", False)):
                if blur is None or cost_acc is None:
                    print("⚠⚠ [trim-V/C] **設了 trim_by_mean_contribution 但拿不到 V 或 C，本機制不會生效**", flush=True)
                else:
                    _beta = float(getattr(self, "trim_cost_beta_px", 0.0)) * float(n_used)
                    _mc = blur / torch.clamp_min(cost_acc + _beta, 1.0)
                    if not getattr(self, "_mc_announced", False):
                        self._mc_announced = True
                        _r = self.prune_ratio
                        _mv = contribution <= torch.quantile(contribution, _r)
                        _mm = _mc <= torch.quantile(_mc, _r)
                        _ov = float((_mv & _mm).sum()) / max(float(_mv.sum()), 1.0)
                        _sm = module.gaussian_model.get_scales().max(dim=-1).values
                        _tiny = _sm < 1e-4
                        _fr = lambda m: 100.0 * float((m & _tiny).sum()) / max(float(m.sum()), 1.0)
                        print(f"[trim-V/C] ✅ 首次觸發 step {step}：判準 v -> V/(β·{n_used} 視角 + C)，β={float(getattr(self, 'trim_cost_beta_px', 0.0)):g} px"
                              f"｜底部 {100*_r:.0f}% 與現行 v 重疊 **{100*_ov:.2f}%**（~100% = no-op）"
                              f"｜被剪的微小顆粒（最大軸<1e-4）佔 {_fr(_mm):.1f}%（v 判準 {_fr(_mv):.1f}%、全體 {100*float(_tiny.float().mean()):.1f}%）", flush=True)
                    score = _mc
            if _tau is not None:
                # 門檻剪：判準固定用 v（contribution），不用 v/c —— 剪了不補時 v/c 最差（實驗分析/06）
                tile = torch.tensor(_tau, device=contribution.device)
                prune_mask = (contribution <= tile)
            else:
                tile = torch.quantile(score, self.prune_ratio)
                prune_mask = (score <= tile)
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
            # ★ 必須在 `_prune_points` **之前**先套遮罩取出存活者的成本 ——
            #   剪枝會改變 N，之後再切就對不上了。
            # ⚠ 2026-09-07：我第一次把這段加到了 `before_training_step`（起始 trim），
            #   那裡沒有 `cost_acc`/`n_used` => step 1 就 NameError。`_prune_points` 在本檔
            #   出現**兩次**，替換時必須用「後面的 print 有 step=」來區分週期 trim。
            _surv_cost = cost_acc[~prune_mask] if cost_acc is not None else None
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            if _surv_cost is not None:
                # 除以視角數 => **每視角平均**的渲染成本，與命題 `(1/K)Σc_i <= B` 同單位。
                module.density_controller._exact_cost = _surv_cost / max(n_used, 1)
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
