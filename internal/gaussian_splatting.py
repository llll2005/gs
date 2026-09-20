import os.path
import queue
import threading
import traceback
from typing import Tuple, List, Dict, Union, Any, Callable, Optional
from typing_extensions import Self

import time
import os
import torch
import torch.optim
import torchvision
import wandb
import csv
from lightning.pytorch.core.module import MODULE_OPTIMIZERS
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, LRSchedulerPLType, STEP_OUTPUT
import lightning.pytorch.loggers

import internal.mp_strategy
from internal.viewer.training_viewer import TrainingViewer
from internal.configs.light_gaussian import LightGaussian

from internal.models.gaussian import Gaussian, GaussianModel
from internal.models.vanilla_gaussian import VanillaGaussian
from internal.renderers import Renderer, VanillaRenderer, RendererConfig
from internal.metrics.metric import Metric
from internal.metrics.vanilla_metrics import VanillaMetrics
from internal.density_controllers.density_controller import DensityController
from internal.density_controllers.vanilla_density_controller import VanillaDensityController
from internal.initializers.gaussian_initializer import Initializer
from jsonargparse import lazy_instance

from internal.utils.sh_utils import eval_sh
from internal.utils.graphics_utils import store_ply


# ── 逐段成本計時（預設關閉；`CITYGS_STEP_PROFILE=1` 開啟）────────────────────────
# 為什麼要在**真實迴圈**裡量，而不是再寫一個微基準：
#   `tools/vram_pressure.py` 臂 A 量到 render+backward+optimizer ≈ 155 ms/百萬顆 + **21 ms**
#   固定成本，而 ledger 68 個跑次擬合的全訓練是 `178.9 + 100.5*N`（§11.18）
#   => 那 **158 ms 的固定成本活在微基準涵蓋不到的地方**，只能在真實迴圈裡抓。
# 三個踩過的坑（`tools/step_breakdown.py` 的 docstring 有完整版）：
#   1 cProfile 把非同步工作記到下一個同步點 => `.to()` 看起來 0.17 秒，其實在等光柵化
#   2 40 步的視窗被「只跑一次」的 trim pass 佔掉 80% => 週期性事件必須按頻率攤平
#   3 跨跑次比 wall time 的噪音約 5% => 小於 10% 的差異量不出來
# 做法：**CUDA event 記時間戳（不強制同步、不破壞重疊）+ 同時記 wall clock**。
#   兩者的差就是 CPU 側的阻塞（dataloader、H2D 搬運）—— 那正是「是不是搬運」的答案。
class _StepProfiler:
    def __init__(self):
        self.on = os.environ.get("CITYGS_STEP_PROFILE", "") not in ("", "0")
        self.gpu = {}          # 段落 -> 累計 GPU ms
        self.cpu = {}          # 段落 -> 累計 wall ms
        self.count = {}        # 段落 -> 觸發次數（週期性事件要用它攤平）
        self.steps = 0
        self.warm = None
        self.real_step_ms = 0.0
        self._step_t0 = None
        self._marks = []       # [(名稱, cuda_event, perf_counter)]
        self._prev_end = None  # 上一步結束的 wall 時間 => 迴圈外的開銷
        # 逐段 VRAM（2026-09-17）：每個標記點讀 max_memory_allocated 後 reset_peak => 得到「段內峰值配置」
        #   ⚠ 只在 profiler 開啟時；reset 會讓台帳的「峰值實佔」只剩最後一段 => 這類跑次的台帳 VRAM 不可引用
        self.pk_alloc = {}     # 段落 -> 段內峰值配置 MiB（取最大）
        self.pk_resv = {}      # 段落 -> 段末保留 MiB（取最大）
        self.global_pk = 0.0   # 全程最大段內峰值配置 MiB（不被 reset 影響）

    WARMUP = int(os.environ.get("CITYGS_STEP_PROFILE_WARMUP", "20"))

    def start_step(self):
        if not self.on:
            return
        import time
        now = time.perf_counter()
        self._step_t0 = self._prev_end if self._prev_end is not None else now
        if self._prev_end is not None:
            self._acc("0 迴圈外（dataloader/Lightning）", None, (now - self._prev_end) * 1e3)
        self._marks = []
        self.mark("__begin__")
        if self._prev_end is not None:          # __begin__ 讀到的峰值 = 上一步結束到這裡（迴圈外）
            self._accmem("0 迴圈外（dataloader/Lightning）", self._marks[0][3], self._marks[0][4])

    def mark(self, name):
        """⚠ 這裡**必須同步**，否則歸屬是歪的。

        CUDA 呼叫是非同步的：CPU 會跑在 GPU 前面，所以早期段落的 event 差很小
        （GPU 還沒開始做），後面的段落把前面的等待全部吸走。實測（N=338k，不同步）：
            forward  GPU 25.93 / wall 23.37      backward GPU 110.75 / wall **3.46**
        backward 的 wall 只有 3.46 ms 是因為它只是把工作排進佇列就回來了，
        而它的 GPU 110.75 ms 其實含了 forward 與 loss 還沒做完的部分。
        這正是 cProfile 把 48 秒記在 `.to()` 上的同一個機制。
        => 每個標記點同步，換到正確的逐段歸屬。
        代價：**消除了 CPU/GPU 重疊**，所以各段之和會略高於真實步時間（報告裡另外印真實步時間）。
        """
        if not self.on:
            return
        import time
        torch.cuda.synchronize()
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        _a, _r = self._mem()
        self._marks.append((name, ev, time.perf_counter(), _a, _r))

    def _mem(self):
        """段內峰值配置與當下保留（MiB），讀完 reset 峰值統計。只在 profiler 開啟時被呼叫。"""
        a = torch.cuda.max_memory_allocated() / 2 ** 20
        r = torch.cuda.memory_reserved() / 2 ** 20
        torch.cuda.reset_peak_memory_stats()
        return a, r

    def _accmem(self, name, a, r):
        self.pk_alloc[name] = max(self.pk_alloc.get(name, 0.0), a)
        self.pk_resv[name] = max(self.pk_resv.get(name, 0.0), r)
        self.global_pk = max(self.global_pk, a)

    def end_step(self):
        if not self.on:
            return
        import time
        self.mark("__end__")
        torch.cuda.synchronize()                       # 每步只同步一次
        for i in range(1, len(self._marks)):
            name, ev, t, a, r = self._marks[i]
            _pn, pev, pt, _pa, _pr = self._marks[i - 1]
            if name.startswith("__"):
                continue
            self._acc(name, pev.elapsed_time(ev), (t - pt) * 1e3)
            self._accmem(name, a, r)
        self.steps += 1
        _now = time.perf_counter()
        if self._step_t0 is not None:
            self.real_step_ms += (_now - self._step_t0) * 1e3
        self._prev_end = _now
        if self.warm is None and self.steps == self.WARMUP:   # ⚠ 只能做一次：
        # 重置後 steps 歸零，少了 `self.warm is None` 就會每 WARMUP 步再重置一遍，
        # steps 永遠到不了報告門檻（實測 250 次迴圈後 steps 只有 10）。
            # 前 WARMUP 步含一次性成本（起始 trim 掃全部 284 台相機、CUDA/cuDNN 初始化）。
            # 記憶 `feedback_measure_before_explaining`：40 步的視窗被只跑一次的 trim 佔掉 80%
            # => 把它們單獨留一份，然後歸零重算穩態。
            self.warm = {"gpu": dict(self.gpu), "cpu": dict(self.cpu),
                         "count": dict(self.count), "steps": self.steps,
                         "pk_alloc": dict(self.pk_alloc), "pk_resv": dict(self.pk_resv)}
            self.pk_alloc, self.pk_resv = {}, {}   # 穩態另計（暖機含起始 trim 的一次性峰值）
            self.gpu, self.cpu, self.count, self.steps = {}, {}, {}, 0
            self.real_step_ms = 0.0

    def _acc(self, name, gpu_ms, cpu_ms):
        if gpu_ms is not None:
            self.gpu[name] = self.gpu.get(name, 0.0) + gpu_ms
        self.cpu[name] = self.cpu.get(name, 0.0) + cpu_ms
        self.count[name] = self.count.get(name, 0) + 1

    def report(self, out_path, n_gaussians):
        if not self.on or self.steps == 0:
            return
        lines = []
        A = lines.append
        A(f"逐段成本（真實訓練迴圈，CUDA event + wall clock）")
        A(f"  穩態取樣 {self.steps:,} 步（已排除前 {self.WARMUP} 步的一次性成本）"
          f"   N = {n_gaussians:,}")
        if self.warm:
            w = sum(self.warm["cpu"].values())
            A(f"  暖機 {self.warm['steps']} 步共 {w/1e3:.2f} s"
              f"（含起始 trim 掃全部相機 + CUDA 初始化）=> 攤到 60,000 步是 {w/60000:.2f} ms/步")
        A("")
        A(f"{'段落':<34} {'GPU ms/步':>10} {'wall ms/步':>11} {'觸發次數':>9} "
          f"{'每次 wall ms':>12} {'佔 wall':>8} {'段內峰值配置MiB':>15} {'段末保留MiB':>12}")
        tot_cpu = sum(self.cpu.values()) / self.steps
        for k in sorted(self.cpu, key=lambda x: -self.cpu[x]):
            g = self.gpu.get(k, 0.0) / self.steps
            c = self.cpu[k] / self.steps
            n = self.count[k]
            A(f"{k:<34} {g:>10.2f} {c:>11.2f} {n:>9,} "
              f"{self.cpu[k]/max(n,1):>12.2f} {100*c/max(tot_cpu,1e-9):>7.1f}% "
              f"{self.pk_alloc.get(k, 0.0):>15.0f} {self.pk_resv.get(k, 0.0):>12.0f}")
        A(f"{'（各段之和·序列化）':<34} {sum(self.gpu.values())/self.steps:>10.2f} {tot_cpu:>11.2f}")
        A(f"{'（真實步時間·含重疊）':<34} {'':>10} {self.real_step_ms/self.steps:>11.2f}"
          f"   = {1000*self.steps/max(self.real_step_ms,1e-9):.2f} it/s")
        if self.pk_alloc:
            A(f"  VRAM：穩態最大段內峰值配置 {max(self.pk_alloc.values()):,.0f} MiB（{max(self.pk_alloc, key=self.pk_alloc.get)}）"
              f"／最大段末保留 {max(self.pk_resv.values()):,.0f} MiB／全程含暖機 {self.global_pk:,.0f} MiB")
            if self.warm and self.warm.get("pk_alloc"):
                _w = self.warm["pk_alloc"]
                A(f"        暖機期最大段內峰值 {max(_w.values()):,.0f} MiB（{max(_w, key=_w.get)}）")
            A("  ⚠ 保留 - 配置 的差 ≈ 配置器快取＋碎片；這個跑次的台帳 VRAM 不可引用（每段 reset 了峰值統計）")
        A("")
        A("判讀：")
        A("  wall >> GPU 的段落 = **CPU 側阻塞**（dataloader、H2D 搬運、Python 開銷）")
        A("  wall ~= GPU 的段落 = 真的在算")
        A("  『觸發次數』遠小於步數的是**週期性事件**（trim 每 500 步、densify 每 150 步）")
        A("  ⚠⚠ 週期性事件的『wall ms/步』是按**取樣視窗**攤平的，**不是按真實週期** ——")
        A("     取樣視窗不是週期的整數倍時會低估。正確的攤平要自己算：")
        A("       該事件的每步成本 = 『每次 wall ms』 / 真實週期（trim = contribution_prune_interval）")
        A("       全跑次佔比       = 每次成本 x 觸發總次數 / 總跑次時間")
        A("     例：本工具量到 trim 一次 50.9 s、視窗 1,000 步只抓到 1 次 => 印出 50.91 ms/步，")
        A("        但週期是 500 步 => 真實是 **101.8 ms/步**（densify 期間），差 2 倍。")
        txt = "\n".join(lines)
        print("\n" + txt, flush=True)
        try:
            with open(out_path, "w") as f:
                f.write(txt + "\n")
        except Exception:
            pass


class GaussianSplatting(LightningModule):
    def __init__(
            self,
            light_gaussian: LightGaussian,  # TODO: may be should implement as a hook
            save_iterations: List[int],
            gaussian: Gaussian = lazy_instance(VanillaGaussian),
            background_color: Tuple[float, float, float] = (0., 0., 0.),
            random_background: bool = False,
            output_path: str = None,
            correct_color: bool = False,
            test_speed: bool = False,
            save_val_output: bool = False,
            save_val_metrics: bool = None,
            max_save_val_output: int = -1,
            renderer: Union[Renderer, RendererConfig] = lazy_instance(VanillaRenderer),
            metric: Metric = lazy_instance(VanillaMetrics),
            density: DensityController = lazy_instance(VanillaDensityController),
            save_ply: bool = False,
            web_viewer: bool = False,
            initialize_from: str = None,
            overwrite_config: bool = True,
            initializer: Optional[Initializer] = None,
            renderer_output_types: Optional[List[str]] = None,
            train_strips: int = 1,
            dynamic_strips: bool = False,
            strip_vram_target_gb: float = 5.4,
            strip_v_os_gb: float = 0.8,
            strip_safety: float = 0.6,
            strip_max: int = 8,
            grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # setup models
        self.gaussian_model = gaussian.instantiate()
        self.frozen_gaussians = None

        self.light_gaussian_hparams = light_gaussian

        # instantiate renderer
        if isinstance(renderer, RendererConfig):
            renderer = renderer.instantiate()
        self.renderer = renderer

        self.renderer_output_types = renderer_output_types

        # instantiate density controller
        self.density_controller = density.instantiate()

        # metrics
        self.metric = metric.instantiate()

        # initializer (optional, opt-in). None => fall back to legacy initialize_from path.
        self.gaussian_initializer = initializer.instantiate() if initializer is not None else None

        # background color
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        if random_background is True:
            self.get_background_color = self._random_background_color
        else:
            self.get_background_color = self._fixed_background_color

        self.web_viewer: TrainingViewer = None

        self.batch_size = 1
        self.restored_epoch = 0
        self.restored_global_step = 0
        self._floats_per_point = None  # cached for dynamic-K load estimate
        self._best_val_psnr = -1.0     # best-val tracking (surfaces if final != best)
        self._best_val_step = -1

        self.max_image_saving_threads = 16
        self.image_queue = queue.Queue(maxsize=self.max_image_saving_threads)
        self.image_saving_threads = []

        self.val_metrics: List[Tuple[str, Dict]] = []

        # hooks
        self.on_train_start_hooks: List[Callable[[GaussianModel, Self], None]] = []
        self.on_after_backward_hooks: List[Callable[[Dict, Any, GaussianModel, int, Self], None]] = []
        self.on_train_batch_end_hooks: List[Callable[[Dict, Any, GaussianModel, int, Self], None]] = []

    def log_metrics(
            self,
            metrics: dict,
            prog_bar: dict,
            prefix: str,
            on_step: bool,
            on_epoch: bool,
            name_prefix: str = "",
    ):
        for name in metrics:
            self.log(
                f"{prefix}/{name_prefix}{name}",
                metrics[name],
                # .get, not [name]: a metric without a prog_bar entry used to KeyError here, and
                # for a validate-only metric that surfaces at the FIRST val -- hours into a run.
                # Not showing it in the progress bar is the harmless default.
                prog_bar=prog_bar.get(name, False),
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=self.batch_size,
            )

    def _fixed_background_color(self):
        return self.background_color

    def _random_background_color(self):
        return torch.rand(3)

    def _initialize_from_trained_model(self, load_path: str = None, overwrite_config: bool = None):
        # assert self.hparams["gaussian"].extra_feature_dims == 0
        # load_path/overwrite_config default to hparams for backward compatibility
        # (legacy initialize_from path). CheckpointInitializer passes them explicitly.
        if load_path is None:
            load_path = self.hparams["initialize_from"]
        if overwrite_config is None:
            overwrite_config = self.hparams["overwrite_config"]

        from internal.utils.gaussian_model_loader import GaussianModelLoader
        load_from = GaussianModelLoader.search_load_file(load_path)

        if load_from.endswith(".ply") is True:
            # Load tensor parameters from PLY into the already-configured model.
            # Reuse self.gaussian_model so the full optimization config (LR, scheduler,
            # optimizer class) set by jsonargparse is preserved.
            from internal.utils.gaussian_utils import GaussianPlyUtils
            ply_data = GaussianPlyUtils.load_from_ply(load_from).to_parameter_structure()
            n = ply_data.xyz.shape[0]
            gaussian_model = self.gaussian_model
            gaussian_model.setup_from_number(n)
            gaussian_model.to(self.device)
            state_dict = {
                "gaussians.means": ply_data.xyz.to(self.device),
                "gaussians.opacities": ply_data.opacities.to(self.device),
                "gaussians.shs_dc": ply_data.features_dc.to(self.device),
                # shs_rest not loaded: PLY has sh_degree=0, leave zeros and let training learn.
                "gaussians.scales": ply_data.scales.to(self.device),
                "gaussians.rotations": ply_data.rotations.to(self.device),
            }
            gaussian_model.load_state_dict(state_dict, strict=False)
            renderer = self.renderer
        else:
            # load from ckpt
            gaussian_model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
                load_from,
                device=self.device,
                eval_mode=False,
                pre_activate=False,
            )

        # Expand shs_rest when configured sh_degree > checkpoint's sh_degree
        # (e.g., coarse trained at sh_degree=0, fine-tuning at sh_degree=2)
        cfg_sh = self.gaussian_model.config.sh_degree
        ckpt_sh = gaussian_model.config.sh_degree
        if cfg_sh > ckpt_sh:
            cfg_rest_dim = (cfg_sh + 1) ** 2 - 1
            shs_rest_param = gaussian_model.gaussians["shs_rest"]
            padded = torch.zeros(
                shs_rest_param.shape[0], cfg_rest_dim, 3,
                device=shs_rest_param.device, dtype=shs_rest_param.dtype,
            )
            if shs_rest_param.shape[1] > 0:
                padded[:, :shs_rest_param.shape[1]] = shs_rest_param.data
            gaussian_model.gaussians["shs_rest"] = torch.nn.Parameter(padded)
            gaussian_model.config.sh_degree = cfg_sh
            print(f"SH degree expanded: {ckpt_sh} -> {cfg_sh} "
                  f"(shs_rest: {list(shs_rest_param.shape)} -> {list(padded.shape)})")

        # replace config
        if overwrite_config:
            self.hparams["gaussian"] = gaussian_model.config
            self.gaussian_model = gaussian_model
        else:
            org_config = self.gaussian_model.config
            self.gaussian_model = gaussian_model
            self.gaussian_model.config = org_config

        # call for renderer
        # ⚠⚠ 2026-09-14：從 .ckpt 初始化時，上面拿到的是 **ckpt 裡存的 renderer**，會把 config/CLI 的
        #   renderer 整個換掉（官方原始碼同一行也是這樣）。官方 aerial 的 trim 階段吃的 coarse 設了
        #   `diable_trimming: true` => **trim 階段其實一次都沒 trim**（本機 official_ft_blk5 60k 步
        #   `Trimming` 出現 0 次）；`--model.renderer.init_args.*` 也全部無效，而 config.yaml 仍記下要求值。
        #   `CITYGS_KEEP_CFG_RENDERER=1` => overwrite_config=False 時保留 config 的 renderer（只取權重）；
        #   預設不設 => 行為與先前逐位元相同。
        _keep = overwrite_config is False and load_from.endswith(".ply") is False \
            and os.environ.get("CITYGS_KEEP_CFG_RENDERER", "") == "1"
        _cfg_r = self.renderer
        # ⚠⚠ 2026-09-18：ckpt 裡的 renderer 是**當時那份程式碼**反序列化出來的實例 =>
        #   之後新增的旗標它**沒有**，而 renderer 是 nn.Module，直接 `self.新旗標` 會走到
        #   `nn.Module.__getattr__` 丟 AttributeError ⇒ **跑到第一次週期 trim 才當場死**
        #   （實測：lab/stepcost2_lab_hi 吃 09-13 的 cs_base ckpt，step 1,000 死在
        #    `'SepDepthTrim2DGSRenderer' object has no attribute 'trim_by_value_per_cost'`，
        #    而它已經跑完 800 步、報告都寫出來了 => 是「舊 ckpt ＋ 新程式碼」的相容性問題）
        #   ⇒ 用 config 建好的那個 renderer 回填缺少的屬性（只補不覆蓋），並印出補了什麼。
        if renderer is not _cfg_r and isinstance(renderer, type(_cfg_r)):
            _bf = []
            for _k, _v in vars(_cfg_r).items():
                if _k.startswith("_") or _k in vars(renderer):
                    continue
                setattr(renderer, _k, _v)
                _bf.append(f"{_k}={_v!r}")
            if _bf:
                print(f"[init-renderer] ⚠ ckpt 的 renderer 缺 {len(_bf)} 個較新的屬性，已用 config 預設回填："
                      f"{', '.join(_bf[:8])}{' …' if len(_bf) > 8 else ''}", flush=True)
        def _rflags(r):
            return (f"{type(r).__name__} diable_trimming={getattr(r, 'diable_trimming', '-')} "
                    f"trim_by_value_per_cost={getattr(r, 'trim_by_value_per_cost', '-')}")
        if _keep:
            print(f"[init-renderer] ✅ 保留 config 的 renderer：{_rflags(_cfg_r)}（ckpt 裡的是 {_rflags(renderer)}）", flush=True)
        else:
            if renderer is not _cfg_r and _rflags(renderer) != _rflags(_cfg_r):
                print(f"[init-renderer] ⚠⚠ **renderer 被 ckpt 覆蓋**：config 要求 {_rflags(_cfg_r)} => 實際 {_rflags(renderer)}"
                      f"（要保留 config 的就設 CITYGS_KEEP_CFG_RENDERER=1）", flush=True)
            self.renderer = renderer

        print(f"initialize from {load_from}: sh_degree={self.gaussian_model.max_sh_degree}, overwrite_config={overwrite_config}")

    def setup(self, stage: str):

        self.renderer.setup(stage=stage, lightning_module=self)

        if stage == "fit":
            if self.gaussian_initializer is not None:
                # new opt-in module path
                self.gaussian_initializer.initialize(self, stage)
            elif self.hparams["initialize_from"] is None:
                # legacy path: SfM point-cloud cold start
                self.gaussian_model.setup_from_pcd(xyz=self.trainer.datamodule.point_cloud.xyz, rgb=self.trainer.datamodule.point_cloud.rgb / 255.)
            else:
                # legacy path: load from ckpt / ply
                self._initialize_from_trained_model()
        else:
            if self.hparams["save_val_metrics"] is None:
                self.hparams["save_val_metrics"] = True

        self.metric.setup(stage=stage, pl_module=self)
        self.density_controller.setup(stage=stage, pl_module=self)

        # use different image log method based on the logger type
        self.log_image = None
        if isinstance(self.logger, lightning.pytorch.loggers.TensorBoardLogger):
            self.log_image = self.tensorboard_log_image
        elif isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.log_image = self.wandb_log_image

    def on_load_checkpoint(self, checkpoint) -> None:
        # reinitialize parameters based on the gaussian number in the checkpoint
        self.gaussian_model.setup_from_number(checkpoint["state_dict"]["gaussian_model.gaussians.means"].shape[0])
        if "frozen_gaussians.means" in checkpoint["state_dict"]:
            from internal.utils.gaussian_containers import TensorDict
            self.frozen_gaussians = TensorDict({
                k: torch.empty_like(checkpoint["state_dict"]["frozen_gaussians.{}".format(k)])
                for k in self.gaussian_model.property_names
            })

        # get epoch and global_step, which used in the output path of the validation and test images
        self.restored_epoch = checkpoint["epoch"]
        self.restored_global_step = checkpoint["global_step"]

        # call for renderer
        self.renderer.on_load_checkpoint(self, checkpoint)
        # call density controller's hook
        self.density_controller.on_load_checkpoint(self, checkpoint)

        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint) -> None:
        # store some extra parameters
        # checkpoint["gaussian_model_extra_state_dict"] = {
        #     "max_radii2D": self.gaussian_model.max_radii2D,
        #     "xyz_gradient_accum": self.gaussian_model.xyz_gradient_accum,
        #     "denom": self.gaussian_model.denom,
        #     "spatial_lr_scale": self.gaussian_model.spatial_lr_scale,
        #     "active_sh_degree": self.gaussian_model.active_sh_degree,
        # }
        super().on_save_checkpoint(checkpoint)

    def tensorboard_log_image(self, tag: str, image_tensor):
        self.logger.experiment.add_image(
            tag,
            image_tensor,
            self.trainer.global_step,
        )

    def wandb_log_image(self, tag: str, image_tensor):
        image_dict = {
            tag: wandb.Image(image_tensor),
        }
        self.logger.experiment.log(
            image_dict,
            step=self.trainer.global_step,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if batch[0].device != self.device:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        camera, image_info, extra_data = batch
        image_name, gt_image, masked_pixels = image_info

        if extra_data is not None:
            extra_data = super().transfer_batch_to_device(extra_data, device, dataloader_idx)
        if gt_image is not None:
            gt_image = gt_image.to(device)
        if masked_pixels is not None:
            masked_pixels = masked_pixels.to(device)

        return camera, (image_name, gt_image, masked_pixels), extra_data

    def forward(self, camera):
        if self.training is True:
            if self.hparams.get("grad_checkpoint", False):
                # Gradient checkpointing (orthogonal VRAM lever to K-strips): don't
                # retain the render's forward activations (the ∝N backward buffers +
                # SB color graph, measured 2.4G@1.45M); recompute them during backward.
                # Trades ~1 extra forward for lower peak. Render is deterministic given
                # params+camera (MCMC noise is added elsewhere) so recompute is exact.
                import torch.utils.checkpoint as _ckpt

                def _render():
                    return self.renderer.training_forward(
                        self.trainer.global_step, self, camera, self.gaussian_model,
                        bg_color=self.get_background_color().to(camera.R.device),
                        render_types=self.renderer_output_types,
                    )
                return _ckpt.checkpoint(_render, use_reentrant=False)
            return self.renderer.training_forward(
                self.trainer.global_step,
                self,
                camera,
                self.gaussian_model,
                bg_color=self.get_background_color().to(camera.R.device),
                render_types=self.renderer_output_types,
            )
        return self.renderer(
            camera,
            self.gaussian_model,
            bg_color=self._fixed_background_color().to(camera.R.device),
            render_types=self.renderer_output_types,
        )

    def optimizers(self, use_pl_optimizer: bool = True):
        optimizers = super().optimizers(use_pl_optimizer=use_pl_optimizer)

        if isinstance(optimizers, list) is False:
            return [optimizers]

        """
        IMPORTANCE: the global_step will be increased on every step() call of all the optimizers,
        issue https://github.com/Lightning-AI/lightning/issues/17958,
        here change _on_before_step and _on_after_step to override this behavior.
        """
        for idx, optimizer in enumerate(optimizers):
            if idx == 0:
                continue
            optimizer._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
            optimizer._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        return optimizers

    def lr_schedulers(self) -> Union[None, List[LRSchedulerPLType], LRSchedulerPLType]:
        schedulers = super().lr_schedulers()

        if schedulers is None:
            return []

        if isinstance(schedulers, list) is False:
            return [schedulers]

        return schedulers

    def is_final_step(self, step: int = None):
        if step is None:
            step = self.trainer.global_step
        if self.trainer.max_steps > 0 and step >= self.trainer.max_steps:
            return True
        # TODO: make it works when max_epochs set
        return False

    def on_train_start(self) -> None:
        super().on_train_start()

        # kernel/color-representation banner: which rasterizer + color rep + floats/point
        if self.trainer.global_rank == 0:
            try:
                floats_per_point = sum(
                    int(torch.tensor(p.shape[1:]).prod()) if p.dim() > 1 else 1
                    for p in self.gaussian_model.properties.values()
                )
                color_rep = "SH{}".format(self.gaussian_model.max_sh_degree)
                if hasattr(self.gaussian_model, "get_sb_params"):
                    color_rep += "+SB{}".format(self.gaussian_model.config.sb_number)
                print("[kernel] renderer={} model={} color={} floats/point={}".format(
                    self.renderer.__class__.__name__,
                    self.gaussian_model.__class__.__name__,
                    color_rep,
                    floats_per_point,
                ))
            except Exception as e:  # banner must never break training
                print("[kernel] banner unavailable: {}".format(e))

        if self.hparams["web_viewer"] is True and self.trainer.global_rank == 0:
            if self.trainer.datamodule.hparams["parser"].__class__.__name__.lower() in ["blender", "nsvf", "matrixcity"]:
                up = torch.tensor([0., 0., 1.])
            else:
                c2w = self.trainer.datamodule.dataparser_outputs.train_set.cameras.world_to_camera[:, :3, :3]
                up = c2w[:, :3, 1].mean(dim=0)
                up = -up / torch.linalg.norm(up)
            self.web_viewer = TrainingViewer(
                camera_names=self.trainer.datamodule.dataparser_outputs.train_set.image_names,
                cameras=self.trainer.datamodule.dataparser_outputs.train_set.cameras,
                up_direction=up.cpu().numpy(),
                camera_center=self.trainer.datamodule.dataparser_outputs.train_set.cameras.camera_center.mean(dim=0).cpu().numpy(),
                available_appearance_options=self.trainer.datamodule.dataparser_outputs.appearance_group_ids,
            )
            self.web_viewer.start()

        for i in self.on_train_start_hooks:
            i(self.gaussian_model, self)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        if self.web_viewer is not None:
            self.web_viewer.training_step(
                self.gaussian_model,
                self.renderer,
                self._fixed_background_color(),
                self.trainer.global_step,
            )
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        # ★ 2026-09-18 驗證用（預設關）：CITYGS_BATCH_HASH_STEPS=N => 前 N 步印出送進訓練的 GT 影像雜湊
        #   用來證明資料管線改動（uint8 快取等）對訓練輸入逐位元不變 —— 不受 GPU 運算非確定性影響
        _hn = int(os.environ.get("CITYGS_BATCH_HASH_STEPS", "0") or 0)
        if _hn > 0 and self.trainer.global_step < _hn:
            import hashlib as _hl
            try:
                _cam, (_nm, _gt, _mk), _ex = batch
                _h = _hl.sha1(_gt.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()[:16]
                print(f"[batch-hash] step {self.trainer.global_step} {_nm} {tuple(_gt.shape)} {str(_gt.dtype).replace('torch.','')} {_h}", flush=True)
            except Exception as _e:
                print(f"⚠⚠ [batch-hash] 取不到影像：{type(_e).__name__} {_e}", flush=True)
        camera, image_info, _ = batch
        # image_name, gt_image, masked_pixels = image_info

        global_step = self.trainer.global_step + 1  # must start from 1 to prevent densify at the beginning

        # get optimizers and schedulers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        # zero grad
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        # save checkpoint
        # checkpoint will always be saved after final step, so do not save for final step here
        if global_step in self.hparams["save_iterations"] and self.is_final_step(global_step) is False and self.trainer.global_step != self.restored_global_step:
            self.save_gaussians()

        # call renderer hook
        _prof = getattr(self, "_step_prof", None)
        if _prof is None:
            _prof = self._step_prof = _StepProfiler()
        _prof.start_step()
        self.renderer.before_training_step(global_step, self)
        # ⚠ 這裡**只有起始 trim**（`if step != 1: return`）。週期性的 trim pass 在
        #   `after_training_step`（每 `contribution_prune_interval` 步、僅 densify 期），
        #   它不在 training_step 裡 => 另外標記，見 on_train_batch_end。
        _prof.mark("1 起始 trim（僅 step 1，一次性）")

        n_strips = self._decide_num_strips(camera, global_step)
        _prof.mark("2 decide_num_strips")
        if n_strips > 1:
            # K-strip tiled step: per-strip forward + backward bounds the rasterizer
            # buffers to ~1/K (budget-formula supply lever on the main-line kernel).
            outputs, metrics, prog_bar = self._strip_forward_backward(camera, batch, global_step, n_strips)
            _prof.mark("3-6 條帶 forward+loss+backward")
            self.log_metrics(metrics, prog_bar, prefix="train", on_step=True, on_epoch=False)
            self._finish_training_step(optimizers, schedulers, outputs, batch, global_step)
            _prof.end_step()
            return

        # forward
        outputs = self(camera)
        _prof.mark("3 forward（光柵化）")
        # metrics
        metrics, prog_bar = self.metric.get_train_metrics(self, self.gaussian_model, global_step, batch, outputs)
        _prof.mark("4 loss（L1+SSIM+正則）")
        self.log_metrics(metrics, prog_bar, prefix="train", on_step=True, on_epoch=False)
        _prof.mark("5 log_metrics")

        # invoke `before_backward` interface of density controller
        self.density_controller.before_backward(
            outputs=outputs,
            batch=batch,
            gaussian_model=self.gaussian_model,
            optimizers=self.gaussian_optimizers,
            global_step=global_step,
            pl_module=self,
        )
        # backward
        if self._split_backward_needed(metrics):
            self.manual_backward(metrics["loss"], retain_graph=True)
            grad_norm_avg = torch.norm(outputs["viewspace_points"].grad[outputs["visibility_filter"], :2], dim=-1, keepdim=True).mean()
            org_grad = outputs["viewspace_points"].grad.detach().clone()
            self.manual_backward(metrics["extra_loss"])
            grad_norm_avg_final = torch.norm(outputs["viewspace_points"].grad[outputs["visibility_filter"], :2], dim=-1, keepdim=True).mean()
            outputs["viewspace_points"].grad = org_grad * max(self.hparams["density"].densify_grad_scaler * grad_norm_avg_final / grad_norm_avg, 1.0)
        elif "extra_loss" in metrics:
            self.manual_backward(metrics["loss"] + metrics["extra_loss"])
        else:
            self.manual_backward(metrics["loss"])
        _prof.mark("6 backward")

        self._finish_training_step(optimizers, schedulers, outputs, batch, global_step)
        _prof.end_step()
        if _prof.on and _prof.steps % 200 == 0:
            _prof.report(os.path.join(self.hparams["output_path"], "step_cost.txt"),
                         self.gaussian_model.get_xyz.shape[0])

    def _split_backward_needed(self, metrics) -> bool:
        """Does anything downstream read the viewspace gradient this step?

        `CityGSV2Metrics` splits the objective while densification runs so that the viewspace
        gradient, read between the two backwards, carries the SSIM term only -- the signal
        gradient-based ADC densifies on (CityGaussianV2's DGD).

        MCMC does not read that gradient; `mcmc_2dgs_density_controller.py:58` states it outright.
        For those configs the split costs an extra backward and a retained graph every step, and
        the retained graph is the "backward activation" term of the VRAM peak model on 6GB.
        Backwarding the sum instead is gradient-identical -- the split is a partition of one sum,
        asserted in `tests/split_backward_equivalence_test.py`.

        Measured on `oreg_0p002_b12` across `densify_until_iter` at constant N (it hits cap 1M at
        step 29,999): 1.420 it/s while splitting, 2.620 it/s after. That boundary also stops
        relocation and contribution pruning, so the split is only part of the 84.5%, but an extra
        backward is roughly +67% of a step's cost on its own.

        Conservative by default: any controller that reads the gradient keeps the old path.
        """
        if "extra_loss" not in metrics:
            return False
        reader = getattr(self.density_controller, "READS_VIEWSPACE_GRAD", True)
        return bool(reader)

    def _decide_num_strips(self, camera, global_step):
        """Number of K-strips for this step. Fixed (train_strips) unless dynamic_strips
        is on, in which case predict it from the CURRENT geometry's render load so most
        steps/views run K=1 (fast) and only high-load moments pay for strips (no quality
        change — K is loss-invariant). Predictive (this step's positions), not reactive,
        with a safety margin absorbing the projection-proxy error -> keeps robustness."""
        if not self.hparams["dynamic_strips"]:
            return self.hparams["train_strips"]
        from internal.utils.strip_cameras import estimate_render_load, predict_num_strips
        gm = self.gaussian_model
        if self._floats_per_point is None:
            self._floats_per_point = sum(
                int(torch.tensor(p.shape[1:]).prod()) if p.dim() > 1 else 1
                for p in gm.properties.values())
        load = estimate_render_load(gm.get_xyz.detach(), gm.get_scales().detach(), camera)
        k = predict_num_strips(
            load, gm.get_xyz.shape[0], self._floats_per_point,
            self.hparams["strip_vram_target_gb"], self.hparams["strip_v_os_gb"],
            self.hparams["strip_safety"], self.hparams["strip_max"])
        # live, flushed K log (conda-run buffers stdout, so [dynamic-K] prints are
        # invisible mid-run; this dedicated file is written+flushed every 100 steps
        # with the actual VRAM so we can watch K adapt and compare estimate vs reality)
        if self.trainer.global_step % 100 == 0 and self.global_rank == 0:
            import torch as _t
            alloc = _t.cuda.memory_allocated() / 2**30
            peak = _t.cuda.max_memory_allocated() / 2**30
            load_str = "MONSTER" if load == float("inf") else "%.0fM" % (load / 1e6)
            line = ("step %d N=%d load=%s -> K=%d | VRAM alloc=%.2fG peak=%.2fG\n"
                    % (global_step, gm.get_xyz.shape[0], load_str, k, alloc, peak))
            try:
                with open(os.path.join(self.hparams["output_path"], "dynk_K.log"), "a") as _f:
                    _f.write(line); _f.flush()
            except Exception:
                pass
            _t.cuda.reset_peak_memory_stats()
        # fragmentation defense: a high-K (monster) step allocates then frees a big
        # transient buffer; empty_cache returns it to CUDA so a later step's small
        # contiguous alloc doesn't fail on fragmentation (dynk's actual death: 50MB
        # request failed with 5.18G reserved). Only when K>1 (rare) -> negligible cost.
        if k > 1:
            import torch as _t
            _t.cuda.empty_cache()
        return k

    def _strip_forward_backward(self, camera, batch, global_step, K):
        """K-strip tiled forward+backward (route-2 camera crop, see strip_cameras.py).

        Per-strip backward frees each strip's activations before the next render, so
        the rasterizer/activation footprint is bounded by the largest strip (~1/K).
        Per-point regularizers stay exact: strip losses are weighted by h/H and the
        weights sum to 1. Image-space SSIM is computed per strip (boundary-window
        approximation, validated equivalent on the gsplat probe line). Notes:
        - density_controller.before_backward is not invoked (no-op for MCMC family);
        - the viewspace grad-rescale dance is skipped (MCMC ignores viewspace grads);
        - radii / visibility_filter / viewspace-grad are aggregated (max / or / SUM).

        ⚠ 2026-09-11：`viewspace_points.grad` 的跨條帶加總是**後來補的**。在那之前
        `outputs` 只保留最後一條帶，而 `absgrad_densify`（2026-08-25 加入，現行最佳配方的
        核心機制）正是從 `outputs["viewspace_points"].grad[:, 2]` 讀 `|g|`
        ⇒ K=6 時取樣權重只由畫面最下面 1/6 的殘差決定，沒出現在該帶的粒子 `|g|=0`、
        退化成 `probs = o`（等同 absgrad 沒開）。下方的 assert 擋不到，因為它只看
        `READS_VIEWSPACE_GRAD`，而 MCMC 是 False（那個旗標問的是「**兩次 backward 之間**
        要不要拿梯度」，absgrad 是在**全部 backward 之後**才讀，兩件事不同）。
        加總是正確的：條帶是**不相交的像素集合**，且每條的 loss 已乘上 `h/H`
        ⇒ 加權梯度之和 == 全幀梯度。
        """
        from internal.utils.strip_cameras import make_strip_camera, strip_bounds, crop_batch

        H = int(camera.height)
        agg_metrics, prog_bar = {}, {}
        agg_radii, agg_vis, last_outputs = None, None, None
        agg_vsgrad, last_vsgrad = None, None
        for v0, v1 in strip_bounds(H, K):
            cam_s = make_strip_camera(camera, v0, v1 - v0)
            outputs = self(cam_s)
            batch_s = crop_batch(batch, v0, v1, cam_s)
            metrics, prog_bar = self.metric.get_train_metrics(self, self.gaussian_model, global_step, batch_s, outputs)
            w = (v1 - v0) / H
            if self._split_backward_needed(metrics):
                self.manual_backward(metrics["loss"] * w, retain_graph=True)
                self.manual_backward(metrics["extra_loss"] * w)
            elif "extra_loss" in metrics:
                self.manual_backward((metrics["loss"] + metrics["extra_loss"]) * w)
            else:
                self.manual_backward(metrics["loss"] * w)
            for k, v in metrics.items():
                try:
                    v = float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                except (TypeError, ValueError):
                    continue
                agg_metrics[k] = agg_metrics.get(k, 0.0) + v * w
            radii = outputs.get("radii", None)
            if radii is not None:
                agg_radii = radii if agg_radii is None else torch.maximum(agg_radii, radii)
            vis = outputs.get("visibility_filter", None)
            if vis is not None:
                agg_vis = vis if agg_vis is None else (agg_vis | vis)
            # 每條帶的 forward 各自產生一個 means2D（`torch.zeros_like(means, requires_grad=True)`）
            # => 各有自己的 .grad，條帶之間不共用。條帶是不相交的像素集合且 loss 已乘 h/H
            # => **相加**就是全幀梯度（.z 的 |g| 同理，它是逐像素 atomicAdd 上去的）。
            _vp = outputs.get("viewspace_points", None)
            _vg = getattr(_vp, "grad", None) if _vp is not None else None
            if _vg is not None:
                last_vsgrad = _vg.detach()
                agg_vsgrad = last_vsgrad.clone() if agg_vsgrad is None else agg_vsgrad + last_vsgrad
            last_outputs = outputs
        outputs = dict(last_outputs)
        # `outputs` is the LAST strip's dict; only the two fields that have a meaningful
        # cross-strip reduction are replaced. `radii` takes the max (a primitive's screen radius
        # does not depend on which strip saw it) and `visibility_filter` the OR.
        #
        # ⚠ Everything else in `outputs` -- `render`, `surf_depth`, `viewspace_points` and its
        # gradient -- still describes ONE strip. MCMC never reads them (its controllers touch no
        # `outputs[...]` field at all, verified 2026-08-06), so the mainline is unaffected. But a
        # GRADIENT-BASED density controller under K-strip would densify on the last strip's
        # viewspace gradient and silently ignore the rest of the frame. That combination is not
        # used and not supported; assert rather than let it run wrong.
        if agg_radii is not None:
            outputs["radii"] = agg_radii
        if agg_vis is not None:
            outputs["visibility_filter"] = agg_vis
        if agg_vsgrad is not None:
            vp_agg = torch.zeros_like(agg_vsgrad)
            vp_agg.grad = agg_vsgrad
            outputs["viewspace_points"] = vp_agg
            # 可觀測性鐵律：新機制必須印「✅首次觸發」，並出示修正前後的差
            if not getattr(self, "_strip_vsgrad_reported", False):
                self._strip_vsgrad_reported = True
                if getattr(self.hparams["density"], "absgrad_densify", 0) > 0 \
                        or getattr(self.hparams["density"], "absgrad_report", 0) > 0:
                    nz_last = int((last_vsgrad[:, 2] != 0).sum())
                    nz_agg = int((agg_vsgrad[:, 2] != 0).sum())
                    print(f"[K-strip] ✅ viewspace-grad 跨條帶加總首次觸發 K={K}："
                          f"|g| 非零粒子 最後一條帶 {nz_last:,} -> 全幀 {nz_agg:,} "
                          f"({nz_agg / max(nz_last, 1):.2f}x)。"
                          f"未修正前 absgrad 只看得到前者。", flush=True)
        assert not getattr(self.density_controller, "READS_VIEWSPACE_GRAD", True), (
            "K-strip training with a controller that needs the viewspace gradient BETWEEN the two "
            "backwards (READS_VIEWSPACE_GRAD): the strip path runs a single fused backward per "
            "strip, so that intermediate gradient never exists. Use K=1 or an MCMC controller.\n"
            "⚠ This assert does NOT cover `absgrad_densify`, which reads the viewspace gradient "
            "AFTER all backwards -- that case is handled by summing the per-strip grads above."
        )
        return outputs, agg_metrics, prog_bar

    def _finish_training_step(self, optimizers, schedulers, outputs, batch, global_step):
        # log learning rate and gaussian count every 100 iterations (without plus one step)
        if self.trainer.global_step % 100 == 0:
            metrics_to_log = {
                "train/gaussians_count": self.gaussian_model.get_xyz.shape[0],
            }
            for opt_idx, opt in enumerate(optimizers):
                if opt is None:
                    continue
                for idx, param_group in enumerate(opt.param_groups):
                    param_group_name = param_group["name"] if "name" in param_group else str(idx)
                    metrics_to_log["lr/{}_{}".format(opt_idx, param_group_name)] = param_group["lr"]
            self.logger.log_metrics(
                metrics_to_log,
                step=self.trainer.global_step,
            )

        _prof = getattr(self, "_step_prof", None)
        # invoke `after_backward` interface of density controller
        self.density_controller.after_backward(
            outputs=outputs,
            batch=batch,
            gaussian_model=self.gaussian_model,
            optimizers=self.gaussian_optimizers,
            global_step=global_step,
            pl_module=self,
        )
        # invoke other hooks
        if _prof is not None:
            # density controller 的 after_backward：densify 事件（每 densification_interval 步）
            # 與每步都跑的累積（absgrad |g|、screen-size prune 的 _max_radii2D 等）都在這裡
            _prof.mark("7 density_controller.after_backward")

        for i in self.on_after_backward_hooks:
            i(outputs, batch, self.gaussian_model, global_step, self)
        if _prof is not None:
            _prof.mark("8 on_after_backward_hooks")

        # optimize
        for optimizer in optimizers:
            optimizer.step()
        if _prof is not None:
            _prof.mark("9 optimizer.step")

        # schedule lr
        for scheduler in schedulers:
            scheduler.step()
        if _prof is not None:
            _prof.mark("10 scheduler.step")

    def light_gaussian_prune(self, global_step):
        # TODO: move elsewhere

        """
        LightGaussian prune
        """
        if global_step not in self.light_gaussian_hparams.prune_steps:
            return

        # 2DGS surfels have 2D scale and are incompatible with the gsplat-3D score path below
        # (gsplat asserts scales == (N, 3)) -> route to the native 2DGS importance prune.
        # Route by scale DIMENSION, not getter name (the runtime model exposes get_scaling but
        # it returns 2D scales for surfels).
        _get_scales = getattr(self.gaussian_model, "get_scales", None)
        _is_2dgs = callable(_get_scales) and _get_scales().shape[-1] == 2
        if _is_2dgs:
            from internal.utils.importance_prune_2dgs import native_2dgs_importance_prune
            prune_step_index = self.light_gaussian_hparams.prune_steps.index(global_step)
            prune_percent = self.light_gaussian_hparams.prune_percent * (self.light_gaussian_hparams.prune_decay ** prune_step_index)
            native_2dgs_importance_prune(self, prune_percent, self.light_gaussian_hparams.v_pow)
            return

        from internal.utils.light_gaussian import get_count_and_score
        from internal.utils.light_gaussian import calculate_v_imp_score
        from internal.utils.light_gaussian import get_prune_mask

        # try to detect whether anti aliased enabled
        try:
            anti_aliased = self.renderer.anti_aliased
        except:
            anti_aliased = False

        with torch.no_grad():
            count, score, _, _ = get_count_and_score(
                self.gaussian_model,
                self.trainer.datamodule.dataparser_outputs.train_set.cameras,
                anti_aliased,
            )
            v_list = calculate_v_imp_score(
                self.gaussian_model.get_scaling,
                score,
                self.light_gaussian_hparams.v_pow,
            )

            # TODO: `self.light_gaussian_hparams.prune_steps` should be sorted
            prune_step_index = self.light_gaussian_hparams.prune_steps.index(global_step)
            prune_percent = self.light_gaussian_hparams.prune_percent * (self.light_gaussian_hparams.prune_decay ** prune_step_index)
            prune_mask = get_prune_mask(prune_percent, v_list)

            print(f"number_of_gaussian={self.gaussian_model.get_xyz.shape[0]}, "
                  f"number_to_prune={prune_mask.sum().item()}, "
                  f"prune_percent={prune_percent}, "
                  f"anti_aliased={anti_aliased}")

            from internal.density_controllers.density_controller import Utils
            valid_points_mask = ~prune_mask  # `True` to keep
            self.gaussian_model.properties = Utils.prune_properties(valid_points_mask, self.gaussian_model, self.gaussian_optimizers)
            self.density_updated_by_renderer()

            print(f"number_of_gaussian_after_pruning={self.gaussian_model.get_xyz.shape[0]}")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # the value of `trainer.global_step` here
        # is the same as the local variable `global_step` in training_step
        global_step = self.trainer.global_step

        self.gaussian_model.on_train_batch_end(global_step, self)

        _prof = getattr(self, "_step_prof", None)
        if _prof is not None and _prof.on:
            import time as _t
            # ★ 2026-09-17 補：end_step 之後、trim 之前的 hook（gaussian_model.on_train_batch_end 等）
            #   原本**沒有歸到任何一段**（_prev_end 在 trim 後被覆寫）=> 補成第 12 段
            torch.cuda.synchronize()
            _c_tail = _t.perf_counter()
            if _prof._prev_end is not None:
                _prof._acc("12 on_train_batch_end（trim 之前的 hook）", None, (_c_tail - _prof._prev_end) * 1e3)
                _ta, _tr = _prof._mem()
                _prof._accmem("12 on_train_batch_end（trim 之前的 hook）", _ta, _tr)
            _e0 = torch.cuda.Event(enable_timing=True); _e0.record(); _c0 = _t.perf_counter()
            self.renderer.after_training_step(self.trainer.global_step, self)
            _e1 = torch.cuda.Event(enable_timing=True); _e1.record()
            torch.cuda.synchronize()
            _dt = (_t.perf_counter() - _c0) * 1e3
            _pa, _pr = _prof._mem()
            # 只在它真的做事的步數上累計（其餘步是 early-return，~0 ms）=> 觸發次數才有意義
            if _dt > 1.0:
                _prof._acc("11 週期性 trim pass（after_training_step）", _e0.elapsed_time(_e1), _dt)
                _prof._accmem("11 週期性 trim pass（after_training_step）", _pa, _pr)
            _prof._prev_end = _t.perf_counter()
        else:
            self.renderer.after_training_step(self.trainer.global_step, self)

        self.light_gaussian_prune(global_step)

        for i in self.on_train_batch_end_hooks:
            i(outputs, batch, self.gaussian_model, global_step, self)

        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        if self.web_viewer is not None:
            self.web_viewer.validation_step(
                self.gaussian_model,
                self.renderer,
                self._fixed_background_color(),
                batch_idx,
            )

    def validation_step(self, batch, batch_idx, name: str = "val"):
        camera, image_info, _ = batch
        gt_image = image_info[1]

        # forward
        if self.hparams["test_speed"]:
            torch.cuda.synchronize()
            start = time.time()
            outputs = self(camera)
            torch.cuda.synchronize()
            end = time.time()

            metrics, prog_bar = self.metric.get_validate_metrics(self, self.gaussian_model, batch, outputs)
            prog_bar["time(s)"] = True
            metrics["time(s)"] = torch.tensor(end - start)
        else:
            outputs = self(camera)
            metrics, prog_bar = self.metric.get_validate_metrics(self, self.gaussian_model, batch, outputs)
        self.log_metrics(metrics, prog_bar, prefix=name, on_step=False, on_epoch=True)
        self.val_metrics.append((image_info[0], metrics))

        # write validation image
        if self.trainer.global_rank == 0 and self.hparams["save_val_output"] is True and (
                self.hparams["max_save_val_output"] < 0 or batch_idx < self.hparams["max_save_val_output"]
        ):
            output_images = []
            if self.renderer_output_types is None:
                output_images.append(outputs["render"].cpu())
            else:
                for i in self.renderer_output_types:
                    output_images.append(self.renderer_output_visualizers[i](outputs).cpu())
            if "extra_image" in outputs:
                output_images.append(outputs["extra_image"].cpu())
            self.image_queue.put({
                "output_images": output_images,
                "gt_image": gt_image.cpu(),
                "stage": name,
                "image_name": image_info[0],
                "epoch": max(self.trainer.current_epoch, self.restored_epoch),
                "step": max(self.trainer.global_step, self.restored_global_step),
            })

            # if self.log_image is not None:
            #     grid = torchvision.utils.make_grid(torch.concat([outputs["render"], gt_image], dim=-1))
            #     self.log_image(
            #         tag="{}_images/{}".format(name, image_info[0].replace("/", "_")),
            #         image_tensor=grid,
            #     )
            #
            # image_output_path = os.path.join(
            #     self.hparams["output_path"],
            #     name,
            #     "epoch={}-step={}".format(
            #         max(self.trainer.current_epoch, self.restored_epoch),
            #         max(self.trainer.global_step, self.restored_global_step),
            #     ),
            #     "{}.png".format(image_info[0].replace("/", "_"))
            # )
            # os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            # torchvision.utils.save_image(
            #     torch.concat([outputs["render"], gt_image], dim=-1),
            #     image_output_path,
            # )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.hparams["save_val_output"] is True:
            from internal.utils.visualizers import Visualizers
            self.renderer_output_visualizers = Visualizers.get_simplified_visualizer_by_renderer_output_info(self.renderer.get_available_outputs())

            for i in range(self.max_image_saving_threads):
                thread = threading.Thread(target=self.save_images)
                self.image_saving_threads.append(thread)
                thread.start()

    def on_validation_epoch_end(self, name="val") -> None:
        super().on_validation_epoch_end()
        for i in range(len(self.image_saving_threads)):
            self.image_queue.put(None)
        for i in self.image_saving_threads:
            i.join()
        self.image_saving_threads = []
        with open(os.path.join(self.hparams["output_path"], "results.txt"),'w') as file:
            for k in sorted (self._trainer.logged_metrics.keys()):
                file.write("%s: %s, \n" % (k, self._trainer.logged_metrics[k].item()))

        # best-val tracking (2026-07-21): record the peak val/psnr + step so downstream
        # knows if the LAST checkpoint is the best one. For the current 60k recipe the
        # harvest jump makes last==best (no-op), but this is the safety net for held-out
        # eval where late overfitting could make an intermediate step genuinely best.
        # Lightweight by design: records step/psnr only, does not auto-save a checkpoint.
        if name == "val" and "val/psnr" in self._trainer.logged_metrics and self.global_rank == 0:
            cur = float(self._trainer.logged_metrics["val/psnr"].item())
            step = max(self.trainer.global_step, self.restored_global_step)
            if cur > self._best_val_psnr:
                self._best_val_psnr = cur
                self._best_val_step = step
                with open(os.path.join(self.hparams["output_path"], "best_val.txt"), "w") as f:
                    f.write("best_val_psnr: %.4f\nbest_val_step: %d\n" % (cur, step))
                    nearest = min(self.hparams["save_iterations"], key=lambda s: abs(s - step)) \
                        if self.hparams["save_iterations"] else step
                    f.write("nearest_saved_ckpt_step: %d\n" % nearest)

        # save metrics
        if self.hparams["save_val_metrics"] is True and self.global_rank == 0 and len(self.val_metrics) > 0:
            metrics_output_dir = os.path.join(self.hparams["output_path"], "metrics")
            os.makedirs(metrics_output_dir, exist_ok=True)
            step = max(self.trainer.global_step, self.restored_global_step)

            metric_list_key_by_name = {}  # [metric_name] = metric_value_list
            metric_fields = list(self.val_metrics[0][1].keys())
            for i in metric_fields:
                metric_list_key_by_name[i] = []

            with open(os.path.join(metrics_output_dir, f"{name}-step={step}.csv"), "w") as f:
                metrics_writer = csv.writer(f)
                metrics_writer.writerow(["name"] + list(metric_fields))

                for image_name, metrics in self.val_metrics:
                    metric_row = [image_name]
                    for metric_name in metric_fields:
                        metric_list_key_by_name[metric_name].append(metrics[metric_name])
                        metric_row.append("{:.8f}".format(metrics[metric_name].item()))
                    metrics_writer.writerow(metric_row)

                # calculate mean metrics
                metrics_writer.writerow([""] + ["" for _ in range(len(metric_fields))])
                mean_metrics = ["MEAN"]
                for i in metric_fields:
                    mean_metrics.append("{:.8f}".format(torch.stack(metric_list_key_by_name[i]).mean(dim=0).item()))
                metrics_writer.writerow(mean_metrics)

        self.val_metrics.clear()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self.on_validation_epoch_end(name="test")

    def save_images(self):
        while True:
            item = self.image_queue.get()
            if item is None:
                break

            try:
                image_list = [item["gt_image"]]
                for i in item["output_images"]:
                    image_list.append(i)
                image = torch.concat(image_list, dim=-1)

                if self.log_image is not None:
                    grid = torchvision.utils.make_grid(image)
                    self.log_image(
                        tag="{}_images/{}".format(item["stage"], item["image_name"].replace("/", "_")),
                        image_tensor=grid,
                    )

                image_output_path = os.path.join(
                    self.hparams["output_path"],
                    item["stage"],
                    "epoch={}-step={}".format(
                        item["epoch"],
                        item["step"],
                    ),
                    "{}.png".format(item["image_name"].replace("/", "_"))
                )
                os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                torchvision.utils.save_image(
                    image,
                    image_output_path,
                )
            except:
                traceback.print_exc()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        # initialize lists that store optimizers and schedulers
        optimizers = []
        schedulers = []

        def add_optimizers_and_schedulers(new_optimizers, new_schedulers):
            nonlocal optimizers
            nonlocal schedulers

            if new_optimizers is not None:
                if isinstance(new_optimizers, list):
                    optimizers += new_optimizers
                else:
                    optimizers.append(new_optimizers)
            if new_schedulers is not None:
                if isinstance(new_schedulers, list):
                    schedulers += new_schedulers
                else:
                    schedulers.append(new_schedulers)

        # gaussian model optimizer and scheduler setup
        gaussian_optimizers, gaussian_schedulers = self.gaussian_model.training_setup(self)
        self.gaussian_optimizers = gaussian_optimizers
        if isinstance(self.gaussian_optimizers, list) is False:
            self.gaussian_optimizers = [self.gaussian_optimizers]
        add_optimizers_and_schedulers(gaussian_optimizers, gaussian_schedulers)
        # add frozen Gaussians
        if self.frozen_gaussians is not None:
            from internal.utils.gaussian_containers import HasExtraParameters
            self.gaussian_model.gaussians = HasExtraParameters(self.frozen_gaussians, self.gaussian_model.gaussians)

        # renderer optimizer and scheduler setup
        renderer_optimizer, renderer_scheduler = self.renderer.training_setup(self)
        add_optimizers_and_schedulers(renderer_optimizer, renderer_scheduler)

        # metric optimizer and scheduler setup
        metric_optimizer, metric_scheduler = self.metric.training_setup(self)
        add_optimizers_and_schedulers(metric_optimizer, metric_scheduler)

        return optimizers, schedulers

    def density_updated_by_renderer(self):
        self.density_controller.after_density_changed(self.gaussian_model, self.gaussian_optimizers, self)

    def save_gaussians(self):
        is_mp_strategy = isinstance(self.trainer.strategy, internal.mp_strategy.MPStrategy)
        if self.trainer.global_rank != 0 and is_mp_strategy is False:
            return

        if self.hparams["save_ply"] is True:
            from internal.utils.gaussian_utils import GaussianPlyUtils
            # save ply file
            filename = "point_cloud.ply"
            # if self.trainer.global_rank != 0:
            #     filename = "point_cloud_{}.ply".format(self.trainer.global_rank)
            with torch.no_grad():
                output_dir = os.path.join(self.hparams["output_path"], "point_cloud",
                                          "iteration_{}".format(self.trainer.global_step))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                GaussianPlyUtils.load_from_model(self.gaussian_model).to_ply_format().save_to_ply(output_path + ".tmp")
                os.rename(output_path + ".tmp", output_path)

            print("Gaussians saved to {}".format(output_path))

        # save checkpoint
        checkpoint_name_suffix = ""
        if is_mp_strategy is True:
            checkpoint_name_suffix = f"-rank={self.global_rank}"

        checkpoint_path = os.path.join(
            self.hparams["output_path"],
            "checkpoints",
            "epoch={}-step={}{}.ckpt".format(self.trainer.current_epoch, self.trainer.global_step, checkpoint_name_suffix),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.trainer.save_checkpoint(checkpoint_path)
        # ★ 2026-09-17 使用者決定：預設**不再**輸出 -xyz_rgb.ply —— 它只有 xyz＋法線＋RGB（27 B/顆），
        #   不是可渲染的 Gaussian 模型，每個 60k 跑次卻存 6 份（約 300 MB）。要恢復：CITYGS_EXPORT_XYZ_RGB_PLY=1
        if os.environ.get("CITYGS_EXPORT_XYZ_RGB_PLY", "") != "1":
            if not getattr(self, "_xyz_rgb_ply_note", False):
                self._xyz_rgb_ply_note = True
                print("[ckpt] -xyz_rgb.ply 不再輸出（CITYGS_EXPORT_XYZ_RGB_PLY=1 可恢復）", flush=True)
        else:
          with torch.no_grad():
            xyz = self.gaussian_model.get_xyz
            rgb = eval_sh(0, self.gaussian_model.get_features[:, :1, :].transpose(1, 2), None)
            store_ply(os.path.join(
                self.hparams["output_path"],
                "checkpoints",
                "epoch={}-step={}{}-xyz_rgb.ply".format(self.trainer.current_epoch, self.trainer.global_step, checkpoint_name_suffix),
            ), xyz.cpu().numpy(), ((rgb + 0.5).clamp(min=0., max=1.) * 255).to(torch.int).cpu().numpy())
        print("Checkpoint saved to {}".format(checkpoint_path))

    def set_datamodule_device(self, device):
        # whether trainer exists
        try:
            self.trainer
        except RuntimeError:
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        datamodule.set_device(device)

    def _on_device_updated(self):
        self.metric.on_parameter_move(device=self.device)
        self.set_datamodule_device(self.device)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)

        self._on_device_updated()

        return self

    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:
        super().cuda(device)

        self._on_device_updated()

        return self

    def cpu(self) -> Self:
        super().cpu()

        self._on_device_updated()

        return self
