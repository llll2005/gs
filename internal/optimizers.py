from typing import Tuple
from dataclasses import dataclass
from typing import Any
import torch
from internal.configs.instantiate_config import InstantiatableConfig


@dataclass
class OptimizerConfig(InstantiatableConfig):
    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        raise NotImplementedError()


@dataclass
class Adam(OptimizerConfig):
    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        return torch.optim.Adam(
            params,
            lr,
            *args,
            **kwargs,
        )


@dataclass
class SelectiveAdam(OptimizerConfig):
    betas: Tuple[float, float] = (0.9, 0.999)

    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        for group in params:
            if "lr" not in group:
                group["lr"] = lr

        from gsplat.optimizers import SelectiveAdam
        from torch.optim.optimizer import _use_grad_for_differentiable

        class Adapter(SelectiveAdam):
            def on_after_backward(self, outputs, batch, gaussian_model, global_step, pl_module):
                self.visibility = outputs["viewspace_points"].has_hit_any_pixels

            @_use_grad_for_differentiable
            def step(self, closure=None):
                self._cuda_graph_capture_health_check()

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                super().step(self.visibility)

                return loss

        return Adapter(
            params,
            betas=self.betas,
            *args,
            **kwargs,
        )


@dataclass
class SparseGaussianAdam(OptimizerConfig):
    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        from diff_accel_gaussian_rasterization import SparseGaussianAdam
        from torch.optim.optimizer import _use_grad_for_differentiable

        class Adapter(SparseGaussianAdam):
            def on_after_backward(self, outputs, batch, gaussian_model, global_step, pl_module):
                self.visibility = outputs["visibility_filter"]

            @_use_grad_for_differentiable
            def step(self, closure=None):
                self._cuda_graph_capture_health_check()

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                super().step(self.visibility, self.visibility.shape[0])

                return loss

        return Adapter(
            params,
            lr,
            *args,
            **kwargs,
        )

@dataclass
class LowPrecMomentAdam(OptimizerConfig):
    """Adam，但把兩個動量存成低精度 —— **不動模型表達力，只壓純開銷**。

    為什麼是這條而不是 SH2/SB（研究總覽 §11.66/§11.79）：
    ```
    VRAM 拆解  逐顆儲存 928 B（83%）／binning 191 B（17%）
    儲存拆解   參數 232 + 梯度 232 + exp_avg 232 + exp_avg_sq 232
               => **動量佔儲存的一半，且完全不影響模型**
    對照 已測且落敗的兩條（都是拿表達力換顆數）：
      sh2 + cap3.06M（等 VRAM）  PSNR **-16.4sd** => -0.454 dB
      SB-4 + 35% 顆粒            六項指標**全輸**
    ```
    `moment_dtype`：`bfloat16`（預設，指數範圍同 fp32）／`float16`（指數範圍小，有下溢風險）。

    ⚠⚠ **`stochastic_rounding` 預設 True，且不可關掉來跑正式實驗** ——
    參考論文 `2603.16731v1.pdf`（Topollai & Choromanska, *Understanding Quantization of
    Optimizer States*）指出低精度 EMA 會 **stall**：名目更新在量化後捨回同一個值。
    我方對自己的 beta 值推導出**更嚴重的後果**：
    ```
    v += (1-b2)·(g² − v)，b2=0.999 => 相對變化 = 0.001·|g²/v − 1|
    bf16 捨入門檻 = 2^-8/2 = 0.00195  =>  需 |g²/v − 1| > **1.953**
    但 g² >= 0  =>  向下更新時 |g²/v − 1| = 1 − g²/v <= 1 < 1.953
    => **v 永遠無法變小，只能往上跳 = 單向棘輪**
    => Adam 步長 lr·m/sqrt(v) 單調萎縮 => 模型逐漸凍結
    ```
    隨機捨入讓 `E[round(x)] = x`（無偏），棘輪消失，代價是多一點變異數。
    ⚠ `exp_avg`（b1=0.9）門檻只有 2%，本來就不會 stall —— 問題**專屬於第二動量**。

    ⚠ 數學一律在 **fp32** 做，只有**儲存**是低精度。
    ⚠ 需要 `density_controller.cat_tensors_to_optimizers_` 保留 dtype
      （否則 `torch.zeros_like(extension_tensor)` 會在第一次 densify 就把動量升回 fp32，
       而且**不會報錯** —— 已修，見該處註解）。
    ⚠ 非位元等價 => 必須端到端驗四指標。
    """

    moment_dtype: str = "bfloat16"
    stochastic_rounding: bool = True

    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        return _LowPrecMomentAdam(params, lr=lr,
                                  moment_dtype=getattr(torch, self.moment_dtype),
                                  stochastic_rounding=self.stochastic_rounding,
                                  **kwargs)


class _LowPrecMomentAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 moment_dtype=torch.bfloat16, stochastic_rounding=True, **_ignored):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.moment_dtype = moment_dtype
        self.stochastic_rounding = stochastic_rounding

    def _store(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """把 fp32 的 `src` 寫進低精度的 `dst`；隨機捨入使 `E[dst] = src`（無偏）。

        ⚠⚠ 第一版用 `torch.nextafter` **是靜默無效的**：那是在 **fp32** 上取下一個值，
        `.to(bfloat16)` 之後又捨回**同一個 bf16 值** => span=0 => prob=0 => 永遠不動。
        實測兩種捨入給出**完全相同**的 v（比值 1.0000），才發現。
        正解：bf16 就是 fp32 砍掉低 16 位尾數 => **在截斷前加一個 16-bit 隨機數**。
        """
        if not self.stochastic_rounding or dst.dtype is not torch.bfloat16:
            dst.copy_(src)          # fp16 不走這條（位元佈局不同），維持確定捨入
            return
        xi = src.contiguous().view(torch.int32)
        r = torch.randint(0, 1 << 16, src.shape, dtype=torch.int32, device=src.device)
        dst.copy_(((xi + r) & ~0xFFFF).view(torch.float32).to(torch.bfloat16))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["betas"]
            lr, eps = g["lr"], g["eps"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    st["exp_avg"] = torch.zeros_like(p, dtype=self.moment_dtype)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=self.moment_dtype)
                st["step"] += 1
                t = st["step"]
                grad = p.grad.float()
                # 讀出 -> fp32 運算 -> 寫回低精度
                m = st["exp_avg"].float().mul_(b1).add_(grad, alpha=1 - b1)
                v = st["exp_avg_sq"].float().mul_(b2).addcmul_(grad, grad, value=1 - b2)
                self._store(st["exp_avg"], m)
                self._store(st["exp_avg_sq"], v)
                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                p.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(eps), value=-lr)
        return loss
