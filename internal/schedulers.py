from typing import Literal, Optional, Any
from dataclasses import dataclass, field
from internal.configs.instantiate_config import InstantiatableConfig
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
import numpy as np


# `ExponentialDecayScheduler` is copied from NeRFStudio

class Scheduler(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> "SchedulerImpl":
        raise NotImplementedError()


class SchedulerImpl:
    """Base scheduler"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        """Abstract method that returns a scheduler object.

        Args:
            optimizer: The optimizer to use.
            lr_init: The initial learning rate.
        Returns:
            The scheduler object.
        """
        raise NotImplementedError()


@dataclass
class ExponentialDecayScheduler(Scheduler):
    """Config for exponential decay scheduler with warmup"""

    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    lr_final: Optional[float] = None
    """Final learning rate. If not provided, it will be set to the optimizers learning rate."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    max_steps: int = 100000
    """The maximum number of steps."""
    ramp: Literal["linear", "cosine"] = "cosine"
    """The ramp function to use during the warmup."""
    hold_steps: int = 0
    """>0 時（2026-10-24，DashGaussian §7.2）：warmup 之後先維持 lr_init 到這一步，再從這一步起在 [hold_steps, max_steps] 內指數衰減到 lr_final。
    DashGaussian 讓位置學習率在解析度升到全解析度之前不衰減。0＝原行為。"""

    def instantiate(self, *args, **kwargs) -> Any:
        return ExponentialDecaySchedulerImpl(self)


class ExponentialDecaySchedulerImpl(SchedulerImpl):
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """

    config: ExponentialDecayScheduler

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        if self.config.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.config.lr_final
        if int(getattr(self.config, "hold_steps", 0) or 0) > 0:
            print(f"[lr-hold] ✅ 學習率 {lr_init:g} 保持到 step {self.config.hold_steps}，之後到 step {self.config.max_steps} 衰減到 {lr_final:g}", flush=True)

        def func(step):
            if lr_final <= 0 or lr_init <= 0:
                return 1.0

            if step < self.config.warmup_steps:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                            self.config.lr_pre_warmup
                            + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                    )
            else:
                t0 = max(self.config.warmup_steps, int(getattr(self.config, "hold_steps", 0) or 0))
                t = np.clip(
                    (step - t0) / max(1, self.config.max_steps - t0), 0, 1
                )
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
