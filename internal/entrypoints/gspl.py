import sys
from internal.cli import CLI
from jsonargparse import lazy_instance
from lightning.pytorch.cli import ArgsType

from internal.gaussian_splatting import GaussianSplatting
from internal.dataset import DataModule
from internal.callbacks import SaveGaussian, KeepRunningIfWebViewerEnabled, StopImageSavingThreads, ProgressBar, TrainConsole, ValidateOnTrainEnd, StopDataLoaderCacheThread


def _apply_vram_cap():
    """`CITYGS_VRAM_CAP_GB=5.66` => 把本行程的 VRAM 上限鎖在該值，超過就 OOM。

    為什麼需要：lab 主機是 RTX 3090（24 GiB），而本專案整篇的立論是**6GB 可行性**。
    在 24 GiB 上調出來的配方若超過 6 GiB，對命題完全沒有用。
    這個旗標讓 lab 上的跑次留在 6GB 的信封內 —— 換到的是**同組態的平行吞吐**
    （24/6 = 4 個並跑），而不是更大的模型。

    實作用 `torch.cuda.set_per_process_memory_fraction`（torch 2.0.1 就有）：
    它把快取配置器的上限設成 `fraction x 裝置總量`，超過就丟 `OutOfMemoryError`，
    行為與真的小卡一致。
    ⚠ fraction 是**相對於當下裝置**的比例，所以要用「目標 GB / 本機總量」換算，
      這樣同一個環境變數在 6GB 與 24GB 上都給出相同的絕對上限。
    ⚠ 只影響 PyTorch 的配置器；光柵器自己 `cudaMalloc` 的部分不受管
      => 這是**下界**，真實峰值可能略高於設定值。
    ⚠ 本機（5.66 GiB 可用）設 5.66 等於不變，可安全留在腳本裡。
    """
    import os
    cap = os.environ.get("CITYGS_VRAM_CAP_GB", "").strip()
    if not cap:
        return
    import torch
    if not torch.cuda.is_available():
        print(f"[vram-cap] ⚠⚠ CITYGS_VRAM_CAP_GB={cap} 但沒有可用的 CUDA 裝置，忽略", flush=True)
        return
    try:
        want = float(cap)
    except ValueError:
        print(f"[vram-cap] ⚠⚠ CITYGS_VRAM_CAP_GB={cap!r} 不是數字，忽略", flush=True)
        return
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    frac = min(max(want / total, 0.01), 1.0)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"[vram-cap] ✅ 鎖在 {want:.2f} GiB / 本機 {total:.2f} GiB "
          f"=> fraction {frac:.4f}（超過即 OOM）", flush=True)


def cli(args: ArgsType = None):
    _apply_vram_cap()
    CLI(
        GaussianSplatting,
        DataModule,
        seed_everything_default=42,
        auto_configure_optimizers=False,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "auto",
            "devices": 1,
            # "logger": "TensorBoardLogger",
            "num_sanity_val_steps": 1,
            # "max_epochs": -1,
            "max_steps": 30_000,
            "use_distributed_sampler": False,  # use custom ddp sampler
            "enable_checkpointing": False,
            "callbacks": [
                lazy_instance(SaveGaussian),
                lazy_instance(ValidateOnTrainEnd),
                lazy_instance(KeepRunningIfWebViewerEnabled),
                lazy_instance(StopImageSavingThreads),
                lazy_instance(TrainConsole),   # 預設進度條：TTY+rich → 固定底欄面板；否則 fallback tqdm（= 原 ProgressBar）
                lazy_instance(StopDataLoaderCacheThread),
            ],
        },
        save_config_kwargs={"overwrite": True},
        args=args,
    )
    # note: don't call fit!!


def cli_with_subcommand(subcommand: str):
    sys.argv.insert(1, subcommand)
    cli()


def cli_fit():
    cli_with_subcommand("fit")


def cli_val():
    cli_with_subcommand("validate")


def cli_test():
    cli_with_subcommand("test")


def cli_predict():
    cli_with_subcommand("predict")
