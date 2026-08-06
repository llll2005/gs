"""
Gaussian 初始化方式的可換模組（Initializer）。

背景：初始化有兩種「互斥」方式（見 gaussian_splatting.py 舊 setup 邏輯）：
  1. 從 SfM 點雲冷啟動（原 `initialize_from: null`）
  2. 從已訓練 ckpt / depth-init PLY 載入（原 `initialize_from: <path>`）

本模組把它們做成跟 renderer/density/metric 一樣的 class_path + init_args 可換元件，
config 上給出明確開關。**向後相容**：GaussianSplatting 只有在 config 有指定
`initializer` 時才走這裡；沒指定就走原本的 initialize_from 路徑（行為完全不變）。

各 Impl 直接「複用」GaussianSplatting 已驗證的初始化程式（setup_from_pcd /
_initialize_from_trained_model），不重寫邏輯，降低風險。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Initializer:
    """所有初始化方式的基底。子類實作 instantiate() 回傳對應 Impl。"""

    def instantiate(self, *args, **kwargs) -> "InitializerImpl":
        raise NotImplementedError()


class InitializerImpl:
    def __init__(self, config: Initializer) -> None:
        self.config = config

    def initialize(self, pl_module, stage: str) -> None:
        """在 fit setup 階段被呼叫一次，把初始高斯灌進 pl_module.gaussian_model。"""
        raise NotImplementedError()


@dataclass
class PointCloudInitializer(Initializer):
    """從 dataparser 的 SfM 點雲冷啟動（等價於原 `initialize_from: null`）。"""

    def instantiate(self, *args, **kwargs) -> "PointCloudInitializerImpl":
        return PointCloudInitializerImpl(self)


class PointCloudInitializerImpl(InitializerImpl):
    def initialize(self, pl_module, stage: str) -> None:
        point_cloud = pl_module.trainer.datamodule.point_cloud
        pl_module.gaussian_model.setup_from_pcd(
            xyz=point_cloud.xyz,
            rgb=point_cloud.rgb / 255.,
        )


@dataclass
class CheckpointInitializer(Initializer):
    """從已訓練 ckpt 或 PLY（例如 RTG depth-init）載入（等價於原 `initialize_from: <path>`）。

    path: ckpt 檔 / 輸出目錄（會 search_load_file 自動找最新）/ .ply 檔。
    overwrite_config: True=用 ckpt 內存的 config 覆蓋本檔；False=用本檔設定、只取權重
        （本檔 sh_degree 高於 ckpt 時會自動 padding SH）。等同原 model.overwrite_config。
    """

    path: Optional[str] = None
    overwrite_config: bool = True

    def instantiate(self, *args, **kwargs) -> "CheckpointInitializerImpl":
        return CheckpointInitializerImpl(self)


class CheckpointInitializerImpl(InitializerImpl):
    def initialize(self, pl_module, stage: str) -> None:
        assert self.config.path is not None, "CheckpointInitializer.path 不可為 None"
        # 複用 GaussianSplatting 既有、已驗證的載入邏輯（含 .ply/.ckpt 分支 + SH padding）。
        pl_module._initialize_from_trained_model(
            load_path=self.config.path,
            overwrite_config=self.config.overwrite_config,
        )
