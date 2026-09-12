# Configs 導航（本研究專用）

> 本表只列**本研究（6GB city-scale，MatrixCity aerial）自建的 config**。
> 上游 gaussian-splatting-lightning 範例（`blender*`, `deformable*`, `gsplat*`, `mip_splatting*`,
> `distributed*`, `ddp*`, `mcmc.yaml`, `image_on_gpu*`, `vanilla_2dgs` …）與 CityGaussian 官方
> 各場景 config（`citygsv2_*`, `citygs_*`）**不在此列**，請勿與本研究 config 混用。
> 死掉的歷史一次性實驗已移至 `configs/archive/`。

## 本研究 active config

| config | init | density | densify / reset | 用途 / 現況 |
|---|---|---|---|---|
| `RTG_mc_no_ADDM_aerial_sh2_trim.yaml` | depth | RTGStable | 0.0002 / 2000 | **正規 depth-init baseline**。block7=21.967、block16=23.710。多數對照以此為準 |
| `RTG_resetoff_aerial_sh2_trim.yaml` | depth | RTGStable | 0.0002 / **off(999999)** | **reset-off 實驗**（= baseline 關 opacity-reset）。驗證 depth reg 是否足以抑制 floater = 換 MCMC 的前置。詳見 `紀錄/主線_Gaussian效率.md` |
| `RTG_mc_aerial_sh2_trim24.yaml` | depth | RTGStable | 0.0002 / 2000 | base RTG trim（depth_loss 排程到 30k）。歷史上多個實驗的基底 |
| `RTG_mc_aerial_sh2_trim24_coarse.yaml` | coarse | RTGStable | **0.00005** / 2000 | coarse-init 基底（densify 細、immune=false）。**注意：0.00005 在 6GB 對重內容 block OOM** |
| `AB_coarse_md_aerial_sh2_trim.yaml` | coarse | RTGStable | 0.0002 / 2000 | **coarse-init A/B（densify 對齊乾淨版）**。去 confound + 不 OOM。block7=22.373、block16=24.012 |
| `AB_coarse_init_aerial_sh2_trim.yaml` | coarse | RTGStable | 0.00005 / 2000 | ⚠ **已作廢**：confound + OOM 版 coarse A/B（block7=24.069 不可信）。保留作記錄，勿用於新實驗 |
| `dt_admm_gat_mc_aerial.yaml` | depth | **DtAdmm** | 0.0002 / 2000 | **ADMM 正確 config**（吃 admm_state_path/rho）。task2 用。coordinator 必須配這顆 |
| `dt_admm_gat_mc_aerial_v2.yaml` | depth | DtAdmm | 0.0002 | ADMM v2 變體 |
| `ADMM_RTG_mc_aerial_sh2_trim.yaml` | depth | RTGStable | 0.0002 / 2000 | ⚠ 命名誤導：叫 ADMM 但 density 是 RTGStable（**不吃 admm_state_path**）。被 scripts 引用，保留 |

## 論文溯源（component → 論文 → 檔案 → 關鍵 flag）

| component | 來源論文 | 我們的檔案 | 關鍵參數 |
|---|---|---|---|
| Gaussian2D（2D surfel）| 2DGS（`2dgs.pdf`）| `internal/models/gaussian_2d.py` | sh_degree；scale 為 **2 維** |
| SepDepthTrim2DGS renderer | CityGSV2（Trim2DGS）+ RaDe-GS 深度 | `internal/renderers/sep_depth_trim_2dgs_renderer.py` | depth_ratio |
| RTGStable density | RTG-SLAM（`RTG-SLAM`）stable/unstable 管理 | `internal/density_controllers/rtg_stable_density_controller.py` | stable_threshold, depth_init_immune, freeze_opacity, opacity_reset_interval |
| CityGSV2 density | CityGaussianV2（`2411.00771v2.pdf`）| `internal/density_controllers/citygsv2_density_controller.py` | cap_max |
| DtAdmm density | 本研究 DT-ADMM-GAT（+ DOGS 共識）| `internal/density_controllers/dt_admm_density_controller.py` | rho, admm_state_path, feat_std |
| MCMC density（3D，待 2D 適配）| 3DGS-MCMC（NeurIPS'24）| `internal/density_controllers/mcmc_density_controller.py` | cap_max, noise_lr；需 gsplat。詳見 `紀錄/MCMC整合分析.md` |
| CityGSV2 metrics（normal+depth）| CityGSV2 + RaDe-GS（`RaDe-GS.pdf`）| `internal/metrics/citygsv2_metrics.py` | lambda_normal, depth_loss_* |
| depth-init（alpha=0.99 不透明）| RTG-SLAM | `utils/depth_init_blocks.py` | INIT_ALPHA=0.99 |
| scene 分區 | CityGaussianV1（`CityGaussian: Real-time...pdf`）| `internal/utils/citygs_partitioning_utils.py` | block_dim, content_threshold |
| boundary graph / GAT z-update | 本研究 DT-ADMM-GAT | `internal/utils/boundary_graph.py`, `internal/renderers/dt_admm_gat_renderer.py` | d_boundary, gat_lr |

## archive/（死掉的歷史一次性實驗，2026-06-07 移入）
`RTG_s10_*`（5 個 S10 era freeze/b3/b5 消融）、`RTG_mc_aerial_sh2_trim24-ver2`。無 scripts 引用，純歷史記錄。
