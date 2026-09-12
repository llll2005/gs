# CityGaussian 參數速查手冊

> 與 `configs/_REFERENCE_annotated.yaml` **內容一致、互為對照**。
> YAML = 可直接複製去跑的完整版；本 MD = 純速查表，掃讀用。**改參數時兩份都要更新。**
> 所有數值已逐檔對照原始碼（2026-06-30），與 dataclass 預設一致。

## 圖例

- **★** = 關鍵旋鈕（動它最影響顆數 / 品質 / 記憶體）。
- **∅sh0** = `sh_degree=0` 時此參數無作用。
- **預設** = 不寫時程式碼用的值（dataclass 預設）。
- **基底** = 本參考檔（原版 CityGSV2 aerial sh0 fine-tune）實際用的值；`=` 代表與預設相同。
- **相依** = 對應下面「參數相依性地圖」的 D1–D8 規則。

---

## 參數相依性地圖（寫 conf 前先看：改一個要連帶改另一個）

| # | 規則 |
|---|---|
| **D1** | `trainer.max_steps` 應 == `means_lr_scheduler.max_steps` == `depth_loss_weight.max_steps`。不一致 → LR/深度權重衰減曲線對不上總步數。 |
| **D2** | `block_id` 非 null 時才會用 `block_dim` + `content_threshold` + `partition_subdir`；三者須與 partition 目錄名 `partition/<subdir>/partitions-dim_{D0}_{D1}_visibility_{content_threshold}` 一致，否則讀不到分塊檔。`block_id=null` → 全場景。 |
| **D3** | 改 `down_sample_factor` 必須有對應解析度的 `depth_dir`（ds1.2→estimated_depths / ds2→estimated_depths_2 / ds3→estimated_depths_3），否則深度圖 ≠ 相機尺寸 → tensor 形狀不匹配報錯。 |
| **D4** | `sh_degree=0` 時 `shs_rest_lr` 與 `sh_degree_up_interval` 無作用(∅sh0)。 |
| **D5** | `overwrite_config=True` → 用 ckpt 內 config 蓋本檔（本檔多數設定失效）；`False` → 用本檔、只取 ckpt 權重（本檔 sh_degree 高於 coarse 會自動 padding SH）。 |
| **D6** | `densify_grad_threshold` / `opacity_reset_interval` / `densify_from~until` 只在 `step < densify_until_iter` 內有意義；之後顆數凍結，只剩精修。 |
| **D7** | 啟用 MCMC（來源 C）時，density 與 metric 要成對替換（`MCMC2DGSDensityController` + `MCMCCityGSV2Metrics`），`cap_max` 必填；此時 grad-densify 旋鈕（grad_threshold / opacity_reset）不再生效。 |
| **D8** | `depth_rescaling=true` 需要 `depth_scale_name`.json；test 集無此檔會自動跳過深度監督（非報錯）。`depth_dir` 不存在才是真報錯。 |

---

## model（頂層載入）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| `initialize_from` | 從 ckpt 初始化（給檔或目錄，目錄會自動找最新） | 路徑 / null | null | null | D5 |
| `overwrite_config` | 是否用 ckpt 內 config 蓋本檔 | true/false | **True** | False | D5 |

## model.gaussian（`Gaussian2D`；3DGS 改 `VanillaGaussian`）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| **★ `sh_degree`** | 視角相依容量，每通道 (L+1)² 係數；記憶體 sh0=13/sh2=37/sh3=58 floats/顆 | 0..3 (int) | 3 | **0** | D4 |

## model.gaussian.optimization（`OptimizationConfig`）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| `means_lr_init` | 位置(xyz)初始 LR；太大→飄移/floater | >0 | 0.00016 | = | |
| `means_lr_scheduler.lr_final` | 末段位置 LR（慣例 = init/100） | >0 且 <init | 1.6e-6 | = | |
| `means_lr_scheduler.max_steps` | 位置 LR 衰減對齊步數 | >0 (int) | 30000 | 30000 | D1 |
| `spatial_lr_scale` | 位置 LR 場景尺度縮放 | <=0 自動 | -1 | = | |
| `shs_dc_lr` | SH DC（基礎顏色）LR | >0 | 0.0025 | = | |
| `shs_rest_lr` | SH 高階（視角相依）LR，慣例 = dc/20 | >0 | 0.000125 | = | ∅sh0 D4 |
| `opacities_lr` | 不透明度 LR | >0 | 0.05 | = | |
| `scales_lr` | 尺度（surfel 大小）LR；太大→爆球 | >0 | 0.005 | = | |
| `rotations_lr` | 旋轉（法線方向）LR | >0 | 0.001 | = | |
| `sh_degree_up_interval` | 每幾步 active SH 階 +1（暖身） | >0 (int) | 1000 | = | ∅sh0 D4 |
| `optimizer` | 優化器 | Adam/SelectiveAdam/SparseGaussianAdam | Adam | = | |

## model.metric（`CityGSV2Metrics` ← GS2DMetrics ← VanillaMetrics）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| **★ `lambda_dssim`** | RGB 主損失 SSIM 比重：loss=(1-λ)L1+λ(1-SSIM) | 0..1 | 0.2 | = | |
| `rgb_diff_loss` | RGB 差異 loss 型 | l1/l2 | l1 | = | |
| `lpips_net_type` | LPIPS 指標用網路 | vgg/alex/squeeze | alex | = | |
| `fused_ssim` | SSIM CUDA 核加速（需另裝） | true/false | false | = | |
| `lambda_normal` | 法線一致性 loss 權重；大→表面更平 | >=0 | 0.05 | 0.0125 | |
| `normal_regularization_from_iter` | 第幾步開始加法線 loss | >=0 (int) | 7000 | 0 | |
| `lambda_dist` | 2DGS distortion loss 權重（壓深度集中、減 floater） | >=0 | 0 (關) | 0 | 來源 B |
| `dist_regularization_from_iter` | distortion 起始步 | >=0 (int) | 3000 | = | 來源 B |
| `depth_loss_type` | 深度監督 loss 型 | l1 / l1+ssim / l2 / kl | l1 | l1+ssim | |
| `depth_loss_ssim_weight` | 深度 loss 裡 ssim 比重 | 0..1 | 0.2 | 1.0 | |
| `depth_loss_weight.init` | 深度 loss 初始權重（0=不用深度監督） | >=0 | 1.0 | 0.5 | |
| `depth_loss_weight.final_factor` | 末段 = init × 此 | 0..1 | 0.01 | 0.05 | |
| `depth_loss_weight.max_steps` | 深度權重衰減對齊步數 | >0 (int) | 30000 | 30000 | D1 |
| `depth_normalized` | 深度正規化後再算 loss | true/false | false | = | |
| `depth_output_key` | 取哪個渲染深度輸出 | inverse_depth / depth … | inverse_depth | = | |

## model.renderer（`SepDepthTrim2DGSRenderer`）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| **★ `depth_ratio`** | mesh/深度 精度↔召回（0=精度/街景；1=召回/空拍） | 0..1 | 0 | 1.0 | |
| `K` | 每像素取前 K 個高斯算貢獻 | >=1 (int) | 5 | = | |
| `v_pow` | 重要度分數的面積次方 | 0..1 | 0.1 | = | |
| `prune_ratio` | 每次 trim 修掉的低貢獻比例 | 0..1 | 0.1 | = | |
| `contribution_prune_from_iter` | 第幾步開始 trim | >=0 (int) | 1000 | = | |
| `contribution_prune_interval` | 每幾步 trim 一次 | >0 (int) | 500 | = | |
| `start_prune_ratio` | 初期 trim 比例 | 0..1 | 0.0 | = | |
| `diable_start_trimming` | 關閉初期 trim | true/false | false | = | |
| `diable_trimming` | 關閉全部 trim（coarse 設 true） | true/false | false | = | |

## model.density（`CityGSV2DensityController` ← `VanillaDensityController`）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| `percent_dense` | 分裂 vs 複製門檻（尺度>percent_dense×場景尺度→split） | 0.001~0.05 | 0.01 | = | |
| `densification_interval` | 每幾步 densify+prune | >0 (int) | 100 | 200 | D6 |
| **★ `opacity_reset_interval`** | 每幾步重置 opacity（強力剪枝旋鈕；小→剪兇且乾淨） | >0 (int) | 3000 | 6000 | D6 |
| `densify_from_iter` | 第幾步開始 densify | >=0 (int) | 500 | 1000 | D6 |
| **★ `densify_until_iter`** | 第幾步停 densify（之後顆數凍結；太長→floater，28.9 用 12000） | >0 (int) | 15000 | 30000 | D6 |
| **★ `densify_grad_threshold`** | 視角空間位置梯度 > 此值才 densify（越小→越密、越吃記憶體） | 5e-5~5e-4 | 0.0002 | = | D6 |
| `cull_opacity_threshold` | opacity 低於此即剪掉 | 0..1 | 0.005 | = | |
| `camera_extent_factor` | 場景尺度縮放 | >0 | 1.0 | = | |
| `scene_extent_override` | 手動指定場景尺度（<=0 自動） | -1 或 >0 | -1 | = | |
| `absgrad` | 用絕對值梯度累積（AbsGS，受 rasterizer 限制） | true/false | false | = | |
| `densify_grad_scaler` | (CityGSV2) 有 extra_loss(硬深度) 時重新平衡 densify 梯度 | >=0 | 0.0 | = | |
| `axis_ratio_threshold` | (CityGSV2) 長寬比過扁 surfel 門檻 | 0..1 | 0.01 | = | |

## trainer

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| **★ `max_steps`** | 總訓練步數 | >0 (int) | —(Lightning) | 60000 | D1 |
| `check_val_every_n_epoch` | 每幾 epoch 跑驗證 | >0 (int) | 1(Lightning) | 20 | |

## data（頂層）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| `path` | COLMAP 資料根目錄（含 sparse/ + input 或 images_N/） | 路徑 | — | data/matrix_city/aerial/train/block_all | |
| `num_workers` | DataLoader 工作進程數 | >=0 (int) | 0 | 8 | |
| `train_max_num_images_to_cache` | 訓練影像快取上限（-1=全快取；OOM/RAM 不足調低） | int | -1 | (未設) | |
| `val_max_num_images_to_cache` | 驗證影像快取上限 | int | -1 | (未設) | |

## data.parser（`EstimatedDepthBlockColmap` ← ColmapBlock ← Colmap）

| 參數 | 用途 | 值域 | 預設 | 基底 | 相依 |
|---|---|---|---|---|---|
| **★ `down_sample_factor`** | 影像下採樣倍率（影像自動 resize） | >=1 (float) | 1 | 1.2 | D3 |
| `down_sample_rounding_mode` | 下採樣尺寸取整法 | floor/round/round_half_up/ceil | round | = | |
| `block_id` | 只訓練第幾塊的相機（null=全場景） | 0..(N×M-1) / null | null | null | D2 |
| `block_dim` | 分塊網格 N×M | list[int][2] / null | null | [5,5] | D2 |
| `content_threshold` | partition 可見度門檻（對齊目錄名 visibility_x） | 0..1 | 0.08 | 0.08 | D2 |
| `partition_subdir` | partition 輸出多一層子目錄（避免覆蓋）**來源 A** | 字串 / null | null | null | D2 |
| `num_threshold` | 分塊用點數門檻 | >0 (int) | 25000 | = | |
| `min_images_per_block` | 一塊少於此圖數則報錯 | >=0 (int) | 10 | = | |
| `aabb` | 手動分塊包圍盒 [xmin,ymin,zmin,xmax,ymax,zmax] | list[float] / null | null | = | |
| `split_mode` | 訓練/驗證切分模式 | reconstruction / experiment | reconstruction | = | |
| `eval_image_select_mode` | 怎麼挑 val/test 影像 | step/ratio/list/list-optional | step | = | |
| **★ `eval_step`** | step 模式：每 N 張挑 1 當 val（決定 PSNR 比較基準） | >1 (int) | 8 | = | |
| `eval_ratio` | ratio 模式：val 佔比 | 0..1 | 0.01 | = | |
| `eval_list` | list 模式：val 清單檔 | 路徑 / null | null | = | |
| `image_dir` | 影像資料夾覆寫（預設自動找 images/ 或 input/） | 路徑 / null | null | = | |
| `mask_dir` | 遮罩資料夾（做 masked loss） | 路徑 / null | null | = | |
| `depth_dir` | 深度 .npy 目錄 | 路徑 | estimated_depths | = | D3 |
| `depth_rescaling` | 用 per-image scale/offset 校正深度 | true/false | true | = | D8 |
| `depth_scale_name` | scale json 檔名（不含副檔） | 字串 | estimated_depth_scales | = | D8 |
| `depth_scale_lower_bound` | 深度 scale 下界（中位數倍數），超界該圖丟棄 | >0 | 0.2 | = | |
| `depth_scale_upper_bound` | 深度 scale 上界 | >下界 | 5.0 | = | |

## save_iterations

| 參數 | 用途 | 值域 | 預設 | 基底 |
|---|---|---|---|---|
| `save_iterations` | 在這些步數存「可續跑 .ckpt(含 optimizer)+ PLY」 | list[int] | [7000,30000] | [30000,60000] |

---

## 我們的擴充（依來源分類，全部預設關閉）

| 來源 | 名稱 | 功能 | 開啟方式 |
|---|---|---|---|
| **A** | `partition_subdir` | partition 輸出多一層 `<subdir>`，避免不同 coarse/分塊互相覆蓋 | `data.parser.init_args.partition_subdir: my_run` |
| **B** | `lambda_dist` / `dist_regularization_from_iter` | 2DGS distortion 正則，壓深度集中、減 floater | `metric.init_args: { lambda_dist: 100.0, dist_regularization_from_iter: 3000 }`（典型 100~1000） |
| **C** | `MCMC2DGSDensityController` + `MCMCCityGSV2Metrics` | 不看梯度、靠 relocate+add_new 強制長到 cap + Langevin noise；6GB 保證長到指定顆數的可靠法 | 見下「來源 C 細節」（**density 與 metric 成對換**，D7） |
| **D** | `model.light_gaussian`（★在 model: 底下，非頂層） | 指定步數依重要度(transmittance×area^v)剪掉一定比例，回收預算；配 MCMC 時 prune_steps 要 < densify_until 才會 refill=重分配 | `model: { light_gaussian: { prune_steps: [30000], prune_percent: 0.2, prune_decay: 1.0, v_pow: 0.1 } }` |
| **E** | `DBPDensityController` | grad-densify(DGD) + cap 上限 + 超限剪最低重要度；註：6GB 上 grad-densify 本身長不起來 | `density.class_path: internal.density_controllers.dbp_density_controller.DBPDensityController`（讀檔頭註解） |
| **F** | RTG-SLAM depth-init（init 方式，非 config 旗標） | depth 反投影 PLY（α=0.99 不透明 surfel）當 init；同品質更少顆，但梯度低→配 MCMC | `model.initialize_from: data/.../depth_init/block_7.ply`（用 `utils/depth_init_blocks.py` 產生） |
| **F2** | `RTGStableDensityController`（B1/B3/B4/B5 stable 管理） | grad-densify + 凍結α + M_c補點 + 誤差翻轉 + 長期unstable剪枝；與 MCMC 互斥，metric 用 CityGSV2Metrics | 見下「來源 F2 細節」 |
| **G** | edge_aware / atom_normal / sgl_mcmc / scaffold(出局) | 其他實驗性 metric/density | 需要時讀對應檔頭註解；皆預設不啟用 |
| **H** | `model.initializer`（初始化可換模組，opt-in） | 把「SfM冷啟動 vs 載ckpt/PLY」做成明確開關；設了優先於 initialize_from，不設=舊邏輯不變 | 見下「來源 H 細節」 |

### 來源 C 細節（MCMC，取代 density + metric 整段）

**density：**

| 參數 | 用途 | 值域 | 預設 |
|---|---|---|---|
| `cap_max` | **必填**，最大顆數（6GB sh0 可數百萬；sh2 約 1.4M） | >0 (int) | (無，必填) |
| `noise_lr` | Langevin 噪聲強度；α=0.99 點自動豁免 | >=0 | 5e5 |
| `densify_from_iter` | densify 起始步 | >=0 (int) | 500 |
| `densify_until_iter` | densify 終止步 | >0 (int) | 25000 |
| `densification_interval` | 每幾步 relocate/add | >0 (int) | 100 |
| `min_opacity` | 低於此 opacity 視為 dead → 被 relocate | 0..1 | 0.005 |
| `N_max` | relocation 二項式上限（對齊原始 MCMC） | >0 (int) | 51 |

**metric（`MCMCCityGSV2Metrics`，繼承 CityGSV2Metrics 全部欄位）：**

| 參數 | 用途 | 值域 | 預設 |
|---|---|---|---|
| `opacity_reg` | opacity L1 正則（取代 opacity-prune，餵 dead_mask） | >=0 | 0.01 |
| `scale_reg` | scale L1 正則（控球小） | >=0 | 0.01 |
| `immune_opacity_threshold` | 高於此 opacity 豁免 opacity_reg（護 α=0.99） | 0..1 | 0.9 |

### 來源 F2 細節（`RTGStableDensityController`，取代 density；metric 用 `CityGSV2Metrics`）

繼承 CityGSV2 的 grad-densify 參數（`densification_interval`/`densify_from~until_iter`/`densify_grad_threshold`/`percent_dense`/`opacity_reset_interval`/`cull_opacity_threshold` 等，見上面 density 表）。**無 `cap_max`** → 成長靠 grad_threshold + B5 擋。額外參數：

| 參數 | 機制 | 用途 | 值域 | 預設 |
|---|---|---|---|---|
| `freeze_opacity` | B1 | 凍結 opacity（α∈{0.99,0.1}, lr_α=0），停用 opacity-prune/reset，改用 B4+B5 | t/f | false |
| `depth_init_immune` | B1 | depth-init 點開局即 stable、免 reset/opacity-prune | t/f | false |
| `stable_threshold` | — | 幾個 densify interval 後一點轉 stable | >0 (int) | 10 |
| `mc_enabled` | B3 | 開啟 M_c 透明高斯補點（色誤差高像素回投影） | t/f | false |
| `mc_color_threshold` | B3 | δ_c：像素 L1 誤差 > 此值算 M_c | >=0 | 0.1 |
| `mc_sample_ratio` | B3 | 補點取樣比例 | 0..1 | 0.05 |
| `mc_transparent_alpha` | B3 | 新透明點 α | 0..1 | 0.1 |
| `mc_scale_factor` | B3 | 透明點尺度 = depth/focal × 此 | >0 | 0.01 |
| `stable_revert_enabled` | B4 | 開啟 stable→unstable 翻轉（誤差計數） | t/f | false |
| `stable_revert_threshold` | B4 | 翻轉誤差門檻 | >=0 | 0.1 |
| `stable_revert_count` | B4 | e_i 超過此值就翻回 unstable | >0 (int) | 5 |
| `max_unstable_intervals` | B5 | 一點維持 unstable 幾個 interval 後被剪（0=關） | >=0 (int) | 0 |
| `cap_max` | 硬cap | ★硬上限，N 不得超過（0=關）；防 OOM，永遠生效 | >=0 (int) | 0 |
| `soft_cap_enabled` | 軟帶 | 開 cap 之下軟節流帶（需 cap_max>0） | t/f | false |
| `soft_cap_band` | 軟帶 | 軟區 = [cap−band, cap] | >0 (int) | 200000 |
| `soft_densify_scale` | 軟帶 | 軟區內增生保留比例下限（源頭少加） | 0..1 | 0.3 |
| `soft_prune_scale` | 軟帶 | 軟區內 B5 剪枝加強倍率（max_unstable÷此） | >=1 | 1.5 |
| `soft_from_iter` / `soft_until_iter` | 軟帶 | 軟節流生效步範圍（-1=densify 窗口）；硬 cap 不受限 | int | -1 |

### 來源 H 細節（`model.initializer`，初始化可換模組）

設了 `initializer` 則優先於 `initialize_from`；不設 = 走舊邏輯（`initialize_from` null→SfM 冷啟動 / path→載入），**行為完全不變**。

| 選項（class_path） | 對應舊寫法 | init_args |
|---|---|---|
| `...gaussian_initializer.PointCloudInitializer` | `initialize_from: null`（SfM 冷啟動） | （無） |
| `...gaussian_initializer.CheckpointInitializer` | `initialize_from: <path>`（載 ckpt/PLY） | `path`（路徑）、`overwrite_config`（t/f，預設 true） |

實作 `internal/initializers/gaussian_initializer.py`；各 Impl 複用 `setup_from_pcd` / `_initialize_from_trained_model` 既有邏輯。
