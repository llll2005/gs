# 【已歸檔】pipeline.md

> ⛔ **非現行。**架構描述停在 2026-05-10，已被 `完整Pipeline數學詳解.md`（06-18，更完整）與 memory `codebase_architecture` 取代。
>
> 現行入口：`../README.md`；現行公式：`../現行方案與公式.md`。
> 保留原因：內容仍可用/仍是該線的完整記錄，**需要時可直接取用**，只是不再代表當前方向。

---

# 架構與 Pipeline（完整函式調用）
> 最後更新：2026-05-10（S2：depth_init 採樣修正、opacity 修正）

## 標記說明

```
(file.py:N)          — 檔案與行號
[NEW]                — 本專案新增的檔案或步驟
[MOD]                — 本專案修改的行為（有 old/new 對比）
→                    — 直接呼叫
⟶                    — 間接觸發（callback、thread、subprocess）
```

---

## Phase 0：場景分塊（Partition）

### 原版流程（需要 coarse checkpoint，6GB VRAM 無法執行）

```
partition_citygs.py:main()
  → CityGSPartitionableScene (citygs_partitioning_utils.py)
    → get_bounding_box_by_points()
    → build_partition_coordinates()
    → camera_center_based_partition_assignment()
    → [需要 coarse 模型] projection_based_partition_assignment()
          → gaussian_model.pre_activate_all_properties()
          → render each block → visibility fraction
  → write partition/*.txt  (per-block image list)
  → write partitions.pt
```

### 新版流程 [NEW]

```
partition_from_colmap.py:main()                          [NEW]
  → colmap_utils.read_cameras_binary()
  → colmap_utils.read_images_binary()
  → colmap_utils.read_points3D_binary()
  → build c2w matrices from COLMAP extrinsics
  → CityGSPartitionableScene(xyz, cameras)
    → camera_center_based_partition_assignment()         (unchanged)
    → [替代投影] COLMAP track visibility:
          for each camera:
              track_frac = tracks_in_block / total_tracks
              assign if track_frac > content_threshold (0.08)
  → write partition/*.txt                               (格式與原版相同)
  → write partitions.pt                                 (格式與原版相同)
```

---

## Phase 1：Per-block 深度圖點雲初始化

### 原版流程（全局 coarse 訓練，需 24GB VRAM）

```
[不存在 per-block init；全局 coarse checkpoint 直接作為起點]
```

### 新版流程 [NEW / RTG-SLAM style, 2026-05-08]

RTG-SLAM §3.1 compact Gaussian 策略：opaque surface Gaussians、法向量驅動旋轉、M_s mask 採樣。

```
depth_init_blocks.py:main()                              [NEW]
  → colmap_utils.read_cameras_binary()
  → colmap_utils.read_images_binary()
  → load estimated_depth_scales.json
  → for each block (讀 partition/*.txt image list):
      for each image in block:
          → get_intrinsics(camera_id)     → fx, fy, cx, cy
          → npy_path_for(image_name)      → zero-padded path fallback
          → np.load(depth.npy)
          → inv_aligned = scale * inv_depth + offset
          → depth_m = 1.0 / inv_aligned.clamp(1e-6)

          → [RTG-SLAM Pt.2] estimate_normals_cam(depth, fx,fy,cx,cy):
                V = vertex map = [(u-cx)/fx*d, (v-cy)/fy*d, d]
                n = cross(dVdu, dVdv) → normalize
                flip if n_z > 0  (normals must point toward camera)
                → normals_cam [H,W,3], valid_n [H,W]

          → [RTG-SLAM Pt.3] M_s mask:
                ray = [(u-cx)/fx, (v-cy)/fy, 1] / |...|
                cos_ang = |dot(normals_cam, ray)|   [H,W]
                ms_mask = valid_depth
                        & valid_n
                        & (depth in [near, far])
                        & (cos_ang >= 0.5)           ← grazing angle < 60°

          → [RTG-SLAM Pt.3] 保留全部 M_s 像素（S2修正 2026-05-10）:
                ms_idx = argwhere(ms_mask)           → K candidates
                # sample_ratio=1.0 預設保留全部；voxel 負責控制密度
                vs_s, us_s = ms_idx[:, 0], ms_idx[:, 1]
                d_s = depth[vs_s, us_s]

          → backproject:
                pts_cam = [(us-cx)/fx*d, (vs-cy)/fy*d, d]
                pts_world = (pts_cam - t_cw) @ R_cw  (COLMAP T_cw convention)

          → [RTG-SLAM Pt.2] normals cam → world:
                n_cam_s  = normals_cam[vs_s, us_s]
                n_world_s = n_cam_s @ R_cw            (= R_cw.T @ n_cam_s)
                → normalize

      → concat all images' pts_world, normals_world

      → [RTG-SLAM Pt.6] voxel deduplication (offline 版 fusion):
            open3d.PointCloud (points + normals) .voxel_down_sample(voxel_size)
            voxel_size = max(voxel_alpha * scene_scale, voxel_min)
            normals averaged per voxel → re-normalized

      → [RTG-SLAM Pt.2] normals_to_quats(nrm_d):   (vectorized)
            cross([0,0,1], n) = [-ny, nx, 0]
            angle = arctan2(|cross|, nz)
            q = [cos(a/2), ax*sin(a/2), ay*sin(a/2), 0]

      → scale = log(voxel_size) for all points   (voxel_size ≈ depth-based scale)

      → [RTG-SLAM Pt.1] save_2dgs_ply():
            properties: x,y,z,
                        nx,ny,nz (world normals, metadata only — not read by load_from_ply),
                        f_dc_0..2 = 0 (gray),
                        scale_0=scale_1=log(voxel_size),
                        rot_0..3 = normals_to_quats(nrm_d),  ← normal-aligned, IS loaded
                        opacity  = logit(0.1) ≈ -2.197        ← S2修正：半透明起始，讓梯度穿透
  → output: data/.../depth_init/block_{id}.ply
```

**與舊版差異對照：**

| 項目 | 舊版 | 新版 (RTG-SLAM) |
|---|---|---|
| 採樣策略 | stride=4 grid（每 16px 取 1） | M_s mask + 5% 隨機 |
| 法向量 | nx=ny=nz=0（未計算） | 真實 cam→world 法向量 |
| 旋轉四元數 | identity [1,0,0,0] | normal-aligned quat |
| Scale init | KNN distance × knn_gamma | voxel_size（≈ depth-based） |
| Opacity | logit(0.1) = -2.197 | logit(0.99) = 4.595（opaque）|
| Grazing filter | 無 | cos(ray, n) ≥ 0.5 過濾掠射角 |

---

## Phase 2：分 block 訓練編排

```
train_citygs_partitions.py:main()
  → argparse: -n NAME [--depth_init_dir DIR]            [MOD: 新增 --depth_init_dir]
  → load config → block_dim → num_blocks = dim[0]*dim[1]
  → ProcessPoolExecutor(max_workers=num_blocks)
      → for each block_id:
            train_a_partition(block_id, config_args)
              → wait for free GPU (py3nvml.get_free_gpus)
              → args = ["python", "main.py", "fit",
                        "--config", config_path,
                        "--data.parser.block_id", block_id,
                        "--output", outputs/NAME/blocks/block_{id}]
              → [MOD] if depth_init_dir:
                    ply = depth_init_dir/block_{id}.ply
                    args += ["--model.initialize_from", ply]   [NEW branch]
              → subprocess.run(args)
```

---

## Phase 3：main.py fit — 啟動與初始化

```
main.py:4
  → cli()                         (internal/entrypoints/gspl.py:12)
    → CLI(GaussianSplatting, DataModule, ...)
      → parse args → merge YAML config + CLI overrides
      → config.model.save_val_output = config.save_val   (internal/cli.py:150)
      → Trainer.fit(model=GaussianSplatting, datamodule=DataModule)
```

---

## Phase 4：資料管線初始化

```
DataModule.setup("fit")           (internal/dataset.py:367)
  → detect_dataset_type(path)     (dataset.py:351)
  → EstimatedDepthBlockColmap.instantiate()
      → EstimatedDepthBlockColmapDataParser(path, output_path, rank, params)

  EstimatedDepthBlockColmapDataParser.get_outputs()
                                  (estimated_depth_colmap_block_dataparser.py:26)
    → ColmapBlockDataParser.get_outputs() [super]
                                  (colmap_block_dataparser.py:56)
      → colmap_utils.read_cameras_binary(sparse/cameras.bin)
      → colmap_utils.read_images_binary(sparse/images.bin)
      → filter images by block_id image list (partition/*.txt)

      → get_image_dir()           (colmap_dataparser.py:96)
        if image_dir is None  →  path/images_{factor}/     (e.g. images_1.2/)
        if image_dir="input"  →  path/input/

      → for each image in COLMAP:
          img_path = image_dir / extrinsics.name           (e.g. input/0014.png)
          ↓
          [MOD] zero-padded fallback
          ┌──────────────────────────────────────────────────────────────┐
          │ old (colmap_block_dataparser.py:222 原版 / CityGaussianbc):  │
          │   img_path = os.path.join(image_dir, extrinsics.name)       │
          │   (直接用 COLMAP 名稱，若不存在就 FileNotFoundError)         │
          │                                                              │
          │ new (colmap_block_dataparser.py:222-228):                   │
          │   img_path = image_dir / name                               │
          │   if not exists:                                             │
          │       padded = image_dir / name.zfill(6)                    │
          │       if exists: img_path = padded                          │
          │   image_name_list.append(name)        ← 原始名稱作識別符    │
          │   image_path_list.append(img_path)    ← padded 路徑作 I/O  │
          └──────────────────────────────────────────────────────────────┘

      → ColmapDataParser.read_points3D_binary()  (colmap_dataparser.py:135)
      → build_split_indices()      (colmap_dataparser.py:578)
        → eval_step=8: 每 8 張取 1 張作 val set
      → return DataParserOutputs(train_set, val_set, point_cloud)

    → load estimated_depth_scales.json
    → for each image in train_set + val_set:
        depth_path = depth_dir / image_name.npy   (e.g. estimated_depths/0014.png.npy)
        ↓
        [MOD] zero-padded fallback
        ┌──────────────────────────────────────────────────────────────────┐
        │ old (estimated_depth_colmap_block_dataparser.py:38 原版):        │
        │   depth_path = depth_dir / f"{image_name}.npy"                  │
        │   if not exists: WARNING + continue                             │
        │   (結果 loaded_depth_count=0 → AssertionError)                  │
        │                                                                  │
        │ new (estimated_depth_colmap_block_dataparser.py:38-47):         │
        │   depth_path = depth_dir / f"{image_name}.npy"                  │
        │   if not exists:                                                 │
        │       padded = depth_dir / f"{base.zfill(6)}.{ext}.npy"        │
        │       if exists: depth_path = padded                            │
        │       else: WARNING + continue                                  │
        └──────────────────────────────────────────────────────────────────┘
        → check scale bounds: [0.2×median, 5×median]
        → image_set.extra_data[idx] = (depth_path, {scale, offset})
    → image_set.extra_data_processor = EstimatedDepthBlockColmapDataParser.load_depth

  → CacheDataLoader.__init__()    (dataset.py:142)
    → _cache_data(indices)        (dataset.py:223)
      → ThreadPoolExecutor: Dataset.__getitem__(i) per image
          Dataset.__getitem__()   (dataset.py:137)
            → image_cameras[index]   (pre-computed from COLMAP intrinsics × down_sample_factor)
            → get_image(index)       (dataset.py:52)
                → PIL.Image.open(image_path)
                ↓
                [MOD] image resize
                ┌────────────────────────────────────────────────────────┐
                │ old (dataset.py:56 原版):                              │
                │   # TODO: resize                                       │
                │   pil_image = Image.open(path)                         │
                │   numpy_image = np.array(pil_image)                    │
                │   (1920×1080 GT vs 1600×900 render → RuntimeError)     │
                │                                                        │
                │ new (dataset.py:56-59):                                │
                │   pil_image = Image.open(path)                         │
                │   cam_w, cam_h = camera.width, camera.height           │
                │   if pil_image.size != (cam_w, cam_h):                 │
                │       pil_image = pil_image.resize(LANCZOS)            │
                │   numpy_image = np.array(pil_image)                    │
                └────────────────────────────────────────────────────────┘
                → return (name, tensor[3,H,W], mask)
            → get_extra_data(index)  (dataset.py:134)
                → extra_data_processor((depth_path, scale))
                    = load_depth()   (estimated_depth_colmap_block_dataparser.py:72)
                      → depth = np.load(path) * scale + offset
                      → return torch.tensor(depth)   shape [H, W]
            → return (Camera, (name, image, mask), depth_tensor)
```

---

## Phase 5：模型初始化

```
GaussianSplatting.setup("fit")    (gaussian_splatting.py:188)
  → gaussian_model = self.hparams["gaussian"].instantiate()
        = Gaussian2D(sh_degree=2).instantiate()
        → Gaussian2DModel (max_sh_degree=2, active_sh_degree=0)
        → self.gaussian_model = Gaussian2DModel
  → renderer.setup()              (sep_depth_trim_2dgs_renderer.py)
  → density_controller.setup()   (vanilla_density_controller.py:40)
        → prune_extent = datamodule.prune_extent * camera_extent_factor
  → metric.setup()                (citygsv2_metrics.py:40)

  → if initialize_from is not None:
      _initialize_from_trained_model()  (gaussian_splatting.py:131)
        → GaussianModelLoader.search_load_file(.ply)
        → GaussianPlyUtils.load_from_ply(path)  → ply_data (sh_degrees=0)
        ↓
        [MOD] PLY → Gaussian2D model creation
        ┌──────────────────────────────────────────────────────────────────┐
        │ old (gaussian_splatting.py:146 原版):                            │
        │   gaussian_model = Gaussian2D(sh_degree=ply_data.sh_degrees)    │
        │   # sh_degree=0 → max_sh_degree=0                               │
        │   # shs_rest=[n, 0, 3] 永遠不會增長                             │
        │   # SH 訓練全程停在 degree 0                                    │
        │   state_dict["gaussians.shs_rest"] = ply_data.features_rest     │
        │   # 載入空 [n,0,3] 進空模型                                     │
        │                                                                  │
        │ new (gaussian_splatting.py:144-160):                             │
        │   configured_sh_degree = self.hparams["gaussian"].sh_degree  ← config 的 2  │
        │   gaussian_model = Gaussian2D(sh_degree=configured_sh_degree)   │
        │   # max_sh_degree=2，shs_rest=[n, 8, 3] 正確分配               │
        │   # 不載入 shs_rest（從 zeros 開始，training 學習）             │
        │   state_dict = {means, opacities, shs_dc, scales, rotations}   │
        │   # shs_rest 故意省略                                           │
        └──────────────────────────────────────────────────────────────────┘

        → gaussian_model.load_state_dict(state_dict, strict=False)
        → if not overwrite_config:
              org_config = self.gaussian_model.config   (sh_degree=2 from YAML)
              self.gaussian_model = gaussian_model
              self.gaussian_model.config = org_config

  → print: "initialize from ...: sh_degree=2, overwrite_config=False"
```

---

## Phase 6：訓練迴圈（per training step）

```
GaussianSplatting.on_train_batch_start()  (gaussian_splatting.py:352)
  → renderer.before_training_step(step)
      SepDepthTrim2DGSRenderer.before_training_step()  (sep_depth_trim_2dgs_renderer.py:173)
        if step == 1 or step % contribution_prune_interval == 0:
          → 對所有訓練 cameras 渲染 → 計算貢獻度 C_n,k
          → 保留 top-K 貢獻的 Gaussians（trimming）

──────────────────────────────────────────────────────────────
GaussianSplatting.training_step(batch, idx)  (gaussian_splatting.py:362)
  batch = (Camera, (name, gt_image, mask), depth_tensor)

  → GaussianSplatting.forward(camera)  (gaussian_splatting.py:275)
      → SepDepthTrim2DGSRenderer.forward()  (sep_depth_trim_2dgs_renderer.py:43)
          → diff_trim_surfel_rasterization(...)   [CUDA kernel, CityGSV2 修改版]
          → surf_depth = expected*(1-ratio) + median*ratio   (ratio=1.0 for aerial)
          → return {render[3,H,W], rend_alpha, rend_normal,
                    rend_dist, surf_depth[H,W], surf_normal[3,H,W]}

  → CityGSV2MetricsModule.get_train_metrics()  (citygsv2_metrics.py:102)
      → GS2DMetricsImpl.get_train_metrics() [parent]:
          L1_rgb  = |render - gt_image|.mean()
          SSIM    = structural_similarity(render, gt_image)
          dist_loss  = rend_dist.mean()                   (Gaussian 分佈正則)
          normal_loss = (1 - dot(rend_normal, surf_normal)).mean()
          extra_loss  = lambda_dssim * (1 - SSIM)         (給 DGD 用)
      → get_inverse_depth_metric(batch, outputs):
          predicted_inv_depth = 1 / (surf_depth + 1e-8)
          gt_inv_depth = batch[2] (from depth_tensor)
          d_reg = _depth_l1_and_ssim_loss(gt_inv_depth, predicted_inv_depth)
      → d_weight = 0.5 * 0.05^(step/30000)              (exponential decay)
      → if step < densify_until_iter:
            loss = lambda_dssim*(1-SSIM) + dist + normal + d_reg*weight
            extra_loss = (1-lambda_dssim)*L1_rgb           (DGD 分離)
        else:
            loss = L1_rgb + lambda_dssim*(1-SSIM) + dist + normal + d_reg*weight

  → density_controller.before_backward()  (vanilla_density_controller.py:63)
      → accumulate viewspace_point_tensor 梯度統計 (2D screen space)

  → trainer.strategy.backward(loss)                      [lightning manual backward]

  → [DGD] trainer.strategy.backward(extra_loss)
      → scale viewspace grads:
          scaler = max(densify_grad_scaler * grad_after / grad_before, 1.0)
          viewspace_points.grad = org_grad * scaler

  → density_controller.after_backward()  (vanilla_density_controller.py:69)
      if densify_from < step < densify_until and step % densification_interval == 0:
        → _densify_and_prune()  (vanilla_density_controller.py:115)
            → CityGSV2DensityControllerModule._densify_and_clone()  (citygsv2:60)
                → 梯度超閾值且小的 Gaussians → 複製
                → Elongation Filter: axis_ratio = min_scale/max_scale > 0.01
                → cat_tensors_to_properties() → 擴展所有參數 tensor 與 optimizer state
            → CityGSV2DensityControllerModule._densify_and_split()  (citygsv2:16)
                → 梯度超閾值且大的 Gaussians → 分裂為 N=2
                → _split_means_and_scales() → 沿最長軸移動
                → Elongation Filter (同上)
                → _prune_points(原始點)  (vanilla_density_controller.py:236)
                    → Utils.prune_properties(valid_mask, model, optimizers)
            → _prune_points(opacity < cull_threshold, big_ws_points)
      if step % opacity_reset_interval == 0:
        → _reset_opacities(): sigmoid_inv(min(opacity, 0.01))

  → optimizer.step()   (means optimizer + params optimizer)
──────────────────────────────────────────────────────────────

GaussianSplatting.on_train_batch_end()  (gaussian_splatting.py:494)
  → renderer.after_training_step()  (sep_depth_trim_2dgs_renderer.py:209)
      → depth_to_normal() (有限差分叉積 → 偽法向 from depth map)
      → 更新 contribution stats (if interval)
  → if step % sh_degree_up_interval == 0 and active_sh < max_sh:
        gaussian_model.increase_sh_degree()   (active 0→1→2)
  → SaveGaussian callback:
      if step in save_iterations [500, 1500, 15000, 30000]:
        → save_gaussians()  (gaussian_splatting.py:728)
            → GaussianPlyUtils.save_to_ply() → epoch=X-step=Y-xyz_rgb.ply
            → trainer.save_checkpoint()     → epoch=X-step=Y.ckpt
```

---

## Phase 7：Validation（每 15 epochs 一次）

```
GaussianSplatting.on_validation_epoch_start()  (gaussian_splatting.py:584)
  → if save_val_output: spawn save_images() threads

GaussianSplatting.validation_step(batch, idx)  (gaussian_splatting.py:520)
  → forward(camera) → outputs
  → CityGSV2MetricsModule.get_validate_metrics()  (citygsv2_metrics.py:122)
      → GS2DMetricsImpl: PSNR, SSIM, LPIPS
      → get_inverse_depth_metric() → d_reg
  → log_metrics (TensorBoard)
  → if save_val_output:
        image_queue.put({render, gt, stage="val", image_name, epoch, step})
        ⟶ save_images() thread:
              image = concat([gt_image, render], dim=-1)   (side-by-side)
              save to output_path/val/epoch=X-step=Y/{name}.png

GaussianSplatting.on_validation_epoch_end()  (gaussian_splatting.py:595)
  → join save_images threads
  → write results.txt  (所有 val metrics 的 epoch 平均)
  → write metrics/{metric_name}.txt per metric
```

---

## Phase 8：Test（`main.py test`）

```
main.py test → Trainer.test()
  → GaussianSplatting.setup("test")
      → 同 Phase 5 setup（載入 checkpoint via ckpt_path）
  → GaussianSplatting.on_test_epoch_start()  (gaussian_splatting.py:637)
      → 同 on_validation_epoch_start
  → GaussianSplatting.test_step(batch, idx)  (gaussian_splatting.py:681)
      → validation_step(batch, idx, name="test")
          → 同 validation_step，name="test"
          → images saved to output_path/test/epoch=X-step=Y/{name}.png
  → GaussianSplatting.on_test_epoch_end()  (gaussian_splatting.py:641)
      → on_validation_epoch_end()
```

---

## 修改總覽對照表

| 步驟 | 原版問題 | 修正位置 | 行為變化 |
|------|---------|---------|---------|
| Partition | 需要 coarse checkpoint | `utils/partition_from_colmap.py` [NEW] | COLMAP track 可見度替代投影 |
| 點雲初始化 | 無（依賴 coarse） | `utils/depth_init_blocks.py` [NEW] | DA2 深度圖反投影 per-block PLY |
| 訓練編排 | 無 per-block PLY 支援 | `train_citygs_partitions.py:train_a_partition` | 新增 `--depth_init_dir` 分支 |
| 影像路徑 | COLMAP 4-digit vs 磁碟 6-digit | `colmap_block_dataparser.py:222` `colmap_dataparser.py:346` | zfill(6) fallback |
| 深度路徑 | COLMAP 4-digit vs 磁碟 6-digit | `estimated_depth_colmap_block_dataparser.py:40` | zfill(6) fallback |
| 影像 resize | TODO 未實作，1920×1080 GT vs 1600×900 render → crash | `dataset.py:56` | PIL.resize(cam_w, cam_h, LANCZOS) |
| SH degree init | PLY sh_degree=0 → max_sh_degree=0 → SH 永遠不升 | `gaussian_splatting.py:144` | 改用 config sh_degree=2 建模型 |
| SH rest 載入 | 載入空 [n,0,3] 進 [n,8,3] 模型 → shape mismatch | `gaussian_splatting.py:156` | 省略 shs_rest（從 zeros 學習）|
