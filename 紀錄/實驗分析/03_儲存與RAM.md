# 03 儲存與 RAM

> **圖**：`03_儲存與RAM.png`（同名，3 個面板）　｜　**年代**：新年代（2026-10-03 起圖表與表格不再放舊資料時代的數字）
> **資料**：ckpt 拆解（本機與 lab）；`tools/run_storage_audit.py`（不載入張量）；`scripts/task_bitcheck_data.sh`／`task_bitcheck_pipeline.sh`
> **重現**：`python tools/plot_analysis.py`（面板 a=fig4、b=fig6、c=fig5）；`python3 tools/run_storage_audit.py <ckpt 或跑次目錄>`

## 圖表對照

| 面板 | 內容 | 對應本文 |
|---|---|---|
| (a) | ckpt 的組成：2/3 是 Adam；參數裡主要是 SH 高階 | §1.1 |
| (b) | 一個 60k 跑次目錄：中間 ckpt 佔大宗 | §1.2 |
| (c) | CPU RAM 影像快取：舊預設每視角 25.6 MB → 現行 uint8＋不載深度 4.3 MB | §2 |

## 結論

- **現行（10-03）**：影像快取 uint8＋不載未用深度是預設（09-21 起）；ckpt 大小只看 N —— 現行最佳（N 2.34M）的 60k ckpt 約 1.52 GiB。
- **ckpt 的 2/3 是 Adam 狀態**；參數 232 B/顆，其中 **SH 高階 180 B（77.6%）**。`run_storage_audit.py` 在 speed3 b6 60k 重量一致（參數 232 B／Adam 464 B／顆）。
- 一個 60k 跑次 7.8~8.0 GB，**中間 ckpt 佔 78~89%**；`-xyz_rgb.ply` 不是可渲染的模型（2026-09-18 起預設不輸出）。
- lab 被 RAM 綁住的原因：快取每視角存 float32 影像 17.3 MB ＋ 權重為 0 的深度圖 8.3 MB ⇒ **uint8＋不載未用深度**後每視角 4.3 MB（本機 RSS 12.6 → 4.8 GB），**送進訓練的影像逐位元不變**（三層驗證），已設為預設。
- 官方線與我方的儲存對照（同一支工具）已排 lab：`task_citygs_origin.sh storage`。

---

> 本文 §編號沿用原《資源與效能量測彙整》（2026-10-03 拆成 01~04 四份，原檔已歸檔）：§1 時間、§2.x 平行 → `01`；§2 max_split、§3 VRAM → `02`；§4 儲存、§5 RAM、§6 改善項 → `03`；§9 外接盒與成本量測 → `04`。「v2 §x」指 `new_archived/研究總覽_v2_至2026-10-03.md`、「研究總覽 §11.x／§13.x」指舊版 `new_archived/研究總覽_v1_至2026-09-13.md`（現行正文 `研究總覽.md` 的章節編號不同）。

## 4. 成果檔的存儲（面板 a、b）

### 4.1 ckpt 組成（本機 `outputs/cs_refrep/blocks/block_6/checkpoints/epoch=40-step=21920.ckpt`，N 1,850,094）

| 頂層鍵 | 大小 | 佔比 |
|---|---|---|
| state_dict | 0.400 GiB | 33.3% |
| optimizer_states | 0.799 GiB | 66.7% |
| 其他（epoch／loops／hyper_parameters／datamodule_hyper_parameters…） | < 1 MiB | 0 |
| **檔案** | **1.199 GiB** | |

optimizer_states：opt[0]（means 專用）0.041 GiB；opt[1]（其餘 5 群）0.758 GiB；欄位 exp_avg、exp_avg_sq、step

| state_dict 張量 | 形狀 | dtype | 大小 | bytes／顆 | 佔參數 |
|---|---|---|---|---|---|
| shs_rest | (1850094, 15, 3) | float32 | 317.6 MiB | 180 | **77.6%** |
| rotations | (1850094, 4) | float32 | 28.2 MiB | 16 | 6.9% |
| means | (1850094, 3) | float32 | 21.2 MiB | 12 | 5.2% |
| shs_dc | (1850094, 1, 3) | float32 | 21.2 MiB | 12 | 5.2% |
| scales | (1850094, 2) | float32 | 14.1 MiB | 8 | 3.4% |
| opacities | (1850094, 1) | float32 | 7.1 MiB | 4 | 1.7% |
| _active_sh_degree | () | uint8 | 0 | — | — |
| **參數合計** | | | | **232** | |
| ＋Adam 兩個動量 | | | | 464 | |
| **ckpt 每顆** | | | | **≈ 696** | |

### 4.2 跑次目錄

**lab `outputs/lab/speed3/blocks/block_6`（60k，共 7.8 GB）**

| 檔案 | bytes | 約 | 當時 N |
|---|---|---|---|
| epoch=0-step=499.ckpt | 403,003,354 | 384 MiB | |
| epoch=2-step=1499.ckpt | 419,871,514 | 400 MiB | |
| epoch=27-step=14999.ckpt | 1,809,617,434 | 1.69 GiB | 2.6M（cap） |
| epoch=54-step=29999.ckpt | 1,809,617,434 | 1.69 GiB | 2.6M |
| epoch=76-step=41999.ckpt | 1,628,657,434 | 1.52 GiB | 2.34M（cap 只交付 90%） |
| epoch=110-step=60000.ckpt | 1,628,657,434 | 1.52 GiB | 2.34M |
| 六個 -xyz_rgb.ply | 15.6／16.3／70.2／70.2／63.2／63.2 MB | 共約 299 MB | |
| test/ | | 309 MiB | |
| input.ply | | 15 MiB | |
| lightning_logs／cameras.json／其他 | | < 1.4 MiB | |
⇒ 中間 5 個 ckpt 共 5.87 GiB（checkpoints/ 的 78%）；最終模型只存參數約 0.54 GiB。

**本機 `outputs/cs_refrep/blocks/block_6`（21,920 步，共 3.4 GB）**：ckpt 499 385M／1,499 401M／14,999 1.2G／21,920 1.2G；PLY 15M／16M／48M／48M；input.ply 15M；cameras.json 464K

### 4.3 `-xyz_rgb.ply` 的內容

`element vertex 1850094`；`property float x y z nx ny nz`＋`uchar red green blue` ⇒ **27 B/顆**，沒有 opacity／scale／rotation／SH
⇒ **不是可渲染的 Gaussian 模型**（web view 要用 ckpt）。輸出位置 `internal/gaussian_splatting.py`（save checkpoint 之後）。
**2026-09-18 起預設不輸出**（`CITYGS_EXPORT_XYZ_RGB_PLY=1` 恢復）；官方參考線跑的是官方原始碼，照常輸出。
讀者盤點（2026-09-17）：`tools/clean_outputs.py`（刪它）、`utils/train_citygs_partitions.py`（排除它）、`scripts/setup/lab.py`（說明文字）、
`scripts/lab/task_citygs_origin.sh`（讀官方原始碼產生的 PLY 標頭，有 ckpt 退路）；`tools/calibrate_block_caps.py` 讀的是 depth-init PLY，不是它。

### 4.4 lab 磁碟

`/workspace/data/hdd`（/dev/sda1）1.8 TB、已用 425 GB、剩 **1.3 TB**；⚠ `/workspace/data` 本身是 nvme 916 GB 只剩 26 GB（查錯過一次，要打 hdd 路徑）。
`outputs/lab` 151 GB；使用者手動的官方設定跑次：coarse 3.3 GB、trim 13 GB（最終 ckpt 5,217,365,741 B）。

---

## 5. CPU RAM：影像快取（面板 c）

### 5.1 快取格式（`internal/dataset.py`）

- 快取存 `self.dataset.__getitem__(i)` 的**整個 item** =（相機, (名稱, 影像, mask), extra_data）
- 影像：`image_uint8` 預設 False ⇒ `numpy_image.astype(np.float64) / 255.0` 再 `.to(torch.float)`；`image_on_cpu = True`
- extra_data：`EstimatedDepthBlockColmap.load_depth` ⇒ `np.load(...) * scale + offset` 的 float 張量 —— **不論深度 loss 權重是否為 0**
- 影像是 RGBA（抽樣 141 張 alpha 最小值 255 ⇒ 全不透明）
- ⚠⚠ **深度圖實際有沒有被載入，兩台不同**（2026-09-18 查到）：
  - **lab**：`estimated_depths/0000.png.npy` 四位數、配對修正後重新產生 => 找得到、**有載入**（log：`found 6306 depth maps`）=> 下表「現行每視角 25.6 MB」是 lab 的情況
  - **本機**：`estimated_depths/000001.png.npy` 六位數、從 1 起算，是 09-13 配對修正**之前**的舊檔 =>
    每張都印「找不到深度圖 —— 位置對應失敗，不做補零猜測」=> **本機根本沒載深度**
    （現行配方深度權重 0，不影響任何現有結果；但本機不能跑需要深度的實驗，內容也對應錯位前的影像）

| 項目 | 形狀 | dtype | 每張 |
|---|---|---|---|
| 影像（現行） | 900x1600x3 | float32 | 17.28 MB（16.5 MiB） |
| 影像載入暫存 | 900x1600x4 | float64 | 34.56 MB |
| 深度圖（共用 estimated_depths） | 1080x1920 | float32 | 8.29 MB（7.9 MiB） |
| **現行每視角** | | | **25.57 MB** |
| 影像（提案 uint8 RGB） | 900x1600x3 | uint8 | 4.32 MB（4.1 MiB） |
| **提案每視角（不載深度）** | | | **4.32 MB（-83%）** |

| 範圍 | 現行 | 提案 |
|---|---|---|
| b6 訓練視角 548 張 | 14.0 GB | 2.37 GB |
| lab 三槽（x3） | 42.1 GB | 7.1 GB |
lab 實見每跑次 17~20 GB（另含 val 集與 8 條載入執行緒的暫存）；lab 實體 RAM 62 GB，三槽時可用約 2 GB。

**本機實測（2026-09-18，`scripts/task_bitcheck_data.sh`，b6、1,200 步、訓練行程 RSS 峰值取樣；本機沒載深度）**

| 跑次 | 設定 | RSS 峰值 |
|---|---|---|
| bitcheck_A | 原樣（float32 影像快取） | 12,647 MiB |
| bitcheck_A2 | 原樣重跑 | 12,684 MiB |
| bitcheck_D | `--data.image_uint8 true --data.skip_unused_depth true` | **4,819 MiB（-62%，本機只來自 uint8）** |
⇒ 在 lab（有載深度）上預期降幅更大。

### 5.2 影響

- lab 三槽被 RAM 綁住；官方分塊不得不把快取壓到 512 張（`task_citygs_origin.sh`）
- ⬜ 三槽滿載時資料載入是否被拖慢：未量

---

## 6. 改善項目與決策

| # | 項目 | 決策（09-17） | 狀態 |
|---|---|---|---|
| 1 | uint8 影像快取＋深度權重 0 時不載深度 | **做** | ✅ 已驗證並啟用（現行 config，本機＋lab）；本機 RAM 12.6 -> 4.8 GB |
| 2 | 中間 ckpt 只存參數 | **不考慮** | — |
| 3 | 停止輸出 `-xyz_rgb.ply` | **做** | ✅ 已實作並同步 lab（預設不輸出，`CITYGS_EXPORT_XYZ_RGB_PLY=1` 恢復）；只影響之後新開跑的訓練 |
| 4 | 降 SH 階數或壓縮顏色 | **待辦** | 會影響品質，屬研究題目 |

### 6.1 uint8＋不載深度的驗證

**(a) 窮舉 256 個 uint8 值，與現行路徑（CPU float64 / 255 -> float32）比較**

| 轉換方式 | 不同的值 |
|---|---|
| CPU float32 / 255.0 | 0 |
| CPU float32 / 255（整數） | 0 |
| CPU uint8.float().div(255.) | 0 |
| **GPU float32 / 255.0**（原本 `on_after_batch_transfer` 的寫法） | **126**（最大差 5.96e-08） |
| GPU float32 / 255（整數） | 126 |
| GPU float64 / 255 -> float32 | 0 |

**(b) 真實訓練影像（b6 資料夾抽樣，含 LANCZOS 縮放與 alpha 合成）**

| 比較 | 結果 |
|---|---|
| 抽 50 張：天真 uint8（GPU float32 / 255） | 50/50 張有差異 |
| 抽 50 張：修正 uint8（GPU float64 / 255 -> float32） | 0/50 |
| 抽 141 張：alpha 最小值 | 255（全不透明） |
| 抽 141 張：修正路徑 vs 現行（含 alpha 合成原式） | **0/141** |

**(c) 深度的讀者盤點**：只有 `internal/metrics/citygsv2_metrics.py` 與 `mcmc_inverse_depth_metrics.py`，兩者 GT 為 None 時回傳 0；
訓練時權重 0 整段跳過 ⇒ 訓練不受影響。會變的：val 的 `d_reg` 紀錄變 0（連帶 `val/loss`）；最佳 ckpt 按 PSNR 挑（`best_val.txt`），不受影響。

**(d) 實作（預設全關，舊行為不變）**
- `internal/dataset.py`：uint8 路徑遇 RGBA 且 alpha 全為 255 => 丟 alpha；有非不透明像素 => 該張退回 float 路徑；
  `on_after_batch_transfer` 改成 `(gt_image.to(torch.float64) / 255.).to(camera.R.dtype)`
- `DataModule(skip_unused_depth=True)`：metric 的 `depth_loss_weight.init == 0` 時清空 train／val／test 的 extra_data，印 `[depth] ✅`
- 用法：`--data.image_uint8 true --data.skip_unused_depth true`

**(e) 端到端（最終 ckpt）：無法判定** —— `scripts/task_bitcheck_data.sh`（本機 b6、1,200 步）

| 比較 | 25 個張量中不同的 | 最大差（Adam 狀態） |
|---|---|---|
| A vs A2（原樣跑兩次） | **18** | exp_avg 8.0e-05／exp_avg_sq 2.6e-07 |
| A vs D（uint8＋不載深度） | 18 | exp_avg 1.9e-04／exp_avg_sq 2.8e-07 |
⇒ **訓練本身不可逐位元重現**（GPU 運算非確定性），用最終 ckpt 判斷不了；兩組差異同一量級。
⚠ 第一次比對時工具當掉（`No module named 'internal'`），腳本誤報成「原樣重跑就不相同」—— 工具已修（加 repo 根目錄到 import 路徑），上表是修好後重比的結果。

**(f) 管線層級（送進 training_step 的 GT 影像 sha1）：✅ 逐位元相同** —— `scripts/task_bitcheck_pipeline.sh`（本機 b6、60 步 x 2，`CITYGS_BATCH_HASH_STEPS=60`，預設關）

| 項目 | P0 原樣 | P1 uint8＋不載深度 |
|---|---|---|
| 雜湊行數 | 60 | 60 |
| 送進訓練的 dtype／形狀 | float32／(3, 900, 1600) | float32／(3, 900, 1600) |
| 前三步 | step 0 3746.png 088190e2ed0859b3／step 1 2732.png f8362b5f32bef8e1／step 2 0556.png c912ee7dd2d1a9ea | 完全相同 |
| 整份檔案 `cmp` | | **逐位元相同**（60 個不同的 sha1，非同一張重複） |
⚠ 第一版比對腳本取第 2~7 欄，因形狀 "(3, 900, 1600)" 含空格而**漏掉第 8 欄的 sha1**，印出的 ✅ 不算數；已改成比整行並用 `cmp` 重比。

**結論**：訓練輸入逐位元不變（影像：窮舉＋真實影像＋真實管線三層都驗過；深度：只有 metric 讀、權重 0 時訓練整段跳過）。
**啟用（2026-09-18）**：`configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml` 的 data 區加 `image_uint8: true`、`skip_unused_depth: true`；本機與 lab 同步（md5 核對）。
⚠ 副作用：之後的跑次 val 的 `d_reg`（連帶 `val/loss`）記成 0；lab 上跑到一半的 sfmfill 參數比較會出現前後兩種 val/loss，**PSNR／SSIM／LPIPS 不受影響**。
