# 10 速度優化：lean_render、kernel 拆解與候選清單

> **圖**：`10_速度優化_lean與kernel拆解.png`（同名，2 個面板）　｜　**年代**：新年代（時間量測，不經過影像↔姿態配對）
> **資料**：lab [solo] `scripts/lab/task_lean_check.sh`（`logs/lean_check_1002_1031.log`）：`tools/check_lean_render.py`（dump／compare／profile／adamcheck）；
> 真實迴圈 `_StepProfiler`（cs60_conic b6 @14,999 起 1,200 步，含 1 次 trim）；`tools/check_noise_opts.py`
> **重現**：`python tools/plot_analysis.py`（a=`n10_lean_loop`、b=`n10_kernels`）

## 圖表對照

| 面板 | 內容 | 對應本文 |
|---|---|---|
| (a) | 真實訓練迴圈逐段時間：lean 關 vs 開 | §1 |
| (b) 左 | kernel 拆解（profiler）：逐配對的光柵 kernel vs 逐顆／逐像素 | §2 |
| (b) 右 | optimizer.step：foreach（現行）vs fused | §3 |

## 結論

- **現行（10-03）**：lean_render 已裝進 lab（每步 -20.9%，不比時間的臂開啟）；第二批 `record_reduce`／`tile_cull` 的等價驗證已排（[solo]）。
- **lean_render（剃除沒有消費者的幾何通道）：真實迴圈每步 152.3 → 120.5 ms（-20.9%）**；backward -19%、forward -42%、峰值配置 4,022 → 3,821 MiB。
  渲染、radii、覆蓋數逐位元相同；梯度差在浮點加總順序級。**已裝進 lab 共用環境（10-03）**，不比時間的臂開啟（`CITYGS_LEAN=1`）。
- **時間大頭是逐 (像素, 顆粒) 配對的光柵 kernel，約佔 kernel 時間 63%**；Adam 6%、SSIM 4% => 參數壓縮（sh2/sh0）主要收益是 VRAM，不是速度。
- lean 對 trim 的 record pass 幾乎沒省（27.9 → 27.6 s／次）；lean 後 **trim 約佔增生期每步 37%** —— 它每個被評估的配對做 2 個 global atomicAdd，是下一個目標。
- **fused Adam 在 torch 2.0.1 反而慢 2.3~3.2 倍** => 不採用。
- fast_noise（MCMC 噪音的代數改寫）與原算法只差浮點捨入（5 個 ckpt，相對差中位 ~6e-8）；noise_gate 跳過的顆粒中約 1% 噪音 ≥ 自身 Adam 步長 => 要訓練驗（`cnogate` 已排）。
- ⏳ 第二批（光柵器新旗標，預設關）已排 [solo] 驗證：`record_reduce`（trim record 在 block 內先加總）、`tile_cull`（逐 tile 精確剔除，上限去掉 30~42% 配對，`11`）。

---

## §1 lean_render（面板 a）

現行配方 normal／dist／depth 權重全為 0，光柵器卻每步照算：
```
訓練 forward   每像素累積深度／法線／中位深度／distortion 6 個通道＋Python 端法線轉換、nan_to_num 等全幅後處理
訓練 backward  每個 (像素, 顆粒) 配對 17 個 atomicAdd 裡有 3 個是加 0 的法線梯度，外加深度/distortion 梯度算術
trim pass      record 模式照算顏色與 7 個通道，呼叫端只取 T*alpha 與覆蓋數
```
實作：`forward.cu` 的 renderCUDA 加 MODE 模板（0＝原 kernel 原始碼不動／1＝訓練只算顏色＋alpha／2＝record 只算 T*alpha 與覆蓋數）、
`backward.cu` 加 GEOM 模板；renderer 旗標 `lean_train`（預設關）。幾何輸出被接進 loss 時在 backward 當場報錯。

**等價驗證**（cs60_conic b6 60k，6 台相機）

| 比對 | 渲染 | radii | 覆蓋數 | 梯度最大相對差 |
|---|---|---|---|---|
| A 舊 .so 跑兩次（噪音底） | 逐位元同 | 同 | 同 | 1.4~2.2e-6 |
| B 新 .so 舊路徑 vs 舊 .so | 逐位元同 | 同 | 同 | 1.3~2.5e-6 |
| C lean 關 vs 開 | 逐位元同 | 同 | 同 | 2~4e-6（rotations 1.2e-5，約噪音底 5 倍） |

1,200 步真實迴圈兩邊：trim 剪 280,090 vs 280,305、終點 N 2,287,677 vs 2,287,462、val 24.29 vs 24.27 => 等價。

**真實迴圈逐段（ms／步，wall，每標記點同步）**

| 段落 | lean 關 | lean 開 |
|---|---:|---:|
| backward | 84.85 | 68.64 |
| forward | 36.69 | 21.13 |
| 週期 trim（1,000 步窗內 1 次；每次） | 27.87（27,875 ms） | 27.59（27,589 ms） |
| optimizer.step | 15.15 | 15.15 |
| 迴圈外 | 8.00 | 8.03 |
| loss | 5.02 | 5.05 |
| **真實每步** | **152.30（6.57 it/s）** | **120.54（8.30 it/s）** |
| 峰值配置 | 4,022 MiB | 3,821 MiB |

⚠ trim 的「ms／步」是按 1,000 步取樣窗攤平；真實週期是 500 步 => 增生期約 55 ms／步，lean 後約佔每步 37%。

## §2 kernel 拆解（面板 b 左）

profiler，b6 @14,999（N 2.6M），每步換一台相機、12 步，含 1 台相機的 record；**已扣掉 autograd 包裝列**（`_RasterizeGaussians*` 與其 kernel 重複計時）。

| 類別 | lean 關（ms／步） | lean 開 |
|---|---:|---:|
| 光柵 backward（逐配對） | 80.55 | 62.63 |
| 光柵 forward＋record（逐配對） | 74.29 | 58.47（訓練 forward 18.26、record 40.21） |
| Adam（逐參數，foreach） | 14.81 | 14.82 |
| SSIM 卷積（逐像素） | 10.02 | 9.57 |
| SH cat＋binning 排序＋前處理 | 9.08 | 9.07 |
| 其他（小運算、含 runtime API 列） | 55.87 | 58.17 |

=> 逐配對約 63%；「逐顆」的部分（Adam、前處理、SH cat）合計不到 10%。

## §3 fused Adam（面板 b 右）

同一組梯度重複套用、同起點：

| | foreach（現行） | fused | 參數差（以更新量為單位） |
|---|---:|---:|---|
| 1 步 | 20.90 ms | 48.69 ms（+133%） | 1e-5 ~ 2.4e-4 |
| 50 步 | 13.50 ms | 43.32 ms（+221%） | 1.6e-5 ~ 1.3e-4 |

## §4 候選清單（預期收益為估計，除非標「實測」）

| # | 候選 | 打哪一塊 | 收益 | 改結果？ | 狀態 |
|---|---|---|---|---|---|
| 1 | lean_render | 幾何通道＋trim 顏色 | **實測 -20.9%／步** | 否（逐位元／atomic 順序級） | ✅ 已裝 gspl |
| 3 | fused Adam | optimizer | **實測慢 2.3~3.2 倍** | 浮點級 | ⛔ |
| 4 | sh3 → sh2／sh0 | 逐顆＋VRAM | sh2 VRAM -0.79 GB、sh0 -1.68 GB（@2.34M） | **會** | ⏳ `csh2`／`csh0` 60k |
| 5 | block／warp 內先加總再 atomic | 逐配對（trim record 2 個、backward 14 個） | record 版優先 | 浮點加總順序 | ⏳ `record_reduce` 已實作，驗證已排 |
| 6 | PyTorch 升級（只換環境的單變數） | 配置器、optimizer、編譯 | 未知 | 浮點級 | 💤 提案 |
| 7 | 每步 3 個 GPU→CPU 同步點 | CPU/GPU 重疊 | <2% | 否 | 💤 低優先 |
| 8 | tile 精確剔除（StopThePop 2DGS 版） | 逐配對 | **上限 30~42% 配對** | 渲染逐位元相同 | ⏳ `tile_cull` 已實作，驗證已排 |
| 9 | per-splat backward（Taming §4.1） | 光柵 backward | 大（Taming 的主要來源） | 數值等價 | 💤 #5 的強化版 |
| 10 | SH dc／高階分開傳入（不 cat） | forward 逐顆複製 | 小 | 等價 | 💤 |
| 11 | SH 高階每 16 步更新（Taming §4.2） | Adam | Adam 大降，但 Adam 只佔 6% | **會** | 💤 |
| 12 | 誤差引導 tile 抽樣 | 逐配對 | 上限受「何時開始」限制（`11`） | **會** | 💤 排在 #5 #8 之後 |
| 13 | 灰階先訓、最後上色 | 逐顆（VRAM -1.2 GB）＋顏色 | 前提已量：等亮度邊僅約 0.24% 像素 | **會** | 💤 等 sh2/sh0 結果 |
| 14 | YCbCr（亮度 SH3＋色度 SH0） | 逐顆 | 每顆顏色 48 → 18 float | **會** | 💤 等 sh2/sh0 結果 |
| 15 | `skip_surf_normal`（不算 surf normal） | forward 後處理 | **實測 -1%** | 否（`tools/check_skip_surf_normal.py`，b6/b13 8 台：渲染與 loss 逐位元同、梯度在噪音底內） | ✅ 驗過；等目前比較家族跑完再改預設（避免同家族 resolved config 前後不一） |

**判定規則（寫在結果出來前）**：等價類（#5、#8）要求渲染／radii／覆蓋數逐位元相同（record_reduce 只要求覆蓋數相同、T*alpha 浮點級、trim 遮罩重疊接近 100%），
梯度在噪音底內，再看真實迴圈每步時間；會改結果的（#4、#11~14）一律 60k b6/b13 對 cs60_conic 比四項指標。

## §5 設計討論的結論（2026-10-02 問答，原文已歸檔）

- **逐配對的計算不能刪**：它就是可微光柵化本身（forward 的混合與 backward 把每個像素的梯度分回給顆粒）；能動的只有「做多少配對」（trim、v/c、conic、EXACT_SUPPORT、tile_cull）與「每個配對多貴」（lean、block 內加總）。
- **逐個改，不另開乾淨專案**：迴圈外只佔 2%，重寫框架得不到多少；Python 版本不改 kernel 速度，PyTorch 版本可以當「程式碼不動、只換環境」的單變數測；
  重寫要重新移植並重驗幾十個機制，且所有基準要重跑。
- **參數壓縮／latent**：SH 在光柵化前就逐顆解成 3 個顏色 => 壓縮只打得到逐顆那一塊（主要是 VRAM）；把混合搬到 latent／低解析度才打得到逐配對，但細節由解碼器生成，與我方「虛假細節」病灶同型。
- **灰階先訓**：前提（亮度主導幾何）已量：梯度能量亮度 85.6%、色差邊中亮度不變的只有 5.2%；與 YCbCr、sh2/sh0 同屬「顏色參數變少」，先等 sh2/sh0 的品質代價。

## 限制
- lean 的計時是單一塊（b6）、增生期 1,200 步；收割期（沒有 trim）的比例不同。
- profiler 的「其他」含 runtime API 的計時，可能有重複，只當量級。
