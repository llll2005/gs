# 10 速度優化：lean_render、kernel 拆解與候選清單

> **圖**：`10_速度優化_lean與kernel拆解.png`（同名，1 個面板；10-10 起逐段迴圈時間併入 `01`(a)，這裡不再重複）　｜　**年代**：新年代（時間量測，不經過影像↔姿態配對）
> **資料**：lab [solo] `scripts/lab/task_lean_check.sh`（`logs/lean_check_1002_1031.log`）：`tools/check_lean_render.py`（dump／compare／profile／adamcheck）；
> 真實迴圈 `_StepProfiler`（cs60_conic b6 @14,999 起 1,200 步，含 1 次 trim）；`tools/check_noise_opts.py`
> **重現**：`python tools/plot_analysis.py`（`n10_kernels`）

## 圖表對照

| 面板 | 內容 | 對應本文 |
|---|---|---|
| （已移到 `01`(a)） | 真實訓練迴圈逐段時間：lean 關／lean／lean＋record_reduce／＋飽和跳過＋先加總，加上官方兩條線 | §1、§6、§7 |
| 左 | kernel 拆解（profiler，**10-02 資料**）：逐配對的光柵 kernel vs 逐顆／逐像素（⚠ 沒重量 record_reduce 與 10-10 的 backward 改動） | §2 |
| 右 | optimizer.step：foreach（現行）vs fused | §3 |

## 結論

- **現行（10-03）**：lean_render＋**record_reduce** 已裝進 lab；`record_reduce` 讓每次 trim 27.6 → 18.0 s（-35%）、增生期每步 -10.9%，4x4 全場景開啟。
  `tile_cull` 正確但**淨變慢**（配對 -36%，forward 反而 +30%）=> 不開（§6）。
- **lean_render（剃除沒有消費者的幾何通道）：真實迴圈每步 152.3 → 120.5 ms（-20.9%）**；backward -19%、forward -42%、峰值配置 4,022 → 3,821 MiB。
  渲染、radii、覆蓋數逐位元相同；梯度差在浮點加總順序級。**已裝進 lab 共用環境（10-03）**，不比時間的臂開啟（`CITYGS_LEAN=1`）。
- **時間大頭是逐 (像素, 顆粒) 配對的光柵 kernel，約佔 kernel 時間 63%**；Adam 6%、SSIM 4% => 參數壓縮（sh2/sh0）主要收益是 VRAM，不是速度。
- lean 對 trim 的 record pass 幾乎沒省（27.9 → 27.6 s／次）；lean 後 **trim 約佔增生期每步 37%** —— 它每個被評估的配對做 2 個 global atomicAdd，是下一個目標。
- **fused Adam 在 torch 2.0.1 反而慢 2.3~3.2 倍** => 不採用。
- fast_noise（MCMC 噪音的代數改寫）與原算法只差浮點捨入（5 個 ckpt，相對差中位 ~6e-8）；noise_gate 跳過的顆粒中約 1% 噪音 ≥ 自身 Adam 步長 => 要訓練驗（`cnogate` 已排）。
- 第二批（10-03 驗完，§6）：`record_reduce` 採用（4x4 全場景用它）；`tile_cull` 正確但淨變慢、不採用。
- **第三批（10-09／10，§7）：backward 飽和跳過（SAT）＋ warp 內先加總（RED）**：backward 68.75 → **40.86 ms**、真實每步 120.7 → **90.5 ms**（satred，b6 @14,999、N 2.58M、唯一沒被外部佔卡污染的一組）。
  正確性：大元素逐元素相對誤差 p99.9 1.8e-4（噪音底 1.7e-5 的 10.5 倍，形式上超過門檻，原因是加總順序改變）；2,000 步訓練層級的 held-out 都在 base／base2 的噪音 0.47 dB 內。
  ⇒ 共用環境已裝（欄位 `bwd_sat_skip`／`bwd_reduce`，預設關）；**60k 分數判定 `best0fast`（4x4 b6／b12）出來前不進預設**。

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
| 4 | sh3 → sh2／sh1／sh0 | 逐顆＋VRAM | sh2 VRAM -0.79 GB、sh1 -1.35 GB、sh0 -1.68 GB（@2.34M） | **會** | ⏳ `csh2`／`csh1`／`csh0` 60k |
| 5 | block／warp 內先加總再 atomic | 逐配對（trim record 2 個、backward 14 個） | **實測：每次 trim -35%、增生期每步 -10.9%** | 浮點加總順序（trim 遮罩重疊 99.9987%） | ✅ record 版已裝；backward 版 💤 |
| 6 | PyTorch 升級（只換環境的單變數） | 配置器、optimizer、編譯 | 未知 | 浮點級 | 💤 提案 |
| 7 | 每步 3 個 GPU→CPU 同步點 | CPU/GPU 重疊 | <2% | 否 | 💤 低優先 |
| 8 | tile 精確剔除（StopThePop 2DGS 版） | 逐配對 | 配對實測 -35.7%，但 **forward +30%、淨 +3.6%／步** | 渲染逐位元相同 | ⛔ 現行實作不划算（§6） |
| 9 | per-splat backward（Taming §4.1） | 光柵 backward | 大（Taming 的主要來源） | 數值等價 | 💤 #5 的強化版 |
| 10 | SH dc／高階分開傳入（不 cat） | forward 逐顆複製 | 小 | 等價 | 💤 |
| 11 | SH 高階每 16 步更新（Taming §4.2） | Adam | Adam 大降，但 Adam 只佔 6% | **會** | 💤 |
| 12 | 誤差引導 tile 抽樣 | 逐配對 | 上限受「何時開始」限制（`11`） | **會** | 💤 排在 #5 #8 之後 |
| 13 | 灰階先訓、最後上色 | 逐顆（VRAM -1.2 GB）＋顏色 | 前提已量：等亮度邊僅約 0.24% 像素 | **會** | 💤 等 SH 曲線（sh3／2／1／0）結果 |
| 14 | YCbCr（亮度 SH3＋色度 SH0） | 逐顆 | 每顆顏色 48 → 18 float | **會** | 💤 等 SH 曲線（sh3／2／1／0）結果 |
| 15 | `skip_surf_normal`（不算 surf normal） | forward 後處理 | **實測 -1%** | 否（`tools/check_skip_surf_normal.py`，b6/b13 8 台：渲染與 loss 逐位元同、梯度在噪音底內） | ✅ 驗過；等目前比較家族跑完再改預設（避免同家族 resolved config 前後不一） |

**判定規則（寫在結果出來前）**：等價類（#5、#8）要求渲染／radii／覆蓋數逐位元相同（record_reduce 只要求覆蓋數相同、T*alpha 浮點級、trim 遮罩重疊接近 100%），
梯度在噪音底內，再看真實迴圈每步時間；會改結果的（#4、#11~14）一律 60k b6/b13 對 cs60_conic 比四項指標。

## §5 設計討論的結論（2026-10-02 問答，原文已歸檔）

- **逐配對的計算不能刪**：它就是可微光柵化本身（forward 的混合與 backward 把每個像素的梯度分回給顆粒）；能動的只有「做多少配對」（trim、v/c、conic、EXACT_SUPPORT、tile_cull）與「每個配對多貴」（lean、block 內加總）。
- **逐個改，不另開乾淨專案**：迴圈外只佔 2%，重寫框架得不到多少；Python 版本不改 kernel 速度，PyTorch 版本可以當「程式碼不動、只換環境」的單變數測；
  重寫要重新移植並重驗幾十個機制，且所有基準要重跑。
- **參數壓縮／latent**：SH 在光柵化前就逐顆解成 3 個顏色 => 壓縮只打得到逐顆那一塊（主要是 VRAM）；把混合搬到 latent／低解析度才打得到逐配對，但細節由解碼器生成，與我方「虛假細節」病灶同型。
- **灰階先訓**：前提（亮度主導幾何）已量：梯度能量亮度 85.6%、色差邊中亮度不變的只有 5.2%；與 YCbCr、sh2/sh0 同屬「顏色參數變少」，先等 sh2/sh0 的品質代價。

## §6 第二批驗證（10-03，lab [solo]，`logs/speed2_check_1003_0604.log`；cs60_conic b6）

| 比對 | 渲染／radii／覆蓋數 | trim 遮罩重疊 | binning 配對 | 單相機計時（ms，中位） |
|---|---|---|---|---|
| B 共用環境 vs 新 .so（lean 關） | 逐位元相同 | 100% | 不變 | 不變（+0.1~0.2%） |
| N 噪音底（新 .so lean 兩次） | 逐位元相同 | 100% | 不變 | — |
| R lean vs +record_reduce | 逐位元相同 | 99.9987%（9 顆不同） | 不變 | record 20.46 → **12.01（-41%）** |
| T lean vs +record_reduce+tile_cull | 逐位元相同 | 99.9987% | **-35.7%** | forward+loss 15.03 → **22.64（+51%）**、backward -3%、record -41% |

**真實迴圈**（@14,999 起 1,200 步，含 1 次 trim）：

| 設定 | 真實步（不含 trim） | 每次 trim | 增生期每步（trim ÷ 500） |
|---|---:|---:|---:|
| lean | 120.57 ms | 27.64 s | 175.9 ms |
| + record_reduce | 120.69 | **17.97（-35%）** | **156.6（-10.9%）** |
| + record_reduce + tile_cull | 124.96（forward 21.1 → 27.6） | 18.08 | 161.1（-8.4%） |

- record_reduce：整趟 60k（trim 只在前 30k）約省 3%；4x4 開啟。
- tile_cull：省下的配對抵不過逐 tile 判斷本身（每個候選 (tile, 顆粒) 都要做單應映射＋四邊形距離）=> 現行實作淨變慢。
  改良方向（未做）：只對大足跡的顆粒做判斷、或把判斷併進 duplicateWithKeys 免一次額外掃描。
- ⚠ 判定器第一版的 bug：拿來比的「真實步時間」**不含週期 trim**（trim 另列）=> 判成 no_gain；已修為「真實步 + 每次 trim ÷ 500」並重判。

## 限制
- lean 的計時是單一塊（b6）、增生期 1,200 步；收割期（沒有 trim）的比例不同。
- profiler 的「其他」含 runtime API 的計時，可能有重複，只當量級。

## §7 backward 飽和跳過＋warp 內先加總（10-09／10）

實作：`renderCUDA_opt<C,ABS,SAT,RED>`（只走 lean 路徑）。
- **SAT**：每個 tile 從「tile 內最大的最後貢獻者」開始往回走，跳過 forward 時已經飽和（T 低於門檻）之後才輪到的配對 —— 這些配對在 forward 沒有貢獻，backward 對它們的梯度恆為 0。
- **RED**：同一顆粒在一個 tile 內被 256 個像素各自 `atomicAdd` 到同一個全域位址；改成 warp 內 `cg::reduce` → shared atomics → 每批一次全域 atomic，減少同址衝突。

| 量測（lab b6 @14,999、N 2.58M、[solo]） | base | satred |
|---|---|---|
| backward（逐段同步） | 68.75 ms | **40.86 ms** |
| 真實每步（不同步） | 120.7 ms | **90.5 ms** |
| 段內峰值配置（backward） | — | 3,868 MiB |

驗證（`scripts/lab/task_bwdopt_check.sh verify2`）：渲染逐位元相同；梯度大元素 p99.9 相對誤差 1.8e-4 vs 自身噪音 1.7e-5；
2,000 步 held-out 都在噪音內。⚠ 第一輪計時被外部佔卡污染，只有 satred 這組乾淨；sat／red 單獨的貢獻沒有乾淨數字。
60k 判定：`CITYGS_GRID=44 ... task_cmp.sh <6|12> best0fast`（已排在 g44 best0 之後）。

**60k 分數判定（10-10，4x4 [solo]，對照 g44_best0 同塊）**

| | val b6 | val b12 | LPIPS b6／b12 | 塊內 held-out b6／b12 |
|---|---|---|---|---|
| g44_best0 | 30.206 | 34.167 | .1012／.0862 | 20.03／22.42 |
| g44_best0fast | 30.194 | 34.181 | .1014／.0870 | 20.01／22.77 |

=> 差距都在噪音內 => **通過**；下一個比較家族起開成預設（這批 best0 單變數臂沒開，不中途混用）。換塊評測（合併後）重跑中。

