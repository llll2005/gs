---
id: R-pipeline
狀態: 有效（敘述與公式）；**內含的分數需重測**
年代: 跨年代 —— 敘述與公式不經過 GT 影像↔姿態的配對
摘要: 從 Input 到 Output 每一步的數學變換與模組選擇點（分塊/初始化/表徵/渲染/損失/密度控制）。
依賴: [S3-data, S4-init]
時間: 2026-07-28 ~ 2026-08
---

> ✅ **2026-09-13 重驗：搬回主線。** 模組與數學屬程式碼事實 => 倖存。其中 depth-init 那一段要配合 v2 §3.3 讀（那批 PLY 在新資料上不可用）。
> 現行結論見 `研究總覽_v2.md`；本檔是它的**細節依據**，兩者不重複。

# 完整 Pipeline 數學詳解 —— 從 Input 到 Output 的每一步變換

> 撰寫：2026-06-18。目的：把「影像/相機 → 高斯生成 → 渲染 → 損失 → 密度控制 → 剪枝 → 合併 → 輸出」整條流程的**數學變換**、**可選模組**、**我們的嘗試**一次講清楚。
> 數學用 ASCII（||x|| 範數、sum_i 求和、^ 次方、a·b 內積、a×b 外積、prod 連乘）。
> 對應程式：`internal/` 下 models / renderers / density_controllers / metrics 四大資料夾。
> 跨檔脈絡見 `memory/dev_cycles.md`、實驗數字見 `紀錄/本次session待辦.md`。

---

## 0. 符號約定

| 符號 | 意義 |
|---|---|
| `mu` (μ) | 高斯中心，世界座標 3D，∈ R^3 |
| `s` | 尺度。2DGS 是 (s_u, s_v) ∈ R^2；3DGS 是 (s_x,s_y,s_z) ∈ R^3 |
| `q` | 旋轉四元數 ∈ R^4，→ 旋轉矩陣 R = [t_u, t_v, n] |
| `o` (alpha) | 不透明度 ∈ [0,1] |
| `SH` | 球諧係數，決定視角相關顏色。degree D → (D+1)^2 個係數 ×3 通道 |
| `T_i` | 第 i 個高斯的透射率（transmittance）= prod_{j<i}(1-alpha_j) |
| `K` | 相機內參矩陣 [[fx,0,cx],[0,fy,cy],[0,0,1]] |
| `[R|t]` | 相機外參（world→camera 或 camera→world，依上下文） |
| `I` | 渲染影像；`I_gt` ground truth |

整條 pipeline 的「狀態」= 一組高斯 `G = {(mu_i, s_i, q_i, o_i, SH_i)}_{i=1..N}`。所有階段都在**生成、變換、增刪**這組高斯，最後拿它渲染出影像。

---

## 1. 輸入 (INPUT)

### 1.1 原始資料
- **影像** `{I_k}`：MatrixCity aerial，每場景 >4000 張訓練圖、>450 測試圖。
- **相機位姿** `{[R_k|t_k], K_k}`：由 COLMAP SfM 求得（或資料集提供）。
- **稀疏點雲**（可選）：COLMAP 三角化的 sparse points，CityGS V1 用它當初始化；我們的 depth-init 不直接用它。

### 1.2 預處理變換
- **降採樣**：aerial 把長邊降到固定像素（論文 1600px；我們用 `down_sample_factor=1.2`）。
  - ⚠ 這兩者解析度可能不同 → PSNR 不可直接跨設定比較（見實驗記錄）。
- **深度圖估計**：`utils/estimate_dataset_depths.py` 用 **Depth Anything V2** 對每張圖推單目深度 `D_pseudo_k`。
  - 注意：這是**相對深度**（up-to-scale），當「幾何正則的偽標籤」，不是絕對深度。後面 depth loss 會用尺度對齊版本。

**輸出**：`{I_k, [R_k|t_k], K_k, D_pseudo_k}`，COLMAP 目錄結構 `images/ + sparse/0/`。

---

## 2. 場景分塊 (PARTITIONING) — 可選但 city-scale 必要

**動機**：整個城市的高斯放不進單 GPU。divide-and-conquer：切成 N×M 格，每塊獨立訓練、最後合併。

程式：`internal/utils/citygs_partitioning_utils.py:CityGSPartitioning`。

### 2.1 數學
給定所有相機中心 `{c_k}` 在地面 (x,y) 投影，定義場景 AABB（軸對齊包圍盒）：
```
aabb = [min_k x_k, max_k x_k] × [min_k y_k, max_k y_k]
```
切成 N×M 個 cell，第 (i,j) 塊的空間範圍 `B_ij`。

**相機指派 + 可見性過濾**：一張相機 k 指派給塊 (i,j) 的條件是「它看得到該塊內容的比例 > visibility 閾值 tau_vis」。可見性用：
```
vis(k, B_ij) = (B_ij 內被相機 k 看到的點數) / (相機 k 看到的總點數)
若 vis(k, B_ij) >= tau_vis → 相機 k 進入塊 (i,j) 的訓練集
```
我們的設定：`block_dim=[5,5]`（25 塊）、`content_threshold (=tau_vis) = 0.08`。

⚠ **partition 目錄名由 tau_vis 決定**：`partitions-dim_5_5_visibility_0.08/`。改 tau_vis（例如 0.05）必須重新切分，否則 FileNotFound（這 session 踩過）。

### 2.2 可選變換
- `--contract`：把無界場景座標收縮到有界（類似 mip-NeRF360 的 contraction）`contract(x) = x if ||x||<=1 else (2 - 1/||x||) * x/||x||`。
- `--reorient`：把地面對齊到水平，讓分塊軸對齊地面。

**輸出**：每塊一份相機清單 `002_001.txt`（塊 7 = row2_col1）+ AABB；`partitions.pt`。

---

## 3. 初始化 (INITIALIZATION) — ★ 重大模組選擇點

每塊需要一組初始高斯。**兩條路線，差異巨大**：

### 3a. depth-init（RTG-SLAM 策略）★ 我們的主用
程式：`utils/depth_init_blocks.py`。把每張圖的深度圖**反投影**成 3D 點，直接當初始高斯。

**反投影數學**：像素 (u,v)、深度 d、內參 (fx,fy,cx,cy)、相機到世界 [R_c2w | t_c2w]：
```
相機座標：p_cam = d * [ (u-cx)/fx , (v-cy)/fy , 1 ]^T
世界座標：mu     = R_c2w * p_cam + t_c2w
```

每個反投影點變成一個 2D surfel，屬性設定（RTG-SLAM Sec 3.1）：
```
中心   mu     = 上式
法線   n      = 由深度圖的局部梯度估（或相機 z 軸）；surfel 切平面 ⊥ n
尺度   s_u=s_v ≈ d / f   （深度自適應：遠的點 surfel 大，近的點小；f=焦距）
不透明 o      = 0.99      ★ 關鍵：不透明 surface（不是 3DGS 慣用的 0.1）
顏色   SH_dc  = 該像素 RGB；高階 SH = 0
```

**為何 alpha=0.99**：不透明 surfel「一片蓋一塊表面」，不需要像半透明 3DGS 那樣堆很多層才不透 → **同品質用更少高斯**（這是 depth-init 的省點來源之一）。

**體素降採樣**（★ 為何必要）：depth-init **不是搬 COLMAP 點，是無中生有** —— 把每張圖**每個有效像素**（M_s mask：有效深度 + 掠射角<60°，100% 保留）反投影成一個 3D 點。
```
原始點數 = sum_images (有效像素數) ≈ 上百萬像素 × 上百張圖 → 輕鬆 10^8 個點
且高度重複：空拍影像大量重疊 → 同一塊表面被幾十張圖看到 → 幾十個近乎重合的點
```
voxel downsample 就是收這坨洪水：把 3D 空間切成邊長 voxel_size 的方格，**每格只留一點** → (a) 去重（多視角重合→一個 surfel）、(b) 把數量壓到目標、(c) 覆蓋均勻。
```
voxel_min = 0.015（數學逆推：目標 ~50萬點/塊；0.02→30萬；太小 → 千萬點炸 VRAM）
中間 flush（每 chunk_size frame voxel 一次）→ peak RAM 49GB 降到 ~11GB
```
→ **voxel 是「反投影點洪水」的去重+降密度機制，depth-init 內在必要，與 COLMAP 無關（COLMAP 只給位姿）。**
塊 7 depth-init PLY ≈ 430k 點。

### 3b. coarse-init（CityGaussian 原版）
先在**低解析度全場景**訓練一個 coarse 全域模型（`aerial_train_block_all_3x`，30k步），再把它**切**給每塊當初始化。
- 優點：全域一致性好，grad-densify 從一個已收斂的全域模型出發，densify 引導準 → 容易長到高品質。
- 缺點：多一道全域預訓練（~1.5h 全域，攤提到 25 塊）；且 per-block densify 容易長到數百萬點。

### 模組選擇總結
| | depth-init | coarse-init |
|---|---|---|
| 多餘預訓練 | 無 | 需全域 coarse |
| 初始 alpha | 0.99（不透明）| 繼承 coarse |
| 點放置 | 深度反投影（幾何準但無全域一致性）| 全域預訓練（一致性好）|
| 我們的 task1 結論 | 純 init 效應只差 ~+0.4dB PSNR，SSIM/LPIPS 還輸 | — |

**我們的嘗試**：曾深入比較（task1）。結論：init 是次要效應，缺口主因是高斯預算，不是 init。**但後續發現（見 §13）**：要逼近 28 分可能需要 coarse-init + grad-densify，這點還沒定論。

**輸出**：初始高斯集 `G_0`（塊 7 ≈ 430k 個 2D surfel）。

---

## 4. 表徵 (REPRESENTATION) — ★ 模組選擇點

每個高斯「存什麼」由 model 決定。程式：`internal/models/`。

### 4a. Gaussian2D（2D Surfel）★ 我們的主用 (CityGSV2)
程式：`internal/models/gaussian_2d.py:Gaussian2D`。

一個 2D surfel 是**嵌在 3D 空間裡的一片橢圓盤**：
- 中心 `mu ∈ R^3`
- 旋轉 `q ∈ R^4` → R = [t_u | t_v | n]（兩個切向量 + 法線）
- **2D 尺度** `(s_u, s_v)`（盤的長短軸；法線方向厚度 = 0）
- 不透明 `o`
- SH 係數（degree D：(D+1)^2 個 ×3）

**盤上參數化**：local 座標 (u,v) 的點映到世界：
```
P(u,v) = mu + s_u·u·t_u + s_v·v·t_v
法線   n = t_u × t_v
```
**2D 高斯權重**（落在盤上的衰減）：
```
G(u,v) = exp( -(u^2 + v^2) / 2 )
```

每點儲存量（sh2）：mu(3)+o(1)+s(2)+q(4)+SH_dc(3)+SH_rest(8×3=24) = **37 floats**；sh3 = 58 floats。

### 4b. VanillaGaussian（3DGS）
程式：`internal/models/vanilla_gaussian.py`。3D 體積高斯，尺度 3D：
```
G(x) = exp( -1/2 (x-mu)^T Σ^-1 (x-mu) ),  Σ = R S S^T R^T,  S=diag(s_x,s_y,s_z)
```
切到這個 = 走 CityGS V1（3DGS）路線。不堆疊成面、是體積 blob。

### 4c. AppearanceGS2D
2D surfel + **per-image appearance embedding**：每張圖一個外觀向量 a_k，渲染時吸收該圖的曝光/色差。`c_i = SH_i(d) + MLP(a_k, ...)`。適合真實照片集；MatrixCity 合成、跨圖一致 → 效益小。

### 模組選擇 + 我們的嘗試
- 主用 **Gaussian2D**（2DGS 是 CityGSV2 的核心，面對齊、利於 mesh、省點）。
- `sh_degree` 是它的參數不是換模組：**sh2→sh3 只改一個數**，每點 SH 從 24→45 個 rest 係數。我們驗證 sh3 在 720k 點不 OOM、+0.295dB（§13）。

---

## 5. 前向渲染 (RENDERING) — ★ 模組選擇點

把高斯集 `G` + 相機 → 影像。程式：`internal/renderers/`。核心是 2DGS 的**光柵化（rasterization）**。

### 5.1 投影：surfel → 螢幕（ray-splat intersection）
2DGS 不像 3DGS 用近似的 2D 投影協方差，而是**精確求光線與 surfel 切平面的交點**（RaDe-GS / 2DGS 的 ray-splat intersection）。

對螢幕像素 x，其反投影光線與第 i 個 surfel 盤的交點落在盤的 local 座標 `(u_i(x), v_i(x))`。實作上用一個從螢幕到 surfel-local 的**單應 (homography)** H_i（由 mu_i, t_u, t_v, s_u, s_v, 相機矩陣構成），求得 (u,v)，再帶入：
```
G_i(x) = exp( -(u_i(x)^2 + v_i(x)^2) / 2 )
```
並與一個固定的螢幕空間低通高斯取 max（抗鋸齒，避免遠處 surfel 退化成一條線）。

**每個 surfel 對該像素的有效不透明度**：
```
alpha_i(x) = o_i · G_i(x)
```

### 5.2 Alpha 合成（體積渲染方程，front-to-back）
把覆蓋像素 x 的 surfel 依深度排序後合成：
```
透射率   T_i = prod_{j<i} (1 - alpha_j(x))
顏色     C(x) = sum_i  c_i · alpha_i(x) · T_i
其中     c_i  = SH_i 在視角方向 d 上求值（見 5.4）
```

### 5.3 幾何附帶輸出（給正則用）
同一次光柵化還產出：
```
不透明累積 A(x)   = sum_i alpha_i T_i                （= 1 - 最終透射率）
法線圖     N(x)   = sum_i n_i alpha_i T_i             （世界座標法線的加權合成）
期望深度   D_exp(x)= sum_i z_i alpha_i T_i  /  A(x)
中位深度   D_med(x)= 取累積透射率跨越 0.5 那個 surfel 的深度
表面深度   D_surf  = (1-r)·D_exp + r·D_med           （r = depth_ratio）
分佈損失圖 dist(x) = sum_{i,j} w_i w_j |z_i - z_j|,  w=alpha·T   （深度分佈集中度）
```
`depth_ratio` r：1.0 = 用中位深度（有界場景、recall 好、道路完整）；0.0 = 期望深度（無界、抗鋸齒）。我們 aerial 用 **r=1.0**。

### 5.4 SH → RGB（視角相關顏色）
視角方向 `d = normalize(mu_i - camera_center)`，球諧求值：
```
c_i(d) = SH_dc_i + sum_{l=1..D} sum_{m=-l..l} k_{l,m,i} · Y_{l,m}(d)
```
`Y_lm` 是球諧基底。degree D 越高，能表達越強的視角相關效果（高光、反射）。
- 訓練時 active degree 從 0 漸增到 D（每 `sh_degree_up_interval=1000` 步 +1）。
- ⚠ MatrixCity aerial 大多 Lambertian（漫反射屋頂/馬路），但實測 sh3 仍 +0.295dB → 有足夠視角相關內容（§13）。

### 5.5 record_transmittance（每-surfel 貢獻，給剪枝/trim 用）
渲染器有一個特殊模式：不輸出影像，而輸出**每個 surfel 的平均透射率**：
```
trans_i = (該 surfel 在這張圖貢獻的 transmittance 總和) / (它覆蓋的像素數)
```
★ **這是 rasterizer 的一個模式，不是 trimming 功能** → `diable_trimming` 開著也能用 → 我們的 importance-prune 靠它（§9）。

### 模組選擇（renderer，必須與 2DGS 表徵相容）
| renderer | trimming | record_transmittance | 用途 |
|---|---|---|---|
| **SepDepthTrim2DGSRenderer** ★ | 有（可關）| 有 | 我們主用；Trim + 獨立 depth pass |
| Vanilla2DGSRenderer | 無 | 有 | 純 2DGS；sh3 full quality 可用 |
| Appearance2DGSRenderer | — | — | 配 AppearanceGS2D |
| （3DGS/gsplat 系一大堆）| — | — | 與 2D 尺度不相容，不能用 |

`depth_ratio` 和 `diable_trimming` 是這顆的兩個關鍵旋鈕。

**輸出**：渲染影像 `I`、深度 `D_surf`、法線 `N`、分佈 `dist`、可見性 mask `radii>0`。

---

## 6. 損失 / 度量 (LOSS / METRIC) — ★ 模組選擇點

把渲染結果與 GT 比，算總損失。程式：`internal/metrics/`。

### 6.1 光度損失（所有變體共用）
```
L_rgb = (1 - lambda_dssim) · L1(I, I_gt) + lambda_dssim · (1 - SSIM(I, I_gt))
```
lambda_dssim = 0.2（標準 3DGS 值）。L1 對均值好、SSIM 對結構好。

★ **實作細節（DGD 的機制根源，程式 citygsv2_metrics.py:113-118）**：densify 期間 (step < densify_until)，**L1-RGB 項被拆成獨立的 `extra_loss` 走第二條 backward**：
```
loss       = lambda_dssim·(1-SSIM) + dist_loss + normal_loss + d_reg
extra_loss = (1 - lambda_dssim)·L1(I,I_gt)        ← 獨立 backward
```
目的：讓 L1 的螢幕空間梯度可被單獨取出、重新平衡，給 grad-densify 用（§8a DGD，`densify_grad_scaler`）。densify 結束後就合回單一 loss。**MCMC 不用 grad-densify → `densify_grad_scaler=0` 讓這步變 no-op，但兩條 backward 仍各自累積真實參數梯度。**

### 6.2 CityGSV2Metrics（我們的幾何正則底）
程式：`internal/metrics/citygsv2_metrics.py`。在 L_rgb 上加兩項幾何正則：

**(a) 法線一致性**（讓 surfel 法線對齊由深度推出的表面法線；程式 gs2d_metrics.py:28-29）：
```
L_normal = lambda_normal · mean_x ( 1 - N_render(x) · N_surf(x) )
N_render = 渲染合成法線（§5.3）；N_surf = 由 D_surf 空間梯度叉積算的偽表面法線
         （N_surf 已在渲染器乘上 alpha 累積 → 隱含逐像素權重）
lambda_normal = 0.0125（論文把原 0.05 降到 1/4）
normal_regularization_from_iter：預設 7000，我們 config 設 0（從頭開）
```

**(b) 深度正則**（★ 注意：用**逆深度 inverse depth**，不是深度本身；因為 Depth Anything 輸出的是 disparity-like 的逆深度）：
```
渲染逆深度  d_inv      = 1 / (D_surf + 1e-8)
偽標籤      d_inv_gt   = Depth Anything 的逆深度（可選做 mean+2σ 截斷正規化）
L_depth = w_depth(t) · [ (1-w_s)·L1(d_inv, d_inv_gt) + w_s·(1 - SSIM(d_inv, d_inv_gt)) ]
          （depth_loss_type=l1+ssim, w_s=depth_loss_ssim_weight；我們 config w_s=1.0 → 純 SSIM 項）
```
**權重排程**（程式 get_weight，連續指數衰減）：
```
w_depth(t) = init · (final_factor)^( min(t/max_steps, 1) )
   t=0      → w = init                    （= 0.5）
   t=max    → w = init · final_factor      （final）
```
★ **final_factor 是關鍵旋鈕**：30k 舊 recipe 用 0.001（→ final=5e-4，太早衰減 → depth reg 後期失效 → 幾何崩、floater 多、畫面一團糟）；60k 改 **0.05**（→ final=0.025，強 50 倍、全程維持）→ 畫面明顯變好、SSIM/LPIPS 升（§13 重大發現）。

**(c) 分佈損失**（2DGS 深度集中）：`L_dist = lambda_dist · mean_x dist(x)`，壓薄表面。

### 6.3 MCMCCityGSV2Metrics（MCMC 專用，= 6.2 + 兩個 L1）
程式：`internal/metrics/mcmc_citygsv2_metrics.py`。MCMC 不用 opacity-reset/size-prune，改用兩個 L1 取代：
```
L = L_CityGSV2 + lambda_o · sum_i |o_i| + lambda_s · sum_i |s_i|
lambda_o = lambda_s = 0.01
```
- `opacity_reg`：把不透明度往 0 壓 → 沒貢獻的高斯自然死掉（取代 opacity-reset 的清除）。
- `scale_reg`：把尺度往小壓（取代大尺寸剪枝）。
- ★ **免疫護城河**：只對 `o_i <= immune_opacity_threshold (0.9)` 的高斯施加 opacity_reg → **保護 alpha=0.99 的 depth-init 先驗**（驗證 step0 op_reg=0.000，全免疫）。

### 模組選擇 + 我們的嘗試
| metric | 額外項 | 搭配 |
|---|---|---|
| **CityGSV2Metrics** | normal + depth + dist | grad-densify |
| **MCMCCityGSV2Metrics** ★ | + opacity_reg + scale_reg | MCMC |
| GS2DMetrics | normal + dist（無 Depth-Anything 深度）| 基礎 2DGS |

**⚠ 耦合**：metric 與 density controller 要配對 —— MCMC 的 L1 是用來「取代」grad-densify 的 opacity-reset/prune 的。若用 grad-densify 就不需要 L1（它自己有 reset）。

**輸出**：純量 `L`（+ 可能的 `extra_loss`，CityGSV2 的 hard-depth 走獨立 backward path）。

---

## 7. 反向 + 優化 (BACKWARD + OPTIMIZE)

程式：`internal/gaussian_splatting.py:training_step`。

### 7.1 梯度
對 L 反傳，得每個高斯各屬性的梯度：`∂L/∂mu, ∂L/∂s, ∂L/∂q, ∂L/∂o, ∂L/∂SH`。
另外光柵化會回傳**螢幕空間位置梯度** `∂L/∂x_2d`（給 grad-densify 判斷哪裡欠重建用，見 §8a）。

### 7.2 Adam 更新（每屬性不同 lr）
```
mu:  means_lr_init=6.4e-5，指數衰減到 6.4e-7（max_steps 對齊總步數）
s:   scales_lr=0.004
其餘 q/o/SH 各有固定或排程 lr
```
⚠ **我們踩過的坑**：ADMM 實驗時只有 means_lr 有排程，shs/opacity/scales **是常數無排程** → 多訓練步數會持續改善外觀，造成「ADMM +0.2dB 其實是多訓練買的」confound。

### 7.3 時序（一個 training step 內）
```
render → 算 L (§5,§6)
→ before_backward     （density 介面；ADMM 在此注入 AL penalty，MCMC 不動）
→ backward            （算梯度 §7.1）
→ after_backward      ★ grad-densify / MCMC relocate+add 在這裡（§8）
→ optimizer.step()    （套梯度 §7.2）
→ scheduler.step()
接著 on_train_batch_end：
→ renderer.after_training_step   （Trim 在此剪 surfel §9）
→ light_gaussian_prune           （importance-prune 在此 §9）
→ on_train_batch_end_hooks       ★ MCMC Langevin noise 在這裡（每步）
```

---

## 8. 密度控制 (DENSITY CONTROL) — ★★ 最重要的模組選擇點

決定高斯如何「增生 / 搬移 / 死亡」。程式：`internal/density_controllers/`。**這是決定最終點數與品質的核心，三條路線數學完全不同。**

### 8a. grad-densify（Vanilla / CityGSV2DensityController）
程式：`citygsv2_density_controller.py`。**靠梯度在欠重建處狂長點 → 能長到數百萬。**

**累積螢幕空間位置梯度**（多視角平均）：
```
g_i = mean_views || ∂L/∂x_2d,i ||
```
**CityGSV2 的 DGD（Decomposed-Gradient Densification）**：優先用 SSIM 的梯度（對模糊敏感），公式：
```
grad_densify = max( omega · |∇L|_avg / |∇L_SSIM|_avg , 1 ) · ∇L_SSIM,  omega=0.9
```
**增生規則**（每 densification_interval，在 densify_from~densify_until 之間）：
```
若 g_i >= tau (densify_grad_threshold):
   - 小高斯 (max scale <= percent_dense·scene_extent) → CLONE（複製一份）
   - 大高斯 → SPLIT（裂成 N=2，程式 vanilla_density_controller.py:162-168）：
        新尺度  s_new = s / (0.8·N)            （N=2 → s/1.6）
        新位置  x_new = R · sample + μ,  sample ~ N(0, diag(s))   （從父高斯協方差取樣）
   - 額外：CityGSV2 限制過度細長高斯（axis_ratio = s_min/s_max > 0.01 才 split）
```
**opacity reset**（週期清 floater + 間接驅動 densify）：
```
每 opacity_reset_interval 步：o_i <- min(o_i, 0.01)
```
此舉壓低不透明度 → 渲染變差 → 位置梯度升 → 更多 clone（**reset 是「隱性 densification 驅動器」，不只是清 floater**，我們實驗證實關掉 reset 掉 -1.5dB）。
**剪枝**：太透明 (o < cull_opacity_threshold=0.005) 或太大的高斯刪掉。

★ **產生 28 分模型的確切設定**（從那個 4M 點 run 挖出）：
```
densify_grad_threshold=0.0002, densify_until_iter=12000,
opacity_reset_interval=2100, percent_dense=0.01, densification_interval=300
→ 長到 ~397萬 點
```

### 8b. MCMC（3DGS-MCMC 適配 2DGS）★ 我們大量用
程式：`mcmc_density_controller.py` + `mcmc_2dgs_density_controller.py`。**把高斯集看成馬可夫鏈樣本，靠「回收死點 + 受控長到 cap + 朗之萬噪聲」探索。**

**三個動作（不同時機）**：

**(A) Relocate + Add**（after_backward，densify 窗內每 interval）：
```
死亡判定：dead_i = (o_i <= min_opacity=0.005)
relocate：把死掉的高斯，搬到「按 opacity 取樣的活高斯」位置上（回收）
add_new ：活點數每次 ×1.05，長到 cap_max 為止
```
**relocation 的精確公式**（從 CUDA 源碼驗證，gsplat compute_relocation）：一個高斯被取樣 N 次要分裂成 N 個等效高斯，保持渲染積分不變：
```
o_new   = 1 - (1 - o)^(1/N)
denom   = sum_{i=1..N} sum_{k=0..i-1} C(i-1,k)·(-1)^k / sqrt(k+1) · o_new^(k+1)
coeff   = o / denom
s_new   = coeff · s_old      （所有尺度維度同一個 coeff → 與尺度維度無關）
```
★ **2D 適配**：gsplat 要 3D 尺度 → 把 2D 補一個 0 第三維呼叫，結果截回 2D（因 coeff 只看 opacity，對真實兩維精確）。

**(B) Langevin 噪聲**（on_train_batch_end，**每一步**，到最後一步才停）：
```
mu_i <- mu_i + lr_noise · Σ_i · epsilon · sigmoid_k(1 - o_i),   epsilon ~ N(0, I)
sigmoid_k(x) = 1 / (1 + exp(-100·(x - 0.995)))
lr_noise = noise_lr · xyz_lr = 5e5 · (當前 means 學習率)
```
低 opacity 的點被搖得多 = 隨機游走探索。
★ **2D 適配（切平面化）**：協方差 Σ 用「補 0 法線維度」算 → 噪聲只在 surfel 切平面內，不沿法線把點推離表面。

**(C) 兩個 L1 reg**：在 metric 裡（§6.3），取代 opacity-reset 與 size-prune。

★★ **我們 cap2M 的關鍵發現**：設 cap_max=2M，**實際只長到 969k** → cap 沒咬住，因為 **Trim 剪枝 + 兩個 L1 把 add_new 的成長吃掉了** → MCMC+Trim+L1 這套**設計上就把點數壓低**，反而擋住我們用 6GB 餘裕長到 4M。**這是「MCMC 難逼 28」的根因。**

### 8c. RTGStableDensityController
程式：`rtg_stable_density_controller.py`。depth-init 專用的「穩定化」：用 `eta` 信心計數器，達 10 後免 reset / 免 opacity prune（保護初始 surfel 不被 opacity-reset 清掉）。`depth_init_immune` 讓初始點立即穩定。無 cap（densify 自然長到 453k 左右）。

### 模組選擇 + 我們的嘗試對照
| controller | 增生機制 | 點數規模 | 我們的結果 |
|---|---|---|---|
| **grad-densify** | 梯度 clone/split + reset | 可達數百萬 | 28 分那條（但 4M 點）|
| **MCMC** ★ | relocate + add到cap + noise | 自限 ~1M（trim+L1 壓）| 22-23 分，省點但難長 |
| **RTGStable** | depth-init + eta 穩定 | ~453k | baseline 21.97 |
| DtAdmm（我們寫的）| MCMC + GAT consensus | — | net-zero，已放棄 |

**⚠ 這格是當前戰略核心**：MCMC 自限點數 → 難逼 28；grad-densify 能長到 4M/28 但點多。見 §13。

**輸出**：更新後的高斯集 `G_t`（數量隨步數變動）。

---

## 9. 修剪 / 壓縮 (PRUNING / COMPRESSION) — 我們的核心貢獻牌

### 9.1 Trim（訓練中即時，2DGS 內建）
程式：`sep_depth_trim_2dgs_renderer.py:after_training_step`。densify 窗內每 interval：
```
對每個 surfel 算跨視角 top-K transmittance 平均當貢獻 C_i
剪掉 C_i <= quantile(C, prune_ratio) 的（貢獻最低的一批）
```

### 9.2 Importance Pruning（2DGS-native，★ 我們的演算法貢獻）
程式：`internal/utils/importance_prune_2dgs.py`。脫胎 LightGaussian，但**換成 2DGS-native 準則、零 gsplat 依賴**：
```
貢獻   C_i = mean over top-K views of trans_i   （用 §5.5 的 record_transmittance）
面積   A_i = s_u,i · s_v,i                        （2D surfel 面積 = 3D 體積的類比）
重要度 V_imp,i = C_i · A_i^v_pow,  v_pow=0.1
剪掉 V_imp 最低的 k%
```
觸發點：`light_gaussian_prune` 在 on_train_batch_end，於指定 `prune_steps`（如 [20000]）一次性砍。

★ **路由**：用 **scale 維度 == 2** 判 2DGS（不能用 getter 名稱，runtime 模型有 get_scaling 但回 2D）。

**我們的剪枝實驗結論**（都從 30k 弱 recipe 的 720k 天花板剪）：
```
剪 10%→648k=22.207 / 30%→504k=22.180 / 50%→360k=22.208
→ PSNR 剪到 50% 仍無損；vs 直接低 cap 訓練 +0.46dB（explore-then-prune 贏）
```
★⚠ **重大但書**（§13）：這個「無損」是從**已飽和**的低天花板剪。從**還在爬、沒飽和**的 4M 剪會掉比較多（每點都在貢獻）。

### 9.3 VecTree Quantization（部署壓縮）
程式：`tools/vectree_lightning.py`。對最不重要的 SH 做向量量化（K-means），重要的 SH + 幾何屬性存 float16。論文宣稱 2DGS 上省 50% 儲存。這是**部署期**壓縮，不影響訓練。

**輸出**：點數更少的高斯集（部署版）。

---

## 10. 合併 (MERGE) — city-scale 必要

程式：`utils/merge_citygs_ckpts.py`。把 25 塊各自訓練好的高斯**合成一個全場景模型**。

### 數學 / 策略
- 簡單版：每塊只保留其 AABB 內的高斯（避免重複），union 起來。
- visibility-based：某相機看得到的區域用「最適合該區的塊」的高斯渲染。

### ⚠ 我們的已知問題：Gaussian drift
depth-init 的高斯訓練後會**大量漂出原塊 AABB**（位置被優化推走）→ merge filter 只留 13-15% → **全域 merged PSNR 不可信（曾測 19.5）**。
→ 所以我們**目前只信 per-block PSNR**，全域評估尺待修（選項：AABB position penalty 或 visibility-based merge）。這是任務 0，未做。

**輸出**：全場景 PLY；可跑全域 test。

---

## 11. 輸出 + 評估 (OUTPUT + EVALUATION)

### 11.1 渲染品質
```
main.py test --save_val → 存出 val 影像
PSNR  = 10·log10( MAX^2 / MSE(I, I_gt) )      （像素均值，會被結構問題騙）
SSIM  = 結構相似度（亮度·對比·結構）          （抓 floater / 結構崩）
LPIPS = 深度特徵感知距離（越低越好）           （最接近人眼）
```
★ 我們 22-23 分但 SSIM 0.69 / LPIPS 0.40，CityGSV2 0.857 / 0.169 → **PSNR 騙人，視覺差距比 PSNR 顯示的更大**。

### 11.2 幾何品質（mesh）
```
gs2d_mesh_extraction.py：用 TSDF fusion（voxel_size=0.01, sdf_trunc=0.04）從深度圖融出 mesh
eval_tnt：與 GT 點雲算 Precision / Recall / F1（Tanks&Temples 式，含 crop volume 對齊）
```

**最終輸出**：渲染影像、PLY、mesh、PSNR/SSIM/LPIPS/F1 數字。

---

## 12. 完整流程一張圖（文字版）

```
影像+位姿+深度估計 (§1)
   │
   ▼ 分塊 N×M + 可見性過濾 (§2)   ← 可選：contract / reorient
   │
   ▼ 初始化 G_0 (§3)   ★模組：depth-init(α=0.99反投影) | coarse-init(全域預訓練)
   │
   ▼ 表徵 (§4)         ★模組：Gaussian2D(2D surfel) | 3DGS | Appearance；參數 sh_degree
   │
   ▼ ┌──────────── 每個 training step 迴圈 ───────────┐
     │ render (§5) ★renderer: Trim2DGS|Vanilla2DGS; depth_ratio, diable_trimming
     │   → I, D_surf, N, dist, trans_i
     │ loss (§6)  ★metric: CityGSV2 | MCMC-CityGSV2(+2 L1); final_factor 旋鈕
     │ backward + Adam (§7)
     │ density control (§8) ★★ grad-densify | MCMC | RTGStable
     │   → 增生/搬移/死亡 → G_t
     │ trim / importance-prune (§9)  → 砍低貢獻
     └────────────────────────────────────────────────┘
   │
   ▼ 合併 25 塊 (§10)   ⚠ Gaussian drift → 全域尺待修
   │
   ▼ 輸出 + 評估 (§11)   PSNR/SSIM/LPIPS/F1；可選 VecTree 壓縮
```

---

## 13. 我們的嘗試地圖 + 當前開放問題

### 13.1 試過的東西（按 pipeline 階段）
| 階段 | 嘗試 | 結果 |
|---|---|---|
| §3 init | depth-init vs coarse-init (task1) | init 次要(+0.4dB)，SSIM 還輸 |
| §8 density | **GAT-ADMM consensus**（原核心）| net-zero（+0.2 是多訓練買的），放棄 |
| §8 density | **MCMC 適配 2DGS** | 平手 baseline；省點但自限 ~1M |
| §8 density | **gradient-guided MCMC**（grad 主導 relocate）| 負結果，放棄 |
| §4 表徵 | **sh2 → sh3** | +0.295dB，720k 不 OOM，SH 是真槓桿 |
| §6 loss | **depth final_factor 0.001→0.05** | 大幅救畫面（+0.5dB，SSIM/LPIPS 升）|
| §7 訓練 | **30k → 60k 步** | +0.57dB（30k 是訓練不足）|
| §9 prune | **2DGS importance-prune** | 從飽和天花板無損砍 50%，贏低cap +0.46dB |
| §9 combo | **sh3 + prune50** | 360k/22.4，半點數贏 sh2 滿點天花板 |

### 13.2 三個被推翻/修正的舊結論（重要）
1. **「720k = 6GB 天花板」錯**：那是自設 cap_max，非硬體。6GB 對塊7 真實上限 ~4M（CityGSV2 在這機器上跑到 4M/28）。
2. **「414k 飽和 / 剪枝無損」是窄範圍假象**：只採樣 200k-800k，沒看 800k 以上。強 recipe 下品質還在線性爬（cap2M：532k=22.75→969k=23.24，~0.48dB/44萬點）。
3. **「CityGSV2 舊本機數字 28.7」作廢**：provenance 壞（config 指向別資料夾、日期混亂）。論文真值 = **27.23 全域**（CityGS V1=27.46，2DGS 原版只 21.35）。

### 13.3 當前的硬約束 / 物理事實
**這個 regime 裡品質 ∝ 點數，沒有免費午餐。** cap2M 證實線性爬，**28 需要 ~4-5M 點**。所以：
- 「直接訓練低點數模型」拿不到 28（點少品質就低）。
- 「低點數 28」唯一可能 = **train-high-then-prune**（長到 4M 拿 28，再 importance-prune 砍點，看留多少）。但從未飽和的 4M 剪會掉比較多（預期 4M→1M ≈ 25-26，非 28）。
- 真正「少 primitive 本質高品質」需換表徵：**anchor/neural（Scaffold-GS / SOGS）**，每 anchor 經 MLP 生成多高斯。大方向、高風險。

### 13.4 待決策的三條路（A/B/C）
- **(A) train-high-then-prune**：用現成剪枝牌，接受 ~25-26 @ 1M（省 75% 點、略掉品質）。需先跑 grad-densify 拿 4M 素材。
- **(B) 換 anchor/neural 表徵**：賭「少 primitive 高品質」，大改，可能是真論文貢獻。
- **(C) 先驗證 depth-init + grad-densify 真能到 28**（A 的前置 + 證明 6GB 能到 28）。config 已建：`configs/graddensify_depthinit_aerial.yaml`。

### 13.5 論文化未完成項
- 25-block 端到端 6GB + merge（先修 §10 的 drift 尺）。
- baseline（2DGS/3DGS/CityGSV2/DOGS）壓進 6GB + 統一 eval（解析度對齊 1600px）的乾淨對照。
- 目前 per-block(我們) vs 全域(論文 27.23) 不可直接比；per-block→全域映射未在健康 pipeline 上量過。

### 13.6 一句話現況
我們有兩張驗證過的牌（**importance-prune 省點不掉太多 + sh3 表徵槓桿**）和一個被修正的世界觀（**6GB 其實能裝 4M/到 28、品質 ∝ 點數**）。核心未解問題：**能否在「明顯少於 4M」的點數下逼近 28** —— 這決定我們是「效率前緣」論文（A）還是「新表徵」論文（B）。

---

## 14. 採用的 module / 函數 / 外掛的數學式

> 上面 §1-13 是「我們的 pipeline 流程」。這節補齊**每個外部函式庫 / 演算法零件 / 我們自寫元件的內部數學**。

### 14.A 外部光柵化器 (CUDA rasterizers)

#### 14.A.1 `diff_trim_surfel_rasterization`（Trim2DGS，我們主用 renderer 的後端）
2DGS 的 **ray-splat intersection**（精確交點，非 3DGS 的 EWA 近似）：
一個 surfel 由 splat-to-world 變換 H 定義（local (u,v,1) → world）：
```
H = [ s_u·t_u | s_v·t_v | 0 | mu ]   (4×4 齊次)
```
螢幕像素 x 反投影光線與盤的交點：用兩個正交平面 h_x, h_y（盤在螢幕空間的兩條齊次線），perspective-correct 求 local 座標：
```
u(x) = (h_u · x_h) / (h_w · x_h),   v(x) = (h_v · x_h) / (h_w · x_h)
```
其中 x_h 為像素齊次座標、h_* 由 H 與相機投影矩陣合成。再帶 G(u,v)=exp(-(u^2+v^2)/2)，並與固定螢幕低通 max（抗鋸齒）。
**輸出通道 allmap**（程式驗證）：`[0]=expected depth, [1]=alpha, [2:5]=normal, [5]=median depth, [6]=distortion`。

#### 14.A.2 `record_transmittance` 模式 + Trim 貢獻（CityGSV2 paper eq.3,4）
單視角 k 下，第 n 個 surfel 的貢獻（沿用 LightGaussian/Fan et al. 形式）：
```
C_{n,k} = (1/|P_k|) · sum_{p in P_k}  α_n(p) · prod_{j=1}^{n(p)-1} (1 - α_j(p))^(1-γ)
```
- `P_k` = surfel n 在視角 k 的 2D 投影覆蓋像素集
- `n(p)` = 像素 p 上 surfel n 的深度排序位置
- `γ = 0.5`（default）
跨指派視角集 V_m 平均：
```
C_n = (1/|V_m|) · sum_{k in V_m} C_{n,k}
```
Trim：用百分位閾值砍 `C_n <= quantile(C, ratio)`。**這個 C_n 就是 record_transmittance 回傳值的基礎，也是我們 importance-prune 的貢獻項。**

#### 14.A.3 `diff-gaussian-rasterization`（3DGS EWA，VanillaRenderer 後端，我們沒主用）
3D 高斯投影用 **EWA splatting**：世界協方差 Σ 投到螢幕：
```
Σ_2D = J W Σ W^T J^T   （取左上 2×2）
W = 視圖旋轉, J = 投影的仿射近似 Jacobian
α_i(x) = o_i · exp( -1/2 (x - μ_i')^T Σ_2D^-1 (x - μ_i') )
```
與 2DGS 的差別：EWA 是「投影近似」，2DGS 是「精確 ray-splat」。

### 14.B 優化器 / 排程器

#### 14.B.1 Adam（`internal/optimizers/Adam`）
```
g_t = ∂L/∂θ
m_t = β1·m_{t-1} + (1-β1)·g_t                  （一階動量）
v_t = β2·v_{t-1} + (1-β2)·g_t^2                 （二階動量）
m̂ = m_t/(1-β1^t),  v̂ = v_t/(1-β2^t)            （偏差校正）
θ_t = θ_{t-1} - lr · m̂ / (sqrt(v̂) + ε)
β1=0.9, β2=0.999, ε=1e-8
```
每個高斯屬性（mu/s/q/o/SH）各自一組 Adam state（→ 記憶體 = 參數 × 3：參數+m+v）。

#### 14.B.2 ExponentialDecayScheduler（`internal/schedulers`）
log 線性插值（3DGS 標準）：
```
w(t) = clamp(t/max_steps, 0, 1)
lr(t) = exp( (1-w)·ln(lr_init) + w·ln(lr_final) )
      = lr_init · (lr_final/lr_init)^w
```
means lr：6.4e-5 → 6.4e-7。

### 14.C 損失輔助函式

#### 14.C.1 SSIM / fused-ssim（`internal/utils/ssim`）
高斯窗（11×11, σ=1.5）逐窗計算後平均：
```
SSIM(x,y) = [ (2 μ_x μ_y + c1)(2 σ_xy + c2) ] / [ (μ_x^2 + μ_y^2 + c1)(σ_x^2 + σ_y^2 + c2) ]
μ = 窗內高斯加權均值, σ^2 = 加權變異, σ_xy = 加權協變異
c1 = (0.01·L)^2, c2 = (0.03·L)^2, L = 動態範圍(=1 for [0,1] 影像)
DSSIM loss = 1 - SSIM
```
fused-ssim = 同公式的 CUDA 融合版（快、省記憶體）。

### 14.D 採用 / 改編的外部方法（數學）

#### 14.D.1 LightGaussian（原版重要度，我們 §9.2 的來源）
原版 global significance（3DGS）：
```
GS_j = sum_{i=1}^{MN} 1(G_j, r_i) · σ_j · γ(Σ_j)
1(G_j,r_i) = 高斯 j 是否擊中第 i 條光線, σ_j = opacity, γ(Σ_j) = 正規化體積 = (det Σ_j)^(1/2)
V_imp_j = GS_j · (volume_j)^v_pow
```
**我們的 2DGS 改編**：`1·σ·體積` → `transmittance 貢獻 C_j · 面積 A_j^v_pow`（A=s_u·s_v），脫離 gsplat 3D 依賴（§9.2）。

#### 14.D.2 3DGS-MCMC（relocation 推導 + SDE）
把訓練看成從後驗取樣的 **SDE（朗之萬動力學）**：
```
dθ = ∇ log p(θ|I) dt + sqrt(2) dW    （W 為 Wiener 過程）
離散化 → 梯度下降 + 高斯噪聲（§8b 的 Langevin 項）
```
relocation 保持「N 個新高斯的合成 ≈ 原 1 個高斯」的渲染不變式 → 解出 §8b 的 o_new / coeff 公式（CUDA `utils.cu` 二項式展開驗證）。

#### 14.D.3 RTG-SLAM（depth-init + eta）
- 反投影 + α=0.99 + scale=d/f（§3a）。
- eta 信心：穩定計數器，`eta_i += 1` 每 interval，`eta_i >= 10 → 免 reset / 免 prune`（保護已收斂 surfel）。

#### 14.D.4 RaDe-GS（depth 正則設計）
2DGS 的 rasterized Gaussian depth：用 ray-splat 交點的 z 當每高斯深度（精確，非中心近似）→ 餵 §6.2 的 depth/normal 正則。

#### 14.D.5 Depth Anything V2（外掛，偽深度來源）
```
d_inv = DPT_head( DINOv2_ViT( I ) )
```
單目、輸出**相對逆深度（disparity-like）**，up-to-scale（故 §6.2 用逆深度 + 尺度對齊/正規化）。當弱幾何標籤，非絕對深度。黑箱網路，不參與我們的梯度。

#### 14.D.6 VecTree Quantization（部署壓縮，K-means VQ）
```
對 SH 向量集 {sh_i} 做 K-means → codebook {c_1..c_K}
每點：idx_i = argmin_k ||sh_i - c_k||,  重建 sh_i ≈ c_{idx_i}
儲存 = idx(log2 K bits) + codebook；重要 SH（高 V_imp）留 float16 不量化
```
壓 SH 維度（48→低），幾何屬性 float16。論文宣稱 2DGS 省 50% 儲存。

#### 14.D.7 gsplat 函式（外掛）
- `compute_relocation(o, s, N, binoms)`：§8b 公式的官方實作（要 3D scale → 我們 2D pad/truncate）。
- `gsplat.hit_pixel_count`：3DGS 版 LightGaussian 的 hit count（我們 2DGS 不走這條，改 record_transmittance）。

### 14.E 我們自己寫的元件（數學）

#### 14.E.1 DT-ADMM-GAT consensus（核心原創，已驗證 net-zero 放棄）
**節點特徵**（12 維 2DGS）：`h_v = [μ(3), normal(3), scale_2d(2), opacity(1), SH_dc(3)]`
**邊界圖**：相鄰 block 的 boundary surfel，雙向 KNN + 法線 cos filter（`internal/utils/boundary_graph.py`）。
**三步交替**（每 outer iteration）：
```
Step1 Primal（每 block 平行，z 凍結）：
   x_v <- argmin_x  L_render(x) + (ρ/2)·|| h(x) - z_v + y_v/ρ ||^2

Step2 z-update（GAT message passing）：
   e_vu = LeakyReLU( a^T [ W·h_v || W·h_u ] )        （注意力分數）
   α_vu = exp(e_vu) / sum_{u' in N(v)} exp(e_vu')      （softmax 正規化）
   z_v  = σ( sum_{u in N(v)} α_vu · W·h_u )            （加權聚合）
   訓練 z 的損失：L_z = (ρ/2)||x_v - z_v||^2 - y_v^T·z_v

Step3 Dual：
   y_v <- y_v + ρ·(x_v - z_v)
```
**動態拓撲對偶繼承**：`Clone → y'=y`；`Split → y1=y2=y/2`（嚴格耗散）；`Prune → 丟棄 y`。
**weighted-ADMM 預條件**（修 6 屬性尺度差 10^4 的病態）：`feat_std` 對角縮放 + geomean 正規化，三檔（primal penalty / dual / GAT y_norm）自洽。
**為何失敗**：depth-init 後相鄰 block boundary 幾何本就一致 → x-z 殘差趨 0 → ||y|| 建不起來 → AL 對 primal 無施力 → +0.2dB 是多訓練買的（§13.1）。

#### 14.E.2 gradient-guided MCMC relocation（消融，負結果）
標準 MCMC relocate 取樣 ∝ opacity。我們改 grad 主導：
```
g_i = ||xyz_gradient_accum_i|| / (mean_j ||grad_j|| + ε)     （正規化視空間梯度）
mode=grad_op:  w_i = g_i · (o_i + floor),  floor=0.1          （修 Gemini 低-opacity 餓死瑕疵）
relocate 取樣 ∝ w_i  → 死點回收導向高誤差(高梯度)區
```
需重加 `xyz_gradient_accum` buffer（MCMC 原本不維護），且 self-heal 對齊 count+device（扛 Trim/relocate/add 的拓撲變動）。結果：cap300 21.50 vs base 21.51 → 無效，放棄。

#### 14.E.3 importance_prune_2dgs（已驗證有效，§9.2）
`V_imp = C_i · A_i^v_pow`，C=record_transmittance top-K 跨視角均、A=s_u·s_v、v_pow=0.1。詳 §9.2 + §14.D.1。

#### 14.E.4 MCMC2DGS / MCMCCityGSV2Metrics（2D 適配，已驗證）
- relocation 2D pad/truncate（§8b）、noise 切平面化（§8b）、L1 免疫護城河（§6.3）。純工程適配，不改收斂理論。

---

## 15. 速查：每個式子 ↔ 程式位置

| 數學 | 程式 |
|---|---|
| 2DGS ray-splat + allmap | `renderers/sep_depth_trim_2dgs_renderer.py:forward` |
| 光度 + DGD 拆 extra_loss | `metrics/citygsv2_metrics.py:get_train_metrics` |
| 逆深度正則 + 權重排程 | `metrics/citygsv2_metrics.py:get_inverse_depth_metric / get_weight` |
| 法線/分佈損失 | `metrics/gs2d_metrics.py` |
| MCMC relocate/add/noise | `density_controllers/mcmc_density_controller.py` + `mcmc_2dgs_density_controller.py` |
| MCMC L1 + 免疫 | `metrics/mcmc_citygsv2_metrics.py` |
| grad-densify clone/split/reset | `density_controllers/citygsv2_density_controller.py` + `vanilla_density_controller.py` |
| importance prune | `utils/importance_prune_2dgs.py` + `gaussian_splatting.py:light_gaussian_prune` |
| depth-init 反投影 | `utils/depth_init_blocks.py` |
| 分塊 | `utils/citygs_partitioning_utils.py` |
| GAT-ADMM | `density_controllers/dt_admm_density_controller.py` + `renderers/dt_admm_gat_renderer.py` + `utils/dt_admm_coordinator.py` + `utils/boundary_graph.py` |
| 合併 | `utils/merge_citygs_ckpts.py` |
| 訓練主迴圈 | `gaussian_splatting.py:training_step / on_train_batch_end` |
```
（完）
```
