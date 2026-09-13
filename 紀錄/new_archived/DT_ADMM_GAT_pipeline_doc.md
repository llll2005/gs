> ⛔ **2026-09-13 重驗後仍不可用。**
> DT-ADMM-GAT（原始提案）的 pipeline 文件。該線已關；對偶數學活下來並用在 VRAM 約束上（v2 §1.2、§8）。
> 判準見 `../倖存清單_2026-09-13.md`；現行權威是 `../研究總覽_v2.md`。

# CityGaussian + 新初始化管線 技術文檔
> 供 Gemini 分析用。包含 CityGSV1/V2 完整 pipeline、2DGS 表示、RTG-SLAM 初始化機制、以及我們提出的 Depth-Prior Per-Block 初始化設計。

---

## 1. 基礎表示：3DGS 與 2DGS

### 1.1 3D Gaussian Splatting (3DGS)

場景以 N 個 3D 高斯橢球表示：G_N = {G_n | n=1,...,N}

每個 Gaussian 含可學習參數：
- 位置 mu_n in R^3
- 共變異矩陣 Sigma_n = R_n * S_n * S_n^T * R_n^T（R 為旋轉，S 為縮放對角矩陣）
- 不透明度 sigma_n in [0,1]
- 球諧係數 f_n in R^(3x16)（sh_degree=3 時，view-dependent color）

渲染（alpha blending，由前到後）：
  c_p = sum_{i in gamma(p)} c_i * alpha_i * prod_{j=1}^{i-1} (1 - alpha_j)
  alpha_i = sigma_i * exp(-1/2 * (x - mu_i)^T * Sigma_i^{-1} * (x - mu_i))

密度控制（Adaptive Density Control）：
- Clone：梯度大且 scale 小 → 複製
- Split：梯度大且 scale 大 → 分裂為兩個
- Prune：opacity < threshold → 移除
- 依據 view-space position gradient nabla_densify = dL/d_mu 觸發

### 1.2 2D Gaussian Splatting (2DGS)

將 3D 橢球「壓平」為 2D 定向圓盤（surfel），使 normal 方向有明確定義。

#### 參數化

每個 2D Gaussian 由以下參數定義：
- 中心 p_k in R^3
- 主切向量 t_u, t_v（在局部平面上）
- 縮放 S = (s_u, s_v)，控制橢圓半軸
- 旋轉矩陣 R = [t_u, t_v, t_w]（t_w = t_u x t_v 為 normal）
- 齊次變換矩陣 H in R^{4x4}：

  H = | s_u*t_u  s_v*t_v  0  p_k |
      | 0        0        0  1   |

局部 uv 座標系中的高斯值：
  G(u) = exp(-(u^2 + v^2) / 2)

世界座標點 P(u,v) = p_k + s_u * t_u * u + s_v * t_v * v

#### Ray-Splat Intersection（核心創新）

不同於 3DGS 用 affine projection（在邊緣不準確），2DGS 用解析射線-平面交點：

給定畫素 (x,y)，射線 r(u) = (R_g * K^{-1} * u_hat) * theta + t_g

交點由三平面聯立求解：
  h_u = (WH)^T * h_x,   h_v = (WH)^T * h_y

  u(x) = (h_u^1 * h_v^0 - h_v^1 * h_u^0) / (h_u^0 * h_v^1 - h_u^1 * h_v^0)
  v(x) = (h_u^0 * h_v^2 - h_v^0 * h_u^2) / (h_u^0 * h_v^1 - h_u^1 * h_v^0)

此方法提供 perspective-correct、multi-view consistent 的深度。

#### 正則化損失

深度失真損失（Depth Distortion Loss）：
  L_d = sum_{i,j} omega_i * omega_j * |z_i - z_j|
  omega_i = alpha_i * G_i(u(x)) * prod_{j=1}^{i-1} (1 - alpha_j)
目的：使沿 ray 的 splat 分布集中（防止半透明疊加）

法向一致性損失（Normal Consistency Loss）：
  L_n = sum_i omega_i * (1 - n_i^T * N)
  N = (nabla_x * p_s x nabla_y * p_s) / ||nabla_x * p_s x nabla_y * p_s||
目的：讓 surfel normal 與深度梯度推算的 normal 一致

總損失：L = L_c + alpha * L_d + beta * L_n
（alpha=1000 bounded scenes，alpha=100 unbounded，beta=0.05）

---

## 2. CityGaussianV1（ECCV 2024）

### 2.1 核心問題

- 3DGS 直接訓練大場景：24G RTX3090 在 Gaussian 數超過 1100 萬時 OOM
- 渲染瓶頸：MatrixCity 2.7km^2 場景 23M Gaussians → 只有 21 FPS

### 2.2 Pipeline（三段式）

#### 階段 1：全域粗訓練（Global Coarse Pretrain）

對全場景影像跑一遍標準 3DGS 訓練（SH degree 3），生成粗略全域 Gaussian prior。
- 輸入：COLMAP sparse + 所有影像
- 輸出：全域粗 Gaussian checkpoint

#### 階段 2：場景分塊（Scene Partitioning）

將場景等分為 N x M 個非重疊方塊（block）。

關鍵設計：
- contracted space（針對無界場景）：將遠處空間收縮到有界立方體
- 攝影機分配：攝影機中心在 block 內，或對該 block 有足夠 content contribution（content_threshold=0.08）

#### 階段 3：並行 Fine-tune + 合併

每個 block 獨立 fine-tune（可多 GPU 並行）：
- 初始化：從 coarse ckpt 中提取屬於該 block 的 Gaussian 子集
- coarse Gaussian 作為 prior 引導訓練（防止相鄰 block 訓練干擾）
- SH degree：先 pruning（SH3→SH2），再 distillation（額外 10K iter）

後處理：VecTree 量化壓縮，LoD 渲染（Level-of-Detail）

### 2.3 LoD 渲染策略

- 依 block 到相機距離，選擇不同精度的 Gaussian 版本
- 遠離相機的 block 使用壓縮版（更少 Gaussians）
- 達到 MatrixCity small_city 20M+ Gaussians 下仍能 real-time 渲染

---

## 3. CityGaussianV2（ICLR 2025）

### 3.1 與 V1 的關鍵差異

| 面向 | CityGSV1 | CityGSV2 |
|------|----------|----------|
| Gaussian 表示 | 3DGS（橢球） | 2DGS（surfel） |
| 幾何品質 | 中等 | 高（有 normal/depth 損失） |
| 後處理 | Pruning + Distillation | 無（SH2 from scratch + trimming） |
| Densification | 標準 ADC | DGD（SSIM-only gradient） |
| 數量爆炸問題 | 無特殊處理 | Elongation Filter |

### 3.2 優化機制（Sec 3.2）

#### DGD（Decomposed-Gradient-based Densification）

問題：2DGS 在大場景並行訓練早期容易出現 blurry reconstruction。
原因：L1 loss 梯度對 blurriness 不敏感，SSIM loss 梯度才是 densification 的關鍵信號。

解法：densification gradient 只使用 SSIM loss 部分，但用全局 gradient norm 做縮放對齊：

  nabla_densify = max(omega * |nabla_L_avg| / |nabla_L_SSIM_avg|, 1) * nabla_L_SSIM

其中 omega 為常數，確保 SSIM gradient 的尺度對齊原始 threshold。

#### Elongation Filter

問題：2DGS 的極度狹長 surfel 在平行訓練中造成 Gaussian 數量指數爆炸 → OOM。

原因：極度細長的 surfel 投影到螢幕小於 1 像素，無法通過 scaling/rotation gradient 自我修正，
      位置梯度累積過大，持續觸發 clone/split。

解法：densification 前先評估每個 surfel 的 elongation ratio：
  eta_n = min(s_n,u, s_n,v) / max(s_n,u, s_n,v)

  if eta_n < threshold: 不允許 clone/split（但允許 prune）

效果：穩定 Gaussian 數量曲線（見論文 Figure 3），避免 OOM，不影響渲染品質。

#### 深度監督（Depth Supervision）

使用 Depth Anything V2 預測的 inverse depth D_hat_k（已對齊至場景 scale）。

損失（L1 on inverse depth）：
  L_Depth = |D_hat_k - D_k|

訓練過程中 weight alpha 指數衰減（減小不精確深度預測的負面影響）。

#### 法向損失（Normal Loss）

  L_Normal = sum_i omega_i * (1 - n_i^T * N)

surfel normal n_i 需與深度圖梯度推算的表面 normal N 一致。

### 3.3 並行訓練管線（Sec 3.3）

改動重點（相較 CityGSV1）：
1. SH degree 從頭用 2（不需 SH3 → distillation）：SH feature 維度從 48 → 27，省記憶體 40%
2. 用 trimming 取代 post-pruning：contribution-based pruning 整合進 per-block fine-tune

Contribution（Gaussian 重要性）：
  C_{n,k} = (1/|P_k|) * sum_{p in P_k} (alpha_n)^gamma * prod_{j=1}^{n(p)-1} (1-alpha_j)^{(1-gamma)}
  C_n = (1/|V_m|) * sum_{k in V_m} C_{n,k}

gamma=0.5（預設），C_n 低於百分位閾值的 Gaussian 自動移除。

### 3.4 完整 Pipeline 圖

  [Posed Images + COLMAP Sparse]
            |
            v
  [Global Coarse Pretrain（2DGS + DGD + Elongation Filter + Depth Loss）]
            |
            v
  [Scene Partitioning（N x M blocks）]
            |
            v
  [Per-Block Parallel Fine-Tune（SH2, trimming, 2DGS losses）]
            |
            v
  [Merge + VecTree Compression]

---

## 4. RTG-SLAM 的初始化策略（SIGGRAPH 2024）

RTG-SLAM 解決的問題：RGBD 相機在線掃描場景，需要即時（16 FPS）初始化新 Gaussians。

### 4.1 Compact Gaussian 表示

兩類 Gaussian：
- 不透明（Opaque, alpha=0.99）：擬合表面幾何（depth rendering 用）
- 半透明（Transparent, alpha=0.1）：擬合殘差顏色（color refinement 用）

depth 渲染不用 alpha blending，而是將 opaque Gaussian 視為 **ellipsoid disc**，
直接計算 ray-plane intersection → single Gaussian 即可 fit 一個局部 surface region。

### 4.2 Gaussians Adding 觸發條件

每個 frame 計算三張 map（已有 Gaussian 渲染）：
- Color Error Map：|C_k(u) - C_hat_k(u)| > delta_c
- Depth Error Map：|D_k(u) - D_hat_k(u)| > delta_d
- Light Transmission Map：T_hat_k(u) > delta_T（尚未被任何 Gaussian 覆蓋）

Mask M_s（需新增 surface Gaussian）= 光穿透率大 OR 深度誤差大
Mask M_c（需新增 color Gaussian）= 顏色誤差大 且 u not in M_s

從 M_s 和 M_c 各 sample 5% pixels 新增 Gaussian（避免記憶體爆炸）。

### 4.3 Gaussian 初始化（從深度圖）

新增 Opaque Gaussian 的初始化：
- 位置 p_i：從相機 pose T_{g,k} + pixel depth D_k(u) 反投影到世界座標
  p = T_{g,k} * K^{-1} * [u; v; 1] * D_k(u)
- 法向量 n_i：從局部頂點 map V_k^l 的法向 map N_k^l 轉換到世界座標
  N_k^g = R_{g,k} * N_k^l
- 大小：初始化為足以覆蓋場景的 scale（少重疊）
- 顏色（SH DC）：從該 pixel 的 RGB 值初始化

### 4.4 Stable vs Unstable 管理

Stable Gaussian：confidence count eta_i > delta_eta
Unstable Gaussian：新加入或 poorly fit 的 Gaussian

優化時只更新 Unstable Gaussians → 大幅減少每 frame 的計算量

Loss = w_c * L_color + w_d * L_depth + w_reg * L_reg
L_color = |C_k - C_hat_k|,  L_depth = |D_k - D_hat_k|

---

## 5. 我們提出的新初始化方法：Depth-Prior Per-Block Initialization

### 5.1 動機

CityGSV1/V2 的瓶頸：全域粗訓練（Coarse Pretrain）是最大 OOM 來源。
- MatrixCity aerial 5621 張影像 × 全分辨率 = 無法放入 6GB VRAM
- 就算分批訓練，single-GPU 24G 也要 4+ 小時

RTG-SLAM 提供了啟發：**不需要全域訓練，可從深度圖直接初始化 Gaussian**。

### 5.2 設計目標

替換全域粗訓練，直接生成「足夠好的 per-block 初始點雲」：
- 輸入：DA2 estimated depth maps（已有） + COLMAP camera poses（已有） + block partition AABB
- 輸出：每個 block 的初始 Gaussian point cloud（替代從 coarse ckpt 提取 subset）
- 約束：6GB VRAM（無法同時處理所有 block）

### 5.3 Per-Block 深度反投影算法

#### Step 1：Block-Camera Assignment

沿用 CityGSV2 的攝影機分配策略：
  對每個 block B_ij，選出 camera pose set V_ij 滿足：
  - camera center in B_ij（核心相機），或
  - camera 對 B_ij 的 content contribution > content_threshold

#### Step 2：深度圖反投影（per assigned camera）

對每張 assigned camera k 的 estimated depth map D_hat_k：

1. 載入 DA2 npy（normalized inverse depth）並用 scale/offset 還原為絕對深度：
   D_k(u) = 1 / (D_hat_k_norm(u) * scale_k + offset_k)
   其中 scale_k, offset_k 來自 estimated_depth_scales.json

2. 從深度反投影到世界座標：
   p_world(u) = T_{g,k} * K^{-1} * [u; v; 1] * D_k(u)

3. 過濾：只保留 p_world 落在 B_ij 的 AABB 內的點

4. 計算 normal（RTG-SLAM 方式）：
   用相鄰像素的 3D 位置叉積：
   n(u) = (p(u+1,v) - p(u,v)) x (p(u,v+1) - p(u,v))
   n(u) = normalize(n(u))
   將 normal 轉換到世界座標

#### Step 3：Gaussian 屬性初始化

每個保留的反投影點初始化為一個 2DGS surfel：
- mu = p_world（位置）
- t_w = n（normal，作為 surfel 的法向量方向）
- t_u, t_v：從 n 計算正交基（Gram-Schmidt）
- s_u = s_v = 初始 scale（= 相鄰點平均距離的一半，或固定為 0.005 * scene_scale）
- opacity = 0.1（初始低，讓 SplAT 從頭學）
- SH_dc = RGB 值從對應像素初始化
- SH_rest = 0

#### Step 4：去重與稀疏化

多張 camera 的反投影點雲合併後會有重疊：
- 用 voxel grid downsample（voxel size = min_scale * 2）去除重複
- 或用 GaussianSpa 的 sparsification 策略（L0-norm based pruning）

#### Step 5：存成 PLY 檔供 fine-tune 載入

儲存為標準 3DGS PLY 格式（CityGSV2 的 `initialize_from` 介面直接吃）。

### 5.4 與 RTG-SLAM 的差異

| 面向 | RTG-SLAM | 我們的方法 |
|------|----------|-----------|
| 場景規模 | 單室內場景（<100m^2） | city-scale（1.5 km^2） |
| 輸入深度 | GT RGBD 深度（精確） | DA2 estimated depth（有噪聲） |
| 初始化時機 | online per-frame | offline batch per-block |
| Gaussian 類型 | 3DGS（opaque+transparent） | 2DGS（surfel） |
| 主要目的 | 即時重建 | 取代 coarse pretrain，節省 OOM |

### 5.5 與 CityGSV1/V2 Coarse Pretrain 的差異

| 面向 | Coarse Pretrain | Depth-Prior Init |
|------|----------------|-----------------|
| VRAM 需求 | 訓練全場景 → 24GB+ OOM | 只做前向反投影 → <1GB |
| 時間 | 4-8 小時 | <10 分鐘（CPU 可跑） |
| 輸出品質 | 訓練收斂的 prior Gaussian | 幾何對齊的初始點雲（opacity 低） |
| 全域一致性 | 有（訓練過一遍） | 無（各 block 獨立初始化） |
| 全域一致性替代方案 | N/A | DT-ADMM-GAT 的 consensus 機制 |

---

## 6. 完整新管線（DT-ADMM-GAT）

### 6.1 前處理（不變）

1. COLMAP sparse reconstruction（提供 camera poses 和 sparse 3D points）
2. Depth Anything V2 inference → estimated_depths/*.npy
3. get_depth_scales.py → estimated_depth_scales.json（scale/offset per image）

### 6.2 場景分塊（不變）

partition_citygs.py → 5x5 block AABB 定義 + camera assignment

### 6.3 新增：Per-Block Depth-Prior Initialization

對每個 block B_ij：
1. 讀取 assigned cameras V_ij
2. 逐 camera 做深度反投影 → 世界座標點雲
3. 過濾至 AABB 內 → surfel 屬性初始化
4. Voxel downsample → 稀疏點雲
5. 存為 block_ij_init.ply

### 6.4 DT-ADMM-GAT 訓練（核心貢獻）

替代原本的「per-block 獨立 fine-tune」：

三步交替更新（每 100 gradient steps 一個 ADMM 週期）：

Step 1 - Primal（per-block 並行，完全獨立）：
  x_v^{k+1} = GD[L_render(x_v) + (rho/2) * ||x_v - z_v^k + y_v^k/rho||^2]
  z_v^k 為常數（detached），無跨 block 梯度流

Step 2 - z-update（GAT，跨 block 訊息傳遞）：
  邊界 Gaussian 特徵 h_v = [mu(3), normal(3), scale_2d(2), opacity(1), SH_dc(3)]（共 12 維）
  
  GAT attention:
    alpha_vu = Softmax(LeakyReLU(a^T * [W*h_v || W*h_u]))  for u in N(v)
  
  z-update:
    z_v^{k+1} = sigma(sum_{u in N(v)} alpha_vu * W * h_u)
  
  GAT 訓練 loss（x_v, y_v 均為 detached）：
    L_z = (rho/2) * ||x_v^{k+1} - z_v||^2 - (y_v^k)^T * z_v

Step 3 - Dual Update：
  y_v^{k+1} = y_v^k + rho * (x_v^{k+1} - z_v^{k+1})

### 6.5 動態拓撲規則（Densification 期間）

- Clone：y'_v = y_v（能量翻倍，保持責任一致）
- Split：y_v1 = y_v/2, y_v2 = y_v/2（能量嚴格耗散至 1/2）
- Prune：丟棄 y_v，從圖中移除節點及邊

### 6.6 後處理（不變）

Merge blocks → VecTree compression

---

## 7. 參考論文摘要

| 論文 | 關鍵貢獻（對本專案的意義） |
|------|--------------------------|
| **3DGS** (Kerbl et al., 2023) | 基礎表示：Gaussian 參數、alpha blending、ADC |
| **2DGS** (Huang et al., 2024, SIGGRAPH) | Surfel 表示、ray-splat intersection、depth distortion loss、normal loss |
| **CityGaussianV1** (Liu et al., 2024, ECCV) | Divide-and-conquer：coarse pretrain → block partition → parallel fine-tune → LoD |
| **CityGaussianV2** (Liu et al., 2025, ICLR) | DGD、Elongation Filter、Depth Supervision、SH2 from scratch + trimming |
| **DOGS** (Chen & Lee, 2024, NeurIPS) | 競爭者：3DGS + ADMM 分散訓練，dynamic topology 靠啟發式，需 24GB+ |
| **RTG-SLAM** (Peng et al., 2024, SIGGRAPH) | 深度反投影初始化 Gaussian、opaque/transparent 二分表示、stable/unstable 管理 |
| **GS-Scale** (2025, ASPLOS) | Host offloading：非邊界 Gaussian 參數存 CPU RAM（46GB），VRAM 只跑計算 |
| **GaussianSpa** (2025, CVPR) | Optimizing-Sparsifying：L0-norm based 動態剪枝，控制 Gaussian 數量不降品質 |
| **GAT** (Velickovic et al., 2018, ICLR) | Graph Attention Network：attention coefficient alpha_vu = Softmax(LeakyReLU(a^T [Wh_i||Wh_j])) |
| **Online ADMM** (Suzuki, 2013, ICML) | 訓練過程建模為 Online Optimization，densification 為動態目標轉換 |
| **Dist. ADMM w/ Node Error** (IEEE TSP 2019) | 邊界節點數量變化可視為有界 Node Error → 共識估計誤差有界（收斂理論支撐） |
| **Learning to Warm-Start** (ICLR 2020) | ADMM 具備 Bounded Perturbation Resilience（Lyapunov 框架的理論起點） |

---

## 8. 開放問題（供 Gemini 分析）

1. DA2 estimated depth 的噪聲對反投影點雲品質的影響估計？
   （相較 COLMAP sparse points 噪聲更大，但密度高 5-10 倍）

2. Per-block 初始化缺乏全域一致性：只靠 DT-ADMM-GAT 的 consensus 能否補足？
   若 block 初始化太差，ADMM 的收斂是否有保障？

3. 初始 scale 的設定策略：固定值 vs. 根據相機距離自適應 vs. 根據相鄰點距離？

4. Voxel downsample 的 voxel size 如何選：太大丟失幾何細節，太小點雲過密耗記憶體？

5. 與 GaussianSpa 的整合點：Sparsification 應在初始化後（整理點雲）還是訓練過程中動態執行？
