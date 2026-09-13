> ⛔ **2026-09-13 重驗後仍不可用。**
> DT-ADMM-GAT 的邊界圖規格。同上。
> 判準見 `../倖存清單_2026-09-13.md`；現行權威是 `../研究總覽_v2.md`。

# C3 規格文件：boundary_graph.py
> 供 Gemini 評估可行性、修正潛在錯誤、並對開放問題提供建議。

---

## 0. 背景與依賴鏈

### 0.1 DT-ADMM-GAT 三步交替中 C3 的位置

```
Step 1 Primal（per-block 並行，當前已實作）
  x_v^{k+1} = GD[L_render(x_v) + (rho/2) * ||x_v - z_v^k + y_v^k/rho||^2]

Step 2 z-update（C3 + C4，跨 block 訊息傳遞）← C3 建圖，C4 做 GAT
  h_v = [mu(3), normal(3), scale_2d(2), opacity(1), SH_dc(3)]  # 12 維節點特徵
  alpha_vu = Softmax(LeakyReLU(a^T * [W*h_v || W*h_u]))
  z_v^{k+1} = sigma(sum_{u in N(v)} alpha_vu * W * h_u)

Step 3 Dual（C5）
  y_v^{k+1} = y_v^k + rho * (x_v^{k+1} - z_v^{k+1})
```

**C3 的唯一職責**：為 Step 2 維護圖 G=(V,E)，其中 V 為各 block 的邊界 Gaussians，E 為跨 block 的空間鄰近關係。

### 0.2 依賴關係

```
C1（depth-init PLY）── 產生訓練用初始點雲
C3（boundary_graph）─┬─ 依賴 partition config（AABB per block）
                     ├─ 依賴各 block checkpoint（讀取 Gaussian 位置）
                     └─ 輸出 edge_index 給 C4（GAT）
C4（GAT z-update）── 依賴 C3 的圖結構
C5（ADMM 訓練迴圈）─ 依賴 C3 + C4 共同完成 z-update
```

### 0.3 當前架構的根本衝突（最大工程障礙）

**現況**：`train_citygs_partitions.py` 為每個 block 啟動獨立的 `main.py fit` subprocess，process 之間完全隔離。每個 block 的 Gaussian 只存在自己的進程記憶體中。

**ADMM z-update 的需求**：block i 的 z-update 需要讀取相鄰 block j 的 Gaussian 特徵 h_u。這需要跨 block 的資料存取。

**解法選項**（見第 5 節）。

---

## 1. 模組介面設計

### 1.1 輸入

```python
BoundaryGraph(
    block_aabbs: dict[int, tuple[np.ndarray, np.ndarray]],
    # block_id -> (min_xyz shape=(3,), max_xyz shape=(3,))
    
    grid_dim: tuple[int, int] = (5, 5),
    # 5x5 grid，row-major ordering（block_7 = row 1, col 2）
    
    d_boundary: float = 0.15,
    # COLMAP units：Gaussian 到 block 邊緣的距離閾值，視為邊界 Gaussian
    # 0.15 unit ≈ 7.5m 真實距離（1 unit ≈ 50m）
    
    k_neighbors: int = 8,
    # 每個邊界 Gaussian 在相鄰 block 中找的 KNN 數量
    
    connectivity: str = "4",
    # "4"：上下左右；"8"：含對角。aerial 場景推薦 "4"
)
```

### 1.2 輸出（PyTorch Geometric 相容格式）

```python
graph_data = boundary_graph.build(block_gaussians: dict[int, torch.Tensor])
# block_gaussians: block_id -> Gaussian 參數 tensor，shape (N_i, 12)
#                  欄位順序：[mu(3), normal(3), scale_2d(2), opacity(1), SH_dc(3)]

# 回傳 dict:
{
    "edge_index": torch.Tensor,   # shape (2, E)，[source, target]，跨 block 邊
    "node_features": torch.Tensor, # shape (N_boundary_total, 12)，所有邊界 Gaussian 特徵
    "node_block_id": torch.Tensor, # shape (N_boundary_total,)，各節點所屬 block
    "node_local_idx": torch.Tensor,# shape (N_boundary_total,)，在原 block 內的 Gaussian index
    "block_offsets": dict[int, int],# block_id -> 在 node_features 中的起始偏移量
}
```

---

## 2. 演算法設計

### 2.1 Block 鄰接計算

5×5 grid 的 block 鄰接（4-connectivity）：

```python
def compute_adjacency(grid_dim=(5, 5)):
    rows, cols = grid_dim
    adjacency = {}  # block_id -> list of adjacent block_ids
    for r in range(rows):
        for c in range(cols):
            bid = r * cols + c
            neighbors = []
            if r > 0:       neighbors.append((r-1)*cols + c)   # 上
            if r < rows-1:  neighbors.append((r+1)*cols + c)   # 下
            if c > 0:       neighbors.append(r*cols + (c-1))   # 左
            if c < cols-1:  neighbors.append(r*cols + (c+1))   # 右
            adjacency[bid] = neighbors
    return adjacency
```

### 2.2 邊界 Gaussian 識別

對 block i 的每個 Gaussian，判斷它到任一相鄰 block 方向的 AABB 面的距離是否 < d_boundary：

```
對於 block i 的 AABB = [min_i, max_i]（shape=(3,)）

Gaussian 在 x 正方向（靠近 block i 右邊界）：
  dist_right = max_i[0] - mu[0]
  is_boundary_right = (dist_right < d_boundary) AND (block i 右側有相鄰 block)

以此類推四個方向（±x, ±y）。aerial 場景 z 方向不做邊界（相機高度一致）。

is_boundary_i = is_boundary_right OR is_boundary_left OR is_boundary_up OR is_boundary_down
```

關鍵問題：d_boundary 的選擇決定邊界節點數量。

- 太大（> 0.3 unit ≈ 15m）：邊界節點佔全 block 40%+，GAT 計算成本倍增
- 太小（< 0.05 unit ≈ 2.5m）：物理上影響相鄰 block 渲染的 Gaussian 被遺漏
- 建議初始值：0.1 unit（≈ 5m），依實測邊界節點比例調整目標 5-10%

### 2.3 跨 Block KNN：Morton Code 輔助

**為何用 Morton Code**：
KNN 是 C3 最貴的操作（O(N_boundary^2) 暴力搜尋太慢）。Morton code 將 3D 座標映射到 1D，保留空間局部性（相鄰 Morton code 的點在空間上也接近），允許快速過濾候選點。

**Morton Code 計算**（整數位元交錯）：

```python
def xyz_to_morton(xyz: np.ndarray, n_bits: int = 21) -> np.ndarray:
    """
    xyz: (N, 3)，已歸一化到 [0, 2^n_bits) 的整數
    回傳: (N,) int64 Morton codes
    """
    def expand_bits(v):
        # 3D Morton：位元交錯 x=bit0,3,6...  y=bit1,4,7...  z=bit2,5,8...
        v = (v | (v << 32)) & 0x1f00000000ffff
        v = (v | (v << 16)) & 0x1f0000ff0000ff
        v = (v | (v <<  8)) & 0x100f00f00f00f00f
        v = (v | (v <<  4)) & 0x10c30c30c30c30c3
        v = (v | (v <<  2)) & 0x1249249249249249
        return v
    ix = xyz[:, 0].astype(np.int64)
    iy = xyz[:, 1].astype(np.int64)
    iz = xyz[:, 2].astype(np.int64)
    return expand_bits(ix) | (expand_bits(iy) << 1) | (expand_bits(iz) << 2)
```

**KNN 搜尋流程**：

```
1. 收集 block i 的邊界 Gaussian 集合 B_i（由 2.2 識別）
2. 收集相鄰 block j 的邊界 Gaussian 集合 B_j
3. 將 B_i ∪ B_j 按 Morton code 排序
4. 對 B_i 中每個 Gaussian v：
   a. 在排好序的 array 中找到 v 的位置
   b. 取前後 W 個候選點（W = max(k_neighbors * 8, 64)）
   c. 計算這些候選點到 v 的歐式距離
   d. 取距離最小的 k_neighbors 個（且必須來自 B_j）→ 作為 v 的邊
5. 建立 edge_index
```

注意：Morton code 排序後的滑動窗口只是過濾候選點，最終 KNN 還是需要精確歐式距離。  
比暴力搜尋快的原因：從 O(|B_i| × |B_j|) 降為 O(|B_i| × W)，W << |B_j|。

**替代方案（更簡單）**：直接用 `scipy.spatial.KDTree` 或 FAISS。
Morton code 的優勢在 GPU 上並行化時更明顯；CPU 端 KDTree 實作更直接，且 `scipy.spatial.cKDTree.query()` 有 C++ 加速。

建議：初期用 scipy KDTree，後期如有 GPU 加速需求再改 Morton。

### 2.4 邊（Edge）過濾

並非所有 KNN 結果都應作為圖邊：

1. **距離過濾**：||mu_v - mu_u|| > d_max_edge（建議 d_boundary * 3）時捨棄，避免物理上不相關的連接
2. **法向角過濾**：n_v · n_u < cos(60°)，即法向量夾角 > 60° 時捨棄（表示兩個 surfel 朝向完全不同，不應交換幾何資訊）
3. 過濾後每個節點至少保留 1 條邊（若無鄰居則加入距離最近的邊）

---

## 3. 訓練迴圈整合（架構轉換）

### 3.1 ADMM Outer Loop 的兩種實作方案

**方案 A：離線圖（Offline Graph，較易實作）**

```
ADMM 外迴圈 k = 0, 1, 2, ...:
  Phase 1（K_inner 步 primal 更新，每 block 獨立 subprocess）:
    for block i in all_blocks（可並行）:
      train_block_K_steps(i, z_i=z_prev[i], y_i=y_prev[i])
      save_checkpoint(i)
  
  Phase 2（建圖，CPU only）:
    load all checkpoints → extract boundary Gaussians
    boundary_graph = build_boundary_graph(all_blocks_gaussians)
    save graph to disk
  
  Phase 3（z-update，GAT forward pass）:
    load graph → GAT(h_v, edge_index) → z_new
    save z_new to disk
  
  Phase 4（dual update）:
    y_new[i] = y_prev[i] + rho * (x_current[i] - z_new[i])
    save y_new
```

優點：不需要修改訓練主迴圈（`gaussian_splatting.py`），各 block 仍獨立跑 subprocess。  
缺點：每個 ADMM outer iteration 需要額外的 IO（checkpoint 讀寫）；z,y 在 Phase 1 是過時的（stale）。

**方案 B：線上圖（Online Graph，需要架構改造）**

將所有 block 的 GaussianSplatting 模型同時載入記憶體（利用 46GB RAM + host offloading）：

```python
# admm_coordinator.py
models = {i: load_model(i) for i in range(25)}  # 全部載入 CPU RAM
for step in range(max_steps):
    # Primal：每個 block 在 GPU 上做一步 gradient descent（host offloading 保證 VRAM 夠）
    for i in all_blocks:
        offload_to_cpu(models[i])
    for i in all_blocks:  # 可用 thread pool 並行化
        offload_to_gpu(models[i])
        primal_step(models[i], z[i], y[i])
        offload_to_cpu(models[i])
    
    # z-update：每 ADMM_interval 步做一次
    if step % admm_interval == 0:
        boundary_graph = build_boundary_graph(models)
        z = gat_z_update(boundary_graph)
        y = {i: y[i] + rho * (x[i] - z[i]) for i in all_blocks}
```

優點：z,y 始終是最新的，ADMM 收斂性更好。  
缺點：需要重寫訓練迴圈，工程量大；host offloading（C6）是前置依賴。

**建議先實作方案 A**，等整體 pipeline 驗證後再遷移到方案 B（C6 實作後）。

### 3.2 方案 A 的 K_inner 選擇

K_inner = ADMM 每個外迴圈的 primal gradient steps 數。

- K_inner 太小：頻繁建圖，IO 開銷大
- K_inner 太大：z,y 嚴重過時，ADMM 收斂慢或不收斂

參考 DOGS（NeurIPS 2024）：K_inner ≈ 500 steps（約等於一個 densification_interval）。

這讓 z-update 在每次 densification 後進行，天然對齊 Gaussian 數量的拓撲變化節點。

---

## 4. 動態拓撲處理（Densification 期間的圖更新）

每次 densification（clone/split/prune）後，Gaussian 的數量和位置都改變，圖需要更新。

### 4.1 完整重建 vs 增量更新

- **完整重建**（方案 A 天然做到）：Phase 2 每次從 checkpoint 重新建圖，自動反映最新拓撲。代價是 O(N_boundary * K) 的 KNN 計算。
- **增量更新**（方案 B 需要）：只更新被 clone/split/prune 影響的邊界 Gaussian 的邊。需要追蹤 densification 的增量操作，複雜度高。

建議：方案 A 使用完整重建，因為 densification 後邊界 Gaussian 集合可能大幅變化。

### 4.2 圖更新與 ADMM 拓撲繼承的銜接

C5（ADMM）中規定：
- Clone → y'_v = y_v（子節點繼承父節點的對偶變數）
- Split → y_v1 = y_v / 2, y_v2 = y_v / 2
- Prune → 從圖移除節點和對應的 y

boundary_graph 需要追蹤每個邊界 Gaussian 的「身份」（原 block 內的 local index），讓 C5 在重建圖後正確對齊新舊 y 值。

**關鍵介面**：`node_local_idx`（輸出中的第四項）就是這個 tracking 機制。  
C5 在 densification 後，透過 local_idx 追蹤哪些節點是 clone/split 新增的，哪些是 prune 移除的。

---

## 5. 與 C4（GAT z-update）的介面

C3 的輸出直接作為 C4 的輸入。C4 需要的資料：

```python
# C4 (GAT) 期望的輸入
node_features: Tensor  # (N_boundary, 12)，C3 提供
edge_index:    Tensor  # (2, E)，C3 提供
edge_attr:     Tensor  # (E, d)，可選，C3 可提供距離或相對位置作為初始 edge feature
```

GAT forward pass（C4 的職責，不在 C3 內）：
```python
z_boundary = gat(node_features, edge_index)  # output: (N_boundary, 12)
```

C3 需要提供足夠資訊讓 C4 把 z_boundary 正確寫回各 block 的 Gaussian 參數。這就是 `node_block_id` 和 `node_local_idx` 的用途。

---

## 6. 已知問題與風險

### P1：邊界 Gaussian 比例問題

若 d_boundary 設定不當，可能出現：
- **過多邊界節點**：block_3（2.4M Gaussians）在 d_boundary=0.15 時可能有 30-40% 的 Gaussians 被識別為邊界，GAT 的計算量與 N_boundary^2 成正比，可能超過 VRAM 限制
- **過少邊界節點**：邊界模糊問題仍存在，ADMM consensus 無效果

需要監控的指標：`N_boundary / N_total`，目標維持在 5%~15%。

### P2：邊界 Gaussian 的 z-update 語義

GAT 的 z_v 是對 N(v) 的加權平均（attention sum），物理語義為「相鄰 block 認為 v 應該長什麼樣」。但如果邊界兩側的幾何差異太大（e.g., block_6 右側是建築，block_7 左側是道路），z_v 的加權平均可能是一個無意義的混合。

可能的緩解：使用 edge_attr（e.g., 法向夾角、距離）讓 GAT 學會在差異大時降低 attention weight（即 alpha_vu 自動趨近 0）。

### P3：初始的 z, y 值設定

ADMM 第一個 outer iteration 開始時，z 和 y 需要初始值。

- z_v 初始值：等於初始的 x_v（z = x at start，Lagrangian penalty 為 0）
- y_v 初始值：0（無初始 constraint violation）

這個初始化讓第一輪 primal step 等同於純粹的 L_render 訓練，ADMM penalty 逐步引入。

### P4：場景 AABB 與 COLMAP 座標系

從 `partition_from_colmap.py` 的分塊結果讀取 AABB 時，需要確認單位與 Gaussian 位置的座標系一致（均為 COLMAP units）。MatrixCity aerial 中 1 unit ≈ 50m，block AABB 每格約 0.7 unit × 0.7 unit ≈ 35m × 35m。

---

## 7. 建議實作順序

```
C3.1：BoundaryGraph 類別骨架（block 鄰接計算 + AABB 讀取）
C3.2：邊界 Gaussian 識別（d_boundary mask）+ 邊界節點數量監控
C3.3：跨 block KNN（先用 scipy KDTree，後優化為 Morton code）
C3.4：edge 過濾（距離 + 法向角）+ PyG 格式輸出
C3.5：方案 A 離線座標器（admm_coordinator.py，K_inner=500，接 Phase 2/3/4）
```

C3.5 是 C4 和 C5 得以跑起來的前提，需要優先完成到可以 end-to-end 跑通（即使 GAT 只是 identity function）。

---

## 8. 開放問題（供 Gemini 評估）

**Q1：Morton Code vs KDTree**  
在 CPU 端，scipy `cKDTree` 對 N~10K 邊界節點的 KNN 需要多少時間？Morton code 在 numpy 實作下的優勢是否顯著，還是只有在 GPU 並行化時才值得？

**Q2：d_boundary 的自適應設定**  
能否根據 block AABB 的大小自動設定 d_boundary？例如 `d_boundary = 0.1 * min(dx, dy)`（block 最短邊的 10%）？在 5×5 不均勻 block 大小下，固定值還是自適應值更合適？

**Q3：GAT 的 z_v 語義與 ADMM 收斂**  
ADMM 的 z_v 在文獻中通常是「共識點」（consensus variable），代表所有參與方的折衷。但 GAT 的輸出是一個可學習的非線性映射，不是嚴格意義上的取平均。這是否違反 ADMM 的理論收斂條件？若 z-update 不是凸優化的近端算子，是否仍有收斂保障？

**Q4：方案 A 的 IO 瓶頸**  
每個 ADMM outer iteration 需要讀取所有 25 block 的 checkpoint（每個 ~300MB × 25 = 7.5GB）。在 SATA SSD 上（500 MB/s），這需要約 15 秒。K_inner=500 步在 RTX 4050 上約需 3-5 分鐘。IO 佔比約 5-8%，可接受嗎？是否有更輕量的 IPC 替代方案？

**Q5：Stale z,y 對收斂的影響**  
方案 A 中，Phase 1 的 K_inner 步 primal 更新使用的是上一個外迴圈的 z^{k-1}, y^{k-1}（stale）。當 K_inner 很大時（500 步），primal 可能已大幅偏離舊的 z 值，導致 penalty 項 (rho/2)||x - z^{k-1}||^2 失去約束作用。文獻中「inexact ADMM」對 K_inner 的建議上限是多少？

**Q6：z-update 的 Loss 設計**  
當前規格中 z-update 的 loss 為：  
`L_z = (rho/2) * ||x_v^{k+1} - z_v||^2 - (y_v^k)^T * z_v`

這等價於最小化 augmented Lagrangian 對 z 的凸二次子問題（若不使用 GAT 非線性）。引入 GAT 後，`L_z` 應如何修改以保證 GAT 參數的梯度流正確？特別是 y_v^T * z_v 項的 `-` 號，對 GAT 參數更新的影響？

---

*本文件由 Claude Code 生成，供 Gemini 評估。實作前請針對 Q3, Q6 的收斂理論問題提供明確建議。*

---

## 9. Gemini 回覆（Round 1，Q1-Q3）

### A1：Morton Code vs KDTree
結論：CPU 端一律用 `scipy.spatial.cKDTree`。Morton code 在 pure Python/NumPy 的 Z-order 曲線上有「空間突變斷層」問題，迴圈開銷吃掉排序優勢，比 cKDTree 慢。Morton code 只有在自訂 CUDA Kernel 場合才有顯著優勢。  
**採納：是。C3.3 直接用 cKDTree，不實作 Morton code。**

### A2：d_boundary 的設定
結論：不要用 `0.1 * min(dx, dy)` 相對值（非對稱性陷阱：Block A 算出 5m，Block B 算出 0.5m → 圖變成單向 → ADMM 崩潰）。應使用固定全域物理距離或「最大 Gaussian 半徑 × N 倍」動態版。

⚠️ **規格修正（Gemini 給了 2.0 COLMAP units，但此值有誤）**：
Gemini 建議「d_boundary = 2.0 COLMAP units」，但本場景 1 unit ≈ 50m，block AABB 每格僅 ≈ 0.7 unit × 0.7 unit（≈ 35m × 35m）。d_boundary = 2.0 unit ≈ 100m 遠超 block 本身大小，會把整個 block 都標記為邊界。

正確值估計：
- 目標邊界帶寬 ≈ 5~10m 真實距離 → 0.1~0.2 COLMAP units
- 建議初始值：`d_boundary = 0.1`（≈ 5m），監控 N_boundary/N_total 在 5~15%

**採納 Gemini 的對稱性原則，但修正數值為 0.1 unit。**

### A3：GAT 的 z_v 語義與 ADMM 收斂
結論：屬於「Plug-and-Play ADMM / Unrolled Optimization」範式，確實破壞傳統凸優化收斂保證，但可透過以下結構限制保穩定：
1. **凸組合**：Softmax 保證 sum(alpha) = 1，z_v 永遠是鄰居特徵的加權平均 → 保留「共識」物理語義
2. **殘差連接**：`z_v = x_v + GraphNet(x_v, neighbors)`，初期 GraphNet 權重極小 → 系統先以標準 ADMM 方式收斂，中後期才讓 GAT 調整邊界融合
3. **對偶變數 y_v 牽制**：y_v 追蹤 x_v 與 z_v 的誤差，rho 設定得當時可強力拉扯本地 Gaussians 迎合 GAT 給出的共識

**採納：殘差連接設計加入 C4 規格。Q3 理論問題仍有缺口（見下方 Q6 延伸）。**

---

## 10. 待補充的背景資訊（Round 2 給 Gemini）

以下是 Gemini Round 1 未獲得的關鍵背景，Round 2 提問前需一併提供：

| 資訊項目 | 值 |
|---------|-----|
| 硬體 | RTX 4050 Laptop，6GB VRAM，46GB RAM，SATA SSD |
| 訓練架構 | 當前：per-block 獨立 subprocess（sequential），方案 A（離線圖）為目標 |
| COLMAP 場景尺度 | 1 unit ≈ 50m；block AABB ≈ 0.7 × 0.7 unit；camera z ≈ 2 unit = 100m 高度 |
| 當前 Gaussian 數量 | 平均 378K/block（immune eta 後），block_3 最多 2.4M |
| 訓練步數 | 30,000 steps/block；densification interval 500 steps → K_inner 候選值 500 |
| 實作進度 | C1（depth-init）完成；RTGStableDensityController 完成；C3-C5 未開始 |
| ADMM 公式 | 見 `紀錄/DT_ADMM_GAT_pipeline_doc.md` Section 6.4 |

---

## 11.5 Gemini 回覆（Round 2，Q4-Q6）

### A4：IO 與 Partial Checkpoint
15 秒 IO 可接受（K_inner=500 約 3-5 分鐘，佔比 5-8%）。  
Partial Checkpoint 可行，**關鍵風險**：Densification 改變 Gaussian 總數與索引順序，需同時儲存每個邊界 Gaussian 的 `unique_id`（在 C5 density controller 中透過拓撲繼承規則維護），否則 y_v 會對應到錯誤的 Gaussian。  
**採納**：境界 Gaussian 只 dump 位置+特徵，y_v 儲存在 density controller buffer（隨 ckpt 自動存取）。

### A5：Stale z,y（Inexact ADMM）
Inexact ADMM 收斂條件：primal 更新誤差絕對可和（sum ε^k < ∞）。在非凸 3DGS 下此條件無法保證。  
實用 Heuristic：**Warm-start**——z-update 後的前 50 步將 rho 暫時加倍，強迫 x 快速修正對齊新 z，再恢復正常 rho 繼續 L_render 微調。  
**採納**：在 C5 ADMM coordinator 中加入 rho warm-start（50 steps, 2× rho）。

### A6：L_z 梯度流（完整推導）

**(a) 梯度鏈式法則**  
∂L_v/∂z_v = ρ(z_v - x_v) - y_v = ρ(z_v - z*_v)  
其中 z*_v = x_v + y_v/ρ 為「當前疊代的理想共識目標」。  
GAT 訓練等價於動態監督學習：讓 z_v → z*_v，而 z*_v 隨 y_v 每次 dual update 後移動。  
**梯度方向正確**。

**(b) y_v 的物理角色**  
y_v 儲存歷史不一致性（momentum）。當 x_v > z_v 時 y_v 持續增大，使 z*_v 往 x_v 方向移動，最終 GAT 被引導到讓 z_v 貼近 x_v（共識）。  
L_z 不需要加入鄰居接近項（`mu * sum||z_v - x_u||^2`）。鄰居資訊透過**架構**（GAT 以鄰居為輸入）進入，y_v 的累積機制確保跨塊共識。

**(c) Attention 梯度**  
∂L_z/∂α_vu = ρ(z_v - z*_v)^T (W h_u)  
若鄰居 u 的特徵方向與誤差方向相反，α_vu 梯度為負 → 下一步更新時 α_vu 增大。GAT 自動學習「信任哪個鄰居」。

**(d) 穩定性條件**  
z_v 為鄰居特徵凸組合（Softmax 保證 sum α = 1）。對 W 施加 Spectral Normalization（奇異值 ≤ 1）可確保 z_v 落在鄰居特徵凸包內 → 不發散。  
**C4 實作需加入 `spectral_norm` 到 W。**

### 殘差連接 Cold-Start 問題（補充分析）
Gemini A3 建議 `z_v = x_v + GraphNet(neighbors)`。初期 y_v=0 → z*_v = x_v → L_z 驅使 GraphNet(neighbors) → 0（GAT 初期輸出接近零）。隨 y_v 累積，z*_v 偏移，GAT 逐步學習輸出非零共識修正。**Cold-start 由 L_z 自然處理，無額外設計需要。**

---

## 11. 開放問題（Round 2，Q4-Q6）

以下問題 Gemini Round 1 未回答，需要數學推導驗證。

### Q4：方案 A 的 IO 與 IPC 方案
**背景**：25 block × 300MB/checkpoint = 7.5GB。SATA SSD（≈500 MB/s）讀完需 15 秒。  
K_inner=500 步在 RTX 4050 上 ≈ 3-5 分鐘（block fine-tune 本身）。IO 佔比 5-8%。

**問題**：
- IO 15 秒是否可接受？或建議用 mmap / shared memory 替代 checkpoint 讀寫？
- 若改為只 dump 邊界 Gaussian（N_boundary ≈ 20K × 12 float32 × 4 bytes = 960KB），IO 降至 < 1 秒。這種「partial checkpoint」策略是否有已知陷阱？

### Q5：Stale z,y 對收斂的影響（inexact ADMM）
**問題**：
- K_inner=500 時，primal 可能已偏離舊 z 高達 rho×500 倍的更新量。Inexact ADMM 文獻（e.g., Eckstein & Yao 2018）對允許的 inexactness 有什麼形式的上界？
- 在 3DGS 的非凸 L_render 下，inexact ADMM 的理論是否仍適用？若不適用，是否有實作上的 heuristic（如動態縮小 K_inner 或增大 rho）可替代？

### Q6：L_z 的梯度流設計（最重要，請提供推導）
**當前 L_z 設計**：
```
L_z = (rho/2) * ||x_v - z_v||^2 - y_v^T * z_v
z_v = GAT_θ(h_v, {h_u : u in N(v)})
```

**需要 Gemini 推導的問題**：

(a) 對 GAT 參數 θ 的梯度：
```
∂L_z/∂z_v = rho * (z_v - x_v) - y_v
∂L_z/∂θ = (∂z_v/∂θ)^T * [rho*(z_v - x_v) - y_v]
```
初期 y_v ≈ 0 時，梯度 ≈ rho*(z_v - x_v)*(∂z_v/∂θ)，這告訴 GAT「讓輸出（基於鄰居資訊）貼近本 block 的 x_v」。這個梯度方向是否正確？

(b) 純 ADMM 的 z 解析解：
```
z*_v = argmin_z L_z = x_v + y_v/rho
```
此解不含鄰居資訊。GAT 引入鄰居資訊的額外優化目標去哪了？是否需要在 L_z 加入「鄰居接近項」？
例如：`L_z += (mu/2) * sum_{u in N(v)} ||z_v - x_u||^2`

(c) 殘差連接 `z_v = x_v + delta_v`（Gemini A3 建議）與 L_z 的相容性：
若 delta_v 初始為零，則 z_v^{0} = x_v，y-update 時 y = y + rho*(x-z) = y + 0 → y 不累積。
如何讓系統在殘差連接下正確啟動 ADMM 的 dual 積累過程？

(d) `-y_v^T * z_v` 的符號：
此項的梯度為 -y_v，即告訴 GAT「z_v 應往 y_v 方向移動」。
但 y-update 是 y = y + rho*(x-z)，若 z > x 則 y 減小，若 z < x 則 y 增大。
請驗證：在這個符號約定下，整個 (z,y) 的更新循環是否構成 Lagrangian 的 saddle-point 搜尋（z 最小化，y 最大化）？

*Round 2 提問：請同時提供上述四個子問題的推導，並指出其中哪個子問題需要修改 L_z 的設計。*
