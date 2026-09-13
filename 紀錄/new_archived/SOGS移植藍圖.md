> ⛔ **2026-09-13 重驗後仍不可用。**
> SOGS 移植藍圖；SB/lobe 軸已關閉（v2 §8）。
> 判準見 `../倖存清單_2026-09-13.md`；現行權威是 `../研究總覽_v2.md`。

# SOGS / Scaffold-GS 移植藍圖（2026-06-18）

> 目標：把 anchor+MLP 表徵移進我們的 Lightning/2DGS 框架,解「品質∝點數 + 顏色失真」兩個根問題。
> 來源 code：`../Scaffold-GS`（SOGS 底座,已讀核心）。SOGS = Scaffold-GS + second-order anchor + selective gradient loss。
> 這是全專案最大一次工程 → **分階段、每階可測、fail-cheap**。藍圖供逐階執行,別一次梭哈。

---

## 0. Scaffold-GS 架構（已讀 `../Scaffold-GS/scene/gaussian_model.py` + `gaussian_renderer/__init__.py`）

**參數（存的東西）**：
- `_anchor` [N,3]：anchor 位置（voxel 化 COLMAP 點雲得到）
- `_anchor_feat` [N, feat_dim=32]：每 anchor 一個特徵向量
- `_offset` [N, n_offsets=5, 3]：每 anchor n_offsets 個位置偏移
- `_scaling` [N,6]：anchor 的 scale（前3給 offset 縮放、後3給生成高斯的 base scale）
- **MLPs（共享,小）**：`mlp_opacity`(feat+3+1→n_offsets)、`mlp_color`(feat+3+1+app→3*n_offsets)、`mlp_cov`(feat+3+1→7*n_offsets = scale3+rot4)、可選 `mlp_feature_bank`、`embedding_appearance`

**forward（每幀,`generate_neural_gaussians`）**：
```
對可見 anchor（view-frustum filter）:
  ob_view = (anchor - cam_center); ob_dist=||ob_view||; ob_view/=ob_dist   # 視角方向+距離
  cat = [feat, ob_view, ob_dist]
  neural_opacity = mlp_opacity(cat).reshape(-1,1);  mask = opacity>0       # 動態剔除
  color     = mlp_color(cat).reshape(N*n_offsets,3)
  scale_rot = mlp_cov(cat).reshape(N*n_offsets,7)                          # 3 scale + 4 rot
  xyz = anchor + offset * scaling[:, :3]                                   # 生成高斯位置
  → 套 mask → 得到 (xyz,opacity,color,scale,rot) 一批「神經高斯」→ 丟 3DGS rasterizer
```
**關鍵：高斯是「視角相關、每幀由 MLP 即時生成」的 → 視角依賴顏色天生比靜態 SH 強（=修我們失真）；anchor 結構化 → 少 floater;存 anchor+MLP 比存百萬高斯緊湊（=效率）。**

---

## 1. 對應到我們框架（file-by-file）

我們框架四件套：model / renderer / density / metric（各有 config dataclass + instantiate）。

| 新檔 | 對應 Scaffold | 職責 |
|---|---|---|
| `internal/models/scaffold_gaussian.py` | `scene/gaussian_model.py` | 存 anchor/offset/feat/scaling + 4 MLPs + appearance embedding。**參數=anchor+MLP,不是高斯** |
| `internal/renderers/scaffold_renderer.py` | `gaussian_renderer/__init__.py:generate_neural_gaussians` + render | **每幀:generate_neural_gaussians(model, camera) → 得神經高斯 → 光柵化** |
| `internal/density_controllers/scaffold_density_controller.py` | Scaffold 的 `anchor_growing`/`prune_anchor`/`training_statis`（grep 沒抓到,命名不同,要去 repo 找） | densify **anchor**（靠 offset 梯度累積長新 anchor、剪低 opacity anchor）。**取代 MCMC** |
| `internal/metrics/` 沿用 | — | 先用現有 CityGSV2Metrics;SOGS 的 SGL 之後加 |

### ★ 最大的介面不相容（必須解）
我們框架的 `model.get_xyz / get_opacity` 是**相機無關**的(renderer 直接讀);**Scaffold 是相機相關**(要 camera 算 ob_view)。
→ **解法:不要讓 model.get_* 回高斯。改成「renderer 拿 model + camera 呼叫 generate_neural_gaussians」**。即 ScaffoldRenderer.forward(camera, model, bg) 內部先生成神經高斯再 splat。model 只存 anchor 參數 + MLP,提供 `generate(camera)` 方法。

### 我們 model base 的銜接點（`vanilla_gaussian.py`）
- 用 `setup_from_pcd(xyz,rgb)` 初始化 → 我們改成「xyz voxel 化成 anchor、rgb 不直接用、初始化 feat/offset/scaling/MLP」。
- `training_setup` 要回傳 anchor 參數 + MLP 參數的 optimizer groups（不同 lr:anchor/offset/feat 各一組、MLP 一組）。
- property-dict 系統:anchor/offset/feat/scaling 當 properties;MLP 當 nn.Module 掛 model。

---

## 2. 2DGS 適配（保住幾何/mesh,我們的命脈）

Scaffold 原生 3D 高斯。`mlp_cov` 出 **7 = scale(3)+rot(4)**。2DGS surfel 是 **scale(2)+rot(4)=6** + 法向=切平面外積。
- → `mlp_cov` 改出 **6**（或 7 但第三維 scale 強制極小/0）。
- → renderer 用 **2DGS 光柵器**(`sep_depth_trim_2dgs` 或 `vanilla_2dgs`)而非 3DGS。
- → 保留 depth/normal 正則（surf_depth/normal 照算）。
- 風險:MLP 生成的 surfel 法向一致性、與 depth reg 的互動,要實測。

---

## 3. 連鎖改動（cascade）

| 我們現有 | 在 Scaffold 下變成 |
|---|---|
| depth-init（反投影點雲）| **anchor init**:depth-init 點雲 voxel 化成 anchor（密度由 voxel 控,跟現在類似）|
| MCMC density | **Scaffold anchor densification**（移植）|
| importance-prune / immune-moat | N/A（anchor 機制不同,floater 靠結構化天生少）|
| city-scale partition | anchor per block（partition 邏輯不變,init 改成 anchor）|
| cap_max 預算 | anchor 數 × n_offsets = 有效高斯數;VRAM 看 anchor 數 + MLP + 每幀生成的高斯 |

---

## 4. 分階段執行（每階可測,fail-cheap）

- **S1 — 3D Scaffold 跑通（先不碰 2DGS）**:port model+renderer+density(3D 原生),block_7 depth-init→anchor,跑 30k,對比 2DGS baseline 22.18。**驗證 anchor+MLP 機制在我們 pipeline 能動 + 品質/VRAM/floater 如何。** 這階最關鍵,先別求贏只求「跑得動且合理」。
- **S2 — 2DGS 適配**:mlp_cov→6、換 2DGS 光柵器、接 depth/normal 正則。保住 mesh。
- **S3 — SOGS 增強**:
  - second-order anchor:對 `_anchor_feat` [N,D] 算 D×D 協方差→特徵分解 top-M→per-anchor 2層 MLP 增強特徵（縮小 feat_dim 還維持品質）。**純加在 model 的特徵處理,自含。**
  - selective gradient loss:Sobel(render) vs Sobel(GT),梯度差當權重圖加權 L1。**純 metric 加項,表徵無關,甚至 S1 前就能單獨測**（`internal/metrics/` 加一顆 subclass）。

**便宜的早期摘果**:SGL 可獨立先做（不等 anchor）→ 在現有 2DGS 上測「focus 難畫紋理」有沒有改善細節。

---

## 5. 風險 / 誠實提醒
- 這是多週工程,S1 就不小（anchor model + 神經渲染 + anchor densification 三件）。
- Scaffold 3D→2DGS 適配無現成參考,S2 有不確定性。
- 「anchor 是否在 6GB 用更少有效高斯達高品質」**未驗證**,是賭注核心,S1 出數字才知道。
- 移植優於盲寫:逐檔對照 `../Scaffold-GS`,別憑論文重寫。

---

## 6. 下一步
S1 開工順序:先 `scaffold_gaussian.py`(model,含 4 MLP + anchor 參數 + generate 方法)→ `scaffold_renderer.py`(generate+splat)→ `scaffold_density_controller.py`(移植 anchor_growing)→ config → block_7 smoke。
（SGL 可平行先做,便宜。）
