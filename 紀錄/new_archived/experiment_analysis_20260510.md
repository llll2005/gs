> ⛔ **2026-09-13 重驗後仍不可用。**
> 最早期的實驗分析（2026-05-10），舊資料集年代。
> 判準見 `../倖存清單_2026-09-13.md`；現行權威是 `../研究總覽_v2.md`。

> **[已歸檔]** 內容已合併至 `紀錄/實驗記錄.md`（2026-05-10 block_0/1/2 實驗段落）。

# 實驗結果分析報告
**日期：** 2026-05-10  
**目的：** 比較三種初始化策略對 fine-tuning 品質的影響，並分析與 CityGSV2 baseline 的差距來源。

---

## 一、三次實驗參數確認

| 實驗 | block_id | initialize_from | SH degree（實際） | n_gaussians (final) |
|------|----------|-----------------|-------------------|---------------------|
| block_0 | 0 | depth-init PLY (block_0.ply) | 2 ✓ | **84,417** |
| block_1 | 1 | coarse ckpt (epoch=6-step=30000) | **0 ✗（BUG）** | 1,267,006 |
| block_2 | 2 | coarse ckpt (epoch=6-step=30000) | 2 ✓（SH展開修復後） | 2,648,741 |

共同參數（fine-tuning config `RTG_mc_aerial_sh2_trim24.yaml`）：
- max_steps: 30,000
- densify_grad_threshold: 0.00005
- densify_until_iter: 15,000，interval: 500
- depth_loss_weight: init=0.5, final_factor=0.05 → 終值 ≈ 0.025
- down_sample_factor: 1.2
- block_dim: [5, 5]（共 25 塊）

---

## 二、定量結果對比

### 我們的三次實驗（per-block test, step=30000）

| 實驗 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | d_reg | 說明 |
|------|--------|--------|---------|-------|------|
| block_0（depth-init, sh2）| 24.78 | 0.703 | 0.624 | 0.210 | 無 coarse，點雲嚴重不足 |
| block_1（coarse, sh=0）   | 27.19 | 0.729 | 0.568 | 0.541 | SH未展開（舊 bug） |
| block_2（coarse, sh=2）   | 27.96 | 0.804 | 0.331 | 0.467 | 正確流程（修復後）|

### 參考 baseline（本 repo 既有結果）

| 實驗 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 說明 |
|------|--------|--------|---------|------|
| test_block_1（無 coarse, sh2, 全場景影像）| 22.97 | 0.634 | 0.715 | block_id=null，全部 5620 張訓練 |
| aerial_test_block_1_test（兩階段, sh2, step≈30k）| **29.73** | **0.863** | **0.210** | 從 test_block_1 ckpt 繼續，block_id=null |

> 重要注意：baseline 的 `block_id=null` 代表使用全部 5620 張影像訓練，不做 partition 切割。我們的實驗每個 block 僅使用約 330~420 張。這是一個不公平對比基準。

---

## 三、差距原因分析

### 3.1 block_0：點雲嚴重不足（根本原因）

- **n_gaussians = 84,417**，遠低於 block_1/block_2 的 120 萬～260 萬。
- Depth-init PLY（來自 RTG-SLAM 風格點雲）對 block_0 的初始點數過少。
- 即使有 28 次 densification（15000/500），84K 的最終規模顯示 depth 點雲初始品質極差，或 block_0 的 depth 估計本身誤差大（訓練時出現 4 張 "depth scale out of bound" 警告）。
- **缺乏 coarse 初始化 → 缺乏全域結構 → 點雲增長受限 → PSNR 最差（24.78）。**

### 3.2 block_1：SH degree 實際為 0（已知 bug，已修復）

- `overwrite_config=True`（預設）導致 coarse 的 `sh_degree=0` 覆蓋 YAML 的 `sh_degree=2`。
- 全程以 sh_degree=0 訓練，shs_rest 維持 shape `[N, 0, 3]`（全部零）。
- 結果：LPIPS=0.568，遠高於 block_2（0.331），差距主要來自無法表現視角相依外觀。
- **修復：** `gaussian_splatting.py` 新增 SH 展開邏輯（cfg_sh > ckpt_sh 時自動 zero-pad）。

### 3.3 block_2 vs baseline：仍有 ~2 PSNR 差距

block_2 是當前最佳（PSNR=27.96），但仍低於 baseline（29.73）。原因分析：

**（1）fine-tuning 評估框架不同（非主因，屬比較方法差異）**  
- 我們的 pipeline：coarse 用全部 5620 張，fine-tune 每個 block 用 partition 後約 330～420 張 —— 這是 CityGSV2 設計本意。
- baseline（aerial_test_block_1_test）fine-tuning 階段同樣用全部 5620 張（block_id=null），不做分塊。
- 兩者 coarse 都覆蓋全場景；差別僅在 fine-tuning 的影像集範圍。
- CityGSV2 divide-and-conquer 本身就假設 per-block fine-tuning 影像足夠，此差異不應視為缺陷，不是品質差距的根本原因。

**（2）d_reg 仍然偏高（0.467）**  
- depth_loss_weight 終值 ≈ 0.025，在 30k steps 時仍顯著影響 loss。
- 深度圖（DA2 estimated）的相對誤差在細節區域（房屋邊緣、樹木）引入錯誤梯度。
- 推測：coarse 模型初始化帶入正確全域結構後，depth loss 反而在細節層面造成衝突。

**（3）densify_grad_threshold 過小（0.00005 vs 標準 0.0002）**  
- 造成過度激進的分裂：block_2 最終有 2.64M Gaussians（出發點 1.27M）。
- 過多 Gaussians → 訓練後期記憶體壓力增加 → 可能觸發自動剪枝（prune），破壞良好結構。
- CityGSV2 標準 fine-tuning config 使用 0.0002（本 YAML 保留了我們實驗設定的 0.00005）。

**（4）coarse 模型本身偏弱**  
- 本次 coarse（RTG_mc_aerial_coarse_sh2）同樣使用 densify_grad_threshold=0.00005（激進），sh_degree=0。
- 標準 CityGSV2 coarse 使用 sh_degree=0 + 標準 densification，但 fine-tuning 用 sh_degree=2/3。
- coarse 品質偏低 → fine-tuning 起始點結構不夠好。

**（5）block_2 的 Gaussian 數量是否已超過合理上限**  
- 2.64M Gaussians，以 RTX 4050 6GB VRAM 訓練，接近記憶體上限。
- 未設置 `cap_max`（Gaussian 數量上限），訓練後期可能有效 batch size 下降或品質退化。

---

## 四、各指標對比小結

| 對比維度 | block_0 vs block_2 | block_1 vs block_2 | block_2 vs baseline |
|----------|-------------------|--------------------|---------------------|
| PSNR 差距 | -3.18 | -0.77 | -1.77 |
| SSIM 差距 | -0.101 | -0.075 | -0.059 |
| LPIPS 差距 | +0.293 | +0.237 | +0.121 |
| 主因 | 無 coarse，初始點雲不足 | SH degree=0（bug） | 訓練資料量差異 + d_reg偏高 + densification激進 |

---

## 五、待驗證假設（給 Gemini 查證）

1. **DA2 depth scale "out of bound" 的影響**：block_0 有 4 張深度圖超出合法範圍，這些影像的 depth loss 計算是否被跳過還是輸入了錯誤 scale？

2. **densify_grad_threshold=0.00005 是否對此場景合適**：MatrixCity aerial 場景的 Gaussian gradient 統計是否與 outdoor street 不同？CityGSV2 論文有無場景特定建議值？

3. **block_2 的 2.64M Gaussians 是否合理**：CityGSV2 論文對 aerial 場景每個 block 的 Gaussian 數量是否有報告？DOGS（NeurIPS 2024）對 per-block Gaussian 規模有何描述？

4. **two-stage 流程（coarse→partition→fine-tune）在 block_id=null vs block_id=N 上的品質差異是否在論文中討論過**：CityGSV2 是否也使用了完整影像集作為 fine-tuning 資料？還是只使用 partition 影像？

5. **正確的 SH degree 升維流程**：CityGSV2 是 coarse(sh=0) → fine-tune(sh=2 zero-padded) 嗎？還是 coarse 本身就用 sh=2？

---

## 六、已執行改動（2026-05-10，依 Gemini 回覆）

### `utils/depth_init_blocks.py`
| 項目 | 改前 | 改後 | 原因 |
|------|------|------|------|
| `INIT_ALPHA` | 0.99（不透明） | **0.1**（半透明） | 0.99 造成梯度遮蔽，Densify 停滯，block_0 只長到 84K |
| `sample_ratio` 預設 | 0.05（5%隨機採樣） | **1.0**（全部 M_s 像素） | RTG-SLAM 的 5% 是線上模式記憶體妥協，離線批次初始化不適用；幾何密度由 voxel 決定 |

### `configs/RTG_mc_aerial_sh2_trim24.yaml`
| 項目 | 改前 | 改後 | 原因 |
|------|------|------|------|
| `depth_loss_weight.final_factor` | 0.05 | **0.001** | DA2 深度圖邊緣模糊，後期與 RGB loss 互斥撕裂 |
| `depth_loss_weight.max_steps` | 30_000 | **15_000** | 15k 步後 depth weight ≈ 0.0005（幾近於零），RGB 拿回主導 |
| `densify_grad_threshold` | 0.00005 | **0.0002** | 過激導致 block_2 暴增至 2.64M Gaussians；改回 CityGSV2 標準值 |

### 待執行驗證
```bash
python utils/depth_init_blocks.py data/matrix_city/aerial/train/block_all --block_dim 5 5
python utils/train_citygs_partitions.py -n RTG_mc_aerial_sh2_trim24 --init_mode depth --depth_init_dir data/matrix_city/aerial/train/block_all/depth_init --blocks 0
```
預期：block_0 Gaussian 數量從 84K 大幅提升，PSNR 應顯著改善（目標 >27）。

---

## 七、下一步建議

**立即可修的問題（無需重大設計變動）：**
1. 將 fine-tuning config 的 `densify_grad_threshold` 從 0.00005 改回 0.0002（CityGSV2 標準值）。
2. 降低 `depth_loss_weight.final_factor`（0.05 → 0.01），減少 depth loss 在後期的干擾。
3. 新增 `cap_max`（建議 200 萬），防止單 block Gaussian 過多。
4. 驗證 block_0 的 depth-init PLY 實際包含多少點，以確認是 PLY 品質問題還是 densification 參數問題。

**需要進一步實驗的問題：**
- 跑一個完整 25 blocks 的 coarse-init + sh=2 訓練（修復後的版本）再 merge，做全場景 PSNR 評估，才能與 CityGSV2 官方數字公平對比。
- 目前三個 block 各屬不同區域，樣本偏差大，不宜作最終結論。
