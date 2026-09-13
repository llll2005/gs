> ⛔ **2026-09-13 重驗後仍不可用。**
> 2026-06-07 的戰略轉向文件；其後命題再次演進成成本感知密度控制（v2 §1）=> 已過時。
> 現行權威：`../研究總覽_v2.md`。

# 主線設計：6GB 內有效 Gaussian 管理（2026-06-07 戰略轉向）

> 來源：task1 + task2 跑完後的戰略討論（2026-06-07）。
> 這是 ADMM/init 兩條原路線出局後浮現的**新主線**。
> 待辦對照：`紀錄/本次session待辦.md`、`紀錄/ADMM_改法清單.md`（ADMM 降級後的復活藍圖）。

---

## 0. 為什麼轉這條（兩個實驗的鐵證）

| 原本下注 | 實測 | 是否大槓桿 |
|---|---|---|
| coarse-init 優勢（task1，densify 對齊乾淨版）| 同算力 +0.4dB PSNR，SSIM/LPIPS 還輸 | ❌ 次要、混合 |
| ADMM 跨塊共識（task2，rho=0.5 ×27 outer）| net-zero（+0.2dB 是多訓練買的、||y|| 惰性）| ❌ 無效 |
| **Gaussian 預算/密度**（意外發現）| densify 0.00005→0.0002 值 **1.7dB**，但 0.00005 在 6GB OOM | ✅ **真正大槓桿** |

**結論：4.64dB 缺口主因 = CityGSV2 的 Gaussian 數遠超 6GB 能負擔（硬體稅），不是演算法選擇。**
→ 主線命題改成：**在固定 6GB 預算下，最大化「有效 Gaussian」的利用率**，讓有限數量發揮高密度效果。

---

## 1. 四軸預算框架（攻擊 6GB 的四把刀）

| 軸 | 原理 | 技術 | 狀態 | 救不救 block_16 OOM |
|---|---|---|---|---|
| **源頭效率**（每顆物盡其用）| opaque init → 一顆 surfel 蓋一塊表面 | **alpha=0.99**（RTG）| ✅ 已在 depth-init 用 | 間接（densify 壓力小→數量少）|
| **減總數** | 移除低貢獻 Gaussian | Prune / GaussianSpa / LightGaussian | LightGaussian 已掛 config 未啟用 | ✅ 同時降優化器+渲染 |
| **回收/搬移** | 死 Gaussian 搬到高需求區，不浪費 | **3DGS-MCMC relocation** | 待導入（新 density controller）| ✅ 固定預算最大化利用 |
| **釋放優化器** | frozen Gaussian 移出 optimizer 省 Adam 狀態 | RTG stable 擴展成 freeze | 待做（現 stable 只做幾何管理，無記憶體效益）| △ 間接讓出 headroom |
| **省儲存** | base + 預測殘差，不存完整屬性 | ResNet 殘差原理 → anchor+offset（Scaffold/HAC）| 待評估 | ✗ 主要降儲存/inference |

最上游最便宜、且**已經有**的是 alpha=0.99。

---

## 2. alpha=0.99 為什麼讓「更少 Gaussian 擬合區塊」（源頭效率）

alpha-compositing：像素 = `sum_i c_i * alpha_i * prod_{j<i}(1 - alpha_j)`。表面要 opaque 需累積 alpha→1。
- **alpha=0.1（標準 3DGS）**：單顆貢獻小，要堆 ~`log(0.01)/log(0.9)≈44` 層才 opaque；透明起手→重建誤差大→瘋狂 clone→Gaussian 暴增。
- **alpha=0.99（RTG）**：單顆即 opaque，一顆 surfel 蓋一塊 patch；densify 壓力小→數量少。

**已驗證的兩個推論（從 task1 A/B）：**
1. **這就是 depth-init 塞得進 6GB 而 coarse-init OOM 的原因**：depth-init Gaussian 起手 opaque+貼表面，densify 壓力小；coarse-init 靠堆量→爆。
2. **重新解讀 A/B：depth-init 的 SSIM 完勝**（block_16: 0.845 vs coarse 0.659）正因 alpha=0.99+準深度 = 少、不透明、貼合表面 → 結構保真度高。coarse 用半透明 blob 堆，PSNR 勉強拉近但結構糊。
   → **depth-init 不只「比較省」，結構品質實質更好**；PSNR 小缺口該用「預算內讓 depth-init 用更多 Gaussian（三把刀讓出空間）」補，不是換回 coarse。

**alpha=0.99 的三個協同陷阱：**
1. **強依賴準確深度**：opaque 放錯位是硬錯誤（沒法靠混合看穿修正），透明的較寬容。故本是 RGBD-SLAM 技巧。
2. **打架 opacity-reset**（見 §3，這是換 reset 的動機）。
3. **densify 啟發式要重調**：opaque 起手後位置梯度行為變了，densify=0.0002 未必對 alpha=0.99 最佳。

---

## 3. opacity-reset 是承重結構（非冗餘）—— MCMC 整套接管而非刪除

> 註：本節原標題「原始且冗餘」**已作廢**（2026-06-07 實驗 + 讀程式推翻），見下方 ⚠ 段。
> 結論反轉：reset 不能直接拔，要用 MCMC 整套有原則替換。

**reset 原始動作**：原版 3DGS 每 ~3000 步把 opacity 壓到 ~0.01。我們的 RTGStable 版讓 stable（含 alpha=0.99 init）豁免，只打 densify 子 Gaussian。**替代方案 = 3DGS-MCMC（relocate+noise+兩個 L1+cap_max），對齊主線「固定預算下有效利用率最大化」，且我們 codebase 已有 controller（見 `紀錄/MCMC整合分析.md`）。**

**⚠ 2026-06-07 實驗 + 讀程式：「reset 冗餘」假設被推翻，且我的實驗有 confound。**

實驗：`RTG_resetoff_aerial_sh2_trim`（opacity_reset_interval=999999）vs baseline：block7 **−1.66dB + Gaussian −35%**、block16 **−1.50dB**（數量持平）。

**讀程式後發現的 confound（重要）**：`opacity_reset_interval` 這個參數**同時 gate 兩件事**：
- `_reset_opacities` 呼叫（vanilla:85 `step % interval == 0`）
- **大尺寸剪枝**（vanilla:78 `size_threshold = 20 if step > interval else None` → 設 999999 則永遠 None → rtg:252-255 的 `big_vs`半徑>20 / `big_ws`scale>0.1·extent 剪枝**完全不跑**）
→ **我的 reset-off 同時關了 reset + 大尺寸剪枝，−1.5dB 是混合效應，非純 reset。**

**程式事實（reset 在我們 pipeline 實際做什麼）**：
- `_reset_opacities`（rtg:216）= `opacity = min(opacity, 0.01)`，**但 stable 豁免**。`depth_init_immune=true` → alpha=0.99 初始 Gaussian 是 stable → 本就不被 reset；reset 只打 **densify 子 Gaussian**。
- densify 機制：cloning（vanilla:143）看的是**位置梯度** `xyz_gradient_accum`，不是 opacity。reset 透過「壓低 opacity → 該區渲染變差 → loss 升 → 位置梯度升 → 過 densify_grad_threshold → 更多 clone」**間接**驅動 densify。關掉 → opacity 不擾動 → 早擬合完 → 梯度飽和 → 少 clone → 欠擬合（解釋 block7 −35%，且壓過「少剪枝會加 Gaussian」的反向）。

**MCMC 對應（讀程式後更精確：四機制，含先前漏掉的 scale_reg）**：

| 我們現在的機制 | 程式位置 | MCMC 替代 |
|---|---|---|
| opacity 擾動 → 間接驅動 densify | reset + 位置梯度 | **relocate + noise** |
| opacity 剪枝（<0.005，非 stable）| rtg:246 | **opacity_reg L1 + dead_mask** |
| **大尺寸剪枝（scale/半徑）** | rtg:252-255 | **scale_reg L1**（讀程式才看出這條對應）|
| 數量上限 | 無（靠 densify_until）| **cap_max** |

→ **必須整套 MCMC 接管，不能只關 reset。** MCMC 兩個 L1（opacity_reg+scale_reg）正好替兩種剪枝，relocate+noise 替 reset 的 densify 驅動。MCMC 的 **opacity_reg 要調保守**（opacity 動態對品質敏感）。

**可選乾淨化實驗**：加 `disable_opacity_reset: bool` flag（只關 reset 呼叫、保留 size-pruning）才能純歸因 reset。但主線是 MCMC（整套替換），ROI 低，結論「不能只關 reset」不受 confound 影響。

**修正後邏輯鏈：**
```
alpha=0.99 源頭少 Gaussian（已驗證：depth-init 少 Gaussian + SSIM 更好）
   → 但 opacity-reset + 它 gate 的剪枝是承重結構（densify 驅動 + 兩種剪枝），不能直接拔
   → 用 MCMC 整套（relocate + noise + opacity_reg + scale_reg + cap_max）有原則接管
   → 固定 6GB 預算下有效 Gaussian 利用率最大化（= 主線命題）
```

---

## 4. 一個重要的記憶體分軸澄清（freeze vs 渲染 OOM）

block_16 OOM 在 `rasterize_gaussians_backward`（**渲染峰值記憶體**），需要 414MB 只剩 406MB —— **就差 ~8MB**。
- 6GB 被三塊吃：(a) 參數 (b) 優化器狀態 ~2×參數 (c) rasterizer forward/backward activations。
- **RTG freeze 擴展**（detach + 移出 optimizer）省的是 (b)，frozen Gaussian **照樣 render** 不降 (c)。但因共用 6GB 池，省 (b) 間接給 (c) 讓 headroom（block_16 差 8MB，省任何優化器空間都能過）。
- **要直接降 (c)**：Prune（減總數）或讓 rasterizer 跳過 frozen Gaussian 梯度（CUDA kernel 改）。
- **Prune 是唯一同時降 (b)+(c) 的** → 對 OOM 最直接。

---

## 5. 便宜實驗清單（驗證主線，按 ROI）

1. **opacity-reset 冗餘性**（零成本）：關掉 reset，看 depth reg 撐不撐 floater。→ 驗證 §3。
2. **alpha 0.99 vs 0.1 的 Gaussian 數對照**（便宜）：同 block，兩種 init 各需多少 Gaussian 達同 PSNR/SSIM。→ 量化 §2 源頭效率。
3. **LightGaussian prune 啟用**（config 已有）：看定向 prune 能砍多少數量 / 掉多少品質。
4. **（待源碼）MCMC relocation 導入**：使用者抓源碼中，Claude 分析接進 density controller。

## 6. ADMM 尾巴（不忘）
- 對照組（純訓練到 45000）釘死 task2 net-zero，寫負結果章節。
- ADMM 若復活走 `紀錄/ADMM_改法清單.md`（ResNet 殘差式 z + 條件式觸發），只在「分歧大」regime。
