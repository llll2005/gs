# 3DGS-MCMC 整合分析（源碼對照後，2026-06-07）

> 源碼：`../3dgs-mcmc`（官方 repo，Kheradmand et al. NeurIPS 2024）。
> 目的：取代 opacity-reset、對齊主線「6GB 內有效 Gaussian 管理」（見 `紀錄/主線_Gaussian效率.md`）。
> 結論：**中等工程量、且無需碰 CUDA**（純 torch 重寫 relocation 是關鍵繞法）。

---

## 1. MCMC 實際做的四件事（train.py）

1. **兩個 L1 正則**（train.py 102-103）：`loss += opacity_reg*|opacity|.mean() + scale_reg*|scaling|.mean()` → 推 opacity/scale 變小、稀疏化。
2. **relocate dead**（每 densification_interval，train.py 124-126）：`dead_mask = (opacity <= 0.005)`，把死 Gaussian 搬到 alive 的位置（`_sample_alives` 機率 ∝ opacity 多項式取樣）。
3. **add_new 到 cap**（train.py 127）：`target = min(cap_max, int(1.05*current))`，新點也從 alive 按 opacity 取樣。
4. **noise injection**（每步 optimizer.step **之後**，train.py 140-142）：xyz 加 covariance-shaped 高斯噪聲，乘 `op_sigmoid(1-opacity, k=100, x0=0.995)` × noise_lr × xyz_lr。Langevin/MCMC 探索項。

`relocate_gs` / `add_new_gs` / `_update_params`：gaussian_model.py 450-524。relocate 後 `replace_tensors_to_optimizer` 重置該批的 Adam 狀態。

---

## 2. 【2026-06-07 修正】我們 codebase 已有 MCMC controller，阻點不是 relocation

> ⚠ 本節原寫「compute_relocation 燒在 rasterizer fork、要純 torch 重寫」**作廢**。那是分析外部
> `../3dgs-mcmc` 的結論，但**沒先看自己 codebase**。實情：

- **框架已內建** `internal/density_controllers/mcmc_density_controller.py`（完整 MCMCDensityController：relocate_gs / add_new_gs / _add_xyz_noise / compute_relocation / cap_max / noise_lr，且用 `pl_module.on_train_batch_end_hooks` 掛 noise）。
- **relocation CUDA op 由 gsplat 套件獨立提供**：`from gsplat.relocation import compute_relocation`，**不綁 rasterizer**。→ 純 torch 重寫沒必要。

**真正的兩個阻點：**
1. **gsplat 沒裝**（`ModuleNotFoundError: No module named 'gsplat'`）。需 `pip install -r requirements/gsplat.txt`（CLAUDE.md 有記）。
2. **內建 controller 是 3DGS 專用**：`_get_new_params` 第 141 行 `new_scaling.reshape(-1, 3)`、noise 用 `compute_cov_3d`（mcmc_density_controller.py:100,118）。而 **Gaussian2D 是 2D surfel scale**（`gaussian_2d.py` `scales[..., :2]`）→ 要適配 2 維。

---

## 3. 整合點對照我們的 stack

| 項目 | 對接 | 工程量 |
|---|---|---|
| relocate + add_new | 新 `MCMCDensityController`，放 `after_backward` 週期檢查（現有 densify 同位置）| 中 |
| noise injection | 需 optimizer.step **之後** 的 hook。我們 `density_controller.after_backward` 在 step **之前**（gaussian_splatting.py:436 vs step 450）；**用 `gaussian_model.on_train_batch_end`（:509，在 step 之後）** 或加 post-step hook | 小 |
| opacity_reg / scale_reg L1 | metric 的 extra_loss 或 controller 加 | 小 |
| **cap_max** | **CityGSV2DensityController 已有 cap_max，天然對齊** | 免 |
| 2DGS 適配 | MCMC `_update_params` 假設 **3D scale**（`reshape(-1,3)`）；surfel 是 **2D scale** → relocation scale 數學改 2 維 | 小-中 |
| relocate 後 optimizer 狀態 | MCMC 用 `replace_tensors_to_optimizer`；我們有 `replace_tensors_to_optimizers_`（density_controller.py:149）現成 | 免 |

我們 density controller hook 介面（internal/density_controllers/density_controller.py）：`before_backward` / `after_backward`（step 前）/ `after_density_changed` / `setup` / `on_load_checkpoint`。tensor↔optimizer 工具齊全（cat/prune/replace）。

---

## 4. 協同分析：好消息為主（印證主線）

- ✅ **noise 自動豁免 alpha=0.99**：噪聲乘 `op_sigmoid(1-opacity)`，opacity→0.99 時 ≈0 → 高 opacity 的 depth-init Gaussian 幾乎不被擾動，只擾動低 opacity（不確定）的。**設計上就跟 alpha=0.99 不打架。**
- ✅ **完全無 opacity-reset** → 解決原衝突，`depth_init_immune` 補丁可退役。
- ✅ **cap_max 固定預算 + 死 Gaussian 回收** = 「6GB 內有效利用率最大化」**就是主線命題本身**。
- ✅ **depth reg 共存**：MCMC 不依賴 opacity-reset 做 floater 清除，跟我們的幾何 floater 抑制（depth/normal loss）疊加。
- ⚠ **唯一要調**：`opacity_reg` L1 把 opacity 往下推，**輕微**跟 alpha=0.99 對拉 → 調小 opacity_reg，或讓 immune Gaussian 豁免這項 reg。

---

## 5. 【2026-06-07 修正】落地計畫

1. **先做零成本實驗**（不需 MCMC）：關掉 opacity-reset 看 depth reg 撐不撐 floater。config `RTG_resetoff_aerial_sh2_trim.yaml` 已建（opacity_reset_interval=999999，黑底故無白底一次性 reset）。撐得住 → reset 冗餘、alpha=0.99 免費不被打架、MCMC 無 reset 前提成立。
2. **裝 gsplat**：`pip install -r requirements/gsplat.txt`，確認 `from gsplat.relocation import compute_relocation` 可用。
3. **複製內建 `MCMCDensityController` 改 2DGS 版**（不是從零寫）。**2 處 3D-hardcode 要改（已驗證 Gaussian2D `get_scaling` 回 2D、`compute_cov_3d` 吃 3D scale 會壞）：**
   - **(a) noise 切平面化（Gemini Q2，已驗證正確）**：內建 `_add_xyz_noise` 用 `bmm(cov_3d, randn)` 在 3D 加噪 → 2D surfel 法向 scale≈0，直接套會讓新點**沿法向飄離表面、破壞貼合幾何**。改成：在局部切平面生成 2D 噪聲 `[N(0,s_u), N(0,s_v), 0]`，用四元數轉回世界座標、法向分量強制 0。
   - **(b) relocation scale（Gemini 漏、Claude 補）**：`_get_new_params` 呼叫 `gsplat.compute_relocation(scale_old)` 後 `reshape(-1,3)`，gsplat op 假設 3D scale。改：呼叫前把 2D scale **pad 成 3D（法向補極小值）**，結果**截回 2D**；或確認 gsplat 對退化 scale 的行為。
4. **小場景 smoke test**（block 6 或 7）：固定 cap_max，比 PSNR/SSIM/Gaussian 數 vs 現有 RTGStable+reset 版。重點看「同 cap 下品質」與「alpha=0.99 是否被 opacity_reg 拉壞」。
5. 量化主線指標：固定 6GB（cap_max）下 MCMC vs baseline 的 PSNR/SSIM、有效 Gaussian 利用率。

> 內建 controller 的 relocate/add_new/noise/sample 邏輯（mcmc_density_controller.py）與外部 3dgs-mcmc 一致，
> 故 §1/§3/§4 的機制與協同分析仍有效，只是「怎麼來」從「自己寫」變「改現成 3DGS 版成 2D」。

## 6. 待確認 / 風險
- **Gemini 查證（2026-06-07）**：(1) relocation 隨機過程數學**不需重推**，只 noise 幾何要降維（切平面）—— 與 §5.3 一致；(2) **無已發表的 2DGS/surfel MCMC** → 我們做的是新適配（也是 novelty 一塊）。
- gsplat compute_relocation 對 2D/退化 scale 的實際行為（裝了 gsplat 後對拍驗證）。
- opacity_reg / scale_reg 對 alpha=0.99 的對拉幅度（實測調參，opacity_reg 要保守）。

## 8. 實作狀態（2026-06-07：已寫 + smoke 通過）

- **gsplat build 問題已解**：根因 `CUDA_PATH=/opt/cuda`(系統 13.2) 蓋過 conda env 的完整 11.8 toolkit。修法：`CUDA_HOME=/home/LnoArch/miniconda3/envs/gspl CUDA_PATH=同 TORCH_CUDA_ARCH_LIST=8.9 pip install -r requirements/gsplat.txt`。（4050=sm_89）。可選永久化：`conda env config vars set CUDA_HOME=... -n gspl`。
- **relocation 公式已驗證**（從 `shakibakh/diff-gaussian-rasterization@gs-mcmc` `cuda_rasterizer/utils.cu`）：`opacity_new=1-(1-o)^(1/N)`；`denom=sum_{i=1}^{N} sum_{k=0}^{i-1} binom(i-1,k)(-1)^k/sqrt(k+1) o_new^(k+1)`；`coeff=o/denom`；`scale_new=coeff·scale_old`（**所有維度同一 coeff，scale-dim 無關 → 2D 直接 pad/truncate 無損，已數值對拍 diff=0**）。
- **檔案**：`internal/density_controllers/mcmc_2dgs_density_controller.py`（subclass 內建，走官方 gsplat relocation）。Override：`setup`（hook 不論 initialize_from 都註冊 + 跳過清 alpha=0.99 的 init）、`compute_relocation`（pad 2D→3D / 截回）、`_get_new_params`（去 reshape(-1,3)）、`_add_xyz_noise`（pad 法向=0 → 切平面噪聲）、`_prune_points`（Trim renderer 需要）。config `configs/mcmc_2dgs_mc_aerial.yaml`。
- **3 個框架耦合（B 層警訊，已修）**：(1) Trim2DGS renderer 呼叫 `density_controller._prune_points`（補）；(2) `gaussian_splatting.py:433` 在 metric 有 extra_loss 時讀 `density.densify_grad_scaler`（加欄位=0 → no-op，MCMC 不靠 viewspace grad densify）；(3) Gaussian2D `means/scales/properties` 是 runtime 屬性（setup 後才有）。
- **smoke test（block_7, depth-init, →1200 step, cap 480k）：exit 0，Gaussian=428,183，VRAM 2000MiB，val/psnr@1200=17.54（早期，非判決）。relocate/add_new/noise/Trim 全程不崩。**
- **兩個 L1 regs 已加（step 1 完成，2026-06-07）**：`internal/metrics/mcmc_citygsv2_metrics.py`（`MCMCCityGSV2Metrics`，subclass CityGSV2Metrics）加 `opacity_reg*|opacity| + scale_reg*|scale|` 到 loss。**免死金牌用 opacity-threshold（>0.9 豁免 opacity_reg）而非 origin-tracking** —— 保護「當下承重」、不需 buffer、衰減的 depth-init 點不會永久免疫。config `mcmc_2dgs_mc_aerial.yaml`：opacity_reg=0.01, scale_reg=0.01, immune_opacity_threshold=0.9。**reg smoke（250 step）驗證：op_reg step0=0.000（全 alpha=0.99 免疫，先驗零壓力）→ 0.200（不確定點出現才受壓）= 護城河生效；sc_reg 0.026→0.019。exit 0。**
- **待辦**：(a) block_7 完整 30k run vs depth baseline（21.967）—— 指令見下；(b) 調 cap_max / noise_lr / opacity_reg；(c) gradient-guided relocation 當步驟3消融（Gemini 提案，需先重加 xyz_gradient_accum + 修正權重公式 `opacity×grad` 在低 opacity 仍餓死的瑕疵 → grad 主導；**等步驟1數字出來、確認有「預算浪費」症狀才做**）。
  - 30k 指令（cap≈baseline 460k 求公平）：`python main.py fit --config configs/mcmc_2dgs_mc_aerial.yaml -n mcmc_2dgs_b7 --data.parser.block_id 7 --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply --model.density.init_args.cap_max 460000`

## 7. Q3 reset 文獻引用靠山（Gemini 提供，投稿前逐篇核對原文）
- **3DGS-MCMC (Kheradmand et al., NeurIPS 2024)**：明指原版 clone/split/opacity-reset 是 ad-hoc heuristic，MCMC 用 relocation+noise 自然取代 reset。**主引用**。
- **Revising the Densification of 3DGS / AbsGS (2024)**：分析 view-space 位置梯度依賴 alpha-blending 權重；opacity 歸零→渲染誤差飆→反傳成大位置梯度→人為跨 densify 門檻。**直接佐證「reset 是隱性 densify 驅動器」**。
- **Mini-Splatting (CVPR 2024) / Compact-3DGS**：原版生長+修剪產生大量低貢獻冗餘點，需額外正則或替代驅動維持緊湊。
- 可寫法：「opacity-reset 並非單純冗餘清理，而是透過破壞渲染結果強行刺激位置梯度的 **implicit densification driver**；在 6GB 嚴格預算 + alpha=0.99 設定下此破壞性驅動導致資源浪費，故我們移除 reset 改用 MCMC relocation。」
