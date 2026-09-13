> ⛔ **2026-09-13 重驗後仍不可用。**
> gradient-guided MCMC 規格；該方向的有效部分已由 absgrad_densify 取代並採用（v2 §1.1）。
> 判準見 `../倖存清單_2026-09-13.md`；現行權威是 `../研究總覽_v2.md`。

# Gradient-Guided Surfel MCMC —— 實作規格（步驟3 消融，2026-06-07）

> 來源：Gemini 提案「梯度引導輪迴」+ Claude 的瓶頸診斷修正。
> **狀態：設計就緒，待 gate 放行才實作。** Gate = 預算掃描（cap=300k/200k，見 `本次session待辦.md`）。
> 詳細脈絡見 `紀錄/MCMC整合分析.md`、`紀錄/主線_Gaussian效率.md`。

---

## 0. 動機與 Gate（先講清楚何時才做）

base MCMC（cap≈460k）vs baseline = **平手**（22.04 vs 21.97），但 453k 已近飽和。
**核心未解問題：22 dB 天花板是 (a) 點數錯置 還是 (b) 容量/sh2 表徵極限？**
- gradient-guided 只在 **(a)** 成立時有用（把點導向高誤差區）。若 (b)，它突破不了天花板。
- **Gate（先量再賭）**：先看預算掃描——
  - 300k 仍守 ~22 或陡降（count-limited / 有分配餘裕）→ **放行做 gradient-guided**。
  - 300k 與 460k 都卡 ~22 打平（容量天花板）→ **不做**，轉 sh3/表徵 或 systems 框架。

---

## 1. 機制：改 relocate/add_new 的取樣權重（核心改動）

現況（`mcmc_density_controller.py`）：
- `relocate_gs`：`probs = get_opacities()[alive_indices, 0]` → 死點被傳送到「高 opacity」的活點旁。
- `add_new_gs`：`probs = get_opacities().squeeze(-1)` → 新點長在高 opacity 處。
- 問題（Gemini「富者愈富」）：高 opacity = 已經好的區域 → 回收預算堆在不需要的地方；真正破洞（誤差大）分不到點。

**改成 gradient-aware：** 用 `xyz_gradient_accum`（view-space 位置梯度累積 = 局部重建誤差代理）當權重。

### 權重公式（修正 Gemini 的瑕疵）
Gemini 提 `P ∝ opacity × grad`。**瑕疵：破洞牆壁 opacity≈0 時 opacity×grad≈0 → 照樣餓死**（正是他想解的問題沒解）。
**修正：grad 主導 + opacity 只當防護 floor。** 提供可切換模式（消融用）：

```
g = grad / (grad.mean() + eps)            # 正規化的相對誤差（避免尺度問題）
o = opacity                                # in (0,1)

mode = "opacity"   →  P ∝ o                 # base MCMC（對照組）
mode = "grad"      →  P ∝ g                 # 純誤差引導
mode = "grad_op"   →  P ∝ g * (o + floor)   # grad 主導 + opacity 軟防護（建議主打，floor~0.1）
mode = "powered"   →  P ∝ g**alpha * o**beta  # 通用，alpha/beta 可調
```
- `floor` 防止低 opacity 高誤差區被歸零（修掉 Gemini 瑕疵的關鍵）。
- 兩處（relocate 的 alive-probs、add_new 的 probs）都套同一公式。

---

## 2. 工程：重加 xyz_gradient_accum（MCMC 拔掉的東西要接回來）

MCMC controller 不維護 per-gaussian 梯度狀態。需仿 Vanilla 重加（`vanilla_density_controller.py` 可抄）：

**Buffers**（per-gaussian，跟著拓撲變動維護）：
- `xyz_gradient_accum [N,1]`、`denom [N,1]`。`grads = xyz_gradient_accum / denom`。

**Hook 接法**：
1. `before_backward`：`outputs["viewspace_points"].retain_grad()`（讓 after_backward 拿得到 grad）。
2. `after_backward`（在 relocate/add_new **之前**）：`update_states(outputs)` 累積 —— 取 `viewspace_points.grad[visibility_filter, :2]` 的 norm 累加進 accum、denom+=1（抄 vanilla `_add_densification_stats`）。
3. densify step（每 interval）：用當前 `grads` 算取樣權重 → relocate + add_new → **重置 accum/denom**（仿 vanilla densify 後 `_init_state`）。

**拓撲變動下維護（關鍵，否則 shape mismatch）**：
- `add_new_gs`：append 新點 → accum/denom 補零並 cat。
- `relocate_gs`：死點搬移後其 accum 該歸零（它在新位置是「新生」）。
- `_prune_points`（Trim renderer 也會呼叫）：accum/denom 跟著 mask 砍（`self.xyz_gradient_accum = self.xyz_gradient_accum[valid]` …）。
- `_densification_postfix` / `after_density_changed`：重建/對齊 buffer 大小。
- `on_load_checkpoint` / setup：初始化 buffer。

> 注意：這等於把 MCMC 當初為了乾淨而拔掉的 vanilla 狀態管理接回來一部分 —— 複雜度上升，務必用 smoke test 驗證 buffer 在 relocate/add_new/trim 後 shape 對齊不崩。

---

## 3. 檔案與類別

- 新檔 `internal/density_controllers/gg_mcmc_2dgs_density_controller.py`：
  - `GGMCMC2DGSDensityController(MCMC2DGSDensityController)` + config 加 `relocation_weight_mode`（opacity/grad/grad_op/powered）、`grad_floor=0.1`、`grad_alpha=1.0`、`grad_beta=1.0`。
  - Impl：override `setup`（加 accum/denom）、`before_backward`（retain_grad）、`after_backward`（update_states → super 的 relocate/add_new → reset accum）、`relocate_gs`/`add_new_gs`（改 probs）、`_prune_points`（同時砍 accum/denom）、`_densification_postfix`/`after_density_changed`（buffer 對齊）。
- config `configs/gg_mcmc_2dgs_mc_aerial.yaml`（= mcmc_2dgs 換 density class + weight mode 參數）。

---

## 4. 消融設計（論文的關鍵表）

固定其餘一切（depth-init、cap、sh2、30k），只換取樣權重模式，block_7：

| 變體 | relocation_weight_mode | 預期 |
|---|---|---|
| base MCMC | opacity | 22.04（已有）|
| grad-only | grad | 看純誤差引導會不會反而抖 |
| **grad_op（主打）** | grad_op (floor 0.1) | 期望 > 22.04 = novelty |
| powered 掃 | grad^a o^b | 找最佳 a,b |

- **量**：PSNR/SSIM/LPIPS + 最終點數。**同 cap 下**比，證明「同預算、更好分配 → 更高品質」。
- 若 grad_op > base 穩定 +Δ → 論文核心貢獻成立（演算法層），且有消融表堵 reviewer。
- 若全部 ≈ base → 確認是容量天花板（gate 沒攔住的話），收手轉 systems。

---

## 5. 風險 / 待查
- **novelty 真實性**：投稿前查「gradient-weighted / error-guided MCMC relocation」是否已有人做（Gemini「絕對」要降一檔）。error-guided densification 本身是老路（AbsGS/Pixel-GS），新點在於接到 MCMC relocation + 2DGS surfel + city-scale。
- buffer 維護 bug（拓撲變動 shape mismatch）—— smoke test 必過。
- grad 訊號在 2DGS 的意義（viewspace grad 對 surfel 是否良好誤差代理）——實測。
