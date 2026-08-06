# 本次 Session 待辦

> **最新活躍實驗在最上方（MCMC 預算掃描）。下方 task1/task2 是已完成的歷史。**
> 主線脈絡：`紀錄/主線_Gaussian效率.md`、`紀錄/MCMC整合分析.md`、`紀錄/gradient_guided_mcmc_spec.md`。
> Pipeline 全圖：`紀錄/完整Pipeline數學詳解.md`。

---
6/30
---
病因 A：floater / 半透明疊影(conf 可修)★ 先打這個

半透明覆蓋 = 低不透明度的浮游高斯飄在表面前面,從訓練視角看剛好對、換個視角就穿幫疊上去。針狀拉絲 = 被拉長的各向異性 surfel。這次 aggr17 的 recipe 為了衝顆數,把抑制 floater 的旋鈕全鬆綁了,代價就是這個。

對應 conf 區塊 + 建議:

1. metric 區塊 — 開啟 distortion(目前關著,這是 2DGS 對付半透明疊影的標準解)
lambda_dist: 0  →  100   (dist_regularization_from_iter: 3000)
distortion loss 把沿光線散開的權重壓回單一表面 → 直接消半透明分層。這是 source B,你的擴充清單裡有,正是為這症狀設計的。值域 100~1000,先試 100。

2. metric 區塊 — opacity_reg / scale_reg 調回去(aggr17 從 0.01 鬆到 0.007)
opacity_reg: 0.007  →  0.01    # 多砍低透明度 floater(半透明疊影)
scale_reg:   0.007  →  0.01    # 收縮被拉長的 surfel(斜向拉絲)
aggr17 把這兩個鬆綁是「少砍點換顆數」;既然你現在要的是畫質、而且顆數本來就撞 6GB 天花板,換回來划算。

3. density 區塊(MCMC)— 提高剪枝門檻
min_opacity: 0.005  →  0.01    # dead 判定放寬,多回收 floater
注意 MCMC 沒有 opacity_reset(不像原版 CityGSV2 每 6000 步重置逼 floater 現形),全靠這條 + opacity_reg 清,所以這條更關鍵。

4. (可選)light_gaussian 頂層 — 後期 importance-prune
light_gaussian: { prune_steps: [45000], prune_percent: 0.4, v_pow: 0.1 }
紀錄裡這招「explore-then-prune」在 60k 量過 +0.28dB,專砍低重要度浮游點。source D。

---
病因 B：高樓/斜視/邊緣視角欠觀測(結構問題,單一旋鈕修不掉)

1708 那種崩法不是 floater 而已 —— 是那棟高樓在 block_7 邊緣、又是斜視大深度,訓練相機根本沒幾台看過它 → 幾何欠約束 → MCMC 在沒有多視角約束下亂長。這就是紀錄裡那 ~3.7dB 殘差 = coverage / 全域一致性 / scene-scale 的部分,跟 cheap_lever 結論一致。

可緩解、但別期待單一 conf 解決:

5. renderer 區塊 — depth_ratio 試著往下調(精度↔召回)
depth_ratio: 1.0  →  0.5(甚至 0)
1.0=median 深度(偏召回/完整,但對大深度高樓會把表面抹開成霧);往 0=expected 深度(偏精度,表面更銳),代價是薄結構可能缺角。1708 的「整片霧」有部分是 median blending 造成,值得一試。

6. 真正的解(非本 conf 範圍):這些欠觀測視角要靠更密的相機覆蓋 / coarse 全域一致性種子 / 分塊邊界重疊補,屬於下一步要測的主線,不是調 aggr17 能補的。

---
我的建議優先序

先做病因 A 的 1+2(開 distortion + 收回 opacity/scale_reg)—— 最便宜、直打你最在意的「半透明覆蓋 + 拉絲」,而且不太損 PSNR(可能還小升,因為 floater 本來就傷 SSIM/LPIPS)。3、4、5 當第二批疊加。病因 B 留給主線。

要不要我直接複製一份 configs/mcmc_60k_sh3_aggr17_clean_b7.yaml,把病因 A 的 1–4 套好(其餘不動),讓你直接重跑對比?還是你想先只開 distortion 單變量試?














## 🟡 TODO：floater 處理（2026-06-18，使用者讀完 pipeline 後一併處理）

**問題**：floater 嚴重 —— `tools/diag_multiview_consistency.py`（GT-free 有效診斷）量出我們 floater 像素 **4.7%** vs CityGSV2 **1.1%**（4 倍）。傷 SSIM/LPIPS（畫面一團糟）。

**已試 + 死路（別重走）**：
- 偽深度診斷（`diag_depth_consistency.py`）→ **無效**（CityGSV2 對照證明量的是 Depth Anything 自身誤差）。
- post-hoc 硬砍（`floater_cleanup.py` 逐高斯多視角深度一致性 prune）→ **死**：砍 2.3% 就 -1.28dB、砍 17.8% 就 -5.46dB（指標選到邊緣/高頻/遮擋邊界=重要點，非 floater；同 RTG B4/B5 失敗模式；無 finetune 破洞）。

**任何方案必過 5 道關卡**：① 便宜(6GB) ② 2DGS 相容 ③ GT-free ④ 訓練期 soft 非 post-hoc 硬砍 ⑤ 不純刪點(點數飢餓，relocate/壓 opacity 優於 delete)。

**候選（撈論文 + 我打分），按 ROI**：
1. ★ **Distortion loss（2DGS 原生，我們有但 `lambda_dist=0` 關著）** —— 把沿光線 alpha 權重逼到單一表面 → 半透明 floater 無法存在。過全部 5 關、零刪點、訓練期 soft。**免費解，第一個試**。測試：config metric 加 `lambda_dist`（試 100 或 1000，3DGS/2DGS 慣用）+ `dist_regularization_from_iter`，重跑 block_7 量 floater + SSIM。
2. **Opacity 正則 + immune moat 互動**：MCMC opacity_reg 想壓死無支撐 floater，但 immune moat(o>0.9)正在保護 alpha=0.99 的 floater init → 考慮「immune 只保護『多視角一致』的高 opacity 點」（soft，光度梯度自動保護真點、floater 無支撐自然死）。
3. **修 densification**（AbsGS / Revising Densification / Pixel-GS）：floater 常是 densify 失當生的，從源頭少生。要動 density controller。
4. **soft 多視角一致性**：把 `diag_multiview_consistency` 的一致性當「opacity 壓力權重」（非硬砍）→ 光度梯度保護重要點、純 floater 死。硬版已死、soft 版未試。

**搜尋方向**：「2DGS distortion loss」「GS floater removal opacity regularization / Mini-Splatting / binary opacity」「revising densification / AbsGS / Pixel-GS floater」。**別找**：Mip-Splatting(鋸齒非 floater)、純 LightGaussian(不對症)、RTG B4/B5(已死)。

**工具狀態**：`tools/diag_multiview_consistency.py` ⚠**被 depth_ratio 實驗證明會被「深度平滑度/密度」污染**（見下），不是乾淨 floater 隔離器，跨設定比較無效、CityGSV2 對照也要打折；`diag_depth_consistency.py`(偽深度,無效)、`floater_cleanup.py`(硬砍,死路)留作教訓。

**depth_ratio 小實驗（2026-06-18，30k cap800 同 config 只改 r）—— 定案 + 揭穿診斷缺陷**：
| r | PSNR | SSIM | LPIPS | 診斷floater | 雙向不一致 |
|---|---|---|---|---|---|
| 0.0 | 21.461 | 0.651 | 0.474 | 4.6% | 18.7% |
| 0.5 | 21.644 | 0.656 | 0.464 | 4.4% | 18.9% |
| 1.0 | 22.182 | 0.672 | 0.434 | 5.7% | 31.9% |
- **品質 r=1 全面最好（含 SSIM/LPIPS）→ depth_ratio 定案 r=1，沒得再優化，不是 floater 解。**
- ⚠ **r=1 品質最好但診斷 floater 最高 = 矛盾 → 診斷被污染**：`surf_depth` 本身=depth_ratio，r=0 expected 平滑→重投影誤差小(假低)、r=1 median 銳利→誤差大(假高)。**鐵證：真 floater 多 SSIM 會差，但 r=1 SSIM 最好。**
- **影響：診斷數字受「深度平滑度/點密度」干擾 → CityGSV2 1.1% vs 我們 4.7% 有一部分只是點數密度（4M vs 720k）非獨立 floater 病 → 隱約把 floater 往「點數不夠」拉、繞回品質∝點數。但不就此斷定 floater 非問題（視覺/SSIM 上是真差），只是這支診斷沒那麼可信。** 真 floater 旋鈕仍是 distortion loss(關著)+ GES 表徵層。

---

## 🔵 當前活躍：MCMC 預算掃描（2026-06-07）—— 回答「MCMC 有沒有效率優勢」

**背景**：base MCMC（cap≈460k）vs baseline = 平手（22.04 vs 21.97），但 453k 已近飽和 → 高點數比是「假平手」。**真考場在低預算**：把 MCMC 強壓到飽和點以下，看管理能力。baseline 無 cap（453k 自然長成），故是 **MCMC 曲線 vs baseline 單點 (453k, 21.967)**。

### 要跑的（離線可跑，各 ~4h，先 300k 再 200k）

cap=300k：
```
python main.py fit --config configs/mcmc_2dgs_mc_aerial.yaml -n mcmc_2dgs_b7_cap300 --data.parser.block_id 7 --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply --model.density.init_args.cap_max 300000
```

cap=200k（上行 `300000`→`200000`、`-n` 改 `mcmc_2dgs_b7_cap200`）：
```
python main.py fit --config configs/mcmc_2dgs_mc_aerial.yaml -n mcmc_2dgs_b7_cap200 --data.parser.block_id 7 --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply --model.density.init_args.cap_max 200000
```

> 註：MCMC 的 cap 只限「往上長」，densify 後點數會從 cap 往下漂 ~10%，所以最終實際點數會 < cap。判讀看**最終實際點數**（叫 Claude 讀，或自己從 ckpt 的 means.shape[0]）。

### 結果填寫區（品質 vs 預算曲線）

| cap | 最終實際點數 | PSNR | SSIM | LPIPS |
|-----|-------------|------|------|-------|
| **800k（6GB 天花板）**| **720,000** | **22.182** | **0.672** | **0.434** |
| 460k | 414,000 | 22.043 | 0.653 | 0.467 |
| 300k | 270,000 | 21.510 | 0.627 | 0.512 |
| 200k | 180,000 | 21.136 | 0.606 | 0.540 |
| baseline 參考點 | 452,989 | 21.967 | 0.661 | 0.448 |

**飽和拐點 ~414k（2026-06-08）**：斜率 180→414k ≈0.4dB/100k，**414→720k 暴跌 0.045dB/100k**（多 306k 點只 +0.14dB，幾乎冗餘）。→ 6GB 天花板 ≈22.18@720k（無 OOM）；**414k 甜蜜點 = 天花板 99.4% 品質、57% 點，「同品質低損耗」未剪枝已成立**；720k 三項全勝 baseline；**預測 prune 720k→~414k 近乎無損**；誠實天花板（vs CityGSV2 28.7 差 6.5dB 是 sh3/完整 pipeline，非點數）。

### 看到什麼 → 代表什麼 → 下一步（= gradient-guided 的 gate）

| 結果 | 瓶頸 | 下一步 |
|------|------|--------|
| 300k 仍守 ~22（曲線平）| 點數有餘裕 | ✅ 效率保底成立（少 ~40% 點同品質）+ **放行 gradient-guided**（有重分配空間）|
| 300k 陡降 | count-limited，點數珍貴 | **放行 gradient-guided 且更該做**（把稀缺點放對地方價值高）|
| 300k & 460k 都卡 ~22 打平 | 容量/sh2 天花板 | ❌ gradient-guided 突破不了 → 轉 sh3/表徵 或老實走 systems 框架 |

**gradient-guided 規格已寫好**（`紀錄/gradient_guided_mcmc_spec.md`）。

---

## 🟢 gradient-guided 消融（2026-06-07，gate 已綠燈 + controller 已實作+smoke 通過）

預算掃描確認 count-limited（180k=21.14 / 270k=21.51 / 414k=22.04，斜率 ~0.4dB/100k 仍在爬，非天花板）→ gradient-guided 前提存活。`GGMCMC2DGSDensityController` 已寫好（`internal/density_controllers/gg_mcmc_2dgs_density_controller.py`，config `configs/gg_mcmc_2dgs_mc_aerial.yaml`，預設 mode=grad_op）。smoke（1300 step）exit 0、buffer 拓撲維護不崩。

**消融設計**：同 cap、同 init、只換 relocation_weight_mode。先打**點數稀缺的 cap=300k**（分配價值最大）：

```
python main.py fit --config configs/gg_mcmc_2dgs_mc_aerial.yaml -n gg_mcmc_b7_cap300 --data.parser.block_id 7 --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply --model.density.init_args.cap_max 300000
```

可選續跑：cap=460k（`-n gg_mcmc_b7_cap460 --cap_max 460000`）；純 grad 模式（加 `--model.density.init_args.relocation_weight_mode grad`）。

### 消融結果表

| cap | mode | 最終點數 | PSNR | SSIM | LPIPS | vs base |
|-----|------|---------|------|------|-------|--------|
| 300k | opacity (base) | 270,000 | 21.510 | 0.627 | 0.512 | — |
| 300k | **grad_op** |  |  |  |  | ← 跑這個 |
| 460k | opacity (base) | 414,000 | 22.043 | 0.653 | 0.467 | — |
| 460k | grad_op（可選）|  |  |  |  |  |

判讀：**grad_op @300k 穩定 > 21.51** → gradient 引導有效 = 演算法 novelty 成立（有消融表撐）。≈21.51 → MCMC 分配其實沒錯、是缺點數 → 收手拿 systems 保底（reset-free + 預算控制）。

---

## 🟢 冗餘掃描：2DGS-native 重要度剪枝（2026-06-08，已實作+smoke 通過）

`internal/utils/importance_prune_2dgs.py`（Gemini 演算法 grounded 版：harvest 用 Trim `record_transmittance` 算貢獻 × 面積^v_pow 砍底部 k%，零 gsplat）。接進 `light_gaussian_prune`（路由用 **scale 維度==2** 判 2DGS）。smoke：431k→剪30%→302k，exit 0。重用 `model.light_gaussian.prune_steps/prune_percent/v_pow` config，CLI 傳。

**實驗（2026-06-08 更新：從 6GB 天花板 800k 剪，不是 460k）**：MCMC cap=800k，step 20000 剪 10/30/50% → 跑到 30k fine-tune（add_new 15000 就停，不回長）。動機：飽和曲線顯示 414→720k 幾乎冗餘 → 預測 prune 720k→~414k 近乎無損。
```
python main.py fit --config configs/mcmc_2dgs_mc_aerial.yaml -n mcmc_capmax_prune30_b7 --data.parser.block_id 7 --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply --model.density.init_args.cap_max 800000 --model.light_gaussian.prune_steps "[20000]" --model.light_gaussian.prune_percent 0.3
```
10%/50%：改 `prune_percent`（0.1/0.5）+ `-n`（mcmc_capmax_prune10/50_b7）。

### 結果表

| 剪枝% | 剪後點數 | PSNR | SSIM | LPIPS | native 內插 | 優勢 |
|------|---------|------|------|-------|-----------|------|
| 0（no-prune, 720k 天花板）| 720,000 | 22.182 | 0.672 | 0.434 | — | — |
| 10%（2026-06-10）| 648,000 | 22.207 | 0.672 | 0.434 | 22.149 | +0.058 |
| 30%（2026-06-10）| 504,000 | 22.180 | 0.669 | 0.441 | 22.084 | +0.096 |
| 50%（2026-06-10）| 360,000 | 22.208 | 0.665 | 0.449 | 21.843 | **+0.365** |

native 訓練曲線對照：720k=22.182 / 414k=22.043 / 270k=21.510 / 180k=21.136。

**【2026-06-10 完整剪枝曲線結論：三大發現】**
1. **PSNR 剪到 50%(360k)仍無損,打破「~42% 崩潰」預測**。10/30/50% PSNR=22.207/22.180/22.208 差 0.028=純雜訊 → **720k→360k 全程 PSNR 持平**,一半的點對 PSNR 冗餘。
2. **成本在感知指標,優雅遞減**：SSIM 0.672→0.665(剪 50% 才 −0.007)、LPIPS 0.434→0.449(+0.015),單調平緩 = graceful degradation。
3. **importance-prune vs native 優勢「越剪越大」(核心賣點)**：+0.058→+0.096→+0.365 單調放大。物理意義：**預算越緊,「先探索 720k 全域再砍」比「直接低 cap 訓練」價值越大**(留越少,看過完整解空間越關鍵)。compounding 曲線。
4. **✅ 嚴格對照組釘死(2026-06-10):優勢實測比內插更大,三指標全贏。** `mcmc_native360_ctrl_b7`(cap400→落點正好 360k):native 直接低 cap = **21.751 / 0.641 / 0.484**。vs 剪枝 50% 同點數(22.208/0.665/0.449)→ **PSNR +0.457、SSIM +0.024、LPIPS −0.035 全贏**,全遠超雜訊地板 ~0.03。內插預測 21.843、實測 21.751(更低)→ native 曲線微凹、內插**低估**我方優勢(+0.365→+0.457)。對照同 config/同 30k 步/同 densify-until 15k,且對剪枝組不利(native 最終點數跑滿 15k finetune、剪枝只 10k)→ 吃虧還三項全贏 = 硬。**核心賣點完整閉環:importance-prune 用半數點(360k)達天花板品質(無損)+ 乾淨贏直接低 cap 0.46dB。explore-then-prune 不只省點,是找到更好的解。**

---

## 🟢 方向 D：sh3 表徵槓桿（2026-06-11）

**首發（同 cap800/720k、只翻 sh2→sh3、trim/步數照舊）**：`mcmc_sh3_cap800_b7`

| cap800 @ 720k | PSNR | SSIM | LPIPS |
|---|---|---|---|
| sh2（舊天花板）| 22.182 | 0.672 | 0.434 |
| **sh3** | **22.477** | **0.680** | **0.425** |
| Δ | **+0.295** | +0.008 | −0.009 |

**結論**：(1) **sh3 在 720k 沒 OOM → 「22.18 天花板」是 sh2 表徵天花板、非硬體**，真 6GB 天花板 ≥22.48；(2) **SH 是真槓桿,Lambertian 擔憂錯**,方向 D 假設成立;(3) **⚠ 非 VRAM-matched**:sh3@720k 多吃 ~180MB params+Adam → +0.295=「SH+多 VRAM」混合,「同預算」claim 還缺 VRAM-matched 一步。

**✅ 組合牌完成（2026-06-17）：閉環成立，雙贏（更少資源 + 更高品質）。** `mcmc_sh3_cap800_prune50_b7`

| config | 點數 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| sh2 720k（舊天花板）| 720k | 22.182 | 0.672 | 0.434 |
| sh2 360k（prune50）| 360k | 22.208 | 0.665 | 0.449 |
| sh3 720k | 720k | 22.477 | 0.680 | 0.425 |
| **sh3 360k（prune50）** | **360k** | **22.403** | 0.669 | 0.444 |

三對照：(1) **vs sh2 720k 天花板 → +0.221dB，用一半點數 + ~78% VRAM**（sh3 58float×360k=20.9M vs sh2 37×720k=26.6M）→ **兩軸都更省、品質還更高 = strictly 雙贏，繞過 VRAM-matched 質疑**；(2) vs sh2 360k 同點數 → **+0.195dB**，SH 槓桿在 360k 仍成立（720k 時 +0.295，點少略降合理）；(3) **sh3 剪枝 720k→360k = −0.074dB（不像 sh2 完全無損）**，因 sh3 每點帶更多 SH 資訊、單點價值高、砍掉損失稍大，但極小。

**閉環句**：6GB 上 sh3 + 50% importance-prune = 360k/22.403，贏 sh2 滿點天花板 +0.22dB，同時半點數 + ~78% VRAM。效率（剪枝）+ 表徵（sh3）一次閉環，且是「更少資源+更高品質」非 trade-off。

**剩餘可選**：(a) 完整論文還需 25-block 端到端 6GB 驗證 + baseline 壓進 6GB 對照；(b) VRAM-matched sh3 from-scratch（cap~510k→460k）做純表徵 claim（非必要，組合牌已 strictly 雙贏）。

---

### ✅ 嚴格對照組（2026-06-10，已完成，見上 §剪枝結論 #4）

> 已跑完 `mcmc_native360_ctrl_b7`，+0.457dB 釘死。以下為當時規劃留存。

### （歷史規劃）嚴格對照組釘死 +0.096dB

**動機**：+0.096dB「剪枝優於低 cap」目前是**內插值**，是 novelty 最脆的一環。先跑真 504k native 點釘死它，再補 10/50。
**算術修正**：Gemini 寫 cap=504k 但 MCMC 漂 ~10%（cap460→414k、cap800→720k 都 90%）→ 要落 504k 須 **cap=560k**（504/0.9）。同 config/同總步數 30k/同 densify-until 15k，只差「直接低 cap vs 先 800k 再剪」。
```
python main.py fit --config configs/mcmc_2dgs_mc_aerial.yaml -n mcmc_native504_ctrl_b7 --data.parser.block_id 7 --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply --model.density.init_args.cap_max 560000
```
判讀：落點 ~504k 且 PSNR << 22.18 → +0.096dB 鐵的演算法優勢、主線鎖死；≈22.18 → 內插高估、剪枝只是點少 → 退回純工程故事。註：此對照對剪枝組不利（native 最終點數跑滿 15k finetune，剪枝組只 10k）→ 剪枝吃虧還贏則優勢更硬。落點偏離 504k 太多（<470/>540）調 cap 重跑。

### Gemini 學術裁決（2026-06-10，置信「絕對」）
- **§3 剪枝 = micro-novelty 未被佔領**：3DGS 剪枝成熟（LightGaussian/Compact-3DGS）；2DGS 壓縮近期多走 K-means VQ / 分割遮罩 / 特徵網格，**沒人提「3D體積→2D surfel面積 + 原生 transmittance 跨視角累積」且零 gsplat 依賴**。配 6GB MCMC 框架 = 紮實演算法貢獻。
- **6GB city-scale = 絕對藍海**：SOTA（CityGS/VastGS/MatrixCity）全預設 24-80GB 叢集；消費級多停在單房間/小物件或純推論加速（VkSplat）。可標榜 *Resource-Constrained City-Scale Reconstruction*。
- **方向 ROI 排序 A > D > B > C > E**：A 剪枝主線（最快變現）；D 回收 30% VRAM 升 sh2→3 或加密高誤差區（若破 22.18 天花板 → 故事完美閉環「剪枝是為騰資源做更高表徵」）；B host offloading（純系統貢獻強但 C++/CUDA 工程風險高）；C surfel-merge（2DGS 重疊率低、手寫合併易 alpha-blend 失真，CP 低）；E ADMM 徹底放棄（三振停損）。
- **新 2DGS-specific 剪枝準則（Gemini 提案，待評估）**：利用 2DGS 顯式法向+切平面 → **Normal-Consistency Penalty**：空間極近但法向反平行/分歧的 surfel = 物理無意義的 Z-fighting 冗餘堆疊 → 把「法向與局部表面法向夾角變異」併入 V_imp 優先剔除。完全 specific to 2DGS（3DGS 沒法向）。可當 importance-prune 的擴充準則做 ablation。

**【2026-06-10 結論：剪枝兩個判準雙雙過關，是全專案最強正面結果】**
1. **vs no-prune 22.18 → −0.002 dB（雜訊地板內，等於無損）**。720k 剪掉 216k（30%）PSNR 不動 → **這 216k 點是純冗餘**，直接證實飽和曲線預測。SSIM −0.003 / LPIPS +0.007 也都極小。
2. **vs native 同點數曲線 → 504k 剪枝(22.180) vs native 504k 內插(22.084) = +0.096 dB**。importance-prune 在同點數下品質明顯高於「原生低 cap 訓練」→ **剪枝不是只是點少,是真的更會留品質**(先讓 800k 看過完整最佳化、再砍掉貢獻最低的)。這是 importance-prune 的真演算法價值。
3. 機制：cap=800k → densify 到 800k(@15k) → 漂到 ~720k → step 20000 剪 30% → 504k → fine-tune 到 30k(add_new 15000 已停,不回長)。
4. **只跑了 30% 一個點**(使用者沒跑 10/50)。一個乾淨無損點已足以立論;10/50% 可補成完整「剪枝率 vs 品質」曲線找真正的崩潰拐點(預測 ~414k 即 ~42% 處才開始掉)。

---

## 環境前置（每次開新 terminal 先做）

```
conda activate gspl
```

所有指令都在專案根目錄 `/home/LnoArch/Projects/專題/CityGaussian` 下執行。

已驗證可用的關鍵路徑：
- coarse 模型：`outputs/RTG_mc_aerial_coarse_sh2/checkpoints/epoch=6-step=30000.ckpt`
- partitions.pt：`data/matrix_city/aerial/train/block_all/partition/partitions-dim_5_5_visibility_0.08/partitions.pt`
- coarse-init config：`configs/RTG_mc_aerial_sh2_trim24_coarse.yaml`
- depth-init baseline（已有數字）：`outputs/RTG_mc_no_ADDM_aerial_sh2_trim`

---

## 任務 1：A/B —— coarse-init 到底比 depth-init 好幾 dB？

**問題**：診斷 §3 指出 4.64 dB 缺口可能來自「拿掉 coarse pretrain」。直接量。
**設計**：depth-init 那側我們已經有數字（下表），這次只補跑 **coarse-init 的 block 7、16**，同 2 個 block 直接對比。

> 注意：coarse-init 與 depth-init 各自需要不同 config（這是已知的，不是 bug）。
> coarse-init 用 `_coarse.yaml`。所以這是「兩種初始化策略各自最佳 config」的公平對比，
> 不是「同 config 換初始化」。這點報告時要講清楚。

### 指令（單行）

跑 coarse-init 的 block 7、16：

```
python utils/train_citygs_partitions.py -n AB_coarse_init_aerial_sh2_trim --config_dir ./configs -c ./configs --init_mode coarse --coarse_ckpt outputs/RTG_mc_aerial_coarse_sh2/checkpoints/epoch=6-step=30000.ckpt --blocks 7 16
```

> 若上面 `-c` 與 `--config_dir` 衝突報錯，改用最小參數版：
> `python utils/train_citygs_partitions.py -n AB_coarse_init_aerial_sh2_trim --init_mode coarse --coarse_ckpt outputs/RTG_mc_aerial_coarse_sh2/checkpoints/epoch=6-step=30000.ckpt --blocks 7 16`
> （需要 `AB_coarse_init_aerial_sh2_trim` 對應的 config 存在；若腳本抱怨找不到 config，
> 先 `cp configs/RTG_mc_aerial_sh2_trim24_coarse.yaml configs/AB_coarse_init_aerial_sh2_trim.yaml` 再跑。）

跑完讀數字（PSNR/SSIM/LPIPS）：

```
python -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E; import glob,os; [print(b, {t: E(sorted(glob.glob(f'outputs/AB_coarse_init_aerial_sh2_trim/blocks/block_{b}/lightning_logs/version_*'))[-1]).Reload() or E(sorted(glob.glob(f'outputs/AB_coarse_init_aerial_sh2_trim/blocks/block_{b}/lightning_logs/version_*'))[-1])._impl if False else None}) for b in [7,16]]"
```

> 上面那行太繞，直接用這個簡單版：跑完直接叫我讀，或你自己開 tensorboard：
> `tensorboard --logdir outputs/AB_coarse_init_aerial_sh2_trim/blocks`

### 對比基準（depth-init，已有）

| block | depth-init PSNR | depth-init SSIM | depth-init LPIPS |
|-------|-----------------|-----------------|------------------|
| 7  | 21.967 | 0.661 | 0.448 |
| 16 | 23.710 | 0.845 | 0.503 |

### 看到什麼結果 → 代表什麼 → 下一步

| coarse-init 結果（vs depth-init） | 判讀 | 下一步 |
|----------------------------------|------|--------|
| **coarse 高 > +2 dB** | 確認「拿掉 coarse」是 4.64 缺口主因 | 嚴肅考慮**走回 coarse-init + host offloading**，depth-init 降級為加速選項。ADMM 角色重新評估 |
| **coarse 高 +0.5~2 dB** | coarse 有明顯但非決定性優勢 | 試「C-2 輕量全域先驗」——用 depth-init 點雲 merge 出便宜的弱全域 prior，吃回大部分差距 |
| **兩者 < ±0.5 dB（差不多）** | depth-init 沒虧多少，4.64 缺口另有來源（densify/SH/trimming 設定） | 缺口轉去查 config 細節 vs CityGSV2，ADMM 不是重點 |
| **coarse 反而更差** | 意外，可能 coarse config 沒調好或 coarse 模型品質差 | 叫我一起看 coarse 訓練曲線找原因 |

---

## 任務 2：Path B 決定性實驗 —— ADMM 加強版到底能不能贏過雜訊？

**問題**：診斷 §2 指出 ADMM consensus 訊號弱 1000 倍。這次用**高 rho + 多 outer**逼它到有意義的量級，
一次定生死。選相鄰且有真實 boundary 的 block 6+7。

> 重要：現在 coordinator 只支援**固定 rho**（沒有自適應 rho）。自適應 rho 是程式改動，
> 我下次幫你加。這次先用 **固定 rho=0.5（基線 5 倍）+ n_outer=30** 當可跑的代理版本，
> 這已足以把 ||y|| 推高一個量級、回答「方向上有沒有救」。

### 先準備乾淨基線（把 block 6、7 的 step=30000 複製進新實驗，不污染原 baseline）

```
mkdir -p outputs/PathB_admm_67/blocks/block_6/checkpoints outputs/PathB_admm_67/blocks/block_7/checkpoints outputs/PathB_admm_67/admm_state
```

```
cp outputs/RTG_mc_no_ADDM_aerial_sh2_trim/blocks/block_6/checkpoints/epoch=117-step=30000.ckpt outputs/PathB_admm_67/blocks/block_6/checkpoints/ && cp outputs/RTG_mc_no_ADDM_aerial_sh2_trim/blocks/block_7/checkpoints/epoch=157-step=30000.ckpt outputs/PathB_admm_67/blocks/block_7/checkpoints/
```

### 跑 ADMM coordinator（單行）

> **2026-06-06 修正（重要）**：原本這裡寫的 `--config configs/RTG_mc_aerial_sh2_trim24.yaml`
> **是錯的**。那顆 config 的 density 是 `RTGStableDensityController`，不吃 `admm_state_path`，
> 會讓每個 block 的 primal 噴一大串 jsonargparse usage 然後 exit 2（no-op）。
> 正確 config 是 **`configs/dt_admm_gat_mc_aerial.yaml`**（掛 `DtAdmmDensityController`，
> depth-init S2 設定齊全）。
>
> 已做的程式修正：
> 1. coordinator 開頭加 `validate_config()` 預檢 → 用錯 config 會**立刻**一句話報錯，不再噴 usage dump。
> 2. primal 子程序 exit≠0 時改成**印清楚原因並 abort**，不再默默跑完 30 個 no-op outer。
> 3. coordinator 現在把 `--rho` 也注入 primal（`--model.density.init_args.rho`），
>    讓 primal penalty / z-update / dual 三邊用**同一個 rho**（之前 primal 吃 config 寫死的 0.1，
>    跟 CLI 的 0.5 對不上）。

**先跑 dry-run（n_outer=3）確認 per-prop ||x-z|| 有往下掉、沒發散，再開 30：**

```
python utils/dt_admm_coordinator.py --name PathB_admm_67 --config configs/dt_admm_gat_mc_aerial.yaml --partitions_pt data/matrix_city/aerial/train/block_all/partition/partitions-dim_5_5_visibility_0.08/partitions.pt --blocks 6,7 --init_mode depth --depth_init_dir data/matrix_city/aerial/train/block_all/depth_init --admm_state_dir outputs/PathB_admm_67/admm_state --n_outer 3 --k_inner 500 --gat_n_steps 20 --rho 0.5 --device cpu
```

dry-run 健康 → 把 `--n_outer 3` 改 `--n_outer 30` 跑正式版（同一行其他不變）。

> 跑的時候**盯著 coordinator 印出的 `||y||` 和 `||x-z||`**：
> - 健康訊號：`||y||` 隨 outer 穩定上升，越過 ~0.05 進入 0.1 量級
> - 若 `||y||` 卡在 0.01 不動 → rho 還是太弱或 boundary 太少，記下來告訴我

### 跑完比較（ADMM 後 vs 乾淨基線，同 block 6、7）

基線參考（no-ADMM，step=30000）：

| block | 基線 PSNR |
|-------|-----------|
| 6 | 23.498 |
| 7 | 21.967 |

ADMM 後的 block 6、7 PSNR 從最新 ckpt 的 val 讀，或叫我讀。

### 看到什麼結果 → 代表什麼 → 下一步

| ADMM 後（vs 同 block 基線） | 判讀 | 下一步 |
|----------------------------|------|--------|
| **穩定 > +0.3 dB（兩 block 都升）** | **thesis 活了！** consensus 在足夠強度下真的有用 | 擴大到更多相鄰 block 對、加自適應 rho、準備正式 25-block 實驗 |
| **+0.1 ~ +0.3 dB** | 邊緣效果，可能值得但 ROI 低 | 試 z-update 改「gating 選擇」取代「soft 平均」（診斷 §2.2），看能否放大 |
| **< +0.1 dB 或退步** | **consensus-on-raw-features 不 work（即使加強）** | 誠實轉向：ADMM 降為 minor，主力改走 Path A（工程整合）+ C-2（輕量全域先驗） |
| **||y|| 推不上去** | 強度根本拉不起來（boundary 太稀疏） | 重新檢視 boundary graph 構建，或接受「我們的場景結構不適合 consensus」 |

---

## 任務 0（背景，我來改 code，你不用跑）：修評估尺

診斷 §5 的 Gaussian drift 是地基問題，但它是**程式改動**不是跑指令。等你任務 1/2 跑完來找我，
我來做這部分（AABB position penalty 或 visibility-based merge）。你現在只需要知道：
**目前的 global merged PSNR 不可信，所以任務 1/2 都用 per-block PSNR 判讀，先不看 merged。**

---

## 結果填寫區（跑完填這裡，再來找我）

### 任務 1：coarse-init vs depth-init　【2026-06-07 完成（densify 對齊乾淨版）】

**乾淨版（densify 都=0.0002，只有 init 不同，config `AB_coarse_md_aerial_sh2_trim`，兩 block 跑完無 OOM）：**

| block | | coarse_md | depth | Δ |
|---|---|---|---|---|
| 7 | PSNR | 22.373 | 21.967 | **+0.41** |
| | SSIM | 0.638 | 0.661 | −0.023（輸）|
| | LPIPS | 0.496 | 0.448 | +0.048（輸）|
| 16 | PSNR | 24.012 | 23.710 | **+0.30** |
| | SSIM | 0.659 | 0.845 | **−0.186（大輸）** |
| | LPIPS | 0.552 | 0.503 | +0.049（輸）|

**舊的 confound 版（densify=0.00005，已作廢）**：block_7=24.069。從 24.069→22.373 證明 **調粗 densify 掉 ~1.7dB**。

**三個關鍵結論：**
1. **舊 +2.10dB ≈ 80% densify + 20% init。** 純 init 效應只有 +0.3~0.4dB PSNR。
2. **純 init 效應混合**：PSNR 小贏，但 SSIM/LPIPS 兩 block 都輸（block_16 SSIM 0.845→0.659 慘崩）。PSNR/SSIM 背離 → coarse 拉近像素均值但結構/感知更差。**非乾淨勝利，平手偏混合。**
3. **（最重要）4.64dB 缺口主因不是「拿掉 coarse」，是 Gaussian 預算 = 6GB 稅。** 拿掉 coarse 同算力只值 0.4dB；真正大槓桿是 densify 密度（1.7dB），而那密度 6GB 裝不下（block_16 在 0.00005 OOM）。**缺口 = CityGSV2 Gaussian 數遠超 6GB 能負擔。**

**決策樹「coarse 大贏(>2dB)→走回 coarse」被推翻** → 主線改打「6GB 內買更有效的 Gaussian」（GaussianSpa 稀疏化/壓縮/host offloading），不糾結 init。

### 任務 2：Path B ADMM 加強版

**dry-run（n_outer=3）per-prop ||x-z||——先填這個確認沒發散：**

| outer | block | xyz | nrm | scl | opa | sh | 是否往下掉？ |
|-------|-------|-----|-----|-----|-----|-----|-------------|
| 1 | 6 | 0.0405 | 0.0220 | 0.0028 | 0.0095 | 0.0572 | （基準，primal 失敗前）|
| 1 | 7 | 0.0135 | 0.0087 | 0.0008 | 0.0047 | 0.0231 | （基準，primal 失敗前）|
| 2 | 6 |  |  |  |  |  |  |
| 2 | 7 |  |  |  |  |  |  |
| 3 | 6 |  |  |  |  |  |  |
| 3 | 7 |  |  |  |  |  |  |

> 健康判準：outer 2/3 的 per-prop 應**比 outer 1 小或持平**，且沒有任一屬性爆增（>10x）。
> 若 sh 或 xyz 暴衝 → GAT z-range watch item 觸發，停下來叫我調 gat_lr / proj weight-decay。

**【2026-06-06 dry-run 完成（n_outer=3）】判讀：健康，可進 30-outer。**

最終 per-node RMS ||x-z||（/sqrt(N·dim) 正規化，原始 norm 會被 1 萬節點放大誤導）：

| block | xyz | nrm | scl | opa | sh | 判讀 |
|-------|-----|-----|-----|-----|-----|------|
| 6 | 0.0041 | 0.0157 | 0.0024 | 0.0259 | **0.0532** | 每節點殘差都小，無發散 |
| 7 | 0.0041 | 0.0156 | 0.0021 | 0.0287 | **0.0466** | 同上 |

- **沒發散**：每節點殘差皆小（xyz~0.004、sh~0.05），GAT z-range ±400 watch item 沒造成 blowup。
- **||y||**：block6=5.89、block7=5.35（從 outer-1 的 ~0.03 穩定爬升，沒爆沒卡 0）。feat_std 兩 block 一致（預條件器正確）。
- **殘差集中在 sh/opa/nrm，xyz/scl 近 0**：幾何共識成立、外觀（color/exposure 因 appearance embedding 本就不同）共識拉不動 → rendering loss 在 color 上贏。這是預期現象。
- **PSNR（3-outer，非判決）**：block6=23.466（基線23.498，−0.03）、block7=22.012（基線21.967，+0.05），±0.05 雜訊內，符合 3-outer 預期。真正判決看 30-outer。

**【2026-06-06 完成（27 outer，step 31500→45000）】結論：表面 +0.2dB，但幾乎確定是訓練增益非共識，need 對照組釘死。**

| block | 最終 PSNR | 基線 | Δ | 軌跡 | 最終 \|\|y\||(raw) |
|-------|-----------|------|---|------|------------------|
| 6 | 23.650 | 23.498 | **+0.152** | 23.46→23.65 緩慢單調升 | 4.82（dry-run 時 5.89 → 衰減）|
| 7 | 22.176 | 21.967 | **+0.209** | 21.98→22.18 緩慢單調升 | 4.06（dry-run 時 5.35 → 衰減）|

**+0.2dB 幾乎確定不是 consensus 的功勞 —— 致命 confound + 機制惰性，兩條獨立證據：**
1. **LR 沒凍結**：config 只有 `means_lr`(xyz) 排程到 30000、之後 floor；`shs_dc/opacities/scales/rotations` **是常數無排程**。31500→45000 這 13500 步 SH/opacity/scale 仍全速訓練 → PSNR 單調上升是「繼續訓練外觀」的曲線，非共識形狀。
2. **共識惰性**：||y|| 從 dry-run 的 5.89/5.35 **衰減到 4.82/4.06**（27 outer 不但沒建強反而降），per-prop x-z 維持極小。**共識在 PSNR 上升的同時完全沒出力。**
→ `||y|| 惰性 + LR 沒凍結 + 單調訓練曲線` 三者一致 → **+0.2dB 是多訓練 13500 步買的，不是共識。** 與中期診斷一致（raw-feature consensus 在已對齊 boundary 上無效）。

**釘死結論的對照組（便宜，等 coarse 測試後決定補不補）**：block 6,7 從 31500 **純訓練（無 ADMM）跑到 45000**，同 13500 步。
- 對照組也漲 ~+0.2 → 共識貢獻 = 0（全是訓練）；對照組持平只有 ADMM 漲 → 共識有用（但 ||y|| 惰性使這極不可能）。
- 指令（複製乾淨 31500 ckpt 到新實驗名，無 admm_state，跑 main.py fit 到 45000）：待補時叫 Claude 生成。

---

## 一頁版決策樹（給你心裡有底）

```
任務1（coarse vs depth）
 ├─ coarse 明顯贏(>2dB) ──→ 缺口=拿掉coarse造成，考慮走回 coarse+offloading
 ├─ coarse 小贏(0.5~2)  ──→ 補 C-2 輕量全域先驗
 └─ 差不多/coarse輸     ──→ 缺口在別處(config/SH/trim)，深查 CityGSV2 差異

任務2（ADMM 加強版）
 ├─ >+0.3dB 且 ||y||夠高 ──→ thesis 活，擴大+加自適應rho
 ├─ +0.1~0.3            ──→ 改 gating z-update 再試一次
 └─ <+0.1 / 退步 / ||y||上不去 ──→ ADMM 降級，主力轉 Path A + C-2

兩個任務都做完 → 找我 → 我修評估尺(任務0) + 依結果定正式路線
```
