> ⛔ **2026-09-13 重驗後仍不可用。**
> 剪枝結果交接；分數屬錯位期。剪枝的方法論結論已進 v2 §7。
> 判準見 `../倖存清單_2026-09-13.md`；現行權威是 `../研究總覽_v2.md`。

# 交接給 Gemini —— 現況總整理 + 顧慮 + 發想方向（2026-06-10）

> 用途：把這份 + `./紀錄/` 其他檔一起丟給 Gemini，請他 (1) 找能拔高/補強的論文 (2) 找潛在對手（同題目別人做過沒）(3) 對下面「發想方向」給可行性判斷與更好的解法。
> 數學用 ASCII（||x||、sum_i、^、alpha 之類），不要 LaTeX。

---

## 0. 一句話定位

**大專生研究：在 6GB 消費級 GPU（RTX 4050 Laptop, 46GB RAM）上,訓練 city-scale 2DGS,目標是「同品質、低損耗」——不是贏過 baseline 品質,是用更少資源達到同等品質。** 場景 MatrixCity small_city aerial,單 block 為實驗單位(block_7 為主)。底層框架 = CityGaussianV2 (2DGS surfel) on gaussian-splatting-lightning。

硬體是硬約束:所有設計都要塞進 6GB VRAM。46GB 大 RAM 是唯一餘裕(可做 host offloading)。

---

## 1. 走過的路(誠實版,三個演算法點子全平手/負,但摸清了問題結構)

原始計畫核心貢獻是 **GAT-based ADMM Unrolling**(用圖注意力網路取代 DOGS 式 consensus 的硬 Augmented Lagrangian)。實際做下來:

| 嘗試 | 結果 | 關鍵發現 |
|------|------|---------|
| **A. GAT-ADMM consensus**(原核心)| **net-zero**(+0.2dB 但被證明是多訓練 13500 步買的,非共識)| depth-init 後相鄰 block 的 boundary Gaussian 幾何**本就高度一致**→ 沒分歧給 dual 累積 → consensus 無施力空間。\|\|y\|\| 27 outer 不升反降(5.89→4.82)。DOGS 命門也已破:DOGS densify 期間根本不做 consensus(凍結拓撲迴避動態拓撲)。|
| **B. coarse-init vs depth-init**(查 4.64dB 缺口來源)| init 是**次要效應**(+0.4dB PSNR 但 SSIM/LPIPS 輸)| 4.64dB 缺口主因**不是**「拿掉 coarse 預訓練」,而是 **Gaussian 預算 = 6GB 稅**。coarse 的 +2.1dB 舊數字裡 ~1.7dB 來自更密的 densify,而那密度 6GB 裝不下(block_16 在 densify=0.00005 OOM,長到 370 萬點)。|
| **C. base MCMC(3DGS-MCMC 適配 2DGS)** | 平手(cap460→414k = 22.04 vs baseline 453k = 21.97)| 工程成立(取代承重的 opacity-reset、點更少),但 453k 已近飽和 → 高點數比是「假平手」。|
| **D. gradient-guided MCMC relocation**(把回收點導向高誤差區)| **負結果**(cap300 21.503 vs base 21.510)| 重分配同樣數量的點沒用 → 純粹是容量受限,不是「點放錯地方」。|

→ 三個演算法層點子(ADMM / base-MCMC / gradient-guided)全部平手或負。**這逼出一個結論:在這個飽和區,瓶頸是容量(capacity)不是管理(management)。** 使用者據此把目標明確收斂成「同品質低損耗」,不再追求演算法贏品質。

---

## 2. 摸清的問題結構:6GB 天花板 + 飽和曲線(關鍵)

把 MCMC 的 cap_max 從 200k 掃到 800k,得到「品質 vs 實際點數」曲線:

| 實際點數 | PSNR | SSIM | LPIPS | 備註 |
|---------|------|------|-------|------|
| 180k | 21.136 | 0.606 | 0.540 | |
| 270k | 21.510 | 0.627 | 0.512 | |
| 414k | 22.043 | 0.653 | 0.467 | baseline 同級 |
| **720k** | **22.182** | **0.672** | **0.434** | **6GB 天花板(cap800,無 OOM)** |
| baseline 參考 | 21.967 | 0.661 | 0.448 | 453k,RTGStable 無 cap 自然長成 |

**飽和拐點 ~414k**:斜率 180→414k 約 0.4 dB/100k(仍在爬);**414→720k 暴跌到 0.045 dB/100k**(多 306k 點只 +0.14dB,幾乎冗餘)。

三個推論:
1. **414k 是甜蜜點**:天花板 99.4% 品質、57% 點數 →「同品質低損耗」未剪枝就成立。
2. 720k 三項指標全勝 baseline 453k。
3. **414→720k 的點幾乎冗餘 → 預測 importance-prune 720k→~414k 近乎無損。**(已驗證,見 §3)

誠實天花板:vs CityGSV2 的 28.7dB 還差 6.5dB,但這差距來自 **sh3 + 完整 pipeline**(我們跑 sh2 + 單 block 簡化),**不是點數**(點數已飽和)。這是「6GB 上能達到的最佳」,不是「無限 VRAM 下的最佳」——命題要講清楚。

---

## 3. 🟢 最強正面結果:2DGS-native importance pruning(2026-06-10)

**Gemini 上次推的 2DGS 版 LightGaussian 重要度剪枝,Claude 落地實作,實驗雙判準過關。**

演算法(`internal/utils/importance_prune_2dgs.py`,零 gsplat 依賴):
- 貢獻度 contribution:用 Trim renderer 的 `record_transmittance=True` 取每個 surfel 的 per-view transmittance,跨所有 train view 取 top-K 平均(複製 Trim 自己的貢獻定義)。
- 大小 size:`prod(scales, dim=1)` = surfel 面積(2D 類比 LightGaussian 的 3D 體積)。
- 重要度 `V_imp = contribution * area^v_pow`(v_pow=0.1),砍掉底部 k%。

**實驗**:MCMC cap800(→720k 天花板)→ step 20000 剪 30% → fine-tune 到 30k(densify 在 15000 已停,不回長)。

| 剪枝% | 剪後點數 | PSNR | SSIM | LPIPS |
|------|---------|------|------|-------|
| 0(no-prune 天花板)| 720,000 | 22.182 | 0.672 | 0.434 |
| **30%** | **504,000** | **22.180** | 0.669 | 0.441 |

**兩個判準雙雙過關:**
1. **vs 天花板 22.18 → −0.002 dB(雜訊地板內,等於無損)。** 剪掉 216k(30%)點 PSNR 完全不動 → **這 216k 點是純冗餘**,直接證實 §2 的飽和曲線。
2. **vs native 同點數曲線 → 504k 剪枝(22.180) vs native 504k 內插(22.084) = +0.096 dB。** importance-prune 在同點數下品質明顯高於「原生低 cap 訓練」→ **剪枝不是只是點少,是真的更會留品質**(先讓 800k 看過完整最佳化、再砍掉貢獻最低的)。這是 importance-prune 的真演算法價值,不只是 budget control。

→ **論文賣點成形:6GB 上 720k→504k 無損減 30% 點,且 importance-prune 優於同點數的低 cap 訓練。**

(註:只跑了 30% 一個點,使用者沒跑 10/50%。一個乾淨無損點已足立論;補 10/50% 可畫完整「剪枝率 vs 品質」曲線找真正崩潰拐點,預測 ~42% 處(即剪到 ~414k)才開始掉。)

---

## 4. 目前掌握的「成品牌」(已驗證能跑的零件)

1. **depth-init(RTG-SLAM 策略)** + alpha=0.99 不透明 surfel 初始化:省記憶體(稀疏點雲 + densify=0.0002),且 alpha=0.99 讓同品質用更少 Gaussian(不透明 surfel 一片蓋一塊表面,vs 透明的疊很多層)。源碼對照無 bug。
2. **MCMC density controller 適配 2DGS**(`mcmc_2dgs_density_controller.py`):取代承重的 opacity-reset,提供顯式 cap_max 預算控制 + reset-free 平滑曲線。relocation 公式從 CUDA 源碼驗證、2D pad/truncate 數值對拍 diff=0;noise 切平面化(沿 surfel 切平面加噪,法向=0);兩個 L1 reg(opacity_reg/scale_reg)用 opacity>0.9 免疫護城河保護 alpha=0.99 先驗。
3. **2DGS-native importance pruning**(§3):無損減 30% 點。
4. depth regularization(RaDe-GS 式)+ normal loss:幾何抑制 floater(讓 opacity-reset 的 floater 清除功能變冗餘)。

這四個都服務同一目標:6GB 內塞下更有效的 Gaussian。

---

## 5. 我的顧慮(Claude 的誠實 flag,請 Gemini 幫忙判斷)

1. **演算法 novelty 薄。** 三個演算法點子全平手/負,目前最強的是 importance-pruning,但那是 Gemini 推的 LightGaussian 適配,**「2DGS 版 importance pruning」是否已有人發表?** 若有,我們的貢獻退化成「工程整合 + 6GB 可行性驗證」。**請 Gemini 重點查:2DGS/surfel-specific pruning、transmittance-based importance、24H2~25 arXiv 的 city-scale 壓縮。**

2. **單 block 的結論能不能推到全場景?** 所有數字都是 block_7(或 6+7)單 block。25-block 合併時有 Gaussian drift 問題(merged PSNR 不可信,評估尺待修)。**「6GB 上 720k 單 block」乘 25 block 是否仍是 6GB 命題?** 我們靠 divide-and-conquer 逐 block 訓練,每次只有一個 block 在 VRAM,所以理論上成立,但沒端到端驗證過完整 25-block 在 6GB 跑通 + 合併。

3. **「同品質低損耗」的對手是誰?** 我們省的是 VRAM/點數,但 vs 什麼 baseline?CityGSV2 本身要 24GB+?DOGS 要 24GB+?**請 Gemini 確認:目前 city-scale 3DGS/2DGS 論文的最低 VRAM 門檻是多少,有沒有人專門做「消費級 GPU(<=8GB)city-scale」這個卡位。** 上次查說是藍海,但投稿前要釘死。

4. **importance-prune 的 +0.096dB「優於低 cap」會不會只是 fine-tune 步數差?** 剪枝版先訓到 800k 再剪再 fine-tune,native 版直接低 cap 訓練 —— 兩者總訓練步數/有效梯度更新不完全對等。這個 +0.096dB 是不是乾淨的「演算法優勢」需要更嚴格的對照(同總步數)。請 Gemini 想想怎麼設乾淨對照。

5. **天花板距 SOTA 6.5dB。** 即使「6GB 同品質低損耗」成立,審稿人會問「那品質呢」。我們的答案是「飽和、瓶頸是 sh3/pipeline 不是點數」,但這需要把 sh3 也在 6GB 上試一把確認(目前沒做,sh3 記憶體更大可能塞不下,反而強化「6GB 的代價」論點)。

---

## 6. 發想方向(請 Gemini 評可行性 + 找對應論文)

### 方向 A:把 importance-pruning 做成主線貢獻(目前最有料)
- 完整化「剪枝率 vs 品質」曲線(補 10/50%),找崩潰拐點。
- 嚴格對照:剪枝 504k vs native 504k,控制總訓練步數,釘死 +0.096dB 是演算法優勢。
- **求 Gemini**:(a) 2DGS/surfel importance pruning 有無前作?(b) 有沒有比 `contribution*area^v_pow` 更好的 2DGS 重要度準則(例如考慮法向一致性、深度貢獻、surfel 重疊)?(c) 迭代式剪枝(prune-finetune 多輪)vs 一次性剪枝在 city-scale 上誰好?

### 方向 B:host offloading 把「6GB 跑 city-scale」做成系統貢獻
- 46GB RAM 是餘裕。GS-Scale 式 host offloading:VRAM 只放當前 active 的 Gaussian/optimizer state,其餘 offload 到 host RAM。
- 結合 freeze(穩定 surfel 凍結、釋放 optimizer state)+ 殘差壓縮。
- **求 Gemini**:消費級 GPU 上 3DGS 的 offloading/out-of-core 訓練最新論文(GS-Scale 之外),哪些能直接接 2DGS。

### 方向 C:surfel merge(EAGLES 式合併)取代/補充 pruning
- 上次討論 surfel-merge vs LightGaussian 哪個 ROI 高。pruning 已驗證有效;merge 是把相鄰共面 surfel 合成一個(2DGS 特別適合,因為 surfel 有明確法向+切平面)。
- **求 Gemini**:(a) EAGLES (arXiv 2024) 的 merge 機制細節,是否 2DGS-native?(b) 「共面 surfel 合併」有無專門前作?這個若 novel 且配合 2DGS 的幾何特性,可能比 pruning 更有故事。(c) merge vs prune 的品質/壓縮率 trade-off 文獻怎麼說。

### 方向 D:回收容量做別的(已知瓶頸是容量)
- §2 證明瓶頸是容量。pruning 省下 30% 容量後,這 30% 可以拿來:升 sh degree(sh2→sh3 局部)、或加密高誤差區、或塞更大 block。
- **求 Gemini**:在固定 VRAM 預算下,「省點數換更高 SH」vs「省點數換更多點」哪個 ROI 高,有無 ablation 前作。

### 方向 E:ADMM 不死,改 regime(留作備案)
- ADMM 在「已對齊 boundary」上 net-zero,但若改成 **ResNet 殘差式 z-update**(z = x + delta,delta 由 GAT 學)+ **乾淨 boundary 指派(YOLO 式)**,可能在「拓撲/分歧大」的 regime 復活。目前 depth-init 後 boundary 太一致 → 沒分歧。
- 詳見 `紀錄/ADMM_改法清單.md`。**低優先**,除非 Gemini 認為 consensus 還有救。

---

## 7. 給 Gemini 的具體問題(最想要的回答)

1. **「2DGS/surfel-specific importance pruning」是否已被發表?** 若有,列出論文 + 我們和它的差異點。這直接決定 §3 是不是 novel。
2. **消費級 GPU(<=8GB)city-scale 3DGS/2DGS 訓練,目前有沒有專門的論文卡位?** 最低 VRAM 門檻的 SOTA 是誰?
3. **§5.4 的 +0.096dB「剪枝優於低 cap」,怎麼設乾淨對照釘死它是演算法優勢而非訓練步數差?**
4. **方向 A~E 哪個最可能在大專生時間/算力內做出「可投稿的單一清楚貢獻」?** 給排序 + 理由。
5. **有沒有我們沒想到的、利用「2DGS surfel 有顯式法向+切平面+面積」這個特性的壓縮/剪枝/合併準則?** 這是 2DGS vs 3DGS 的結構差異,可能是 novelty 來源。

---

## 8. 配套檔案(都在 ./紀錄/,一起給 Gemini)
- `本次session待辦.md` —— 所有實驗的指令 + 結果表(含本次剪枝)
- `主線_Gaussian效率.md` —— 四軸預算框架(效率 / prune / relocation / freeze / 壓縮)
- `MCMC整合分析.md` —— 3DGS-MCMC 機制 + 2DGS 適配細節 + 學術查證(§7 引用清單)
- `gradient_guided_mcmc_spec.md` —— gradient-guided 消融規格(負結果)
- `ADMM_改法清單.md` —— ADMM 復活藍圖(方向 E)
- `期末診斷.md`、`DOGS_源碼對照與命門驗證.md` —— 方向批判 + DOGS 命門
- `實驗記錄.md` —— 完整實驗數字 + 根因分析
- `reference_papers.md`(在 memory/) —— 12 篇現有參考論文對應表
