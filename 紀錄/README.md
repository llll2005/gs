# 紀錄總覽（單一來源入口）— 2026-07-06

> **維護規則**：**機制／公式變動 → `現行方案與公式.md`（現行的唯一來源，看那份就知道我們在用什麼）**；
> 新實驗結果 → `實驗總表.csv`（跑 `python3 紀錄/tools_make_master_csv.py`
> 重新收割 outputs/ 後補註記）＋當期開發日誌；線路退役 → `廢案彙整.md`；
> 現況/規劃變動 → 只改本檔。其他文檔不再各自維護「現況」。
>
> **三份入口，各一個問題**：「我們在用什麼」→ `現行方案與公式.md`；
> 「有哪些問題、怎麼應對」→ `問題與應對總表_2026-07-23.md`；「接下來跑什麼」→ `待辦與半成品清單.md`。

## ★方法論定案（2026-07-12，塑造整篇論文比較結構）
**不用 GT → SfM held-out + 自跑 CityGS baseline，27.23 降為脈絡參考。** 詳 memory `eval_protocol`。
- **不用 GT**：目標真實場景重建（無 GT 位姿）；GT-free 是賣點。官方 GT test set 框比我們 COLMAP 框
  大 ~150×(Sim3 不對齊,繞開)。現有 `outputs/*/test/` 其實是 val 影像非真 held-out。
- **CityGS V1/V2 baseline**：用「自跑」（同 SfM held-out/6GB/GT-free）不用「論文 27.23」（8×A100/GT test,不可直接比）。
- **新北極星**：GT-free 6GB held-out 協議上**贏自跑 CityGS V1/V2**（非命中 27.23,後者寫 related work 背景）。
- **held-out 取法**：`split_mode=experiment`（排 every-8th 出訓練,在排除的上評,同 SfM 框免對齊）;
  我方+CityGS baseline 都在此模式重跑（一次性定稿）。⚠ 自跑 CityGS 小心 densify config。

## 0. 命題（權威=`方向重定調_2026-07-03.md`）
消費級 6GB 筆電上的 city-scale 高斯重建：以每顆 primitive / 每 byte VRAM 的品質為中心。
硬約束：**GT-free**（方法端只用 images+SfM+偽深度）、只用當前資料（2026-05-29 後）、
RTX 4050 6GB / 46GB RAM。六缺陷 D1-D6 見權威文件 §2。

**論文敘事候選（2026-07-06 新增，Gemini 背書）**：
「**成本感知密度控制（Gaussian 經濟學）**」——現有機制只看價值訊號（梯度/貢獻/opacity），
無人看成本訊號（screen-area×可見度 ∝ 相交數 ∝ VRAM/時間）。b12 近相機怪物 = 活證據；
screen-size prune = 第一塊磚；統一 formalism = 預算約束下 value-per-cost 分配（影子價格
λ 可 dual ascent——ADMM 數學復活在活約束上）。⚠ 寫作前必過 G2 查證：最近先行者
**Taming 3DGS**（預算=顆數）；我方切割點 = **顆數是錯誤計價單位**
（b12 0.32M 顆 OOM vs b8 1.53M 顆存活 = 同顆數 VRAM 差 3 倍）。

## 1. 現況快照
### ★2026-07-13~17 大事記（詳=`archive/統一預算框架_演化史_2026-07-12~28.md` 末段）
- **封閉預算公式驗證+漏洞**：`N_max=(Vt−V_os)/(4MF+γτ/K)` K=1 預測 1.70M 零校準命中實死點；
  但 **τ 訓練中漂移 73→~120**（K2 凍結測試 1.8M 爆）→ static init-τ 偏樂觀，須 τ_growth 修正（~1.65）。
- **K8 v2-v5 迭代（使用者跑,gsplat 線）**：K=8 供給側技術成功（2.5M 顆、97.9M 相交怪物 peak 3.35G）；
  但品質負結果（60k+幾何正則 21.5 < beta 500k 的 21.83）；動態天花板+ω 風險溢價=churn 淨負。
- **⚠使用者抓出 confound（07-17）**：「b12 更多點無益」=跨 kernel 比較不乾淨；trim 半數顆數(1.0M=22.45)
  贏 gsplat(1.9M=21.5)=kernel 品質主導；**同 kernel count ablation 從未做過=開放問題**。
- **SB color 移植主線完成（07-17）**：DBS 的 Spherical Beta color（**不含**淨負的 beta falloff）
  → `Gaussian2DSB`+`SepDepthTrim2DGSSBRenderer`+`configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml`；
  **F 59→25 floats/pt**（模型狀態省 58%）；smoke b7 1500 步全綠（機制/densify/ckpt/色彩路徑）。
  kernel 顯示橫幅 `[kernel] renderer=... color=... floats/point=...`。指令=手冊 §9。
- **逐塊 cap 校準工具**：`tools/calibrate_block_caps.py`（taus/probe-trim/caps 三子命令）
  → `紀錄/block_taus.csv`（病理排名）+`紀錄/block_caps.csv`（cap 表）。
- 收割器重建：`紀錄/tools_harvest.py`（07-06 大整理誤失）；總表全註記（跑 `tools_make_master_csv.py` 重生）。

### ★★2026-07-18~20 大事記（凝結偵探 + 標尺事件 + 怪物防禦）
- **★★殭屍經濟/兩相動力學（07-19,五回合 Gemini 對撞,原始五回合已歸檔 `archive/雙峰opacity_*`,結論綜合於本檔+`問題與應對總表` B1/B2）**：從「雙峰 opacity 提案」
  一路 debug 出 MCMC 收斂的**兩相結構**——densify 期=分散表徵(15 萬顆扛 95% 畫面,零殭屍)、
  **42k relocation 關閉後大凝結**(質量濃縮進 7 千顆,+2.75dB 跳升,輸家被雙 L1 碾成塵埃且無人收屍→87% dead weight)。
  證據鏈:直方圖(97%霧)→法向一致性(fog 0.971>anchors,隨機 init 自對齊)→質量審計(0.8% 點扛 95%)→時間線審計。
  假說三代皆斃(千層酥/停車位錯位/最後=收割凝結)。**手術臂實錘:收割引擎=外觀收斂,凝結是副現象**
  (freeze/noreg≈平,我方「凝結阻斷-2.4dB」預測錯/Gemini 對=相關≠因果教訓)。
- **★塵埃處理落地(07-20)**：部署級 `tools/cull_dust.py`(o<0.05&subpx→**7.4× 無損**,位元級相同;
  25 塊 merge 22.5M→3M 入場券,加 `--min_view_dist` 保守模式+`--ply` 目視)、訓練中 `harvest_dust_trim_interval`
  (收割期週期剪,待驗)、`tools/audit_working_set.py`。⚠清塵只驗訓練分佈內視角,novel-view 拉近未驗(sub-pixel 是視距相關)。
- **★★標尺斷裂事件+癒合(07-19,權威=`新標尺重錨_2026-07-19.md`)**：K-strip 需主點偏移→trim rasterizer
  cx/cy 寫死→pp_shifty 補丁重編→發現**三月實裝二進位≠requirements pin(9eefc03)**(差 0.4-1.5dB)。
  考古:三月裝的是當時 HEAD **4c6c576**(比 pin 新兩 commit:NEAR_PLANE 修復+**螢幕邊界剔除=trim 抗怪物真機制**)。
  **解=4c6c576+pp_shifty 合體**→dust 對板 17.93 全復原→**歷史數字全部有效**。源存
  `submodules/diff-surfel-rasterization-trim-pp`+`pp_shifty.patch`。教訓:**pin≠實裝;wheel cache=時光機**。
- **K-strip 主線機制完成**：`internal/utils/strip_cameras.py`+`gaussian_splatting.py train_strips`+per-strip
  manual_backward(h/H 加權 reg 精確)+pp_shifty rasterizer。條帶等價 1.4e-4。**但 K=2 對 b12@2M 不足**。
- **★count ablation 收官(結論明確)**：A=SB@1.0M=21.99(舊尺)/A' 重訓中;**B(2M K=1)與 B'(2M K=2)皆 OOM**
  (單幀 4.74G binning). **Y「更多點有沒有用」早在 b7 答完**(見 [[cheap-lever-scaling]]:乖塊 count 見頂
  ~1.5M/24.3,SB→+0.78 推估). **b12 的缺口=怪物(工程)非 count**→count ablation 該在乖塊(b8)不在 b12。
- **★怪物防禦 CPU 預測(07-20,`tools/predict_monster_defense.py`)**：b12@2M=**全體高 overdraw(scenario S)
  非少數巨獸**(最壞視角 tiles/pt 平均 182/top 0.1% 只扛 3.4%)→①殺超標點失敗(50%幀只殺 8 顆)、③半徑鉗制傷正常幾何。
- **★★K-strip 掃描=正解(07-20,使用者追問「K 調大不就好」→對)**：worst view **concentration 只 1.07-1.25x**
  =負載幾乎完美均勻→**K 精確按 1/K 縮**。死亡點全幀~424M→**K=8→1.2GiB 配 SB 省的模型記憶體容得下**。
  **對 b12 分散且合法負載,K 是主力(零品質損失,代價=時間 K×preprocess≈55h/60k),緊急剪(`screen_prune_emergency_px`)
  只是廉價保險**。修正:曾誤判「空間集中打穿 K」(舊 gsplat 誤植)+曾推薦緊急剪為主力,皆錯。
  **當前 kernel=SB 25 floats/pt(SH3 是 58),07-17 已移植,A/B/B' 都在用**→SB 省模型記憶體+大 K=一致正解。

### ★★2026-07-21~24 大事記（動態 K 估計器 + 兩次修正，權威=`現行方案與公式.md` §2 供給側 + `問題與應對總表`）
- **A'(SB@1.0M K=1)完成=22.12**（合體二進位,乾淨錨點;≈舊 A 21.99;SB 比 SH3 低 0.33=F 槓桿代價,待顆數補）。
- **動態 K 實作+兩次修 bug**：`dynamic_strips`（每步投影估負載→預算公式反解 K）。
  (1)**v1 量錯維度**：只估 binning load,漏 backward activation(∝N)→全程 K=1 撞死。標定峰值模型
  `peak=model+A·N+(B·N+γload)/K`(A≈500,B≈1550)修正→死點判 K=3/怪物 K=8。
  (2)**v2 近軸近似**：`r=3fx·s/z` 漏離軸透視 Jacobian 拉伸→低估 1.35×(離軸極端 10×)→死於怪物區 34k/1.55M。
  (3)**v3 Jacobian 拉伸修正**：`×sqrt(1+(x/z)²+(y/z)²)` 物理項→est-vs-real 翻成保守過估 1.5×(安全上界)。
- **✅ 已驗證**：K 砍 render 有效(VRAM 三桿表 K=4 −54%)、Jacobian 拉伸(est-vs-real 保守上界)、
  checkpoint 排除(CUDA buffer 管不到,+3.7%)、b12=scenario S(無單一巨獸,半徑偵測證偽)。
- **🟡 進行中**：stretch 版 dynamic-K 能否讓 b12@2M 活到 60k(前兩版低估→死)。**結論未定,勿引用。**
- **踩坑分類**：死路方法(半徑偵測/checkpoint/offload-SB/動態天花板)→`廢案彙整` §11;
  純程式/流程 bug(stale pycache/量錯維度中間版/Gemini 降模型算錯)→不入檔(與理論無關)。
- **2026 3DGS 對照**：診斷驅動(趨勢4=Gaussian ID/每像素貢獻者)我方領先(audit_working_set 等);
  訓練/推論 VRAM 分報(趨勢3)該補入敘事(cull_dust=推論側/dynamic-K=訓練側);pose-free/多感測器正交。

### ★★★2026-07-28 大事記（單日：churn 決定性結果 + 11 篇論文核對）
- **★★churn 消融決定性開獎**：`reg000`(opacity_reg=0) ＝ **23.158 / 0% churn / 4.34 it/s / 1.97G**
  vs A'(0.007) 22.117 / 63% / 0.40 it/s / ~5G → **+1.04dB 且快 10.9× 且省 3G，三軸全贏**。
  **理論解釋（讀 MCMC 原文後）**：opacity L1 的角色是「製造死點供 relocation 回收」，拿掉＝relocation 實質關閉；
  而 MCMC 的賣點是**初始化不變性**，我方 depth-init 已解掉初始化 ⇒ **relocation 與 depth-init 是替代品不是互補品**。
  ⚠ 保留：`cap_max` 綁住顆數才無害；更高 cap／merge／怪物情境會失控 → **這正是 DAR 的立足點**。
- **DAR 統一臂**：smoke 過（churn 8.2% vs 63%），60k 全長臂跑中，**要贏的是 reg000 的 23.158 不是 A' 的 22.117**。
- **★★11 篇論文逐篇核對（`論文核對表.md`）**：找到 **LightGaussian Eq.4 的 `min(·,1)` 截斷整個缺失**（已修，
  正是我方怪物病理的解藥）；**MCMC Eq.8 印刷符號與其敘述相反**（我方實作正確）；
  **DBS Table 3 沒有任何一列是我方組態**（SB −0.46dB 有原文根據：distant background + 無 specular）；
  修正我方**五個 overclaim**；**3DGS 自己命名近相機 floater 並有三道解藥、被 MCMC 全刪**。
- **★★★今日最大槓桿（`論文核對表.md` §11）**：**精確支撐半徑 `d=sqrt(2ln(255·o))` 的 tile 剔除**。
  無損已驗（forward/backward 都已跳過 alpha<1/255）；估 **4.30× 少相交 → N_max ×1.80（b12 1.70M→3.05M）**
  ⇒ **可能直接解開 b12@2M 且不需 tiling**。~2 行 CUDA。⚠ 先前「收緊框救不了 b12」太寬已修正
  （框形狀 1.14× vs 貢獻剔除 4.30× 是兩種機制）。
- **統一形式定稿（`現行方案與公式.md` §7）**：密度控制 ＝ **背包問題** `max Q(θ_S) s.t. (1/K)Σc_i ≤ B`；
  **九篇全部 `c_i≡1`，差別只在 `v_i`；我方是唯一有非平凡 `c_i` 與供給側 `K` 的**。
  三個誠實邊界：背包涵蓋不了「長」（成長需要不存在之點的邊際價值）／價值不可加（alpha-blending 次模）／預算非靜態。

### 最佳數字（block_7, 當前資料, eval down1.2/0.08）
| run | PSNR/SSIM/LPIPS | 點數 | 備註 |
|---|---|---|---|
| **mcmc_60k_sh3_aggr17_prune_b7** | **24.47/0.770/0.284** | ~1.2M | ★帳面最佳=aggr17+LightGaussian 20%prune@30k |
| mcmc_60k_sh3_aggr17_b7 | 24.30/0.755/0.309 | 1.53M | 主基線（quad 的 arm0） |
| sf_mcmc_60k_sh3_aggr17_b7 | 24.24/0.757/0.303 | 1.53M | SF-MCMC arm1（統計平手） |
| citygs_ft60k_coarse3x (b7) | 24.16/0.666/0.454 | — | 官方式 coarse+ft 對照（SSIM/LPIPS 大輸） |

論文錨點：CityGSV2 全域 27.23（8×A100）、2DGS 原版全域 21.35（論文 Table 1，解析度未對齊）。

### quad {7,8,12,13} ✅ 完結 + merge 量測完成（2026-07-08）
- b7 24.30/0.755/0.309、b8 23.62/0.705/0.430、b12 22.45/0.594/0.740、b13 23.87/0.690/0.474
  （per-block 平均 23.56；偏差：b13=+screenprune300、b12=+screenprune300+cap1.0M）
- **★D4 首次量化**：merge（官方裁切,1.18M 顆）後 naive 全圖 val 12.8-14.9（孤島效應污染，
  勿引用）；**公平量測（footprint 覆蓋的 14 視角）= merged 15.87 vs single 22.88 =
  掉 7.0dB**。機制=獨立訓練（depth-init 無共享骨架）的跨塊擁有權不相容。
  工具 `tools/eval_quad_merge_footprint.py`；詳 SF 計畫 §6。
- **核心張力確立**：depth-init 單塊強（24.3）拼接爛（−7dB）vs coarse-init 單塊弱（22.27）
  拼接穩（官方設計）→ D4 研究空間 = 兩者兼得
- **SF 線終局（07-08）**：三臂平手（24.299/24.235/24.283 @1.53M）→ 按預註冊關閉；
  簡併洞見+churn 稅=論文機制素材（SF 計畫 §7）
- **★consolidation 有效（07-08）**：merged.ckpt + 829 聯集純 train 影像 10k 步（54min）
  → 14 held-out 視角掉分 **7.01→3.41（治癒 51%）**；殘差主力=出 quad 缺失背景
  （densify 關閉無法補洞,25 塊版按構造消失）。順手修 merge 工具 optimizer mask bug
  → **merged ckpt 可續訓 = 25 塊 consolidation pass 路線打通**（SF 計畫 §8）
### ★P5 探針酸性測試過關（2026-07-09，probes/p5_cost_aware/）
- 訓練器 = gsplat 1.4 rasterization_2dgs(packed) + MCMCStrategy，資料/val 尺與 CityGS 端同
  （parser 複用）；smoke b7：**峰值 VRAM 3.19G @1.67M**（CityGS 端 1.53M 要 5.7G）
- **b12 酸性測試**：stock MCMC 死 @~1600（isect_tiles 爆,= 病理不是 rasterizer 稅而是
  密度控制成本盲區）；cost-aware 迭代 v1(λ對尖峰遲鈍)→v2(+ceiling,輸在 relocate 保 N)
  →v3(+買/調/賣三檔,SELL 生效仍遲)→**v4(+儲存定價=隱形點變可賣資產) = 活著完跑：
  22.56 PSNR @61,110 顆/1.24h**，vs CityGS 端倖存版 22.45 @0.93M/23.5h（同 val 尺,
  recipe 不同）——**品質平、點數 1/15、時間 1/19；b12 從來不需要百萬顆,價值盲控制
  囤積庫存把自己撐死**
- 控制器設計定型：value/(render_tiles_EMA + storage_cost) < λ 處決（每事件≤10%）+
  cost>2000 tiles 硬天花板（尖峰殺手,不受上限管）+ λ dual ascent on 相交數預算 +
  BUY 僅在 λ=0 且 usage<0.7×budget + SELL（真刪,ops.remove）當 λ>1
- **★b7 前緣結果（07-10）**：stock cap1.7M 死@~4200（真容量,兩底座四驗證=價值盲控制
  在 6GB 不可行）；**cost-aware 23.83@988k/4.09G/3.2h**；**stock 人工配給 1M 子集 =
  24.31@1M/2.91G/3.17h = 反超 cost-aware +0.48（churn 稅+生存預算勒容量）且
  ★★打平 CityGS 冠軍 24.30@1.53M/5.7G/7h——一半 VRAM/時間、無 depth reg**
- **Gate 裁決：過——但過關的是 gsplat 底座（同品質 VRAM −49%）,非 cost-aware 控制器。**
  cost-aware 定位修正：品質非贏家,價值=可行性自動化（b12 自主活/1 次 vs 人工 5 次搶救）
  + 機制素材（stock 無向下管理算子/儲存負債/怪物經濟學）
- **★選點測試終局（07-10）**：b12 stock 隨機 61k = **22.65@0.59G 峰值/1.03h** ≈ cost-aware
  22.56（雜訊級）→ **「固定點數下選點/放置=net-zero」第五次確認**（gg/edge/SF/arm1/economics）。
  b12 病理在「對的點數+對的底座」下蒸發（vs CityGS 端 5 次搶救/23.5h/0.93M 的 22.45）。
  **P5 三結論**：(1) 點數是主變量 (2) 真問題=「每塊多少點」無先驗——cost-aware 定位
  收斂為**預算自動探測器**（λ 均衡一次找到 61k,再用 stock 重訓收割品質）(3) gsplat
  底座效率碾壓。**P5 探針完結。**
- **consolidation v2 判負（07-10）**：重開 densify 的補洞版 92% OOM 死（cap 1.7M 對
  4-cell union 內容過高=設計失誤）且死前 union-val 16.03 << v1 的 19.97 = 重開
  densify churn 在 merged 模型上淨傷害。**v1（10k 純精修）= consolidation 配方定案。**
- **M2 判決局（07-10 深夜）**：depth-SSIM+normal reg 移植版 = 24.16 **< 24.31 無 reg 基線
  （−0.15,負收益）**——CityGS 端「+0.5」本有 confound（與 60k recipe 綁著改）;depth reg
  定位=幾何交付/重 floater 塊用,PSNR 跑預設關。遷移裁決不變（gate 由無 reg 24.31 過）。
- **★★P1 Beta 探針（07-11 凌晨,官方 repo 吃 probe_b7）**：
  | beta@30k 步數對齊 | **24.72**/0.765/0.319 | 1.00M | **30 floats/點** |
  | beta@best(107k 步/3.1h) | **25.81/0.791/0.289** | 1.00M | 30 |
  vs 我們 gsplat 24.31@1M/59f、CityGS 冠軍 24.47@1.2M/59f →
  **同點數同步數 +0.41、每點記憶體減半（sh0+2 Spherical Beta lobes）、加練到 107k 再 +1.1
  （反證我們 30k 欠訓練）**。P1 gate 大幅通過 → Beta kernel+SB 顏色=搬上 surfel 的頭號素材。
  ⚠ iteration_best 選擇準則待查（可能 test-peeking;步數對齊版無此慮照樣贏）。
- **探針工程資產**：`tools/export_probe_scene.py`（COLMAP 塊匯出+val_names 協議）、
  `data/probe_b7`（217k 點）/`probe_b7_small`（70k）、三 repo val 協議 patch、
  **--data_device cpu**（這些 repo 預設把全部影像搬 VRAM ~3.3G=屢敗主因之一）。
  triangle/convex 帶此修復重跑中（round 7）。
- **★kernel 三探針終表（07-11 晨,b7 同 val 尺 30k 步）**：
  beta 24.72（@107k=25.81/0.791/0.289）1M×30f/0.9h ＞ 我們 24.31 1M×59f/3.2h ＞
  CityGS 24.47 1.2M×59f/8h ＞ convex† 23.44 549k×69f/2.8h ＞ triangle† 22.65 ≤500k/1.1h
  （†讓步配置:70k init;triangle 另 500k cap）。**Beta 品質/記憶體/速度三軸制霸。**
  疑點待查:beta iteration_best 選擇準則。工程備忘:--data_device cpu 必帶(全影像進
  VRAM 3.3G)、beta 依賴=fused_ssim+plas+nerfview、fork gsplat 同名→_pylibs 隔離。
- **★★Beta 深挖（07-12,b7 iteration_30000 同尺）**：
  - **b12 酸測=beta 也死**（Iter 3400,isect_tiles 798MB,峰值 5775MiB）→ **近相機怪物病理
    跨表徵/跨實作普遍成立**：三底座（CityGS trim / gsplat stock / beta DBS）× 死亡區間
    ~1600-3800 全中。**唯二活過 b12 的都是成本感知**（我們 cost-aware 22.56 / 人工配給 61k
    22.65）→ cost-aware 論文敘事最強拼圖：品質冠軍 beta 自帶 MCMC densify+cap 仍躲不過。
  - **★消融分解定稿（b7 iteration_30000 同尺）**：
    | config | kernel | color | PSNR | SSIM | LPIPS |
    |完整 beta | 可學-Beta | 2 SB lobes | 24.72 | 0.765 | 0.319 |
    |noSB | 可學-Beta | 純 DC(sh0) | 23.97 | 0.749 | 0.337 |
    |gaussK | 凍結-Beta | 2 SB lobes | **24.88** | 0.767 | 0.318 |
    |我們 gsplat | Gaussian | sh3(48) | 24.31 | 0.753 | — |
    **(1) SB 色=+0.75**（24.72−23.97,2 lobes 打 48 SH 係數,半記憶體）;
    **(2) 可學 kernel=−0.16**（24.72 vs 凍結 24.88）→ **churn 稅第 6 次確認**（beta≈0.00 幾乎沒學離初始,多一個可學參數只添噪）;
    **(3) 30k 最佳=凍結 kernel+SB 色=24.88=+0.57 勝我們 gsplat**。
  - **⚠ gotcha**：beta train.py `--eval` 下 `while True` 無限迴圈(L79-81)→存檔即殺;
    eval.py 要帶對應 `--sb_number`(否則 load_ply assert)。
  - **★底座裁決（待使用者拍板,方向級）**：分解證明**兩個值錢零件（SB 色 + 固定非高斯 kernel profile）
    都不需要 beta fork 的可學-beta 機器 → 推薦「摘零件上 gsplat 1.4 surfel + 保留 cost-aware」
    而非整包搬 fork**（保住控制器/工具鏈/1.4,只加兩配料）。⚠ 保留項:此為 30k 快照,
    full-beta @107k=25.81,凍結版長訓是否同爬未知(但可學 kernel 30k 淨負是強訊號)。
  - **★60k 長訓（07-12）**：cap1m beta 24.70(30k)→**25.14(60k,+0.44)**→25.81(107k)=持續爬
    → **我們全專案數字（皆停 30k/60k）系統性欠訓練**,重跑基線都該拉長步數。
  - **★★★cap 4M 天花板被打破（07-12）**：**beta 在 6GB 塞下 4,000,000 顆,峰值僅 5794MiB,
    存活過 30k**。對照:我們 Gaussian gsplat b7 cap1.7M 直接 OOM 死、CityGS 卡 ~1.5M。
    SB 色省記憶體(30 vs 59 floats)兌現=2.6× 顆數。**改寫 27.23 算術**:先前顆數上限建於 1.5M 封頂
    (推估到頂+0.8);4M 可用→純顆數增益 0.8·ln(4M/1.5M)≈+0.78 且落在真 CityGS regime。
    **4M 品質（07-12,單獨評測）：4M@30k=25.85/0.803/0.272、4M@60k=26.61/0.825/0.246**
    （仍在爬,30k→60k +0.77）。**距論文 27.23 僅 0.6**。beta count 斜率 ~1.06dB/e-fold
    （比我們舊 Gaussian 0.8 陡）→ 外推 ~7M 顆 @60k 可觸 27.23。
    ⚠⚠ **三重保留(勿過度宣稱)**：(1)**val=reconstruction 模式(val⊂train,測到訓練過的圖)→
    對真 held-out 27.23 是樂觀的**,寫作前須跑 b7 test partition(109 視角); (2)per-block b7
    vs 論文 global(per-block 有 premium,非同尺); (3)beta fork 非我們 stack,且 b12 病理塊
    仍死(需 cost-aware)。但「4M 可上 6GB」與「顆數槓桿解封即兌現」兩事鐵證。
  - **★★合體實驗 cost-aware×beta on b12（07-12,進行中）**：把 CityGS 端 screen_size_prune
    移植進 beta fork（`cost_aware_patch.py:ScreenCostState` + train.py flag `--screen_recycle_px`
    + beta_model.py render 加 `radius_clip`）。**v1（僅事件回收）仍死 @Iter3800**——診斷=
    怪物半徑達 **553,153px**（CityGS 端才 3000px）,單顆爆 isect_tiles;33 次事件回收攔不住
    densify 事件「之間」的單幀。**★關鍵洞見：成本控制必須 render-time（每幀）而非 densify-time**。
    v2 加 gsplat `radius_clip`（渲染源頭跳過超標半徑 primitive,每幀生效）跑中。
  - **★MCMC 約束理論定位（回答「原論文怎麼處理 cap」）**：MCMC=**硬 cap +
    opacity L1 + scale L1 + 分佈保持 relocation**（`compute_relocation` opacity 重整化保渲染不變）。
    **三層約束無一看渲染成本**：cap 數顆數(b8 1.53M活/b12 0.32M死=錯計價)、opacity L1 看不透明度
    (怪物 op 0.028 逃過)、**scale L1 是最接近成本但量錯維度**(世界尺度;近相機小世界尺度→大螢幕
    footprint,scale L1 隱形)。→ **cost-aware 補的正是「螢幕 footprint×相交數」這維度**=論文立論核心。
    程式對照:`internal/density_controllers/mcmc_density_controller.py`+`mcmc_citygsv2_metrics.py`。
- 排隊：cost-aware×beta v2 判讀 → **底座裁決拍板**（摘零件 vs 整包搬,+cost-aware 必留）/
  b7 test partition 誠實 held-out 數字 / SB 色移植 gsplat surfel / 25 塊 / 伺服器
- **★VRAM 拆解（07-12,實測+計算）**：cap500k on b12=5417MiB 中 **模型(params+Adam+grad)僅 229MiB(4%)、
  rasterizer+backward 緩衝 ~5190MiB(96%)**。緩衝 ∝ 每視角高斯-像素相交數(overdraw);b12 飛穿密集內容
  每像素數百顆重疊→即使 500k 顆也頂 5.4G。**b7 4M 活 vs b12 1M 死的差別完全在 rasterizer 緩衝(overdraw)非模型大小**。
  → param 壓縮救不了(打 4%);真正砍 96% 的=**LOD(減 per-view 高斯,最對症)/fp16 rasterization(緩衝對半)/
  transmittance 早停(便宜旋鈕)/tiling(CLM-GS)**。
- **⚠ LOD 更正（07-12,讀 code 後）**：`partition_lod_renderer.py`（CityGS V1）是**推論時+合併場景**渲染優化
  （吃 finest→coarsest 多層預訓練模型,按視角選 partition 的 LOD 層）→ **碰不到 b12 單塊訓練的 overdraw,
  且要先訓多層模型=更多成本**。**LOD 對 b12 訓練 VRAM 出局**（我一度過度推銷,已更正）。
  V2 不用 LOD=scope 轉移（V1 即時渲染→V2 幾何精度）非 LOD 壞。**LOD 正確定位=未來「合併 25 塊在 6GB 推論/渲染」**。
- **b12 訓練 VRAM 的對的工具**：cost-aware(減顆數)/fp16 rasterization(緩衝對半)/transmittance 早停(便宜旋鈕)/
  tiling(route2 camera-crop,自己做)。param 壓縮(打 4%)/LOD(推論階段)/**HiGS(inference-only)** 皆不對症。
- **★統一預算框架 + G2 查證（07-12,詳 `archive/統一預算框架_演化史_2026-07-12~28.md`）**：主判別式 `(1/K)Σ[c_iv·1(v/c>=λ)]<=B`,
  cost-aware(λ 減需求)×tiling(K 加供給)共用貨幣 c_iv=拉格朗日對偶。**我方聯網讀原文**:Pocket-SLAM(2606.24796)=真最近鄰
  必引+差異化(他們 area-as-value 保留大點/固定配額/SLAM post-hoc;我們 area-as-cost 審查大點/λ市場/K耦合/訓練期;
  怪物=決策相反臨界證明);HiGS 可引不可用。**兩實證 gate 未過別慶祝**:tiling 1hr camera-crop 驗證 + b7/b12 λ_floor 分界。
- 完整 OOM saga 取證（近相機怪物/trim 族群重整化/覆蓋守恆）：`SF_MCMC_實驗計畫_2026-07-04.md`

### 新機制（2026-07-06，程式碼資產）
`internal/density_controllers/mcmc_2dgs_density_controller.py` 新增 `screen_size_prune_px`
flag（預設 -1 關閉）：每步累積 max_radii2D，densify 事件把螢幕半徑超標的點併入 dead_mask
走 MCMC relocation 回收。**補上 MCMC 缺失的 vanilla max_screen_size 對應物 = D1 一手證據。**

## 2. 近期規劃（2026-07-06 更新）
1. **quad 收齊**（~07-07）→ **merge 組裝 + 掉分量測 + 接縫診斷**（任務 #5；
   symlink 拼 blocks/ → merge_citygs_ckpts.py → merged PSNR vs 加權平均）
2. **SF 冷卻重調臂**（zero-code CLI flags，b7 ~7h）
3. **P5 探針：成本感知控制器 on gsplat**（規格 §3；環境/程式=CPU 側可先備，GPU 排 merge 後）
4. G1（4M sh0 引擎對決，Phase 0 遺留）＋ P1 Beta / P2 Triangle / P3 Offloading / P4 Diffusion
   按 GPU 空檔穿插（探針全開為使用者拍板項）
5. 病理塊普查：07-02 中斷批次顯示 b3 疑似同型 OOM（停 1499 步）——merge 擴大前
   用 screen-prune flag 掃一次重災塊
6. 遠端伺服器（140.119.162.16）：等管理員修 GPU direct access + 掛 ≥80G 磁碟後接管重跑

## 3. P5 探針規格：成本感知密度控制 on gsplat
- **底座**：gsplat 1.4.0（已裝，`rasterization_2dgs`/packed/absgrad 可用）+ MCMCStrategy；
  資料=block_7（對照）+ block_12（酸性測試）
- **臂**：(a) gsplat MCMC-2DGS baseline（cap 對齊 1.7M）；(b) +cost-aware：relocation
  的 dead_mask 加 value-per-cost 準則（value=貢獻度近似, cost=Σ視角 tile 覆蓋），
  預算=顯式 VRAM/相交數上限，λ dual ascent
- **量測**：品質 vs 峰值 VRAM 曲線、b12 是否免 flag 存活（packed mode 假說）、it/s
- **判準（gate）**：同品質峰值 VRAM −30% 或同 VRAM +0.3dB → 論文演算法主體遷 gsplat；
  否則留 CityGS 現路線，探針止於 timebox（≤1 週工程+2 塊 GPU 時間）
- **注意**：兩底座數字不互比（rasterizer 語義不同）；交付=機制證據非可比 PSNR

## 4. 文件索引
### 活文件
| 檔案 | 內容 |
|---|---|
| `README.md`（本檔） | 單一來源：現況/規劃/索引 |
| `實驗總表.csv` + `tools_make_master_csv.py` | 全部實驗數據（含作廢標記）＋收割腳本 |
| **`現行方案與公式.md`** | **【現行公式的唯一來源★★】看這一份就知道我們在用什麼**：只寫現行方案，無時間軸、無被取代版本，每條附「實作在哪個檔＋驗到什麼程度」。§0 一頁總覽表／§1 需求側（拉格朗日×DAR 解耦、成本項兩層壓縮、`C_t`＝存活時間）／§2 供給側（峰值模型、`N_max`↔`K` 對偶、Jacobian 拉伸）／§3 實作對照表／§4 相關工作差異化／§5 現行未驗與已知歧異／**§6 出處與貢獻歸屬**（每條標🟩我方原創/🟨借用改造/⬜直接沿用，並標查證等級——⚠=Gemini 轉述未查證，引用前必查）。**改機制就改這份** |
| `archive/統一預算框架_演化史_2026-07-12~28.md` | 【演化史・非現行】上一份的推導軌跡：被取代的公式（反應式 λ／近軸 load v1-v2）、beta 底座實證、K=8 負結果、Gemini 各次對帳。論文 design rationale 用 |
| **`論文核對表.md`** | **【引用前必查★★】** 11 篇逐篇核對實作 vs 原文：①實作正確性 ②配套是否必須 ③可借用概念。內含 **LightGaussian Eq.4 真缺漏(已修)**／**MCMC Eq.8 印刷符號有誤**／**DBS 從未測過我方組態**／**我方五個 overclaim 修正**／**StopThePop 貢獻式 tile 剔除(今日最大槓桿)**。**寫報告引用任何論文前先看這份** |
| **`問題與應對總表_2026-07-23.md`** | **【問題×解法索引★】** 所有發現的問題+應對機制（狀態標記，✅/🟡/⚫），細節指向理論權威 |
| **`待辦與半成品清單.md`** | **【持久待辦★】** GPU 隊列 + 已實作待跑 + 設計未實作 + 大魚阻塞（取代會斷線的 task 工具，新待辦寫這裡）|
| `方向重定調_2026-07-03.md` | 命題/六缺陷 D1-D6（**仍權威**）；⚠ 其「四探針全開」已過時，現行方向見 `現行方案與公式.md` |
| `archive/雙峰opacity_*`（已歸檔 5 份） | 殭屍經濟/兩相動力學的原始 Gemini 五回合；**結論已綜合於 `問題與應對總表` B1/B2**，此 5 份僅存歷史 |
| **`指標時間線與姿態稽核_2026-08-01.md`** | **【2026-08-01★】** ①b12 三段乾淨拆解（EXACT_SUPPORT −0.118 LPIPS＝最大單一槓桿、要從「無損 VRAM 優化」重新歸類；reg→0 PSNR −0.20 但 LPIPS −0.099）②**平原期**：05-31~07-27 十四次跑次 LPIPS 全距僅 0.022，兩個月機制工作感知指標淨零 ③**姿態稽核＋三次撤回**：`sparse/0` 是 **GT 姿態×1/100，殘差 0.0 px** → 霧不是姿態造成，分支永久關閉 ④**官方 test set 741 幀 100% 可對齊**（「150× 不對齊」前提推翻）→ 評測定位待拍板。圖=`metric_timeline.png`，工具=`tools/plot_metric_timeline.py`、`tools/audit_pose_consistency.py` |
| **`初始化溯源與SfM-init_2026-08-05.md`** | **【2026-08-05★★ 病灶改判｜談 floater/剪枝前必讀】** ①**深度監督已榨乾**：偽深度誤差 0.633＝**174×** 表面解析度，比我方位置誤差(29×)還大 6 倍；62.8% 粒子跟深度圖一致度 <10%（＝其解析極限）⇒ 深度不一致度**不能當剪枝判準**（r=0.253、分離僅 1.3×，繼多視角集中度後第二個失敗）②**起點就偏**：`depth_init` PLY 離表面 **20.2×**，訓練只推到 29.2× ⇒ 80% 偏差在第一步；先前「訓練把粒子推壞」**推翻** ③**溯源**：depth-init 自承 "RTG-SLAM style"，但 RTG 用 **RGB-D 感測器**（公制），我方用 DA2 **單目估計**（8.6% 誤差）——機制搬了精度沒搬 ④**3DGS 論文 init 消融**：init 不良 →「floaters that **cannot be removed by optimization**」⇒ 我方所有下游手段（貢獻度/opacity_reg/塵埃/集中度/深度一致度）失敗是必然 ⑤**SfM-init 零成本**：`--model.initialize_from null` 即逐塊 SfM-init（dataparser 已帶 `selected_image_ids`），b12 得 320,513 顆 ⑥⚠ `surface_distance` 定義＝到最近 SfM 點 ⇒ 不可用它論證 SfM-init 較好（循環），且它**同時在量 SfM 密度**（每格 1-200 點→34.5×、1000-5000→9.8×）不可跨塊比。工具＝`tools/measure_depth_agreement.py` |
| **`信心分級初始化_2026-08-05.md`** | **【2026-08-05★★ 新機制｜Stage 0 跑中】** ①**SfM track 長度＝免費的逐顆「已驗證」標籤**——五個事後判準全失敗是因為「這顆在不在真實表面」訓練完就不存在了，但訓練前存在（`Point3D.image_ids` 一直都在）②**⛔重投影誤差反相關不可當信心**（track<=2 誤差 0.142px < track>=6 的 0.549px；BA 可自由移動 2-track 點滿足兩射線）③Tier A 錨點+Tier B 只補空體素，錨點的 normal/scale/rot 取自最近 depth-init 點 ⇒ **唯一變數是位置** ④**取點區域不可用 partition AABB**（AABB 內 196,631 vs depth-init 足跡內 1,160,384，**約 6 倍**）⇒ **副產物：block 間內容重疊極嚴重** ⑤**自我修正**：depth-init 的 scale 實測**全部相同**（走 `:444` 後備分支非 `:432` 的 `depth_i/focal`）⇒ 顆數與尺寸**都**是 `c_i≡1` ⑥⛔ EDGS/Photometric-SLAM-2026/MDPI-2026 經使用者確認為**幻覺文獻**；RAIN-GS 真實且是**本線反證**。工具＝`tools/make_graded_init.py`、`tools/measure_anchor_drift.py` |
| **`RAIN-GS核對_2026-08-05.md`** | **【2026-08-05★★ 定位翻轉＋prior art 警告】** ①**RAIN-GS 是我方 init 線最強外部支持，不是反證**（先前定位錯）：Table 1 `SfM 27.205 → SfM+ε 24.944`＝**加噪音 −2.26dB**，而**我方 depth-init 20.2× 就是「SfM+ε」那列**；Table 2 移動量 `3DGS(SfM) 0.349 / Random 0.184 / RAIN-GS 12.100`＝原文「optimization **struggles to transport Gaussians**」，這正是 3DGS 論文「floaters cannot be removed by optimization」的機制，也解釋我方 20.2×→29.2× 只漂 1.45× ②**⛔ prior art 撞擊**：其 `s = HW/(9πN)` 就是「每顆的螢幕預算」⇒ **「所有既有方法都是 `c_i≡1`」這句必須修掉**，我方貢獻要改寫成「把螢幕足跡從**優化排程**提升為**資源約束**並逐顆計價」 ③**三機制可用性**：漸進低通在我方 N 下是 **no-op**（N=1.3M → 半徑下界 0.59px；算式在文內）／SLV 大變異＝我方的怪物＝OOM 主因，**不可移植**／**ABE-Split 可借**且對到待辦 #10，關鍵細節是「位移後保留原尺寸不縮小」 ④arXiv ID 更正 `2311.16043`→**`2403.09413`** |
| **`裁切coarse_已刪除_2026-08-06.md`** | **【對照組陷阱★｜找 coarse-init 對照前必讀】** 把全域 coarse 裁切到單塊 AABB 的三個檔案**已刪除**（共 967MB），用過它們的四個跑次分數全部偏低：`e1_citygsv2_24k_b7` **19.60**／`e1b_scaler09_b7` **13.52**／`citygsv2_b7_faithful` 21.78／`mcmc_60k_sh3_aggr17_COARSE_b7` 22.27，CSV 已逐筆註記。**`e1_citygsv2_24k_b7` 正是「27 分那套 24k 參數＋當前資料」的唯一嘗試**，不知情會誤判成「官方配方只值 19.6」。**正確做法＝完整全域 coarse 不裁切**（`_initialize_from_trained_model` 本來就不做 block 過濾，塊外內容交給貢獻度剪枝）→ `official_ft_blk5` **23.342**。附對照組健康度實測：`official_coarse_sh2` 覆蓋 2.97/overdraw 63.3（健康，與 28.9 世系的 3.29/74.3 同級）vs `coarse_mc_aerial_1.2x` 8.60/565.7（病態，不可當 init）|
| **`深度損失覆蓋率bug_2026-08-06.md`** | **【汙染界線★★★｜引用 08-05~06 任何數字前必讀】** `1/(surf_depth+1e-8)` 在無粒子像素回傳 **1e8**；全 log 掃描出**今天的 7 個跑次全中、更早的 60 個全乾淨**，汙染率與結果**單調對應**（v1 2.4%→22.070／cap80k 14.9%→崩到 17.06／sfminit 18.9%）。根因不只是 metric：**24k 線的 `interval 350` > 破平衡點 231.5 ⇒ 顆數必然衰減 ⇒ 製造覆蓋破洞**。❌不可再用：v1 的 **22.070**、77.6k 的**紋理比 0.170**、v2/v3 失敗歸因、cap80k 崩潰（**「少而準」仍未測到**）。✅不受影響：oreg 23.848／reg000 23.39／全部 60k 線／步數vs顆數的邊際比較／起始 trim 全部發現／幾何比較／RAIN-GS／official_ft_blk5。修正＝`depth_coverage_eps=1e-4`（未覆蓋像素填 detached GT，SSIM 才有定義），**已雙面驗證**：健康模型未覆蓋像素 0.0000% ⇒ 靜默。**⚠ 過程教訓：驗「有沒有異常值」要取 max 與計數，不能取樣**（我用 48,091 抽 12 個就回答「零個」）|
| **`剪枝曲線與五個自我否證_2026-08-06.md`** | **【方法論★★】** ①剪枝曲線兩張（oreg 900k／reg000 1.8M）：**剪掉約 30% 幾乎免費**（−0.9%／**+1.0%**）、50% 損失 4~5% ← **這條有效**（部署側測法與用法一致）②**⚠ 使用者指出的缺陷**：剪枝**不重新擬合**，剩下的高斯 scale/opacity 是在別人存在的前提下調的 ⇒ **「少而準為假」「訓練大再剪小較差」兩個結論收回** ③**這一天的五個自我否證與否證方式**（厚殼→PCA 0.81／trim 預測器→模型種類錯／覆蓋倍數→不看 opacity 不可跨類／可見表面容量→曲線無轉折／P7→掃 max 而非取樣）④**`cap_max` 低於現有顆數 ＝ densify 永久關閉**（設計缺陷，與 P7 無關）⑤**「表現能力」與「控制能力」必須分開問** |
| `新標尺重錨_2026-07-19.md` | 標尺斷裂事件（rasterizer 二進位考古） |
| `廢案彙整.md` | 所有死線的結論與教訓（復活前必讀；§11=VRAM 桿/怪物防禦死路） |
| `完整Pipeline數學詳解.md` | input→output 全 pipeline 數學（參考） |
| `pipeline.md` | 函式調用 pipeline Phase 0-8（參考） |
| `完整指令手冊.md` | 指令參考 |
| `開發日誌.md` / `_2026-06下旬` / `_2026-06-30` / `_2026-07-01` | 歷史 session 記錄（append-only） |

### archive/（15 檔，經 `廢案彙整.md` 索引）
ADMM 線 ×4、SOGS、gradient-guided、期末診斷、主線_Gaussian效率、MCMC整合分析、
Gemini交接、DGD反推與DBP、實驗記錄（數據已入 CSV）、s1、experiment_analysis、session待辦。

### 待辦（2026-07-24 更新）
- [ ] **🟡 動態 K stretch 版跑完**（b12@2M 能否活到 60k）— 跑中，唯一活躍實驗，怪物區未到。
- [ ] **harvest_dust_trim_interval 驗證**（收割期週期清塵，已實作未跑）：判讀 PSNR 平/耗時降/VRAM 降，排 dynk 後。
- [ ] **cost-aware densify 乾淨對照**（若要評分表）：同底座同 cap，只變 densify 準則(MCMC relocate vs value-per-cost)，補全 SSIM/LPIPS。現有數字跨底座+PSNR-only 不能當乾淨對照。
- [ ] **2D tile backward（1D→2D，改 CUDA）**：低優先——K 已夠用，checkpoint 已排除，此為錦上添花。
- [x] ~~gradient checkpointing~~ ⚫排除（+3.7% 微負，CUDA buffer 管不到，詳廢案 §11）。
- [x] ~~三軸協同驗證~~ 已做（VRAM 三桿表：K=4 −54%、ckpt +3.7%）。
