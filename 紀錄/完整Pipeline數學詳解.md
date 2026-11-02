---
id: R-pipeline
狀態: 有效（現行配方 A＋dist4 的完整流程與公式）
年代: 新年代（2026-09-13 影像↔姿態修正之後；4x4）
摘要: 從資料到部署的每一步：做什麼、數學、參數值、對應程式碼；另列實驗中的模組與狀態。數字的出處在 `實驗分析/`。
依賴: [研究總覽 §0, 實驗分析/15, 17, 19, 21, 23, 24]
時間: 2026-11-02 重寫（舊版 7～10 月：`new_archived/完整Pipeline數學詳解_v1_至2026-10-03.md`）
最後更新: 2026-11-02（由 tools/stamp_records.py 從 git 產生，勿手改）
---

# 完整 Pipeline —— 現行配方 A＋dist4（每塊 30k、4x4）

> 一句話：**4x4 分塊 × 每塊 30k 步的 2DGS＋MCMC**，用塊外遮罩讓每塊只學自己的內容，二元熵把 opacity 推向 0／1，
> 累加式 V/C trim 邊訓練邊剪低價值顆粒；16 塊各留盒內顆粒合併成全場景。
> 全場景 MatrixCity test 741 幀 **28.058／.8824／.1142**、27.75M 顆、16 塊純訓練 5.04 h（lab 3090）、40.6 FPS（`實驗分析/21`）。
> 跑法：`CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=30000 CITYGS_CAP=2500000 CITYGS_FAMILY=g44s30km25fssh16di4_ bash scripts/lab/task_cmp.sh <塊> best1maskdistrfed10b4`
> ＋共同旗標 `record_reduce absgrad_gate fused_ssim sh_rest_interval 16 lambda_dist_interval 4`；全場景用 `scripts/lab/task_fullscene.sh maskplan`（預設自適應顆數，§9）。

## 0. 符號
```
i            顆粒（2DGS 面元）索引；N＝顆數；cap＝每塊上限
mu_i         中心（3）       s_i＝(s1, s2) 面內兩軸尺度     q_i 旋轉四元數（w,x,y,z）    t1, t2, n＝旋轉矩陣三欄（n＝法向）
o_i          opacity＝sigmoid(raw_o)                         c_i(d)＝SH 3 階顏色（16x3）
T_i(p)       像素 p 的光線到第 i 顆之前的穿透率；alpha_i(p)＝o_i · G_i(p)（G＝面元上的 2D 高斯）
z            相機座標深度（z 軸）；inv＝1/z
A／B／C      塊外遮罩的三類像素（§2）
```
每顆參數 59 個 float（3＋2＋4＋1＋48）＝232 B；ckpt 另含 Adam 的 m／v（佔 ckpt 約 2/3）；半精度部署 122 B／顆（`實驗分析/03`）。

## 1. 輸入與前處理（一次性）
| 項目 | 內容 | 程式 |
|---|---|---|
| 影像 | MatrixCity 空拍 5,621 張 1920x1080；**載入時縮 1.2 倍 => 1600x900**（`down_sample_factor`），uint8 快取 | `internal/dataparsers/`、live config |
| 姿態 | 與 CityGS 相同的輸入：`sparse/0`＝MatrixCity GT 姿態 x1/100（場景 1 單位＝100 m） | — |
| SfM 點 | 以 GT 姿態跑的 COLMAP 三角化，382 萬點（track>=3 且誤差<1.5：270 萬） | `sparse/0/points3D.bin` |
| 單目深度 | Depth Anything V2 ViT-L（輸入 518）逐張逆深度；每張用 SfM 點擬合一組縮放：`inv = s·d + o`（中位數／MAD，hierarchical-3DGS 的 `get_depth_scales.py`）；`s` 超出 [0.2, 5]×中位的那張不載深度 | `utils/estimate_dataset_depths.py`、`utils/get_depth_scales.py` |

- 單目深度在現行配方只用在 **塊外遮罩**（§2）。它的誤差：整體 14.6% 的真表面被說成「更遠」、高樓越高越嚴重（塔頂 68%）；海面錯在每張的整體縮放（`實驗分析/24` §5、`25`）。
- ⚠ 本機 `estimated_depths/` 是 09-13 前的舊命名（作廢）；lab 那份才是訓練實際用的。

## 2. 分塊與塊外遮罩
**分塊**：`utils/partition_from_colmap.py --block_dim 4 4 --content_threshold 0.08`（與 CityGS 4x4 同格）。
SfM 點與相機中心乘 `rotation_transform[:3,:3]` 後取 xy；盒 b＝[xy_b, xy_b + size_b]（外圈塊延伸到場景邊）。
每塊選相機：該相機看到的本塊內容 >= 8%（例：b3 899 張）。

**塊外遮罩**（`tools/block_mask44.py`；每塊每張一張三值圖）：用單目深度把每個像素反投影成 3D 點 X(p)，
```
A（塊內）       X(p) 落在本塊盒內（外擴 m_in）                                  照常算光度 loss
B（塊外且空）   X(p) 在盒外，且這條射線過了 X(p) 到地面為止不再進入盒（外擴 m_c > m_in）  目標＝本步背景色 => 本塊在這格必須透明
C（不計分）     第一表面在塊外但射線之後會進入本塊（鄰塊高樓擋住本塊街道）、深度無效、A/B 交界帶   權重 0
```
B 判得保守（判錯會刪掉本塊真的內容），A 判錯只是多學一點背景。單塊模型原本 73～81% 的顆粒長在盒外（合併時全丟），遮罩讓它們不必長（`實驗分析/17`）。

## 3. 初始化（每塊）
1. SfM 點＋空處補點（`tools/make_sfm_fill_init.py --fill-ratio 0.35 --fill-voxel 0.35`）。
2. 複製 K 份（`--dup K`）：K 取「起始顆數 <= 0.9 cap」的最大值（`scripts/lab/pick_dup.sh`；2.5M 時 b2 6、b3 7、b12 9）。
3. 裁到本塊盒（外擴 m_c；`block_mask44.py --crop_ply`）=> `sfmfill_sweep44_crop/dup<K>/block_<B>.ply`。
4. 每點變一顆面元（`setup_from_pcd`）：`log s = log sqrt(mean dist² 到 3 近鄰)`、opacity 與旋轉用 gspl 預設。

（10-24 試過深度補點初始化：浮塊少、分數不變，`實驗分析/21` §6。）

## 4. 渲染（2DGS 光柵器，`submodules/diff-surfel-rasterization-trim-pp`）
```
面元 i 的局部座標 (u,v)：射線與面元平面交點 => G_i(p) = exp(−(u² + v²)/2)
alpha_i(p) = o_i · G_i(p)                       T_i(p) = Π_{j<i} (1 − alpha_j(p))（依深度由前往後）
C(p)  = Σ_i T_i alpha_i c_i(view)  ＋ T_end · bg                 alpha(p) = Σ_i T_i alpha_i
z(p)  = 面元交點深度的 median／expected（surf_depth）           T 小於 1e-4 就提早結束
```
- 外接盒：conic 精確外接盒（只 bin 真的蓋到的 tile；`實驗分析/04`）；o <= 1/255 的顆粒不進 binning（EXACT_SUPPORT）。
- **精簡路徑（lean）**：訓練步光柵器不算深度／法向通道；同一趟 forward 順便累加 trim 要的 V／C（§7）。
  需要幾何通道的步（法向、單側不能看穿、補洞事件）自動走完整路徑。
- backward：飽和跳過（T 已飽和的像素不回傳）、block 內先加總再原子加（`record_reduce`）；|g|（位置梯度絕對值）順便累加在
  `dL_dmean2D.z`（ABSGRAD；渲染與梯度逐位元不變）。

## 5. 損失
```
L = (1 − λ)·L1 + λ·(1 − SSIM)                                   λ = 0.2；SSIM 用融合 kernel（fused_ssim）
    像素目標：A => GT、B => 本步背景色（隨機背景）、C => 渲染值本身（梯度 0）
  + k·λ_dist · L_dist（每 k=4 步算一次，權重乘 k）                λ_dist = 100
    L_dist(p) = Σ_i Σ_j ω_i ω_j |z_i − z_j|，ω_i = T_i alpha_i    （2DGS 深度失真：同一條光線的貢獻集中到一層）
  + λ_H · mean_i H(o_i)，H(o) = −o ln o − (1 − o) ln(1 − o)        opacity 二元熵（推向 0 或 1）
    step 1,000～9,999：只對 o < 0.5 的顆粒（只往 0 推），λ_H = 0.002
    step >= 10,000（STEPS/3）：全部顆粒（雙向），λ_H = 0.008
  + λ_s · mean(|s|)（全部面元的尺度元素平均）                   λ_s = 0.007（3DGS-MCMC 的 opacity L1 在 best1 起是 0）
```
- 熵只往 0 推是為了不在前 1/3 長出 opacity≈1 的冗餘層（水面、玻璃上方；`實驗分析/20`）；拿掉熵 F1 +0.03／+0.10 但 b0 浮塊 +31%（`23` §2）。
- 現行配方**沒有**法向一致、單目深度 loss（那兩個是 CityGS 的；試過的版本見 §10）。

## 6. 優化
- FusedAdam；學習率排程沿用 live config（`configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml`）。
- SH 高階（shs_rest）每 16 步才走一次 Adam（梯度在中間累積；`sh_rest_interval 16`），SH 分開存放、SH 的 Adam 融進 backward（逐位元相同）。
- 時間順序（一步）：取資料 → forward（lean 或完整）→ loss → backward → density controller `after_backward`（|g| 累積、MCMC 事件）→ optimizer.step → renderer `after_training_step`（trim 事件）。

## 7. 密度控制（MCMC，`internal/density_controllers/mcmc_2dgs_density_controller.py`）
每 150 步一個事件，從 1,000 步到 15,000 步（STEPS/2）：
```
死亡       dead = { i : o_i <= 0.005 }
搬家       每個死顆粒抽一個活顆粒 j 當宿主，機率 ∝ o_j · (1 + 2 · |g_j| / mean|g|)（absgrad_densify 2.0）
           新點複製宿主，兩者平分 opacity 與縮尺度讓畫面不變（3DGS-MCMC Eq. 9）：
           o_new = 1 − (1 − o_j)^(1/(n+1))（n＝j 被抽中的次數），宿主 Adam 歸零、搬家的保留動量
增生       N ← min(cap, 1.05 N)，新點用同樣的抽樣與 Eq. 9
雜訊       每步 Δmu_i = lr_mu · 5e5 · sigmoid(100 (0.005 − o_i)) · Σ_i ε，Σ_i = R diag(s1², s2², 0) Rᵀ
           => 只對幾乎透明的顆粒、**只在面元自己的平面內**推（法向那軸是 0）；gate < 1e-3 的跳過（fast_noise）
```
- cap 2.5M／塊；trim 剪掉的會在之後的增生事件補回 => 實際只交付約 0.9 cap。
- 10-24 修過 relocation 的 NaN（父代 o 恰為 1.0 時 Eq. 9 相消；`程式碼稽核清單` §3）。

## 8. 累加式 V/C trim（renderer，`trim_running_vc`）
```
訓練中每個 lean forward 順便累加：V_i += Σ_p T_i alpha_i（顆粒 i 實際混合的量）   C_i += 它蓋到的像素數
每 500 步（1,000～15,000）：score_i = V_i / C_i（沒被看到的這輪不剪）
                            剪掉 score 最低的 10%（以 N 計）；累加器歸零
```
- 判準 V/C（價值除以成本）而不是 V：同 N 下渲染成本 −1/3、品質平（`實驗分析/13`）。
- **速度來自「累加」**：週期 trim（每次掃相機重算）同品質、慢 19～33%（`23` §2）。
- 15,000 步後不再 trim、不再增生；後半段只做光度精修（＋熵、dist）。

## 9. 合併、自適應顆數、部署
- **合併**（`utils/merge_citygs_ckpts.py`）：每塊只留中心落在自己盒內（enlarge 0）的顆粒，16 塊串接 => 27.75M 顆（2.5M 版）。
- **自適應顆數（11-02 起全場景預設）**：`task_fullscene.sh maskplan <STEPS> <平均 cap>` 讓每塊 cap＝平均 × `configs/cap_alloc_g44.json` 的權重（0.69～1.34）。
  權重來自 10-31 的價格分配：對 A＋dist4 全場景每顆算 V/C，取共同門檻 λ 使總數＝預算，每塊 cap＝留下顆數 / 0.9（`tools/price_alloc.py`）。
  平均 1M 驗證：27.828 vs 均勻 27.805（`24` §6）；平均 2.5M 11-02 排在 lab。單塊實驗不套用（維持與基準可比）。
- **部署**：`tools/half_ckpt.py` 轉半精度、丟 optimizer => 3.15 GiB（1M 版 1.31 GiB）；`web_demo/` 檢視器（讀本機 ckpt）。

## 10. 評測
| 工具 | 量什麼 |
|---|---|
| `tools/eval_official_test.py` | 全場景 MatrixCity test 741 幀 PSNR／SSIM／LPIPS、FPS、峰值 VRAM（與 CityGS release 同一支工具） |
| `tools/swap_block_eval.py`（`task_full44.sh swap`） | 單塊實驗：把候選的第 B 塊換進 full44 合併模型，只看本塊負責的像素（本塊區域／幾何區域）與離接縫距離分帶；雜訊約 ±0.1 |
| `tools/block_f1.py`（`task_block_f1.sh arms`） | 區塊幾何 F1（只用該塊相機融合網格，對 MatrixCity 點雲） |
| `labrun/float2_blk_1029.py`、`orient_blk_1029.py`、`facade_ckpt_1030.py` | 海上／陸上浮塊、面元朝向一致度、立面貼牆度 |
| `labrun/lowangle_1030.py` | 合成低角度視角的透空率（MatrixCity test 全是空拍高度，看不到） |
| `tools/inbox_diag.py` | 馬路地底面元（暗柏油表面在地底的比例） |
| `labrun/time_1024.py`、`CITYGS_STEP_PROFILE=1` | 純訓練時間；逐段時間組成（`step_cost.txt`；開了會同步、不可拿來比總時間） |

## 11. 一張圖
```
MatrixCity 影像＋GT 姿態 ─┬─ COLMAP（GT 姿態）=> SfM 點 ─────────────┐
                          └─ Depth Anything V2 => 逆深度 ─(SfM 擬合縮放)┤
                                                                         ▼
           partition_from_colmap 4x4 ──> 每塊相機 ──> block_mask44：A／B／C 遮罩
                                                     └──> SfM＋補點 ×K、裁盒 => init PLY
                                                                         ▼
  每塊 30k 步 ┌──────────────────────────────────────────────────────────────┐
              │ forward（lean：順便累加 V/C、|g|）                            │
              │ loss = L1/SSIM（A=GT、B=背景、C=0）＋dist（每 4 步）＋熵＋scale │
              │ backward（飽和跳過、先加總）→ FusedAdam（SH 高階每 16 步）     │
              │ 每 150 步（≤15k）：MCMC 搬死顆粒＋增生到 cap（|g| 加權）       │
              │ 每 500 步（≤15k）：剪 V/C 最低 10%                            │
              └──────────────────────────────────────────────────────────────┘
                                                                         ▼
           16 塊各留盒內顆粒 => 合併 => 半精度部署 => 741 幀評測／檢視器
```

## 12. 實驗中的模組與狀態（2026-11-02；全部預設關閉、旗標名在括號）
| 模組 | 做什麼 | 結果（單塊 b3／b12 為主） | 時間 | 狀態 |
|---|---|---|---|---|
| 深度支撐閘門（metric `depth_support`＋density `geom_gate_sampling`） | 顆粒中心投到每個視角，對單目深度的殘差在 400 步視窗內一直 < −5%（在前方）或從不貼面 => 不往 1 推熵、MCMC 抽樣 ×1e-3 | 海上浮塊 −85～95%、PSNR 持平 | +8～9% | 候選 |
| └ SfM 豁免（`depth_support_sfm_radius 0.03`） | 離 SfM 點 0.03 單位內不判 | 修好 b3 塔頂變煙、浮塊壓制保留 | 0 | 候選 |
| └ 投票規則／每 k 步（`depth_support_rule vote`、`depth_support_interval`） | 過半觀測說在前方才判（中位數）；每 k 步才累積 | 11-02 排在 lab | — | 實驗中 |
| 單側不能看穿（`depth_see_through_reg`；`_hole`／`_deeper` 拆項） | A 像素 Σ(1−α)＋渲染比單目深 1.1 倍就罰 | F1 +0.04／+0.09；**好處與 b12 低角度變差都來自補洞項** | +9%（b3） | 被補洞放點取代中 |
| MCMC 補洞放點（density `hole_fill_frac`） | 增生事件把洞／看穿處的像素沿視線放到單目深度（新小面元） | b12 低角度 3.0%（不能看穿 7.8%）；b3 塊內掉分 | ＋6%（待查） | 實驗中（`25`） |
| 遮擋回收（`occl_recycle_min`） | 一直躲在不透明表面後方、從沒被看到的顆粒交給 MCMC | 11-02 排在 lab | — | 實驗中 |
| 增生方向限制（`inplane_offset`、`inplane_hold`） | 新點在母點平面內錯開；出生後只准平面內移動（1k→10k 漸增） | b12 低角度 2.1%（最好）、PSNR 雜訊內 | −3% | 實驗中（`25`） |
| L1 混 L2（metric `rgb_l2_weight 2`） | 光度 loss 加 L2 | 救回超高塔頂、b3 +0.05 | 0 | 候選 |
| 法向每 k 步（`lambda_normal 0.0125`、`lambda_normal_interval`） | 渲染法向對深度法向 | 朝向 0.77 → 0.94；浮塊不減、b12 低角度變差 | k=8 約 +5% | 候選（幾何用） |
| 鄰域平面約束（metric `surf_reg`、`surf_orient`） | 不渲染：Morton 近鄰 PCA 平面，拉近平面、朝向對齊 | 朝向 0.91、浮塊 −22%、b12 低角度不變差 | +5～10% | 候選 |
| 合併後刪除（density `merge_prune`、`merge_*`） | 互為最佳的同平面同色鄰居矩匹配合併，被併的刪掉、cap 同步扣 | 後半合併：顆數 −16%、PSNR 平、浮塊 −24～41% | −3～7% | **輕量化候選**；全程／更積極版排在 lab |
| 合併優先的 trim（renderer `trim_merge_first`） | trim 候選先併進同平面鄰居再剪 | 11-02 排在 lab | — | 實驗中 |
| 每塊 cap 1M／自適應 | 少顆數 | 全場景 −0.23 dB、時間 −36%；自適應 +0.023 | −36% | 輕量版 |

已關閉的（含理由與重開條件）：`new_archived/已關閉的線.md`。

## 13. 時間組成
（待 11-02 的計時跑次 `g44s30km25fssh16di4prof_` b3 的 `step_cost.txt`；上一次拆解是 10 月初的 best1，`實驗分析/01`、`10`。）
