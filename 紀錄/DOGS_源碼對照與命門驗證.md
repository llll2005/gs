---
id: R-dogs
狀態: 有效（敘述與核對）；**內含的分數需重測**
年代: 跨年代 —— 文獻/程式碼事實不經過 GT 影像↔姿態的配對
摘要: DOGS 論文 Eq.(8b) + 官方源碼對照；我方核心批判被雙重坐實，同時暴露我方兩個結構性問題。
依賴: [S9-papers]
時間: 2026-06 ~ 2026-08
---

> ✅ **2026-09-13 重驗：搬回主線。** 論文與**對手源碼**的對照 => 文獻與程式碼事實，不經過 GT 影像配對 => 倖存。
> 現行結論見 `研究總覽_v2.md`；本檔是它的**細節依據**。

# DOGS 論文 + 源碼對照：命門驗證與我方實作審查

> 日期：2026-06-06
> 觸發：把《期末診斷》的口頭推理釘進「可引用文獻 + 對手源碼」。
> 對手源碼位置：`../DOGS`（DOGS 官方 release，含 master/slave ADMM trainer）。
> 結論一句話：**我方的核心批判被 DOGS 論文 Eq.(8b) 與源碼雙重坐實；但同一份證據也把我方 thesis 的舉證責任收得更緊，並暴露 2 個我方實作的結構性問題。**

---

## Part 1：論點 ↔ 參考論文對照（哪些有白紙黑字記載）

| 論點 | 出處 | 狀態 |
|---|---|---|
| AL 懲罰 `(rho/2)||x-z||^2` 是進 backprop 的軟懲罰、z 是共識變數 | DOGS Eq.(5)(8a) | ✅ 直接記載 |
| z = 各 block 局部變數的**平均** | DOGS Eq.(8b) `z=(1/K)Σ_k x_i^k` | ✅ 直接記載 |
| 點數(非 init)驅動 VRAM、可獨立控制 | 3DGS / CityGS；DOGS p.6「larger block can cause OOM」 | ✅ 直接記載 |
| GaussianSpa 設點數上限不掉品質 | Gspa.pdf | ✅ 直接記載 |
| Host offloading（RAM 存高斯、6GB 跑計算） | GS-Scale.pdf | ✅ 直接記載 |
| C-2「廉價全域先驗當 init」=「coarse→finetune」同一光譜 | CityGaussianV1/V2 管線 | ✅ 直接記載 |
| ADMM 需 adaptive rho 才收斂 | DOGS §3.3 adaptive penalty + over-relaxation | ✅ DOGS 已做（非我方創新） |

**七條 load-bearing 論點全部在已有論文中。不需另找論文背書。**

### 我方原創批判，被 DOGS 實證
我方 §2.2 主張：「我們的 GAT z 是錯的共識目標，因為相鄰 block 的邊界高斯不是同一個物理變數」。
- DOGS p.4 原文：*"The shared local 3D Gaussians ... are a copy of the global 3D Gaussians."*
- DOGS 的 `z=mean(x_i^k)` 之所以是**正確 ADMM 共識**，前提是 `x_i^k` 全是**同一顆全域高斯的副本**。
- 我們的 `z_v = sigma(Σ alpha_vu·W·h_u)` 是對**不同鄰居高斯**做 attention → 違反此前提。
- 另：DOGS p.5 明說 `{u,q,s,f,o}` 各屬性**分開**做共識；我們用一個 W 把 12 維混在一起。
→ **批判成立，且非個人意見，是 DOGS 數學前提反推。**

---

## Part 2：命門驗證 —— DOGS 到底怎麼處理「densification 動態拓撲」

CLAUDE.md 把整個 GAT 貢獻的正當性押在「DOGS 處理動態拓撲只靠啟發式」這句話上。源碼給出**比這句更強**的答案：

### 鐵證（`conerf/trainers/master_gaussian_trainer.py`）
```
L684-688:  # The consensus and sharing of gaussian splats only activated
           # when the densification step finished.
           if self.iteration < self.config.geometry.densify_end_iter or \
              not self.config.trainer.admm.enable:
               return                      # ← densify 期間直接跳過 consensus

L305-306:  rho 等狀態只在 iteration >= densify_end_iter 時才存
L512-514:  consensus 只在 iteration > densify_end_iter 且 % consensus_interval==0 時跑
L558-618:  fuse_local_gaussians()：densify_end 時「只跑一次」，
           融合所有 block → prune → select_gaussians_in_each_block 建立
           固定的 global_indices 對應 → setup penalty → setup dual → enable_admm
L538-555:  gaussian_splat_consensus()：global 歸零 → 依固定 global_indices
           plus_gaussians → average_gaussians(count=visibility_count)
           = 對「同一顆全域高斯的副本」取平均（Eq.8b）
```

### 結論
**DOGS 根本不在 densification 期間做 consensus。** 流程是：
```
階段1：各 block 獨立訓練 + densify（拓撲自由變動，零 consensus）
階段2：densify_end → fuse 一次，凍結拓撲，建立固定 global_indices 對應
階段3：拓撲已凍 → 才開始 ADMM，z=平均「同一顆高斯的副本」
```
→ **DOGS 不是「啟發式解決」動態拓撲，而是用「時間分離」直接迴避它。** consensus 永遠跑在**靜態拓撲**上，所以 clone/split/prune 的 y 繼承問題在 DOGS 裡**根本不存在**。

---

## Part 3：DOGS vs 我方實作 —— 逐項對照

| 維度 | DOGS（源碼） | 我方（dt_admm_*） | 風險 |
|---|---|---|---|
| consensus 時機 | densify_end **之後**，靜態拓撲 | 與 densify **交錯**，每個 outer 都做 | 我方走「更難的新路」 |
| 動態拓撲 | 迴避（凍結後才共識） | 正面處理（y 繼承：clone保留/split折半/prune丟棄） | 我方解的可能是**非問題** |
| z 算子 | mean（同一顆高斯副本的平均） | GAT attention（不同鄰居加權） | §2.2 算子合法性 |
| rho 結構 | **per-property**（xyz/fdc/fr/s/q/o 各一個），按點數縮放 | **單一純量** rho，作用在 raw-space AL | ★ 見 Part 4 |
| adaptive rho | 有（residual balancing：mu/tau_inc/tau_dec） | 無（固定 rho） | task2 用固定 rho 是粗代理 |
| 防 trivial fixed point | 不需要（mean≠各副本，天然良置） | 需加 `beta_cross` band-aid（否則 y=0 時 z=x，dual 永不累積） | 算子需人工補丁 |

---

## Part 4：我方實作審查 —— 找到的問題

排除一個誤判、確認一個結構問題、外加兩個設計疑慮：

### ✅ 排除：空間不一致 bug（原本懷疑、查證後無此問題）
- GAT z-update 在 z-score **正規化空間**運作；但 `denormalize`（renderer L152/176-179）把 z 還原回 **raw 空間**才存 `z_consensus`，y 也按 std 縮放。
- density controller 的 AL 懲罰用 raw-space `h(x_bnd)` 對 raw-space `z_consensus` → **空間一致，無 bug。**

### ★ 結構問題 1：單一純量 rho 作用在 raw-space → 4/5 屬性的共識被餓死
- AL 懲罰 `(rho/2)||h(x)-target||^2` 在 **raw 空間**逐維平方和。
- raw 量級：means ~[-100,100]、normal ~[-1,1]、scale ~[0,0.1]、opacity [0,1]、SH_dc ~[-0.5,0.5]。
- 平方和被 **means(xyz) 完全宰制**（量級差 ~10^4）→ 對 opacity/color/scale 的共識梯度近乎 0。
- **這正是「||y|| ~ 0.01、訊號弱 1000 倍」的結構性來源之一**：penalty 幾乎只在拉位置，其他屬性的共識根本沒在發生。
- DOGS 用 per-property rho 正是為了避免這件事（各屬性共識強度各自校準）。
- **對 task 2 的直接警告**：task 2 用單一 rho=0.5，即使 ×5，仍無法修正 raw-space 的 per-property 失衡。
  **若 task 2 得出「ADMM 沒救」，這個結論被單一-rho 餓死效應 confound，不能直接判 ADMM 死刑。**

### ⚠ 設計疑慮 2：consensus 與 densification 交錯（vs DOGS 凍結後共識）
- 我方每個 outer 都在「仍會 densify 的拓撲」上做 consensus → 需要整套 y 繼承機制。
- DOGS 證明：凍結拓撲後再共識，照樣拿 SOTA。**所以「交錯」這條更難的路，要先回答「它到底買到了什麼」。**

### ⚠ 設計疑慮 3：beta_cross 補丁暴露算子非自然
- renderer L204：*"Without it, z*=x and the dual never accumulates."*
- 我方 z=GAT-over-neighbors 在 y=0 時有 trivial fixed point（z=x），必須人工加 `beta_cross` 把 z 推離 x。
- DOGS 的 mean-over-copies 天然 z≠各副本，**不需要這個補丁**。補丁的存在本身就是 §2.2 算子問題的徵狀。

---

## Part 5：雙刃結論（誠實，不挑好聽的講）

**對 thesis 有利（novelty 前提仍在、甚至更強）：**
- 「DOGS 只靠啟發式處理動態拓撲」這句其實**講客氣了**——DOGS 在 densify 期間**完全不做 consensus**。
- 所以「**consensus 與 densification 並行**」確實是 DOGS 沒碰的開放問題，我方 novelty 前提成立。

**對 thesis 不利（舉證責任收緊，必須正視）：**
1. DOGS 用「densify→凍結→共識」拿到 SOTA。**我方必須證明「並行」買到了 DOGS 那條路拿不到的東西**；否則整套動態拓撲機制是在解一個 DOGS 已示範可迴避的非問題。
2. 我方算子（GAT-over-neighbors）需要 `beta_cross` 補丁，且 raw-space 單一 rho 餓死 4/5 屬性——這些問題 DOGS **靠 construction（mean-over-copies + per-property rho）天然避開**。
3. 「adaptive rho」是 DOGS 標配（§3.3），非我方創新。

---

## Part 6：這份審查改變了什麼（行動）

### 對 task 2 的修正（重要，跑之前看）
task 2 現有設計（單一 rho=0.5、與 densify 交錯）**自帶兩個 confound**：
- 單一-rho raw-space 餓死 → opacity/color/scale 共識本來就接近 0；
- 與 densify 交錯 → 拓撲一直動，y 一直被繼承/重置。

→ **建議二選一：**
- **(A) 先修 rho 結構再跑**：把 AL 懲罰改 per-property rho（抄 DOGS setup_penalty_parameters 的 6 個 alpha），這樣 task 2 才是「ADMM 能不能 work」的公平測試。
- **(B) 照現狀跑但謹慎判讀**：若升 → 強訊號（連餓死版都 work）；若不升 → **不可判 ADMM 死刑**，因為被 confound，得補 (A) 再判。

### 候選戰略路線（新增一條，源於 DOGS 源碼）
- **路線 D（DOGS-faithful 移植）**：採用 DOGS 的「densify→凍結→固定對應上做共識」結構，
  **丟掉整套動態拓撲機制**（y 繼承、beta 補丁全部不需要）。
  貢獻重定位為：「**在 6GB 上靠 offloading+sparsification 跑出 DOGS 級共識**」（工程，可達成）
  而非「GAT 解動態拓撲」（可能是非問題）。
- 這條風險最低、最有源碼+論文背書，**且和 C-2/工程棧方向天然相容**。

---

## Part 7：要丟給 Gemini 的問題（它的學術資源比我多，請它查證）

1. DOGS 之後（2024 末~2025），**有沒有論文真的在 densification 並行下做跨 block consensus**？
   若有，我方 novelty 前提（「並行是開放問題」）就破了，必須現在知道。
2. 把 ADMM 的 z-update 換成 **learnable GAT**（algorithm unrolling）在 3DGS 以外領域，
   **有沒有被證明優於 closed-form mean**？還是普遍發現 unrolling 在這種「副本平均」場景沒有增益？
3. per-property/per-block adaptive rho 在 heterogeneous 變數共識上，**有沒有標準的自動定 scale 做法**
   （而非手調 6 個 alpha）？這直接決定 Part 4 結構問題 1 的修法。
4. 「densify→凍結→共識」(DOGS) vs 「並行共識」，**在收斂理論上**前者是否總是 dominate？
   有沒有反例場景證明並行共識能逃離前者收斂不到的局部最優？

---

## Part 8：RTG-SLAM 源碼比對 + rho 修正的精確化（2026-06-06）

### 8.1 RTG-SLAM 幾何 init 比對（`../RTG-SLAM`）——忠實，無 bug
對照 RTG `SLAM/utils.py:compute_vertex_map` + `gaussian_pointcloud.py` vs 我方 `utils/depth_init_blocks.py`：

| 公式 | RTG 官方 | 我方 | 結論 |
|---|---|---|---|
| 反投影 | `(i-cx)/fx · depth`（pinhole） | `(us-cx)/fx · d_s` | ✅ 一致 |
| 法線 | vertex map cross product | `cross(dVdu,dVdv)` + 朝相機翻轉 | ✅ 一致 |
| scale | radius ∝ depth/focal (Sec 3.1) | `scale_i = depth_i/focal · scale_factor` | ✅ 一致 |
| opacity init | `init_opacity`（0.99 opaque） | `INIT_ALPHA = 0.99` | ✅ 忠於 RTG |

→ **depth-init 是 RTG 幾何 init 的忠實實作，無公式 bug，task 1 的 depth-init 數字可信。**
- 附帶：`depth_init_blocks.py` 檔頭 docstring 仍寫 `alpha=0.1`（過時，實際 0.99）；記憶 algorithm_notes 已更正。
- 我們忠實複製了 RTG「0.99 opaque + freeze」，這正是 ablation 量到傷害的部分（B1 -0.197、B4+B5 -1.883）。忠於 RTG ≠ 離線最優，但主兇是 B4+B5。

### 8.2 rho「餓死」說法的精確化（修正我方 §2.1 與 Gemini Q3 的描述）
追完 primal 懲罰的實際梯度後：懲罰是**可分離**的逐維平方和，
```
∂L/∂opacity = rho·(opacity - target_opacity)   ← 與 xyz 獨立
∂L/∂xyz     = rho·(xyz - target_xyz)
```
→ xyz **不會壓制** opacity 的梯度；被 xyz 宰制的只是回報的 ||y|| 純量與殘差尺度。
**精確問題**：單一 raw-space rho 無法**同時校準** 6 個尺度差 10^4 的屬性（讓 xyz 合理的 rho 會讓 opacity 共識可忽略，反之亦然）。不是「歸零餓死」，是「無法一體校準」。

### 8.3 現有 ADMM 其實是自洽的 raw-space ADMM（排除空間 bug，再次確認）
完整推導：GAT 內部 normalize + `y_norm=y_raw/std`，denormalize 後產生的 z\* **正好等於** raw-space ADMM 最優 `z*=x+y/rho`。
→ 正規化只是讓 GAT 在良置空間學習，**不改變 ADMM 本身**。系統自洽。

### 8.4 行動修正：先量、再決定要不要 reweight（不盲改）
- 既然懲罰可分離，rho 0.1→0.5 會**獨立強化每個屬性**的共識梯度 → task 2 很可能本來就會動 opacity/color。
- 故**正確順序＝先量後改**：已在 `dt_admm_coordinator.py` step-5 加 **per-property `||x-z||` 診斷印出**（xyz/nrm/scl/opa/sh 分群，僅加印不動數學，語法已驗證）。
- 跑 task 2 看每群殘差隨 outer 的變化：
  - opa/sh 殘差**有隨 outer 縮小** → 單一 rho 夠用，直接判讀 PSNR。
  - opa/sh **不動、只有 xyz 動** → 才需 reweight，此時再套已備好的 weighted-ADMM。

### 8.5 weighted-ADMM 已實作 + CPU 測試通過（2026-06-06）
**裁決**：理論（Gemini）+ DOGS 源碼皆證明單一 rho ill-conditioned，故跳過 8.4 當決定性測試，直接實作 8.5。
但 naive `1/std²` 有兩個 Gemini 未察的數值陷阱，已在實作中解掉：

**陷阱 1：凍結維爆權重** — opacity 凍結在 0.99 → std≈0 → 1/std² 爆。
**解**：cond-cap，`std.clamp(min=0.1·max(std))`，把屬性間權重比鎖在 ~100。

**陷阱 2：整體強度被 std² 稀釋** — `w=1/std²<<1` → 共識整體弱 1000×（CPU 測試抓到：加權梯度 0.0000 vs 未加權 0.1），會 confound rho sweep。
**解**：geomean 正規化，`std /= geomean(std)` → `geomean(w)=1`，preconditioner 只**重分配**強度不改總量，rho 仍是唯一強度旋鈕。

**實作（`compute_feat_std` + 三檔協同，全部 raw-space 自洽）**：
- `feat_std` = geomean-normed、cond-capped std（同一個 s 用於 GAT 正規化 / 權重 / y_norm）
- **primal**（density controller `_compute_al_penalty`）：`w=1/s²`；`target=z - s²·y/rho`；`L=(rho/2)·Σw·(h_x-target)²`
- **dual**（coordinator `dual_update`）：`y += rho·w·(x-z)`
- **GAT y_norm**（coordinator `run_gat_z_update`）：`y_norm = y_raw·s`（兩路推導確認）
- **plumbing**：`feat_std=(12,)` 存入 admm_state；density controller 加 (12,) buffer（topology-invariant，hasattr 守衛不隨 densify 重置）；warm-up outer `feat_std=None`→退回未加權

**驗證**：`tests/test_weighted_admm_precond.py`（CPU，無需資料/GPU）全通過：cond-cap、geomean、平衡、強度保留、無 NaN、凍結維安全、向後相容。

**⚠ dry-run 必看的 watch item**：CPU 測試發現 geomean-normed 輸入（xyz~±18）下 **GAT z-range 可達 ±400**（資料 ±100），gap 隨步增長 → GAT 的 `proj`（非 spectral-normed）在大幅輸入下可能膨脹。隨機測試圖的 artifact 成分高，但**真資料 dry-run 要盯**：
- per-prop `||x-z||`：opa/scl/sh 應隨 outer 縮小且與 xyz 量級相當（驗證平衡生效）
- `||x-z||` 或 z 不可發散（驗證 GAT 穩定）
- 若發散：降 `gat_lr`、或給 `proj` weight-decay、或把 GAT 正規化用的 std 與 preconditioner s **解耦**（GAT 用 true-std 良置、權重用 geomean-s）。
