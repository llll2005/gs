# ADMM 改法清單（復活藍圖）

> 來源：2026-06-06 與使用者腦力激盪（YOLO / ResNet 兩篇經典）。
> 背景：task 2 壓測（rho=0.5、27 outer）確認 **raw-feature consensus 在「已對齊 boundary」上 net-zero**，
> 機制問題是 (a) GAT z-update 近乎 identity（空轉）、(b) boundary 已對齊到 x−z≈0 沒分歧可用。
> 這份記兩個可能的破法，**只在 ADMM 要復活時才動**（目前主力是 task1 init 效應）。
> 使用者可代找源碼，屆時 Claude 分析引入。

---

## 改法 1（ResNet）：殘差式 z-update —— 主要瞄準「拓撲階段 / 分歧大」regime

**問題**：現在 z 從零重建整個特徵
```
z_v = sigma( sum_u alpha_vu * W * h_u )
```
因為 z 應該 ≈ x（相鄰 block 本就接近），網路被迫先把 x 重學一遍 → 塌縮成 identity、學不到東西（task 2 觀察到的 GAT loss 不動）。

**ResNet 改法**：z 寫成 input + 殘差，GAT 只建模「跨 block 微小修正量」
```
z_v = x_v + GAT_residual( neighbors )      # GAT 輸出 0 時自動退化成 z=x
```
好處：
- 梯度只流向「分歧」這個小信號 → 訊噪比高
- identity 變免費預設，網路只在有分歧時才動
- 對應 ResNet「目標接近輸入時學殘差更易」的核心

**關鍵限定（潑冷水）**：這修的是 **GAT 表達力/惰性**，不是「有沒有分歧」。
- 在**已對齊 boundary**（task 2 那種，x−z≈0）→ 殘差式仍抓不出不存在的分歧 → **大概不翻轉 net-zero**。
- 真正會發光的是 **densify 期間 / 動態拓撲 / blocks 尚未對齊**的 regime（= 我方原始 novelty 主張：DOGS 迴避、densify 期間不做共識）。那裡分歧大，殘差式 GAT 才有意義。
- **行動**：若回到「並行共識 during densification」路線，z-update 先改殘差式再測。可順帶量「殘差 norm」當分歧指標，決定哪些場景值得觸發共識（呼應 pivot 的「條件式觸發」）。

**待分析源碼**：ResNet（殘差連接實作細節，雖然概念已夠用，源碼非必要）。

---

## 改法 2（YOLO）：有原則的 boundary 指派

**問題**：現在 boundary 指派偏 ad hoc（距離帶 `d_boundary`），且觀察到 boundary% 莫名從 0.9% 跳到 5.0%（outer-1 → outer-11），指派不穩定可能稀釋/噪化共識訊號。

**YOLO 原則**：grid-cell 責任制 —— 物件中心落在哪個 cell，**那個 cell 唯一負責**，乾淨無重疊、無歧義。
**借鏡**：把「某個 Gaussian 屬於哪條 block 邊界 / 由哪對 block 負責共識」做成**明確、唯一、穩定**的指派規則（例如以 Gaussian 中心落在哪個 partition 重疊帶、由固定的 (block_i, block_j) pair 負責），而非每 outer 重算距離帶導致數量飄動。

**限定**：這是**細節改善非主線**。YOLO 整體精神（單次全域 reasoning）跟我們 6GB 的 divide-and-conquer **對立** —— 它反而佐證「partition + consensus 是被硬體逼出來的、非選錯」。只取「乾淨指派」這一點。

**待分析源碼**：YOLO（grid 責任分配 + 多項加權 loss 的實作，若要落地指派規則可參考）。

---

## 一句話
- ResNet → z=x+殘差，**留給拓撲/分歧大 regime**，是 ADMM 復活的核心改法。
- YOLO → 乾淨 boundary 指派，細節改善 + 路線正當性對照。
- 兩者都**等 ADMM 確定要復活才動**；現在主力是 task1（densify 對齊的 init 效應）。
