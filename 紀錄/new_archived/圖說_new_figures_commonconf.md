# 通用 config 新舊版對比（2026-09-22）

產生：`python tools/plot_commonconf_ab.py`　數據出處：`紀錄/資源與效能量測彙整.md` §9.9b

## ⚠⚠ 這組圖「是什麼」與「不是什麼」

2026-09-22 的 config 收斂一共改了 **7 項**，但其中 **6 項是行為無變化**的
（`lambda_normal` / `depth_loss_weight` / `opacity_reg` / `cap_max` / `densify_until_iter` /
`absgrad_densify`+`fast_noise`+`noise_gate_eps`）—— 因為**所有任務腳本本來就在傳那些值**，
config 只是被對齊過去，讓新腳本與 `main.py fit --config` 不會吃到 2026-08 的舊配方。

⇒ 對「用任務腳本啟動的跑次」而言，**新舊 config 的差完全等於 `exact_conic_aabb` 的差**。

- ✅ 本組圖 ＝ conic off（舊）vs conic on（新）的乾淨 A/B：**2 塊 x 2 長度、兩臂同 N、同一個 .so**
- ⛔ 本組圖**不是**「舊 config 整體 vs 新 config 整體」。那個比較**沒有乾淨資料**，
  因為新年代裡從來沒有任何一次跑次真的用過那 6 項的舊值（腳本全都覆寫掉了）。
  要有那個比較，得專門去跑一次舊值 —— 目前判斷**沒有必要**（它只影響「忘記傳旗標」的情境）。

## 圖

| 檔 | 內容 | 一句話 |
|---|---|---|
| `cc1_quality_delta.png` | 4 組比較 x 4 指標的 Δ | **16 個差沒有一個是負的**；黃帶＝同配方重複樣本的實測差（噪音底） |
| `cc2_time.png` | forward / forward+backward（`step_breakdown`，`[solo]`，同 N） | forward **−26%**、fwd+bwd **−16~17%** |
| `cc3_load_vram.png` | 離線精確 Σtiles（全部相機）＋ 峰值 VRAM | Load 中位 **−40.1% / −45.8%**；VRAM **−3.9~−6.6%** |
| `cc4_proxy_decoupled.png` | 代理 ÷ 精確 成本比 | ⚠ **副作用**：新版讓代理指標脫鉤（1.07~1.28 → 2.23~2.27） |

## 引用時必須一起講的三件事

1. **不宣稱品質變好。** 60k 沒有重複樣本，+0.06~+0.16 dB 落在同配方重複的量級（0.03~0.05）。
   站得住的說法是「**一致地沒有品質代價**」；4/4 同號是訊號，但指標之間相關，不是 16 個獨立樣本。
2. **訓練牆鐘不可引用**（lab 三槽平行）。時間數字一律取自 `[solo]` 的 `step_breakdown`。
3. **副作用要一起報**：conic 開啟後 `cost_budget` / `vpc` / `cost_aware_densify` 的代理成本會
   高估約 2.3 倍 ⇒ 用到它們時必須改開 `exact_tile_cost`（控制器已加自動偵測警告）。
