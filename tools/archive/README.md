# tools/archive —— 已退役的工具

移到這裡不代表壞掉，代表**不該再被拿來用**：要嘛它服務的假說已被推翻，要嘛已被更好的工具取代，
要嘛它會製造已知有害的產物。**復活任何一個之前先讀 `紀錄/研究總覽.md` 的對應章節。**

## ⛔ 會製造有害產物（2026-08-06 封存）

| 工具 | 為什麼封存 |
|---|---|
| `export_coarse_block_init.py` | 把全域 coarse 裁切到單塊 AABB。**用過裁切 coarse 的四個跑次分數全崩**：`e1_citygsv2_24k_b7` **19.60** / `e1b_scaler09_b7` **13.52** / `citygsv2_b7_faithful` 21.78 / `mcmc_60k_sh3_aggr17_COARSE_b7` 22.27。產物已刪（967MB）。`_initialize_from_trained_model` 本來就不做 block 過濾——**官方不裁切，我們也不該裁**。見 研究總覽 §4.1 |
| `crop_ckpt_to_block.py` | 同上，另一個裁切實作 |
| `visibility_crop_ckpt.py` | 同上，可見度版裁切 |

## 服務的假說已被推翻

| 工具 | 被推翻的假說 |
|---|---|
| `ray_clear_init.py` | 「depth-init 是厚殼、要清掉擋在錨點前的偽前層」。**局部 PCA 厚度 0.81，六個模型最薄 ⇒ 沒有殼**。附帶：它內建的 CPU trim 預測器校準失敗（硬 z-buffer vs transmittance，模型種類錯）。見 研究總覽 §7.1 |
| `make_global_sparse_init.py` | 「把 depth-init 改成稀疏全域」。同顆數下我方深度雲比 coarse **粗 2.3 倍**（間距 0.0658 vs 0.0285），省下的顆數等量從表面解析度扣走 |
| `floater_cleanup.py` | 事後 floater 清理（廢案 #7，2026-06-18 判死路）|

## 已被更好的工具取代

| 舊 | 新 |
|---|---|
| `diag_multiview_consistency.py` | `measure_view_consistency.py` |
| `diag_depth_consistency.py` | `measure_depth_agreement.py` |
| `diag_offsurface_gaussians.py` | `floater_label.py:surface_distance` |
| `plot_metric_timeline.py` | 結論已寫入 `紀錄/研究總覽.md` §2 |

## 一次性事件工具

| 工具 | 事件 |
|---|---|
| `reanchor_eval.py` | 2026-07-19 rasterizer 重錨（`pin ≠ 實裝` 事件），已結案 |
| `compare_server.py` | 本地 web 對照檢視器，沒在用 |
