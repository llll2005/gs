# 歸檔的零散 CSV（2026-08-21 移入）

這些都是**單次分析的中間輸出**，且**全部在 GT 修正（2026-08-12 10:08:37）之前產生** ⇒
數字不可引用，保留只為追溯當時怎麼算的。

| 檔案 | 當時的用途 |
|---|---|
| `rescore_by_content.csv` / `rescore_postfix.csv` / `rescore_matched_step.csv` | 分水面／建築重新計分（bug 期） |
| `rescore_sparse_frontier.csv` / `rescore_uniform.csv` / `rescore_v1clean.csv` | 稀疏前緣、uniform-init、v1clean 的分區計分 |
| `dim_returns_b12_reg000*.csv` | 邊際報酬掃描（bug 期，結論已由 §7.0 取代） |
| `pose_consistency.csv` | 姿態一致性核對（結論已收進 `data_poses_are_gt` 記憶） |

**現行的資料出口只有兩個**：`紀錄/實驗總表.csv`（全歷史，含 era 標記）與
`紀錄/多指標總表_<日期>.txt`（修正後跑次的五指標快照）。
