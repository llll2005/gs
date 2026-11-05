# tools/probe：紀錄引用的一次性分析腳本（2026-11-05 自 session 暫存目錄收進 repo）

這些是 `紀錄/實驗分析/` 各開啟線（31～34、27～29）引用的診斷腳本，原本放在 Claude session 的暫存目錄（重開機就消失）。
收進來只是為了**紀錄有出處、能重跑**；它們不是維護中的工具，路徑與參數多半寫死在檔頭或命令列（例：`--block 3 --models "名稱=ckpt glob"`）。
另有一批在 lab 的 `labrun/`（`time_1024.py`、`float*_blk_1029.py`、`orient_blk_1029.py`、`roof_cmp_1029.py`、`facade_ckpt_1030.py` 等），還沒收進來。

| 腳本 | 量什麼 | 紀錄 |
|---|---|---|
| `ground_layer_1103.py` | 地面格（SfM）上的貼地覆蓋、蓋滿格、地下覆蓋（Σ o·2π·su·sv ÷ 格面積）、躺平；定義寫在檔頭 | 31 |
| `sfm_dist_premise_1105.py` | 不透明顆粒離可靠 SfM 的距離，依類別（地下、海面上方、高處、其他） | 27、31、32 |
| `tower_probe_1105.py` | 高樓（頂高 > 100 m）各高度帶對 CityGS 表面／SfM 點的破洞 | 27、33 |
| `fitgap_tower_1105.py` | 訓練視角 vs held-out 的缺口（依高度帶）、低角度環繞玻璃塔 | 27、33 |
| `lowangle_1030.py` | 低角度透空／破洞 | 22、31 |
| `sea_prior_probe_1105.py` | 海面格的 SfM 點、海平面估計、海面遮罩（在 lab 跑） | 27、32 |
| `monodepth_sea_1102.py`、`sealevel_probe_1102.py`、`calib_mono_1104.py` | 單目深度在海面與地面的誤差、看穿判準校準 | 24、26、32 |
| `giants_1103.py` | 巨大面元（長軸 > 20 m）的位置與遮擋 | 26、31 |
| `holdout_cover_1105.py` | 留出驗證視角後 SfM 觀測數的損失 | 29 |
| `cap_predictors_1105.py` | 訓練前逐塊訊號與探針權重的相關 | 29 |
| `ctile_1105.py` | 塊外遮罩 C 區的像素／tile／warp 比例 | 28 |
