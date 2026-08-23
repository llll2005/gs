# _ctx —— 給 Claude 的精簡提要（2026-08-21）

正文是 `研究總覽.md`。這份只放「開場就該知道、且查一次要花很多 token」的東西。
**有衝突以 `研究總覽.md` 為準。** 更新本檔時只增刪要點，不要寫敘事。

## 一句話

6GB 筆電上的 city-scale 2DGS。論文命題＝**成本感知密度控制**：既有方法用顆數計價，
我方用**渲染成本 `c_i`（螢幕 tile 足跡）**計價並解 `max Q s.t. (1/K)Σc_i ≤ B`。

## 現在的數字

```
現行最佳  coarseft_b12  26.518 / 0.7736 / 0.3522   2.60M   coarse-init + sched30 全部機制
          （中位數 +0.68 dB = 21x 噪音底，但尾巴不動；coarse 用的還是舊 scales，未達上限）
次佳      sched30_b12   26.377 / 0.7711 / 0.3542   2.34M   cap2.6M densify_until30k reg0.002
                                                            lambda_normal 0 depth_loss 0
同配方    sched30_b7    24.993 / 0.8009 / 0.2554   2.34M   （跨塊不可比：b12 半面水）
論文錨點  CityGaussianV2 全域 27.23（8×A100，只有 PSNR）
```
⚠ 我方全是 **val ⊂ train**，27.23 是留出測試集 ⇒ **方向對我方有利，不可直接比。**
現成腳本：`scripts/task_sched30.sh`（b12）／`task_sched30_b7.sh`（b7）。

## 噪音底（效應量一律除以它報倍數）

```
PSNR 0.0325   SSIM 0.00091   LPIPS 0.00200   紋理比 0.00314   低頻誤差 0.00028
```
⚠ n=2 單對，是差值的一次抽樣不是 σ。

## ★ 真正的瓶頸：尾巴（一切決策先看這個）

```
最差視角跨五種配方全距 0.08 dB（平均 0.33 / 中位 0.76 / 最好 0.88）
最差視角從 step 15,000 起 45,000 步只改善 4.5%（同期平均改善 39%）
修好最差 25% = +1.55 dB  <- 現行最強機制(sched30 +0.327)的 4.7 倍，且超過到 27.23 的缺口
```
⇒ **2026-08-23 定案：換起點也沒用。** 四種 init／trim 組態（汙染 depth-init／修正
depth-init／SfM-init／關起始 trim）最差視角全距僅 **0.15 dB**；跨六種配方的**逐視角
難度排名 Spearman ρ = 0.990**（1729 與 2999 永遠是最差兩名）
⇒ **尾巴是內容決定的，「調配方」與「換起點」兩整類介入都碰不到。**
⇒ 剩下的路：①搞清楚那些視角難在哪（內容層級，還沒做過）②對那些區域投不對稱資源
（對得上成本感知命題）③把「平均值對雙峰分布是爛摘要」本身當貢獻。
⇒ **所有實驗除平均外一律報最差 10%**（`tools/tail_analysis.py`）。

## 判準（2026-08-21 改判，六個最常誤用的地方）

1. **一律多指標**（`tools/audit_all.py`）：曾出現五個指標五個不同第一名。
2. **`geometry_health.py` 的 floater% 與結構誤差反向**（r=−0.739）⇒ 不可當結構判準。
3. **紋理比與結構誤差正相關**（r=+0.659）、可用拉高局部對比灌水 ⇒ 不可當優化目標。
4. **結構機制用 `tools/lowfreq_error.py` 判，不要用 PSNR**（小效應時解析度差約 3 倍）。
   跨配方大差異時 PSNR 才是有效代理（r=−0.885）。
5. **相關不是因果，不可預測單一介入**（我因此翻車兩次）。
6. **整體 PSNR 在 b12 被水面稀釋**（一半平坦水面，均勻色塊也 32-40 dB）。
   `tools/lowfreq_split.py` / `region_split_from_test.py` 可分水面／建築（純 CPU，訓練中可用）。

**⚠ 每個配方都要看圖。** 使用者目視發現的疊影，PSNR/LPIPS/建築低頻/floater% **全都沒抓到**。

## ⛔⛔ depth-init PLY 汙染（2026-08-22 發現，未解決前不要下 init 相關結論）

`utils/depth_init_blocks.py:126` 有和 dataparser **一模一樣的 zfill off-by-one**，
而 2026-08-12 的修正只修了 dataparser。**`depth_init/*.ply`（2026-05-29）的每一顆點
都是用鄰幀的深度圖擺位置的**，實測位移中位 **6.18× 表面間距**。
⚠ **但殼的厚度幾乎沒變**（20.0× → 19.5×）⇒ **殼是單目偽深度的固有性質，不是這個 bug**；
bug 做的是「在同樣厚的殼裡把點打亂」。⛔ 我原本說的「35.6% 汙染 = 4.1 倍」量法錯誤已撤回
（比較相鄰深度圖的同一像素，但相機移動了，那是不同內容）。
✅ 程式已修，`depth_init_fix/` 重生中；`scripts/task_fixdepth.sh` 量它值多少 dB。
✅ 臂與臂的比較仍成立（同一混淆在每一臂，且監督是對的）；❌ 絕對值要重測。

## 六個踩過的程式陷阱

1. **⛔ 不要同時開 `screen_size_prune_px` 與 `screen_prune_emergency_px`**：兩者讀同一個
   `_max_radii2D`，但 emergency 會把它從「光柵器實際半徑」覆寫成 `projected_radius()` 的
   **解析上界**（含相機後方的點）⇒ 300px 門檻變成每次 densify 刪 25% 族群。
   實測 `scrprune_b7` **PSNR 13.498**（對照 24.993）。單開 `screen_size_prune_px` 是安全的
   （它一直在跑，只是光柵器半徑從沒超過 300px ⇒ 等於無作用）。
2. **`results.txt` 會被 `main.py test` 從 `val/*` 覆寫成 `test/*`** ⇒ 一律從 tensorboard 讀。
3. **`test/` 可能有多組 ckpt 的圖**（earlyckpt 留四組）⇒ 用 `_final_test_dir()`，
   別用 `glob(test/*/*.png)`（曾汙染建築低頻 +22%）。
4. **批次刪／搬既有檔案**：條件要用「這次產生的子目錄名」正向指定；
   `rm -rf $VAR` 與 `find -newer` 曾兩次毀資料 ⇒ sed 複製腳本後跑 `tools/lint_task_scripts.py`。
5. **`results.txt` 不寫步數** ⇒ 半途 OOM 的跑次留下看似正常的分數
   ⇒ **比較前先跑 `tools/run_status.py`**。
6. 重編 rasterizer 要 `rm -rf build` + 強制 `CUDA_HOME` + `--force-reinstall`，**確認 .so mtime 變了**。

## 破平衡公式（已 4/5 命中）

```
break-even interval = contribution_prune_interval × ln(1.05) / (−ln(1−prune_ratio)) = 231.5
interval > 231.5 → 顆數衰減；< → 成長
```
另：**顆數交付 = 0.9 × cap**（trim 每 500 步剪 10%、densify 每 150 步加 5%，
兩者同時在 densify_until 停 ⇒ 凍結在最後一次 trim 的輸出）。

## 常用指令

```bash
# 單塊訓練（現行最佳）
bash scripts/task_sched30.sh

# 多指標總表（含最差10%、建築低頻；從 tensorboard 讀，不看 results.txt）
python tools/audit_all.py

# 尾巴分析 / 結構誤差 / 分水面建築
python tools/tail_analysis.py <run>[:blk] ...
python tools/lowfreq_error.py <run>[:blk] ...
python tools/lowfreq_split.py <run> ...

# 比較分數前必跑（過濾半途死掉的跑次）
python tools/run_status.py --runs <name> ...

# 進度／死因
tail logs/quad_progress.log
```

## 隊列（2026-08-22）

```
（空）—— 三個實驗都已完成，GPU 閒置

sfminit_b12   尾巴 +0.15（4.6x 底）= 第一個推出 0.08 帶的介入，但平均 −0.097；
              且 depth-init 管線交付平手（起始 trim 砍掉它 72%）
scrprune_b7   ⛔ PSNR 13.498 崩潰，是配置不相容不是機制無效（見程式陷阱 1）
cap30_b7      顆數 +15% 只買到感知（SSIM/LPIPS/紋理比），結構不動、最差視角 −0.61
              => 顆數槓桿在 b12 與 b7 兩塊都關閉，cap 2.6M 是對的
```

## 鐵律

只用當前資料（SfM 2026-05-29 重生）／**GT 修正 = 2026-08-12 10:08:37，之前開跑的一律作廢
且 bug 對內容重的塊傷害 2.13 倍**／outputs 全是這台 6GB 跑的、不准質疑硬體／
繁體中文＋ASCII 數學／指令用腳本或單行／大實驗前先 CPU 驗前提／拿檔案當基準前先查來源／
排程只用 `scripts/runner.sh`＋`queue.txt`，不准自開 watcher。
