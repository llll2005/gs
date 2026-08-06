# _ctx —— 給 Claude 的精簡提要

正文是 `研究總覽.md`。這份只放「開場就該知道、且查一次要花很多 token」的東西。
**有衝突以 `研究總覽.md` 為準。** 更新本檔時只增刪要點，不要寫敘事。

## 一句話

6GB 筆電上的 city-scale 2DGS。論文命題＝**成本感知密度控制**：既有方法用顆數計價，
我方用**渲染成本 `c_i`（螢幕 tile 足跡）**計價並解 `max Q s.t. (1/K)Σc_i ≤ B`。

## 現在的數字

```
單塊最佳   oreg_0p002_b12       23.848 / 0.661 / 0.536   900k   cap1M reg0.002 interval150 60k
次佳       b12_cap2m_reg000     23.39                    1.80M  cap2M reg0
官方對照   official_ft_blk5     23.342 / 0.619 / 0.665   1.18M  官方流程於當前資料(4×4 blk5)
論文錨點   CityGaussianV2 全域  27.23（8×A100，只有 PSNR）
```

## 五個最常誤用的地方

1. **整體 PSNR 在 b12 沒有鑑別力**（一半是平坦水面，均勻色塊也 32-40 dB）。
   唯一驗證過的尺是**建築區紋理比**（`tools/rescore_by_content.py`），現行 kernel 0.315~0.393。
2. **⛔ P7 汙染**：2026-08-05~06 的 7 個跑次全中（v1 的 22.070、77.6k 的紋理比 0.170 都不可引用），
   更早的 60 個全乾淨。判準 `d_reg > 1`。
3. **離表面距離同時在量 SfM 密度** ⇒ 只能同塊比較，且**不可用它論證 SfM-init 較好**（循環）。
4. **覆蓋倍數／overdraw 只在同類間可比**（init opacity 0.99 vs 訓練後 0.03~0.11，指標不看 opacity）。
5. **驗「有沒有異常值」取 max 與計數，不要取樣。**

## 五個踩過的程式陷阱

1. `gaussian_splatting.py:180-215` **ckpt 路徑會連模型與 renderer 一起換掉**；PLY 路徑不會。
2. `add_new_gs = max(0, min(cap,1.05N) − N)` ⇒ **cap 低於現有顆數 = densify 永久關閉**。
3. `_initialize_from_trained_model` **不做 block 過濾**；官方不裁切 coarse，我們也不該裁。
4. 重編 rasterizer 要 `rm -rf build` + 強制 `CUDA_HOME` + `--force-reinstall`，然後**確認 .so mtime 變了**。
5. `main.py validate` 會**覆寫 results.txt**；要用 `main.py test --save_val`。

## 破平衡公式（已 4/5 命中）

```
break-even interval = contribution_prune_interval × ln(1.05) / (−ln(1−prune_ratio)) = 231.5
interval > 231.5 → 顆數衰減；< → 成長
```

## 常用指令

```bash
# 單塊訓練
python -u main.py fit --config configs/<cfg>.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 -n <name>

# 進度／死因
tail logs/quad_progress.log
tr '\r' '\n' < logs/<name>.log | grep -oE "val/psnr=[0-9.]+" | tail

# P7 汙染檢查（新跑次都該做）
tr '\r' '\n' < logs/<name>.log | grep -oE "d_reg=[0-9.e+-]+" | sed 's/.*=//' \
  | awk '{if($1+0>mx)mx=$1+0; if($1+0>1)c++} END{print "max="mx, "spikes="c}'

# 建築區紋理比（唯一有鑑別力的尺）
python tools/rescore_by_content.py --runs <name> [--at_step N]
```

## 隊列（2026-08-06 夜）

```
official_depthinit_blk5   跑中   官方 config 只換 depth-init，對照 23.342
v1clean_24k_b12           排隊   P7 傷害 + 訓練到 7.7 萬顆
```

## 鐵律

只用當前資料（SfM 2026-05-29 重生）／outputs 全是這台 6GB 跑的、不准質疑硬體／
繁體中文＋ASCII 數學／指令用腳本或單行／大實驗前先 CPU 驗前提／拿檔案當基準前先查來源。
