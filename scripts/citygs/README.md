# 官方 CityGaussian 的資料準備腳本（2026-09-12 自上游抓回）

**這四支腳本本專案原本沒有** —— 而 2026-09-12 查出影像↔相機配對壞掉的根因，
很可能就是當初沒有用它們（見 `紀錄/研究總覽.md` §16.11）。
上游路徑是 `scripts/citygs/`（不是 `scripts/`，`doc/data_preparation.md` 裡寫的路徑會 404）。

```
untar_matrixcity_train.sh   把 block_1..10.tar 解開並把 *.png 移進各自的 input/
untar_matrixcity_test.sh    測試集同上
data_proc_mc.sh             ★ 關鍵：蒐集影像 + 用下載的預算 COLMAP 覆蓋 sparse
data_proc_mc_scratch.sh     從頭跑 COLMAP（5000+ 張，很久；有預算結果就不用）
run_citygs_mc_aerial.sh     官方的 aerial 訓練/評測流程（我方沒有在用，留作對照）
```

## `data_proc_mc.sh` 為什麼是關鍵

它呼叫 `tools/transform_json2txt_mc_aerial.py`，而那支的命名邏輯是：
```python
for idx, frame in enumerate(contents['frames']):     # 依 transforms_train.json 的順序
    file_path = '{:0>4d}'.format(idx) + '.png'        # 0000.png, 0001.png, ...
    cp <該 frame 的原始影像> -> input/0000.png
```
⇒ **權威命名 = 4 位數、索引就是 JSON 裡 frames 的順序**，而下載來的預算 COLMAP sparse
  認的就是這組名字（`0000.png`..`5620.png`）。
⇒ 我方 `input/` 是 `000001.png`..`005621.png`，若那是 4 位數集合的單純改名則配對正確；
  實測**不正確**（6 台裡 4 台配錯，偏移 +1/+2/+12/+42/+210 隨區段變化）
  ⇒ 那批檔案是用別的順序重新蒐集的（很可能按檔名 glob 十個 block），
    而十個 block 的邊界處會累積漂移 —— 這就是偏移量不固定的原因。

## 下載位址

```
MatrixCity small_city aerial   https://github.com/city-super/MatrixCity
預算 COLMAP（官方建議直接用）   https://drive.google.com/file/d/1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7/view
                               備用 https://pan.baidu.com/s/1zX34zftxj07dCM1x5bzmbA?pwd=1t6r
                               -> 解到 data/colmap_results/
Depth-Anything-V2-Large        https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

## ⚠ 重建後**必須**驗證才算成功

```
python tools/verify_image_pairing.py --from-name 2900 --to-name 3000
```
現行資料：6 台裡 4 台的指派檔案分數不比隨機檔案好（`0242.png` 排名 2101/5621）。
重建後每一台都該是 **rank 1/5621** 且 margin 明顯。**沒過就不要開始訓練。**
