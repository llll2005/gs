<h1 align="center">單卡重建一座城市：輕量、快速的大規模 2D 高斯場景重建</h1>

<p align="center">作者：黃則綸　·　指導教授：紀明德 教授</p>

<p align="center"><a href="https://llll2005.github.io/gs/"><b>展示網站：飛行影片與測試視角對照</b></a></p>

在一張消費級顯卡上重建整座城市：訓練顯存控制在 6 GB 以內、畫質不輸 CityGaussianV2，並大幅縮短訓練時間。

CityGaussianV2 用 8 張 A100 訓練 3 小時，每張用掉 14.8 GB 顯存；在一張 RTX 3090 上跑它的流程要約 71 小時。
本研究在同一張 RTX 3090 上用 1.33 小時訓練完整座 MatrixCity 空拍城市，訓練顯存峰值 2.7 GB。

## 結果

MatrixCity 空拍測試集 741 張影像。

| 方法 | 硬體 | 訓練時間 ↓ | 訓練顯存 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | F1 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| CityGaussianV2（論文） | 8 × A100 | 3.0 小時 | 14.8 GB | 27.23 | 0.857 | 0.169 | — |
| CityGaussianV2（同卡重跑） | 1 × RTX 3090 | 71.2 小時 | 12～21 GB | 27.26 | 0.867 | 0.154 | 0.597 |
| HUG（ICCV 2025） | 8 × A6000 | — | — | 28.02 | 0.883 | 0.142 | — |
| **我們** | **1 × RTX 3090** | **1.33 小時** | **2.7 GB** | **27.95** | **0.872** | **0.143** | **0.711** |

- 同樣硬體訓練快約 54 倍，訓練顯存只要 2.7 GB。
- 我們與 CityGaussianV2（同卡重跑）用同一支評測程式計算；標「論文」的列與 HUG 取自各自論文的表格。
- 訓練時間是整個訓練任務的牆鐘時間；F1 是幾何準確度。

## 方法

以 CityGaussianV2 的分塊 2D 高斯流程為基礎，拿掉先訓練整個場景的粗模型，場景切成 4×4 共 16 塊各自訓練再合併，並做了四項改良：

1. **用校正後的深度當幾何骨架**：以 SfM 稀疏點逐張校正單目深度（誤差中位數 2.15% → 0.15%），用來決定哪些像素屬於哪一塊、限制高斯不長到地面和屋頂下方、決定高斯的起始位置。
2. **把海面當成一個平面**：偵測水體，用多視角共識決定海平面，把漂浮在海上的高斯收回。
3. **只學看得到的顏色資訊**：空拍相機只看得到球諧顏色的 10 個方向，訓練時只學 8 個，每個高斯的參數從 58 個減到 34 個，訓練顯存峰值降 30%，畫質不變。
4. **讓每一步都更便宜**：精確跳過不影響畫面的計算、前 75% 的步數用低解析度、常數改用單精度。每塊平均 4.5 分鐘訓練完，16 塊合計 1.2 小時。

完整流程、公式、參數、消融與怎麼跑：[`紀錄/現行方法.md`](紀錄/現行方法.md)（§13 怎麼跑）。指令查詢：`bash scripts/help.sh`。

## 參考論文

- CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes（ICLR 2025）
- CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians（ECCV 2024）
- 2D Gaussian Splatting for Geometrically Accurate Radiance Fields（SIGGRAPH 2024）
- 3D Gaussian Splatting as Markov Chain Monte Carlo（NeurIPS 2024）
- Depth Anything V2（NeurIPS 2024）
- MatrixCity: A Large-scale City Dataset for City-scale Neural Rendering and Beyond（ICCV 2023）

## 致謝與授權

本專案修改自 [CityGaussian](https://github.com/Linketic/CityGaussian)（建立在 [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning) 之上），原本的說明保留在 [`README_CityGaussian.md`](README_CityGaussian.md)。
授權沿用上游的 CC BY-NC-SA 4.0，見 [`LICENSE.md`](LICENSE.md)。
