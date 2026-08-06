# 開發日誌 2026-07-01 — distortion 負結果、RTG hybrid cap、init 模組化、統一訓練 console

## 0. 起點
延續 6/30:最佳 = block_7 aggr17（MCMC sh3 60k）= **24.30 / 0.755 / 0.309 @ 1.53M**。跑完 test 看圖:高空近正俯視完美，但**低空/斜視高樓/分塊邊緣**整片糊 + 拉絲 + 半透明疊影。

## 1. v1 = 只開 distortion（負結果，重要）
`mcmc_2dgs_60k_sh3_aggr17_aerial_ver1.yaml` = aggr17 + `lambda_dist:200 / dist_reg_from:4000`（單變量）。
- 結果 **22.63 / 0.702 / 0.396**（全面變差 vs 24.30），而且**原本清楚的圖也糊了**（0238: 25.05→18.31、1308: 29.60→24.48；連最好的 1597 也 31.12→30.28；最爛的 floater 圖 1708 幾乎沒動）。
- **顆數兩版都 1.53M（都撞 cap）→ 同樣點被擺得更糊，不是點變少。**
- **根因**：(1) distortion 定義=懲罰同光線多深度，但**細節住在深度微結構**（立面窗戶=多片略不同深度 surfel）→ 被夾到單深度→融化。(2) **診斷錯**：主病灶是**欠觀測**（高樓/斜視/邊緣訓練相機少），distortion 變不出多視角約束、只會吃真實細節。(3) 空拍**沒有空背景**給 distortion 清（Mip360/2DGS 原設計是無界場景清天空 floater），這裡每條光線都打實體→只能咬細節。**→ distortion 在 dense aerial 該丟；欠觀測是 coverage/全域一致性殘差（對上 6/25 那 ~3.7dB 結論）。**

## 2. v2 第一次 = RTG B1-B5 + distortion（OOM）
使用者要試 RTG 演算法當對照。查到能跑的模板 `RTG_s10_mc_no_ADDM`（B1 freeze + depth_init_immune + B3 mc + B4 revert + B5，**metric 用 CityGSV2Metrics**，無 cap）。
- ver2 = RTGStableDensityController 全 B1-B5 開 + 激烈 densify（interval 200 / grad 0.00015 / until 42000）+ distortion + `scale_reg`（借 MCMCCityGSV2Metrics，opacity_reg 設 0=freeze 下 inert）。
- **OOM**：log 顯示長到 **2,073,310 stable + B3 每步再加幾萬**（+19k/+23k…）→ RTG **無 cap** → 爆（正是先前警告）。

## 3. RTG hybrid soft-cap（新算法，解 OOM）
討論後結論:**硬 cap 控峰值(防 OOM)、軟門檻控穩態(重分配)，兩者管不同東西**。軟門檻單獨用不保證防 OOM（overshoot），故做**混合**。
- `RTGStableDensityController` 新增參數:
  - **硬**:`cap_max`(0=關)。B3 加點限剩餘預算、grad-densify 達 cap 停長、每次 densify 後 `_enforce_hard_cap`（importance=opacity×面積，透明/unstable 天然最低）保底到 cap。**永遠生效**。
  - **軟**:`soft_cap_enabled`/`soft_cap_band`/`soft_densify_scale`/`soft_prune_scale`/`soft_from_iter`/`soft_until_iter`。軟區=[cap−band,cap]，內提高 grad_threshold(少 grad 加)+ 縮 B3 n_add + B5 提早剪(max_unstable÷scale)。
- **收斂分析**:軟節流**關進 densify 窗口**→精修期族群固定→不傷收斂；剪的是**未證明點**(unstable/低重要度)→非 add-then-prune churn→不傷收斂。硬 cap 保證不 OOM。
- ver2 設 `cap_max=1.5M`(6GB sh3 已知安全)+ 軟帶[1.3M,1.5M]、densify 0.3、prune 1.5。
- MCMC 其他功能能否配 RTG:relocate/noise 冗餘或失效(freeze 下 α∈{0.99,0.1} noise auto-exempt)、`scale_reg` 可白拿(純 loss 項)、**唯一值得借的是 cap_max**(已做進 hybrid)。

## 4. init 模組化（opt-in，向後相容）
兩種 init 互斥(SfM 冷啟動 vs 載 ckpt/PLY)做成可換模組:
- 新 `internal/initializers/gaussian_initializer.py`:`Initializer` + `PointCloudInitializer` + `CheckpointInitializer(path, overwrite_config)`，各 Impl 複用 `setup_from_pcd`/`_initialize_from_trained_model`。
- `gaussian_splatting.py` 加 `initializer` 參數;`setup()` 有設就用、否則走舊 `initialize_from`(byte 級不變);`_initialize_from_trained_model(load_path, overwrite_config)` 改可帶參。
- ver2 已改新範式:`model.initializer: CheckpointInitializer(path=depth_init/block_7.ply)`，**指令不再需要 `--model.initialize_from`**。

## 5. 統一訓練 console（去洗版 + 可貼 AI）
- `internal/utils/train_console.py`:全域狀態 + `console_event`(沒 console 時 fallback print)+ `render_plain`(狀態檔)+ `render_rich`(終端固定底欄)。
- `internal/callbacks.py` 加 `TrainConsole`(Callback):rich Live 固定底欄(最近 val + 進度 + **VRAM，>90% 紅字 `!!`**)、上方滾動事件、寫 `outputs/<run>/.../train_status.txt`、非 TTY 降級為定期印快照、例外時停 Live 讓 traceback 完整顯示。
- RTG 的 7 個 `print("[RTGStable]…")` 全改 `console_event`。
- 接法:ver2 設 `trainer.enable_progress_bar:false` + `callbacks:[internal.callbacks.TrainConsole]`。純 opt-in。
- **⚠ 一度製造的 bug 已修**:最初把 callback 建成 `internal/callbacks/` 目錄→遮蔽既有 `internal/callbacks.py` 模組→會壞掉所有訓練的 `from internal.callbacks import SaveGaussian…`。**在任何 run 前抓到**:刪目錄、把 `TrainConsole` 併進 `internal/callbacks.py`。已驗證 entrypoint import 恢復。

## 6. 向後相容（實測）
所有改動 opt-in、預設=舊行為。`--print_config` 實測 exit=0 無錯:舊 MCMC config ✓、舊 CityGSV2 config ✓、**舊「已解析 config」(24.30 baseline resolved，無 initializer 欄位)→ test 指令仍正常 ✓**。console 對訓練時間/GPU 記憶體零實質影響(純 CPU、限流、不 allocate GPU)。

## 7. 參考檔同步
`configs/_REFERENCE_annotated.yaml` + `_REFERENCE_參數手冊.md` 補:來源 F2(RTGStableDensityController B1-B5 + hybrid cap 全參數)、H(initializer 模組)。

## 8. 待辦（部分已完成，見 §9/§10）

## 9. ver2 結果（跑完，負結果）
`mcmc_60k_sh3_aggr17_ver2_b7` = **19.93 / 0.580 / 0.547 @ 1.29M**（RTG B1-B5 + distortion + cap1.5M + scale_reg）。
- **基建全對**：cap 生效（**1.29M < 1.5M，沒 OOM**）、console/initializer/RTG 全跑通。
- **品質最差**（vs aggr17 MCMC 24.30 / v1 distortion 22.63）。根因：(1) **顆數 1.29M < cap** → RTG grad-densify + B3 自己就長不到 cap（比 MCMC 1.53M 還少）= **grad-densify 在 depth-init 長不起來**，cap 非限制只待命防 OOM；(2) **grad-densify 第三次輸 MCMC**（忠實 CityGSV2 21.78 / RTG sh2 ~21 / 今 19.93）+ 疊已知傷的 distortion。val 曲線 19.53→19.93 緩升未收斂但離 24.30 極遠。
- **判讀（不下死結論）**：RTG 線在此場景打不贏 MCMC；要乾淨測需拿掉 distortion 單獨跑，但期望仍 <24.30、低優先。**冠軍仍 aggr17 MCMC 24.30。真殘差 = 欠觀測/全域一致性，非 RTG/distortion 能補。**

## 10. 統一 console 設為預設（2026-07-01）
- `TrainConsole` 改成 `ProgressBar`(TQDMProgressBar) 子類；entrypoint `gspl.py` 預設 callback `ProgressBar`→`TrainConsole`。TTY+rich → `disable()` 靜音 tqdm、跑 rich 固定底欄；非 TTY / rich 失敗 / live=false → `super()` fallback 成原 tqdm（行為不變）。`cli.py` pbar_rate 注入認 `LazyInstance_TrainConsole`。
- **相容**：config 不用再寫 callback（寫了會兩個進度條衝突）；舊 config（citygsv2）自動吃新預設，--print_config exit0。
- 使用者真跑一次：rich 面板（header/最近事件/最近val+進度+VRAM）正常顯示、cap 生效無 OOM。修了 header `init顆數 0`（改在 on_train_start 建 header）+ init 路徑顯示 basename 避免換行。
