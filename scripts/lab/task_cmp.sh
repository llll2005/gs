#!/bin/bash
# ★★★★★★★ 對比實驗的**標準骨架**（使用者 2026-09-13 定調：一律以 2 萬多步為基準長度）
# 用法：task_cmp.sh <block_id> <arm>
#
# ## 為什麼是「2 萬多步」而不是 60k
# 60k 一趟 7.6h（lab）/ 8.8h（本機）。而判準是**四個指標的正負號模式**
# （例如「紋理比↑ 而光度↓」那個虛假細節簽名還在不在），不是誰分數高
# —— 名次要 ~51k 步才穩，但符號模式好分辨得多。⇒ 用 1/3 的時間問對的問題。
#
# ## ★ 排程必須**跟著全長縮放**，否則比較本身是歪的
# 直接把 60k 的配方截斷到 2 萬步，會得到一個**不同的 regime**：
# ```
#   densify_until_iter 30000  >  全長 21,920  =>  整趟都在增生，**沒有收割期**
#   means_lr max_steps 60000                  =>  跑完時 LR 才衰減到 36%
# ```
# 所以本腳本按全長重新配置，而且用的是**已驗證過的比例**而不是隨手挑的數字：
# ```
#   trainer.max_steps        = STEPS
#   means_lr max_steps       = STEPS            （LR 積分在跑完時剛好走完）
#   densify_until_iter       = STEPS / 2        （sched30 驗證過的 50%，兩塊都顯著改善）
# ```
# **刻意不縮放的兩項，以及為什麼（這關係到「不偏向任何選項」）**：
# ```
#   densify_from_iter 1000   暖身，不是全長的比例。而且 `cost_add_densify` 需要
#                            densify 晚於第一次 trim（step 500）=> 縮成 ~365 會讓成本臂
#                            **直接失效** ⇒ 縮放它就是偏向對照組。
#   cap_max 2.6M             容量上限，不是排程。所有臂相同即可。
#   densification_interval   縮放它會改變 densify 事件數，而沒有任何「已驗證的比例」
#                            可依循 ⇒ 隨手挑一個就是引入偏向。固定 = 所有臂相同。
# ```
# ⚠⚠ 連帶後果：這個排程的跑次**不能**與 `lab/speed3`（60k 排程）的 10,960/21,920 直接比
#   —— 所以 arm `base` 存在，它就是這個排程下的基準。**比較一律在同排程的臂之間做。**
#
# ## 步數為什麼不是一個固定數字
# val 的節奏是 `check_val_every_n_epoch: 20` ⇒ 每 20×相機數 步一次。各塊相機數不同：
# ```
#   b6  548 台 -> val 每 10,960 -> 兩期 = 21,920
#   b12 653 台 -> val 每 13,060 -> 兩期 = 26,120
#   b13 667 台 -> val 每 13,340 -> 兩期 = 26,680
# ```
# 取「不少於 20,000 的最小整數期」=> 每塊都剛好停在 val 點上，兩個觀測點可直接對齊。
# 要覆寫用 `STEPS=<n>`。
#
# ## ★ w 的校準值（2026-09-13 在**新資料**上實測，兩塊一致）
# 目標：令 `1 + w*ceiling(訊號)` = 25.8，與 absgrad（已證有效那個）同動態範圍。
# ```
#            ceiling(top5%)/mean      w = 24.8/ceiling
#   1/c      b12 6.32 / b13 6.14      3.92 / 4.04   => 取 **4.0**
#   c        b12 8.97 / b13 8.58      2.76 / 2.89   => 取 **2.8**
# ```
# ⚠ 旗標 docstring 建議的 **2.278** 是用**舊資料**（ceiling(1/ĉ)=10.89x）算的
#   ⇒ 在新資料上只有目標強度的 0.57 倍。而「強度沒對齊」正是我指認為原始否證主因的東西
#   ⇒ 沿用舊值等於用相反方向重蹈同一個錯。**校準值要跟著資料重算。**
set -u
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_cmp.sh <block_id> <arm>}
ARM=${2:?同上}

# ★ B0 = 當前操作點的 Load（`max_view Σ (2r/16)^2`），由 tools/cost_budget_calibrate.py
#   在**已訓練好的 ckpt** 上一次前向量出來（不需要訓練，見該工具檔頭）。
#   b6  speed3@14999   N≈2.60M  B0 = 17,987,534  (B/N 6.92)
#   b12 gate15000@15000 N=2.34M B0 = 14,177,821  (B/N 6.06)
#   ⚠ 不可沿用 tools/cost_budget_probe.py 的 23.1M —— 那支用解析投影，而生效的是光柵器 radii。
case "$BLK" in
  6)  NCAM=548; B0=17987534 ;;
  12) NCAM=653; B0=14177821 ;;
  13) NCAM=667; B0=0 ;;
  *)  NCAM=${NCAM:?未知的 block，請用 NCAM=<相機數> 指定}; B0=${B0:-0} ;;
esac
PERIOD=$((NCAM * 20))
STEPS=${STEPS:-$(( ( (20000 + PERIOD - 1) / PERIOD ) * PERIOD ))}
HALF=$((STEPS / 2))
echo "block $BLK：$NCAM 台相機 / val 每 $PERIOD 步 => 全長 $STEPS 步、densify_until $HALF"

case "$ARM" in
  base)       EXTRA=();                                                        EXP=0 ;;
  # ★ 參考軌跡（2026-09-14）：行為與 base 完全相同，只多兩個純打印旗標 ——
  #   `cost_budget_report 150`（每個 densify 間隔印：區間最壞視角 + 報告區間平均 Load）
  #   `churn_report true`（每個 densify 事件印 dead->relocate／新增數）
  #   用途：預算排程實驗的 Load_ref(t) 與 churn 基準；也可當 base 的重複樣本。
  refrep)     EXTRA=(--model.density.init_args.cost_budget_report 150
                     --model.density.init_args.churn_report true);               EXP=0 ;;
  # ★★ 相對預算 B(t) = rho(t) x Load_ref(t)（2026-09-14；參考軌跡 logs/refrep_b<塊>_traj.csv）
  #   三組的 rho 依 b6 參考軌跡加權算出，使「目標總成本」Σ rho(t)·Load_ref(t)（增生期）相同 => 只比排程形狀：
  #     refc   固定 0.5000
  #     refh1  前鬆後緊 0.8452 -> 0.2452（使用者假設一：可用資源隨訓練減少）
  #     refh2  前緊後鬆 0.1548 -> 0.7548（使用者假設二：資源本就不足，前期寬鬆會 churn／後期無法精修）
  #   ⚠ 閘門只擋加點、擋不到 scale 成長 => 比較時要用**實際花費**（報告的區間平均積分）對齊，不是目標值。
  refc|refh1|refh2)
              REF="logs/refrep_b${BLK}_traj.csv"
              [ -f "$REF" ] || { echo "⛔ 缺參考軌跡 $REF（先跑 refrep 臂並抽出軌跡）"; exit 2; }
              case "$ARM" in
                refc)  RS=0.5000;  RE=0.5000 ;;
                refh1) RS=0.8452; RE=0.2452 ;;
                refh2) RS=0.1548; RE=0.7548 ;;
              esac
              EXTRA=(--model.density.init_args.cost_budget_ref_csv "$REF"
                     --model.density.init_args.cost_budget_ratio_start "$RS"
                     --model.density.init_args.cost_budget_ratio_end "$RE"
                     --model.density.init_args.cost_budget_report 150
                     --model.density.init_args.churn_report true);               EXP=3 ;;
  # ⚠⚠ 2026-09-13 時序：`cs_costdir` 這個**已經在跑的**跑次是 03:15 啟動的，當時腳本寫的還是
  #   w=2.278（旗標 docstring 依**舊資料** ceiling=10.89x 算的）。我在 04:2x 才依新資料重算成 4.0。
  #   ⇒ **磁碟上的 `cs_costdir` = w 2.278**，不是這一行現在寫的值。
  #   不重跑它，改成把兩個強度都留著當**強度掃描**（記憶 observability_saves_runs 記過
  #   「加權過猛」害過一次：權重實現 148.7x 而非設計 25.8x）：
  #     cs_costdir      w 2.278  = 校準目標的 0.57 倍（已跑）
  #     cs_costdir_cal  w 4.0    = 依新資料校準（ceiling(1/c) b12 6.32 / b13 6.14 => 24.8/c）
  # ★ init 第四臂（2026-09-16 lab 20k 零機制：sfmfill 贏 sfm b6 +0.165／b13 +0.187，四項同向）
  #   問題：這個增益在**完整配方**（absgrad + noise + trim）下還在不在。
  #   ⚠ 走 points_from ply 而不是 initialize_from：後者連 scale 初始化也換掉，就不是單變數了。
  sfmfill)    P="sfmfill_init/block_${BLK}.ply"
              [ -f "data/matrix_city/aerial/train/block_all/$P" ] || { echo "⛔ 缺 sfmfill_init/block_${BLK}.ply"; exit 2; }
              EXTRA=(--data.parser.points_from ply --data.parser.ply_file "$P");   EXP=2 ;;
  costdir)    EXTRA=(--model.density.init_args.cost_add_densify 2.278);         EXP=1 ;;
  costdir_cal) EXTRA=(--model.density.init_args.cost_add_densify 4.0);          EXP=1 ;;
  costtaming) EXTRA=(--model.density.init_args.cost_add_densify -2.8);          EXP=1 ;;
  # ★★ 把**週期 trim 的剪枝判準**從價值 v 換成每單位成本的價值 v/c（零額外成本，c 本來就在算）
  #   依據：rho(v,c)≈0（三個模型）=> 兩種排序選出**不同**的一批（top10% 重疊 54~67%）；
  #        按 v/c 貪婪在同成本預算下多拿到 +48~57%（10% 預算）；
  #        而按 contribution 排序剪枝只比**隨機**好 1.27 倍 => 現行判準本身很弱。
  #   ⚠ 這是「破壞集合」那一側；vpc_prune_frac 的 -2.34 dB 是「把低 v/c 的粒子**搬走**」，
  #     移除與搬移不同，但先驗要謹慎 => 判準看四個指標的正負號模式，不是單看 PSNR。
  # ══════════ 2026-09-21 新機制 ══════════
  # ★★★ conic：精確圓錐**不對稱**外接盒，取代上游的線性化 `truncated_R * extent(1)`
  #   離線 Σtile **-41.2%**（無損）、本機 forward **-24.5%** / fwd+bwd -15.1%。
  #   ⚠ **會改變渲染輸出**：它把線性化砍掉的覆蓋補回來（11.3% 的（顆粒,相機）組合被低估，
  #     最大 73,491 px），所以不是免費的加速 —— 這個臂要回答的就是「品質有沒有代價」。
  #   ⚠⚠ 判準不能只看 PSNR：本專案已有**四次**「虛假細節簽名」（紋理比↑ 而光度↓）。
  #     補回邊緣貢獻正好是可能製造那個簽名的形狀 => **紋理比必須與光度一起看**。
  #   ⚠ 與 base 比時**不可引用舊跑次的數字**：加了這段程式碼會擾動 codegen（見 forward.cu 的
  #     長註解，最大 1.8/255、平均 9e-08）=> 對照臂必須用**同一個 .so** 現跑。
  conic)      EXTRA=(--model.renderer.init_args.exact_conic_aabb true);         EXP=1 ;;
  # 疊加臂：conic 會改變每顆的 tile 足跡，而 trimvpc 的 c 正是足跡 => 兩者可能互相影響。
  # ⚠ 只有在 conic 單獨臂**先**判定無品質代價後才有意義，否則分不出是誰的效果。
  conicvpc)   EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.renderer.init_args.trim_by_value_per_cost true);   EXP=1 ;;
  # ★★ tilecal：**純診斷臂**，行為與 base 完全相同（cost_budget=0，沒有任何閘門生效），
  #   只是把 Load 改用**精確的逐顆 tile 數**並同時印出代理值與比值。
  #   要回答的：換算係數隨訓練怎麼走。已知它**會變號** —— 初期代理是精確的 0.280 倍
  #   （低估 3.6 倍，因為忽略 tile 量化），後期 1.340 倍（高估，正方形外接盒主導）。
  #   ⇒ 舊的 `cost_budget` 數值**不能除以一個常數**搬到新單位，要整條曲線。
  #   產物：logs 裡的 [cost-budget] 行 => 抽成 CSV 給後續的相對預算臂當 Load_ref(t)。
  tilecal)    EXTRA=(--model.density.init_args.exact_tile_cost true
                     --model.density.init_args.cost_budget_report 150
                     --model.density.init_args.churn_report true);              EXP=0 ;;
  trimvpc)    EXTRA=(--model.renderer.init_args.trim_by_value_per_cost true);      EXP=1 ;;
  # alpha 0.5：離線掃描說「每犧牲一單位 Σv 回收的 Σc」在這裡有峰（41.09 vs 現行 17.93），
  #   而 alpha=1（上面那個 trimvpc）已越過峰且端到端輸基準 -0.57 dB。
  trimvpc05)  EXTRA=(--model.renderer.init_args.trim_by_value_per_cost true
                     --model.renderer.init_args.trim_value_per_cost_alpha 0.5);     EXP=1 ;;
  dssim05)    EXTRA=(--model.metric.init_args.lambda_dssim 0.5);                EXP=1 ;;
  both)       EXTRA=(--model.density.init_args.cost_add_densify 4.0
                     --model.metric.init_args.lambda_dssim 0.5);                EXP=2 ;;
  # ★★ 命題的**約束端**：把生長條件從顆數 `N<=cap_max` 換成渲染成本 `Load<=cost_budget`
  #   => N 不再是我設的定值，而是**從場景長出來**（使用者 2026-09-13 指出固定 N 會隨場景複雜度失準）。
  #   為什麼排在**緊預算**：tools/rho_value_cost.py 量到背包天花板是預算鬆緊的函數 ——
  #     預算佔總成本 10% => +57%／25% => +22~27%／50% => +6~9%／75% => +1~2%（b12/b13 一致）
  #   而被否證的三個成本感知變體跑在 cap 2.6M ＝**寬鬆端**，天花板只有 2~9%
  #   ⇒ 「它們輸」與「這方向沒用」是兩回事。
  #   ⚠ `cap_max` 保留當安全閥：VRAM = 逐顆儲存（只看 N）＋ binning（只看成本），
  #     本旗標只約束後者（旗標 docstring 明文要求）。
  cb50)       [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 2)));         EXP=1 ;;
  cb25)       [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 4)));         EXP=1 ;;
  # ★★★ 2026-09-13 使用者提的組合（我漏掉的格子）：**緊預算 + 成本感知增生**。
  #   我一直用「①取樣端是在寬鬆預算下量的」替它的三連敗辯護，但 cb50/cb25 只開了
  #   `cost_budget`、**沒有**開 `cost_add_densify` => 那個辯護從來沒被真正測過。
  #   這一格是它的判決實驗：
  #     贏 => 「成本感知取樣需要預算真的綁住才有用」成立，命題活過來
  #     輸 => 藉口用完，取樣端**永久關閉**（三個年代、五種變體、鬆緊兩端都輸過）
  #   ⚠ 使用者原本提的是 `cap 5M`，但實測 lab 同塊 N=1.85M->3.26GB / 2.34M->4.48GB
  #     （2.49 GB/百萬顆）=> 5M 約 **11.1 GB**，是 6GB 信封的兩倍，不可行；
  #     而且調高 cap 是把約束**放鬆**，方向相反。真正綁住的是 cost_budget ——
  #     實測 cs_cb50 在 49% 時只有 **0.45M 顆**而 cap 還是 2.6M。
  #   ⚠⚠ 而 `B0/2` **其實不是緊端**：背包表按「預算佔總成本」分欄，50% 那欄上界只有
  #     +5.4~8.9%，25% 才是 +21.5~26.6%。=> 判決要做成**兩點**，cb25cost 才是主力，
  #     cb50cost 是「上界隨預算放大」這條曲線本身的對照（預期增益更小）。
  # ⚠ 加 `cost_budget_report 500`：讓預算**自證**（每 500 步印 N / Load / 預算）。
  #   它是**純打印**（`_update_load` 在 cost_budget>0 時本來就會跑）=> 不改變訓練行為。
  #   2026-09-13 的教訓：cb50/cb25 跑完也沒辦法從 log 證明預算真的在綁。
  cb50cost)   [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 2))
                     --model.density.init_args.cost_add_densify 4.0
                     --model.density.init_args.cost_budget_report 500);          EXP=2 ;;
  cb25cost)   [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 4))
                     --model.density.init_args.cost_add_densify 4.0
                     --model.density.init_args.cost_budget_report 500);          EXP=2 ;;
  *) echo "⛔ 未知的 arm：$ARM（base|refrep|refc|refh1|refh2|sfmfill|costdir|costdir_cal|costtaming|dssim05|both|cb50|cb25|cb50cost|cb25cost|trimvpc|trimvpc05|conic|conicvpc|tilecal）"; exit 2 ;;
esac
# run_fit 收尾會 diff resolved config；基準就是同排程的 `cs_base`
# ★ 2026-09-21 `CITYGS_FAMILY` 換家族名（預設 `cs_`）。用途＝同一批臂換一個排程再跑一次
#   （例：60k 判決用 `cs60_`）。⚠ 它**保留 RUN_PREFIX** —— 直接用 `CITYGS_RUN_NAME` 會
#   繞過前綴，在 lab 上就把產物寫到 `outputs/` 而不是 `outputs/lab/`，違反命名空間規則。
#   config diff 的基準也跟著換家族，否則會拿 60k 的臂去跟 22k 的 cs_base 比。
export CITYGS_DIFF_VS="${RUN_PREFIX}${CITYGS_FAMILY:-cs_}base"
export CITYGS_DIFF_EXPECT=$EXP
# ★ 2026-09-18：`CITYGS_RUN_NAME` 可覆蓋跑次名。用途＝**同一份配方、換個名字再跑一次**
#   （例：獨佔計時對照，不能讓 run_fit 把既有的 cs_base 搬走）。不設就是原本的 cs_<arm>。
run_fit "${CITYGS_RUN_NAME:-${RUN_PREFIX}${CITYGS_FAMILY:-cs_}${ARM}}" "$BLK" \
  --model.initialize_from null \
  `# ★ 2026-09-21 加入：uint8 影像快取 + 不載入權重為 0 的深度圖。` \
  `#   09-18 已驗證送進訓練的 GT 影像**逐位元不變**（60 步 x2 的 sha1 全同），` \
  `#   只省 RAM（本機 12.6 -> 4.8 GB）。lab 三槽平行原本被 RAM 綁住（每個跑次快取 17~20 GB），` \
  `#   這是讓三槽真的能用的前提。所有臂一律套用 => 家族內仍是單變數。` \
  --data.image_uint8 true \
  --data.skip_unused_depth true \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --trainer.max_steps "$STEPS" \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps "$STEPS" \
  --model.density.init_args.densify_until_iter "$HALF" \
  "${EXTRA[@]}"
