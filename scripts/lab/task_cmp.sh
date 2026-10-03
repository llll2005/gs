#!/bin/bash
# ★★★★★★★ 對比實驗的**標準骨架**（使用者 2026-09-13 定調：一律以 2 萬多步為基準長度）
# 用法：bash scripts/lab/task_cmp.sh <塊> <臂> [額外 CLI 覆寫...]（最後者勝；臂清單見 --help）
#   環境變數：STEPS（全長）／CITYGS_FAMILY（家族前綴，預設 cs_；60k 用 cs60_ 且 STEPS=60000）／CITYGS_LEAN=1（lean_train）／
#             CITYGS_CAP（cap_max）／CITYGS_RUN_NAME（覆寫跑次名）／CITYGS_DRY=1（只印指令不執行）
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
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
source "$(dirname "$0")/_common.sh"
# ⚠⚠ 2026-09-22：這三支先前**都不轉傳多餘參數**（沒有 "$@"）=> 從佇列想覆寫單一設定時
#   會被**靜默吃掉**。2026-09-22 釘 exact_conic_aabb=false 時中過一次（25 行全無效）。
#   ⇒ 一律 `shift` 掉自己的位置參數，把剩下的用 "$@" 傳到 run_fit 的最後（最後者勝）。
#   用環境變數也能做，但**不會出現在 resolved config** => 少一層本專案賴以驗證的保護。
BLK=${1:?用法: task_cmp.sh <block_id> <arm> [額外參數...]}
ARM=${2:?同上}
shift 2

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
  # ★ init 第五臂（2026-09-29）：20k 零機制下 dup4 對 sfmfill +1.38/+1.28 dB，但分數幾乎完全跟著**起始顆數**
  #   （dup 只把同一批 SfM 點複製 4 份＋抖動，沒有新位置資訊；20k 配方 densify 在 10k 停 => 起始少的長不到）
  #   問題：完整配方（60k、densify 到 30k、cap 2.6M，大家都會長到 cap 附近）下這個增益還在不在。
  #   PLY = task_sfmfill_sweep2.sh gen 的 dup4（其餘參數同 sfmfill 基準），同樣走 points_from ply。
  sfmdup4)    P="sfmfill_sweep/dup4/block_${BLK}.ply"
              [ -f "data/matrix_city/aerial/train/block_all/$P" ] || { echo "⛔ 缺 $P"; exit 2; }
              EXTRA=(--data.parser.points_from ply --data.parser.ply_file "$P");   EXP=2 ;;
  # ★ 2026-10-01：dup4 為什麼贏（60k 同 N +0.27/+0.32）—— 兩個候選機制分開測：
  #   fastgrow  預設 SfM init，只把 add_ratio 1.05 -> 1.2（每次增生的倍率），讓族群像 dup4 一樣約 2~3k 步就長滿
  #             （dup4 在 step ~2.3k 滿額、預設要到 ~14.3k；兩者 ∫N dstep 差 13.8%）=> 若 ≈ dup4，贏在「提早長滿」
  #   sfmdup2   dup 2 份（起始 1.2M）=> 看效果是否隨劑量變化
  #   其餘（densify_until 30k、cap 2.6M）全部不動 => 單變數
  fastgrow)   EXTRA=(--model.density.init_args.add_ratio 1.2);                  EXP=1 ;;
  # ★ 2026-10-01：逐 tile 前 K 名做成訓練期 trim 判準（renderer trim_by_tile_topk；事後剪枝同成本下最好）
  tilek)      EXTRA=(--model.renderer.init_args.trim_by_tile_topk true);         EXP=1 ;;
  # ★ 2026-10-01：v/c 與逐 tile 名次合成一個判別式 score = (v/c) / (1 + 最好名次/32)（背包：全場預算＋每 tile 容量）
  vpctilek)   EXTRA=(--model.renderer.init_args.trim_by_value_per_cost true
                     --model.renderer.init_args.trim_by_tile_topk true);         EXP=2 ;;
  # ★ 2026-10-01：opacity 二元熵正則（兩極化；權重與 opacity_reg 同量級，從 densify_from 開始才會被回收機制吃到）
  oent)       EXTRA=(--model.metric.init_args.opacity_entropy_reg 0.002
                     --model.metric.init_args.opacity_entropy_from_iter 1000);   EXP=2 ;;
  # ★ 2026-10-01：dup 系列更激進（scripts/lab/gen_dup_variants.sh 產的 PLY）——
  #   sfmdup5＝複製 5 份（起始 > cap）、sfmdup4j10／sfmdup4j025＝dup4 但抖動 1.0／0.25（jitter 從沒測過）
  sfmdup5|sfmdup4j10|sfmdup4j025)
              P="sfmfill_sweep/${ARM#sfm}/block_${BLK}.ply"
              [ -f "data/matrix_city/aerial/train/block_all/$P" ] || { echo "⛔ 缺 $P（先跑 scripts/lab/gen_dup_variants.sh）"; exit 2; }
              EXTRA=(--data.parser.points_from ply --data.parser.ply_file "$P");   EXP=2 ;;
  sfmdup2)    P="sfmfill_sweep/dup2/block_${BLK}.ply"
              [ -f "data/matrix_city/aerial/train/block_all/$P" ] || { echo "⛔ 缺 $P"; exit 2; }
              EXTRA=(--data.parser.points_from ply --data.parser.ply_file "$P");   EXP=2 ;;
  # ══════════ 2026-10-02 配方各項在新年代重驗（使用者：增生比例／absgrad／opacity_reg／noise_gate／trim）══════════
  #   這些設定都是**舊年代**（影像↔姿態錯開）選的，新年代沒有任何單變數測試（lab resolved config 全掃過：
  #   opacity_reg 一直是 0.002、fast_noise/noise_gate 只在 init 20k 與 absgrad 一起關、densify 只有 50%）。
  #   基準 = cs60_conic（conic 開、預設 SfM init）=> 每臂都帶 conic，只再改一項（diff vs cs_base 應為 2 項）。
  #   cdu100/75/25  densify_until = 全長 x 1.0 / 0.75 / 0.25（50% 就是 cs60_conic）
  #                 ⚠ 週期 trim 預設跟著 densify_until 停（renderer contribution_prune_until_iter=-1）=> 兩者一起動，
  #                   這正是「增生到幾 %」在現行程式裡的完整意義
  #   cag4 / cag0   absgrad_densify 4.0 / 0（2.0 是現行；0 = 新年代從沒量過 absgrad 本身有沒有用）
  #   coreg0        opacity_reg 0（現行 0.002）
  #   cnotrim       週期 trim 全關（diable_trimming；N 停在 cap 而不是 0.9 cap）
  #   cnogate       noise_gate_eps 0（fast_noise 不必訓練：tools/check_noise_opts.py 在 5 個 ckpt 上量到
  #                 與原算法只差浮點（相對差中位 ~6e-8）；gate 則有 ~1% 被跳過者噪音 >= 自身 Adam 步長 => 要訓練驗）
  cdu100|cdu75|cdu25)
              case "$ARM" in cdu100) DU=$STEPS ;; cdu75) DU=$((STEPS * 3 / 4)) ;; cdu25) DU=$((STEPS / 4)) ;; esac
              EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.density.init_args.densify_until_iter "$DU");        EXP=2 ;;
  cag4)       EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.density.init_args.absgrad_densify 4.0);            EXP=2 ;;
  cag0)       EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.density.init_args.absgrad_densify 0.0);            EXP=2 ;;
  coreg0)     EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.metric.init_args.opacity_reg 0.0);                 EXP=2 ;;
  cnotrim)    EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.renderer.init_args.diable_trimming true);           EXP=2 ;;
  cnogate)    EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.density.init_args.noise_gate_eps 0.0);             EXP=2 ;;
  # ★ 2026-10-02 sh3 -> sh2（使用者：參數壓縮省運算／空間）。官方 CityGSV2 空拍本來就是 sh2。
  #   每顆 58 -> 37 個 float（-36%）：Adam／逐顆前處理／VRAM（參數+梯度+2 動量 = 336 B/顆，2.34M 顆約 0.79 GB）；
  #   逐 (像素,顆粒) 配對的工作不變（SH 在光柵化前就逐顆解成 3 個顏色）。config 的「sh3 +0.295 dB」是舊年代量的。
  csh2)       EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.gaussian.init_args.sh_degree 2);                   EXP=2 ;;
  # ★ 2026-10-02 SH 曲線的另一端（使用者轉述 Gemini 建議：先量底線）：sh0 = 不隨視角變化的 RGB，每顆 58 -> 13 float（−78%）。
  #   sh0 若與 sh3 在噪音內 => 高階 SH 在空拍不值它的 VRAM（2.34M 顆約 −1.68 GB），連 YCbCr 都不用做。
  csh0)       EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.gaussian.init_args.sh_degree 0);                   EXP=2 ;;
  # ★ 2026-10-02「硬度」：2DGS 自帶的 depth distortion loss（把每條光線上的權重壓成一薄層＝實心表面），
  #   我方 config 一直是 0、**整個專案從沒開過**。2DGS 原文：有界場景 1000、無界場景 100（從第 3,000 步開始）。
  #   ⚠ 需要幾何通道 => 這兩臂不可開 lean_render（開了會在 backward 當場報錯）。
  cdist100|cdist1000)
              EXTRA=(--model.renderer.init_args.exact_conic_aabb true
                     --model.metric.init_args.lambda_dist "${ARM#cdist}");       EXP=2 ;;
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
  # ★★ 2026-09-30：depth loss 重測。關掉它（0.5 -> 0）的決定是用**舊的、被污染的** scales 量的
  #   （研究總覽 v2「待重測」）；現行 estimated_depths 與 estimated_depth_scales.json 是 2026-09-13
  #   影像↔姿態修正後重生的（依相機名稱存檔）=> 前提已倒，值得重測。0.5 = 修改前我方值 = 官方 aerial 配方值。
  depth05)    EXTRA=(--model.metric.init_args.depth_loss_weight.init 0.5);       EXP=1 ;;
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
  # ★★★ 2026-09-30：**精確成本計價**的預算（約束端從未在精確單位下測過）。
  #   舊 cb 臂用代理 Σ(2r/16)² 計價，而 conic 開啟後代理與真實成本方向相反（精確中位 -40% 時代理 +6%，
  #   資源彙整 §9.9e）=> 舊的否定結果不能延伸到精確單位。這裡：exact_tile_cost=true（閘門與 c 都用 Σtiles），
  #   B0X = conic 開的同排程終點模型（cs_conic@21,920）的**精確**最壞視角 Load 5,149,466（load_compare_b6_0921_1716）。
  #   比較方式同 §9.9d：同工具量終點精確 Load，與 conic 開的 cap 曲線（capc07/12/17，同批新跑）在同成本下比。
  cb50x|cb25x|cb50xcost|cb25xcost)
              case "$BLK" in 6) B0X=5149466 ;; *) echo "⛔ block $BLK 沒有精確 B0X（只標定了 b6）"; exit 2 ;; esac
              case "$ARM" in cb50x*) BX=$((B0X / 2)) ;; *) BX=$((B0X / 4)) ;; esac
              EXTRA=(--model.density.init_args.exact_tile_cost true
                     --model.density.init_args.cost_budget "$BX"
                     --model.density.init_args.cost_budget_report 500)
              case "$ARM" in *cost) EXTRA+=(--model.density.init_args.cost_add_densify 4.0); EXP=3 ;; *) EXP=2 ;; esac ;;
  cb50cost)   [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 2))
                     --model.density.init_args.cost_add_densify 4.0
                     --model.density.init_args.cost_budget_report 500);          EXP=2 ;;
  cb25cost)   [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 4))
                     --model.density.init_args.cost_add_densify 4.0
                     --model.density.init_args.cost_budget_report 500);          EXP=2 ;;
  *) echo "⛔ 未知的 arm：$ARM（base|refrep|refc|refh1|refh2|sfmfill|sfmdup4|sfmdup2|fastgrow|tilek|vpctilek|oent|sfmdup5|sfmdup4j10|sfmdup4j025|cdu100|cdu75|cdu25|cag4|cag0|coreg0|cnotrim|cnogate|csh2|csh0|cdist100|cdist1000|costdir|costdir_cal|costtaming|dssim05|depth05|both|cb50|cb25|cb50cost|cb25cost|cb50x|cb25x|cb50xcost|cb25xcost|trimvpc|trimvpc05|conic|conicvpc|tilecal）"; exit 2 ;;
esac
# run_fit 收尾會 diff resolved config；基準就是同排程的 `cs_base`
# ★ 2026-09-21 `CITYGS_FAMILY` 換家族名（預設 `cs_`）。用途＝同一批臂換一個排程再跑一次
#   （例：60k 判決用 `cs60_`）。⚠ 它**保留 RUN_PREFIX** —— 直接用 `CITYGS_RUN_NAME` 會
#   繞過前綴，在 lab 上就把產物寫到 `outputs/` 而不是 `outputs/lab/`，違反命名空間規則。
#   config diff 的基準也跟著換家族，否則會拿 60k 的臂去跟 22k 的 cs_base 比。
# ★ 2026-10-03 `CITYGS_LEAN=1` => 加 `lean_train true`（光柵器 lean_render：渲染逐位元相同、梯度 atomic 順序級、每步 -20.9%）。
#   使用者規則：同家族已有舊跑次且**要比時間**的臂不開（牆鐘／it/s 會不可比）；需要幾何通道的臂（cdist*）不可開。
#   resolved config 會多一項 lean_train => diff 的預期變數數 +1。
if [ "${CITYGS_LEAN:-0}" = 1 ]; then
  case "$ARM" in cdist*) echo "⛔ $ARM 需要幾何通道，不能開 lean"; exit 2 ;; esac
  EXTRA+=(--model.renderer.init_args.lean_train true); EXP=$((EXP + 1))
  echo "lean_train=true（CITYGS_LEAN=1）"
fi
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
  `# ★ 2026-09-21 CITYGS_CAP 覆寫顆數上限（預設 2.6M）。用途＝**cap_max 掃描**，` \
  `#   它是成本預算的對照基線：現有 7 個 budget 跑次已經有 (N, 分數)，但 cap 控制的只有一個點，` \
  `#   畫不出「純 cap 曲線」就無法判斷 budget 有沒有把點推到曲線上方。` \
  `# ⚠ cap 必須 > 起始 N（b6 SfM init 約 52 萬）：低於起始 N 時 add_new_gs 直接加 0，` \
  `#   N 只被 trim 衰減、停在與 cap 無關的地方 => 那不是 cap 控制，是衰減動力學。` \
  --model.density.init_args.cap_max "${CITYGS_CAP:-2600000}" \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --trainer.max_steps "$STEPS" \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps "$STEPS" \
  --model.density.init_args.densify_until_iter "$HALF" \
  "${EXTRA[@]}" "$@"
