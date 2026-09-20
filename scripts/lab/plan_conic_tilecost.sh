#!/usr/bin/env bash
# ★★★★★★ 實驗排程：精確圓錐外接盒（conic）與精確 tile 成本（tilecost）
#   2026-09-21 設計。**在 lab 上執行**，它只會把任務追加進 scripts/queue.txt，不自己跑訓練。
#
#       bash scripts/lab/plan_conic_tilecost.sh          # 乾跑：只印出會排什麼
#       bash scripts/lab/plan_conic_tilecost.sh --go     # 真的追加進佇列
#       STEPS=60000 bash scripts/lab/plan_conic_tilecost.sh --go   # 改全長（見下方「長度」）
#
# ══════════════════════════════════════════════════════════════════════════════
# 這兩個機制在問什麼
# ══════════════════════════════════════════════════════════════════════════════
# **conic** —— 上游（3DGS/2DGS/Trim2DGS/CityGS 全家族）算 tile 外接盒的方式是
#   `radius = ceil(truncated_R * extent(1))`：先算 r=1 等高線的精確外接盒，再**乘一個純量**，
#   而且中心固定。但 ray-splat 交點對像素是仿射的 => 等高線是真正的圓錐曲線，透視下
#   不同 R 的等高線**不是等比放大** => 那個外推是**線性化**。
#   離線實測（b6, N=1.85M, 30 台）：線性化在 **88.7%** 的（顆粒,相機）組合上**高估**（白付 tile）、
#   在 **11.3%** 上**低估**（漏覆蓋，最大 73,491 px）。改成精確解：Σtile **-41.2%**、
#   本機 forward **-24.5%**、fwd+bwd -15.1%。
#   ⚠ **它會改變渲染輸出**（把漏掉的覆蓋補回來）=> 不是免費加速。本實驗就是要量品質代價。
#
# **tilecost** —— `tiles_touched[idx]`（逐顆 tile 數＝binning 成本本身）光柵器每步都算好，
#   但從未暴露；`cost_budget` / `vpc` / `cost_aware_densify` 一直用 `radii^2` 估它。
#   實測代理的誤差**會在訓練中變號**：初期 0.280 倍（低估 3.6 倍，因為忽略 tile 量化 ——
#   半徑 3px 算成 0.14 個 tile，但可見就至少碰 1 個）、後期 1.340 倍（高估，正方形外接盒主導）。
#   ⇒ 同一個預算數字在訓練前後代表的實際成本差約 5 倍。本實驗先把**換算曲線**量出來。
#
# ══════════════════════════════════════════════════════════════════════════════
# 為什麼這樣排（每一項都是為了「結果可被正確辨識」）
# ══════════════════════════════════════════════════════════════════════════════
# 1. **對照臂必須現跑，不可引用舊數字。** `exact_conic_aabb` 雖然是執行期旗標，但那段程式碼
#    「存在於 kernel 裡」本身就改變了 codegen：實測關閉狀態下渲染也不再與歷史逐位元相同
#    （最大 1.8/255、平均 9.0e-08、5.9% 像素；機制是 ceil() 把 sub-ULP 放大成 ±1px 半徑）。
#    對 PSNR 的影響遠小於 0.001 dB，但「逐位元」這個驗證工具在跨建置時失效。
#    ⇒ base 與 conic **必須同一個 .so**。這也是為什麼旗標做成執行期的：編譯期旗標在 lab
#      三槽平行下兩臂**無法同時跑**，誤用還**不會報錯**（兩邊都吃最後編的那份）。
#
# 2. **兩塊，不是一塊。** 本專案已有多次「b12 贏、b7 反轉」（notrim）與「b6 省、b13 反向」
#    （elongation binning）。單塊結論在這個 repo 裡站不住。
#
# 3. **`tilecal` 同時是第三個對照臂。** 它開的是 `exact_tile_cost` + 兩個純打印旗標，而
#    `cost_budget = 0` ⇒ 閘門不生效、vpc 與 cost_aware_densify 都關著 ⇒ **訓練行為與 base 相同**
#    （唯一成本是多一個 N x int32 的視窗，約 10 MB）。所以它免費提供兩樣東西：
#      (a) **同配方重複樣本** => 這個排程的噪音底，後面判「顯不顯著」要用它
#      (b) 精確 vs 代理的**換算曲線**（[cost-budget] 每 150 步印兩種單位與比值）
#    ⇒ 沒有它，我們既沒有噪音底，也得為了標定另外燒一次跑次。
#
# 4. **判準不能只看 PSNR。** 本專案已記錄**四次**「虛假細節簽名」（紋理比↑ 而光度↓）。
#    conic 補回的正是**邊緣的邊際貢獻**，那恰好是可能製造該簽名的形狀。
#    ⇒ 報表要四項一起看：PSNR / SSIM / LPIPS / **紋理比**。
#
# 5. **成本要用離線同工具量。** 訓練中的 [cost-budget] 是「區間最壞視角」，離線是
#    「終點模型 x 全部相機」，兩者差約 1.5 倍且**跨路徑不可比**。
#    ⇒ 跑完拉 ckpt 回本機跑 `scripts/task_load_compare.sh`。
#
# 6. **疊加臂（conicvpc）是條件式的。** conic 改變每顆的 tile 足跡，而 trimvpc 的 c 就是足跡
#    ⇒ 兩者會互相影響。**先單獨判定 conic**，否則分不出是誰的效果
#    （elongation + trimvpc 疊加 -0.68/-0.95 就是這樣被誤讀過一次）。
#
# ══════════════════════════════════════════════════════════════════════════════
# 長度：預設走 task_cmp.sh 的「2 萬多步」標準
# ══════════════════════════════════════════════════════════════════════════════
# task_cmp.sh 會按塊的相機數把全長對齊到 val 點（b6 21,920 / b13 26,680），並把
# `means_lr max_steps` 與 `densify_until_iter` 一起縮放 —— 直接截斷 60k 會落進不同 regime。
# ⚠ **已知的限制**：trimvpc 在 22k 是平手、到 60k 才顯出 -0.27/-0.54 ⇒ 短配方會**低估**
#   品質代價。所以本計畫的判讀規則是：
#     22k 就已經輸（尤其出現虛假細節簽名）=> 直接否決，不必燒 60k
#     22k 平手或更好                      => **必須**再跑 `STEPS=60000` 才能宣稱無代價
#   conic 與 trimvpc 的先驗不同（trimvpc 是**移除**資訊，conic 是**補回**被誤刪的覆蓋），
#   但先驗不能取代量測。
set -u
cd "$(dirname "$0")/../.." || exit 1
GO=0; [ "${1:-}" = "--go" ] && GO=1
Q=scripts/queue.txt
S=${STEPS:-}
PRE=""; [ -n "$S" ] && PRE="STEPS=$S "
TAG=""; [ -n "$S" ] && TAG="（全長 $S）"

emit () {   # emit <註解> <指令>
  printf '\n# %s\n%s\n' "$1" "$2"
}

PLAN=$(
  for blk in 6 13; do
    emit "★★★★★★ conic 判決 b$blk：對照臂 base$TAG（必須與 conic 同一個 .so，不可引用舊跑次）" \
         "${PRE}bash scripts/lab/task_cmp.sh $blk base"
    emit "★★★★★★ conic 判決 b$blk：精確圓錐不對稱外接盒$TAG（離線 Σtile -41.2%、forward -24.5%；會改變輸出）" \
         "${PRE}bash scripts/lab/task_cmp.sh $blk conic"
    emit "★★★★★ b$blk：tilecal＝base 的重複樣本（噪音底）＋精確/代理成本的換算曲線$TAG" \
         "${PRE}bash scripts/lab/task_cmp.sh $blk tilecal"
  done
  emit "[cpu] ★★★★ 一頁式報表：conic 家族四項指標 + 顆數 + 峰值 VRAM（四項要一起看，只看 PSNR 會漏掉虛假細節簽名）" \
       "[cpu] conda run -n gspl python tools/lab_cmp_report.py"
)

echo "════════ 將追加 $(printf '%s' "$PLAN" | grep -cE '^(\[cpu\]|STEPS=|bash )') 個任務到 $Q ════════"
printf '%s\n' "$PLAN"
cat <<'NOTE'

════════ 跑完之後（不在本腳本內，要自己做）════════
1. 報表：上面那行 [cpu] 會印四項指標 + N + 峰值 VRAM。
   **判讀**：PSNR/SSIM/LPIPS/紋理比四項一起看；base 與 tilecal 的差＝這個排程的噪音底，
   conic 對 base 的差要超過它才算數。
2. 成本：把兩臂的終點 ckpt 拉回本機，用**同一支工具同一組相機**量：
       python scripts/setup/lab.py get lab/cs_base 6 <step>
       python scripts/setup/lab.py get lab/cs_conic 6 <step>
       bash scripts/task_load_compare.sh 6 cs_base cs_conic
3. 換算曲線：從 tilecal 的 log 抽 [cost-budget] 行（含「另一單位」與「比值」），
   畫成 ratio(t)。**這條曲線是把舊 cost_budget 搬到新單位的唯一依據** —— 它會變號，
   所以不能除以一個常數。
4. 條件式第二階段（只有在 conic 判定無品質代價後才排）：
       bash scripts/lab/task_cmp.sh 6 conicvpc
       bash scripts/lab/task_cmp.sh 13 conicvpc
5. 若 22k 平手或更好 => 必須再跑一次 STEPS=60000 才能宣稱無代價（trimvpc 的教訓）。

⚠ 排之前確認 lab 已同步且光柵器已重編（否則 exact_conic_aabb 這個欄位不存在，
  renderer 會**當場擋住**而不是靜默失效）：
       JTOK=<token> bash scripts/setup/lab_pull.sh
NOTE

if [ "$GO" = 1 ]; then
  printf '%s\n' "$PLAN" >> "$Q"
  echo "✅ 已追加到 $Q（runner 會自己接手；佇列可隨時再編輯，只有正在執行的那行是固定的）"
else
  echo "（乾跑。要真的排進去加 --go）"
fi
