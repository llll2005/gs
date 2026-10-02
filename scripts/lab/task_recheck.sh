#!/bin/bash
# ★★★★ 舊結論在新年代補測（2026-10-02 使用者要求「3. 補測 4. 這些也都測」；清單 = 紀錄/倖存清單_2026-09-13.md 的 ⛔／⚠ 段）
#
# 這些結論當初都是在影像↔姿態錯開的資料上量的；工具本身沒有壞（都不經過 GT 配對，或只經過 val 影像），
# 所以這裡**不改工具**，只換成新年代的模型重跑，並且每項都加量**官方合併模型**當參照：
#   geom      幾何稽核（z 分位／懸空％，audit_geometry）＋渲染深度 vs SfM 可見點的 slope/corr（measure_depth_bias）
#             舊結論：我方 5x5 slope 1.082/corr 0.903、官方 0.158/0.355；vanilla 1500 步幾何比 60k 好（z 0.37 vs 0.59）
#             => 量 60k 與 1499 兩個時點，回答「訓練越久幾何越差」在新資料上還在不在
#   heldout   官方 test 中落在本塊訓練相機 AABB 內的視角（eval_official_test --block）：
#             舊結論「held-out 排序與 val⊂train 排序相反」。⚠ 單塊模型在測試視角看得到塊外內容 => 絕對值是下界，
#             只比**同一批視角上**的排序；官方合併模型在同一批視角上當上界參照
#   starttrim step 1 的起始 trim 砍掉多少 init、被砍的離表面多遠（audit_start_trim；舊結論 73~90%）
#             用 dup4（現行最佳 init，起始 ~2.3M）=> 也回答「dup4 的顆數在早期掉過一次」是不是它
#   pics      存 val 渲染圖（failure／oracle 的輸入；task_savepics.sh）
#   failure   失敗區（絕對門檻 GT std>=0.10，failure_map）=> 舊「糊掉 46%」「失敗區跨配方固定」
#   oracle    O0 oracle（凍結拓撲只重訓失敗 tile 的影響集，oracle_o0）
#   tau       訓練後的逐點 tile 覆蓋 tau 與各 K 的 N_max（measure_tau；逐塊 cap 標定的輸入）
#   coarseinit／coarse／coarsegeom   coarse 先驗閘門（見該段註解）
#
# 用法（佇列）：
#   bash scripts/lab/task_recheck.sh geom <塊> <跑次...>          （跑次＝outputs/lab/ 底下的名稱，如 cs60_conic）
#   [solo] bash scripts/lab/task_recheck.sh geomoff <塊>           官方合併模型（2,448 萬顆，獨佔以免撞卡）
#   bash scripts/lab/task_recheck.sh heldout <塊> <跑次...>
#   [solo] bash scripts/lab/task_recheck.sh heldoutoff <塊>
#   bash scripts/lab/task_recheck.sh starttrim <塊> <跑次> <init PLY（相對 dataset）>
#   bash scripts/lab/task_recheck.sh pics <塊> <跑次>
#   bash scripts/lab/task_recheck.sh failure <塊> <跑次...>
#   bash scripts/lab/task_recheck.sh oracle <塊> <跑次...>
#   bash scripts/lab/task_recheck.sh tau <塊> <跑次...>
#   [cpu] bash scripts/lab/task_recheck.sh coarseinit 0 ；[solo] ... coarse null ；[solo] ... coarsegeom <塊>
set -u
cd "$(dirname "$0")/../.." || exit 1
MODE=${1:?用法見檔頭}; BLK=${2:?塊}; shift 2
D=data/matrix_city/aerial/train/block_all
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
OFF_CK=$(ls ../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/checkpoints/*.ckpt 2>/dev/null | tail -1)
BL="$D/partition/partitions-dim_5_5_visibility_0.08/$(printf '%03d_%03d' $((BLK % 5)) $((BLK / 5))).txt"
L=logs/recheck_${MODE}_b${BLK}_$(date +%m%d_%H%M).log
PY="conda run -n gspl --no-capture-output python"
ck_at () {   # ck_at <run> <step|last>
  local d="outputs/${PFX}$1/blocks/block_$BLK/checkpoints"
  if [ "$2" = last ]; then ls "$d"/*step=60000.ckpt 2>/dev/null | grep -v culldust | head -1
  else ls "$d"/*step=$2.ckpt 2>/dev/null | head -1; fi
}
val_of () {  # 該跑次的 val PSNR（train_status.txt 最後一行 val）
  grep -h "val" "outputs/${PFX}$1/blocks/block_$BLK/train_status.txt" 2>/dev/null | tail -1 | cut -c1-160
}

{
case "$MODE" in
  geom)
    for r in "$@"; do
      for s in 1499 last; do
        c=$(ck_at "$r" "$s"); [ -n "$c" ] || { echo "（$r 沒有 step $s 的 ckpt，略過）"; continue; }
        echo "════ $r @ $s：$c"
        # audit_geometry 自己挑目錄裡步數最大的 ckpt => 只在 last 時跑（1499 時它量的還是 60k）
        [ "$s" = last ] && $PY tools/audit_geometry.py "${PFX}$r" --block "$BLK" 2>&1 | tail -5
        $PY tools/measure_depth_bias.py --ckpt "$c" --block "$BLK" --block_dim 5 5 --content_bounds 2>&1 | tail -6
      done
    done ;;
  geomoff)
    [ -n "$OFF_CK" ] || { echo "⛔ 找不到官方合併 ckpt"; exit 2; }
    echo "════ 官方合併模型 @ block $BLK 的視角：$OFF_CK"
    $PY tools/measure_depth_bias.py --ckpt "$OFF_CK" --block "$BLK" --block_dim 5 5 --content_bounds 2>&1 | tail -6 ;;
  heldout)
    for r in "$@"; do
      for s in 1499 last; do
        c=$(ck_at "$r" "$s"); [ -n "$c" ] || { echo "（$r 沒有 step $s 的 ckpt，略過）"; continue; }
        echo "════ $r @ $s　val⊂train：$(val_of "$r")"
        $PY tools/eval_official_test.py --ckpt "$c" --block "$BLK" --block_dim 5 5 2>&1 \
          | grep -E "選視角|模型|PSNR|SSIM|LPIPS|紋理比|⛔" | head -8
      done
    done ;;
  heldoutoff)
    [ -n "$OFF_CK" ] || { echo "⛔ 找不到官方合併 ckpt"; exit 2; }
    echo "════ 官方合併模型（上界參照，同一批視角）：$OFF_CK"
    $PY tools/eval_official_test.py --ckpt "$OFF_CK" --block "$BLK" --block_dim 5 5 2>&1 \
      | grep -E "選視角|模型|PSNR|SSIM|LPIPS|紋理比|⛔" | head -8 ;;
  starttrim)
    r=${1:?跑次}; P=${2:?init PLY}
    c=$(ck_at "$r" 499); [ -n "$c" ] || { echo "⛔ $r 沒有 step 499 的 ckpt"; exit 2; }
    $PY tools/audit_start_trim.py --seed_ply "$D/$P" --ckpt "$c" --block "$BLK" --block_dim 5 5 ;;
  pics)
    bash scripts/lab/task_savepics.sh "${PFX}${1:?跑次}" "$BLK" ;;
  failure)
    rs=(); for r in "$@"; do rs+=("${PFX}$r"); done
    $PY tools/failure_map.py "${rs[@]}" --blk "$BLK" --block-dim 5 5 ;;
  # fail = pics（缺圖才存）-> failure -> oracle 串成一行：三槽平行下分成三行會在圖還沒存完時就開跑
  fail)
    rs=()
    for r in "$@"; do
      rs+=("${PFX}$r")
      ls -d "outputs/${PFX}$r/blocks/block_$BLK/test/"*/ >/dev/null 2>&1 \
        || bash scripts/lab/task_savepics.sh "${PFX}$r" "$BLK" | tail -3
    done
    echo "════ 失敗區（絕對門檻）"
    $PY tools/failure_map.py "${rs[@]}" --blk "$BLK" --block-dim 5 5
    echo "════ O0 oracle"
    $PY tools/oracle_o0.py "${rs[@]}" --blk "$BLK" --model-run "${rs[0]}/blocks/block_$BLK" ;;
  oracle)
    rs=(); for r in "$@"; do rs+=("${PFX}$r"); done
    # ⚠ oracle_o0 用遞迴 glob 找 60k ckpt：多塊跑次會挑到字典序最前的塊（block_13 < block_6）=> 明給 --model-run
    $PY tools/oracle_o0.py "${rs[@]}" --blk "$BLK" --model-run "${rs[0]}/blocks/block_$BLK" ;;
  tau)
    for r in "$@"; do
      echo "════ $r"
      $PY tools/measure_tau.py --run "${PFX}$r/blocks/block_$BLK" --block-list "$BL" 2>&1 | tail -15
    done ;;
  # ── coarse 當空間先驗的閘門（記憶 proposal_coarse_prior：先量 coarse 的幾何值不值得當先驗）──
  #   舊反證「官方 coarse slope 0.158/corr 0.355」是舊年代量的 => 官方 coarse 與我方 coarse 用同工具、同視角重量。
  #   我方 coarse：全場景（block_id null）、sh0、cap 1M（6GB 內；記憶 proposal_coarse_prior 的 ①）、30k（同官方 coarse 步數）、
  #   其餘 = 現行配方（task_cmp conic 臂）。init 用 SfM 點隨機取 90 萬：全場景 SfM 有 ~360 萬點 > cap，
  #   而起始 > cap 時 add_new_gs 加 0（只剩 trim 衰減），也撐不進 6GB。
  coarseinit)
    mkdir -p "$D/coarse_init"
    $PY - "$D" <<'PYEOF'
import sys, numpy as np
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from tools.make_sfm_fill_init import write_xyzrgb_ply
d = sys.argv[1]
P = read_points3D_binary(f"{d}/sparse/0/points3D.bin")
xyz = np.array([p.xyz for p in P.values()], np.float32); rgb = np.array([p.rgb for p in P.values()], np.uint8)
k = min(900_000, len(xyz)); sel = np.random.default_rng(0).choice(len(xyz), k, replace=False)
write_xyzrgb_ply(f"{d}/coarse_init/sfm_sub900k.ply", xyz[sel], rgb[sel])
print(f"SfM 全場景 {len(xyz):,} 點 -> 隨機取 {k:,} -> {d}/coarse_init/sfm_sub900k.ply")
PYEOF
    ;;
  coarse)
    [ -f "$D/coarse_init/sfm_sub900k.ply" ] || { echo "⛔ 先跑 coarseinit"; exit 2; }
    NCAM=1 STEPS=30000 CITYGS_CAP=1000000 CITYGS_RUN_NAME="${PFX}coarse_ours" CITYGS_FAMILY=coarse_ \
      bash scripts/lab/task_cmp.sh null conic \
        --model.gaussian.init_args.sh_degree 0 \
        --data.parser.points_from ply --data.parser.ply_file coarse_init/sfm_sub900k.ply ;;
  coarsegeom)
    OC=$(ls ../cityGS_origin/outputs/citygsv2_mc_aerial_coarse_sh2/checkpoints/*step=30000.ckpt 2>/dev/null | head -1)
    MC=$(ls outputs/${PFX}coarse_ours/blocks/block_null/checkpoints/*step=30000.ckpt 2>/dev/null | head -1)
    for c in "$OC" "$MC"; do
      [ -n "$c" ] || { echo "（缺一個 coarse ckpt，略過）"; continue; }
      echo "════ coarse：$c"
      $PY tools/measure_depth_bias.py --ckpt "$c" --block "$BLK" --block_dim 5 5 --content_bounds 2>&1 | tail -6
    done ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$L"
exit "${PIPESTATUS[0]}"
