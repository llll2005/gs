#!/bin/bash
# 現行配方逐項計時（使用者 11-09「排逐項計時」：全場景純訓練 A＋dist4 3.16 h -> 現行配方 4.06 h，多出的 85～90% 在增生期，
#   v4 全相機投票約佔 26%，其餘約 72% 沒歸因；拆解臂 g44s30ksfm10loo*_ 是 2～3 槽平行跑的，時間不能用）
#   同一塊、同 cap、同 init，從頭訓練到 MT_STEPS 步；排程照 30k（densify_until 15000），所以整段都在增生期。
#   CITYGS_STEP_PROFILE 只取 MT_WARMUP 之後的穩態（預設 1550 -> 4600：20 次增生事件、step 3000／4500 兩次合併、
#   step 2000／3000／4000 三次 px_floor 重算），印逐段 ms 與真實步時間；每個臂只改一項，cur 頭尾各跑一次估雜訊。
#   臂：cur（現行配方）、base（十項全關＝A＋dist4 加兩個無損旗標）、no<項>（只關那一項）、
#       fix_mgchunk（merge_chunk 524288，lab 卡大）、fix_vis64（v4 每次只抽 64 台相機投票）——後兩個只量時間，行為會變
#   結果：logs/mech_time_b<塊>_<時間>.log（最後一節是彙總表）；跑次 outputs/lab/mechtime_<臂>（不進任何家族、不跑圖表資料包）
# 用法（佇列）：[solo] bash scripts/lab/task_mech_time.sh 4
#               MT_ARMS="cur noV4 cur" MT_STEPS=4600 MT_WARMUP=1550 [solo] bash scripts/lab/task_mech_time.sh 4
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
B=$1
STEPS_RUN=${MT_STEPS:-4600}
WARM=${MT_WARMUP:-1550}
DUP=${MT_DUP:-2}
CAP=${MT_CAP:-$(python3 -c "import json;print(int(round(1e6 * json.load(open('configs/cap_alloc_sfm_g44.json'))['weights']['$B'])))")}
ARMS=${MT_ARMS:-cur base noV4 noNU noPX noIP noGT noNM noMG noHF noL2 noLR fix_mgchunk fix_vis64 cur}
# 量之前等卡上沒有任何計算行程（同 task_best1_time.sh：runner 的 [solo] 不看容器外的行程）
for i in $(seq 1 240); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" -eq 0 ] && break
  [ "$i" = 1 ] && echo "⏳ 卡上還有 $n 個計算行程 => 等它們結束再量（每 30 s 查一次，最多 2 h）$(date)"
  sleep 30
done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && echo "✅ 卡上無其他行程，開始量 $(date)" || echo "⚠⚠ 等滿 2 h 仍有其他行程 => 這次時間可能被污染"
gpu () { echo "（整卡已用 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB、計算行程 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) 個）"; }
# 現行配方（＝11-09 全場景 g44s30ksfm10nuPF2DB2_ 的旗標，順序照抄；後面的覆寫會蓋掉前面的值）
CUR=(--model.renderer.init_args.record_reduce true --model.renderer.init_args.absgrad_gate true --model.metric.init_args.fused_ssim true
  --model.gaussian.init_args.optimization.optimizer.init_args.sh_rest_interval 16 --model.metric.init_args.lambda_dist_interval 4
  --model.metric.init_args.lambda_normal 0.0125 --model.metric.init_args.lambda_normal_interval 8
  --model.metric.init_args.depth_support true --model.density.init_args.geom_gate_sampling true --model.metric.init_args.depth_support_sfm_radius_nn 8
  --model.metric.init_args.depth_support_rule vote --model.metric.init_args.depth_support_interval 8
  --model.density.init_args.hole_fill_frac 0.002
  --model.density.init_args.merge_interval 1500 --model.density.init_args.merge_color_tol 0.1 --model.density.init_args.merge_prune true
  --model.density.init_args.merge_from_iter 3000 --model.density.init_args.merge_until_iter 27000 --model.density.init_args.merge_max_frac 0.02
  --model.density.init_args.merge_chunk 65536
  --model.density.init_args.inplane_offset 1.0 --model.density.init_args.inplane_hold 1000
  --model.metric.init_args.rgb_l2_weight 2.0
  --model.renderer.init_args.trim_running_warp true --model.renderer.init_args.bwd_zero_grad_skip true
  --model.density.init_args.sfm_vis true --model.density.init_args.sfm_vis_cams 0 --model.density.init_args.sfm_vis_cell 8
  --model.density.init_args.sfm_vis_px 4 --model.density.init_args.sfm_vis_free_frac 0.5 --model.density.init_args.sfm_vis_rep 0.1
  --model.density.init_args.sfm_vis_noup true
  --model.density.init_args.px_floor 1.0 --model.density.init_args.px_floor_interval 1000
  --model.gaussian.init_args.optimization.means_lr_init 1.28e-4 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 1.28e-6)
# 各項的「關掉」寫法（同 11-09 拆解臂 g44s30ksfm10loo<項>_）
OFF_V4=(--model.density.init_args.sfm_vis false --model.density.init_args.sfm_vis_noup false)
OFF_NU=(--model.density.init_args.sfm_vis_noup false)
OFF_PX=(--model.density.init_args.px_floor 0)
OFF_IP=(--model.density.init_args.inplane_offset 0)
OFF_GT=(--model.metric.init_args.depth_support false --model.density.init_args.geom_gate_sampling false)
OFF_NM=(--model.metric.init_args.lambda_normal 0)
OFF_MG=(--model.density.init_args.merge_interval 0)
OFF_HF=(--model.density.init_args.hole_fill_frac 0)
OFF_L2=(--model.metric.init_args.rgb_l2_weight 0)
OFF_LR=(--model.gaussian.init_args.optimization.means_lr_init 6.4e-5 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 6.4e-7)
L=logs/mech_time_b${B}_$(date +%m%d_%H%M).log
{
  echo "=== 現行配方逐項計時：block $B、cap $CAP、dup $DUP、訓練 $STEPS_RUN 步（排程照 30k）、穩態取 $WARM 步之後 $(date) ==="
  i=0
  for arm in $ARMS; do
    i=$((i + 1))
    case $arm in
      # 現行配方（全場景 g44s30ksfm10nuPF2DB2_ 的旗標）
      cur)         XA=() ;;
      # 十項全關（只留 A＋dist4 與兩個無損旗標）＝多出時間的總量
      base)        XA=("${OFF_V4[@]}" "${OFF_PX[@]}" "${OFF_IP[@]}" "${OFF_GT[@]}" "${OFF_NM[@]}" "${OFF_MG[@]}" "${OFF_HF[@]}" "${OFF_L2[@]}" "${OFF_LR[@]}") ;;
      # 只關視線一致性 v4（連同不准往上，兩者綁在一起）
      noV4)        XA=("${OFF_V4[@]}") ;;
      # 只關不准往上（每步的高度夾；v4 投票照做）
      noNU)        XA=("${OFF_NU[@]}") ;;
      # 只關點精靈下限 px_floor（每步夾尺度＋每 1000 步重算）
      noPX)        XA=("${OFF_PX[@]}") ;;
      # 只關增生方向限面內（新點 1000 步的梯度限制）
      noIP)        XA=("${OFF_IP[@]}") ;;
      # 只關深度支撐閘門＋增生取樣受閘門限制（每 8 步投影全部顆粒；也免載入單目深度）
      noGT)        XA=("${OFF_GT[@]}") ;;
      # 只關法向一致性每 8 步（那一步不走 lean）
      noNM)        XA=("${OFF_NM[@]}") ;;
      # 只關合併後刪除（每 1500 步）
      noMG)        XA=("${OFF_MG[@]}") ;;
      # 只關補洞放點（每次增生事件）
      noHF)        XA=("${OFF_HF[@]}") ;;
      # 只關 L1 混 L2
      noL2)        XA=("${OFF_L2[@]}") ;;
      # 位置 lr 回原值（沒有直接的計算量；看顆數／Load 軌跡造成的差）
      noLR)        XA=("${OFF_LR[@]}") ;;
      # 合併分批改成 524288（65536 是筆電 6 GB 的值）——只量時間
      fix_mgchunk) XA=(--model.density.init_args.merge_chunk 524288) ;;
      # v4 每次只抽 64 台相機投票（現行＝全部 811～899 台）——只量時間，投票結果會變
      fix_vis64)   XA=(--model.density.init_args.sfm_vis_cams 64) ;;
      *) echo "⛔ 不認得的臂 $arm"; continue ;;
    esac
    NAME="${RUN_PREFIX}mechtime_${arm}_$i"
    echo "════ [$i] $arm $(date)"; gpu
    CITYGS_STEP_PROFILE=1 CITYGS_STEP_PROFILE_WARMUP=$WARM CITYGS_POSTDATA=0 CITYGS_RUN_NAME="$NAME" \
      CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=30000 CITYGS_CAP=$CAP CITYGS_DUP44=$DUP CITYGS_FAMILY=g44s30ksfm10nuPF2DB2_ \
      bash scripts/lab/task_cmp.sh "$B" best1maskdistrfed10b4 "${CUR[@]}" "${XA[@]}" --trainer.max_steps "$STEPS_RUN" 2>&1 \
      | grep -E "${MT_SHOW:-Error|error|⛔|❌|Traceback}" | head -${MT_SHOW_N:-10}
    gpu
    echo "── step_cost（$arm）"; sed -n 1,24p "outputs/$NAME/blocks/block_$B/step_cost.txt" 2>/dev/null || echo "⛔ 沒有 step_cost.txt"
    rm -f "outputs/$NAME/blocks/block_$B/checkpoints/"*.ckpt     # 計時跑次的模型用不到（每個約 1 GB）
  done
  echo
  echo "════ 彙總（真實步時間＝含 CPU/GPU 重疊；各段為逐段同步後的 wall ms／步）"
  python3 - "$B" "$RUN_PREFIX" $ARMS <<'EOF'
import re, sys, os
B, pre, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for i, a in enumerate(arms, 1):
    p = f"outputs/{pre}mechtime_{a}_{i}/blocks/block_{B}/step_cost.txt"
    if not os.path.exists(p):
        rows.append((a, None)); continue
    t = open(p, encoding="utf-8").read()
    real = re.search(r"真實步時間·含重疊）\s+([\d.]+)", t)
    n = re.search(r"N = ([\d,]+)", t)
    seg = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\d+[^\n]*?)\s{2,}[\d.]+\s+([\d.]+)\s+[\d,]+", t, re.M)}
    def g(k):
        return sum(v for s, v in seg.items() if s.startswith(k + " "))
    rows.append((a, dict(real=float(real.group(1)) if real else float("nan"), n=n.group(1) if n else "?",
                         out=g("0"), fwd=g("3"), loss=g("4"), bwd=g("6"), dens=g("7"), opt=g("9"))))
curs = [r["real"] for a, r in rows if a == "cur" and r]
ref = sum(curs) / len(curs) if curs else float("nan")
print(f"cur 平均 {ref:.2f} ms/步（{len(curs)} 次：{', '.join(f'{x:.2f}' for x in curs)}）=> 兩次 cur 的差就是這次量測的雜訊")
print(f"{'臂':<12} {'N（結束時）':>11} {'真實 ms/步':>10} {'對 cur':>8} | {'迴圈外':>7} {'forward':>8} {'loss':>6} {'backward':>9} {'density':>8} {'optimizer':>9}")
for a, r in rows:
    if not r:
        print(f"{a:<12} （沒有結果）"); continue
    d = 100 * (r["real"] / ref - 1)
    print(f"{a:<12} {r['n']:>11} {r['real']:>10.2f} {d:>+7.1f}% | {r['out']:>7.2f} {r['fwd']:>8.2f} {r['loss']:>6.2f} {r['bwd']:>9.2f} {r['dens']:>8.2f} {r['opt']:>9.2f}")
print("讀法：no<項> 比 cur 快多少 ≈ 那一項在增生期每步的成本（N 不同時，forward／backward 的差有一部分是顆數造成的，看 N 欄）。")
EOF
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample|colmap dataparser|depth maps' | tee "$L"
echo "log：$L"
