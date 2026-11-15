#!/bin/bash
# ★ 2026-11-14 本機換塊評測（lab 被外部佔卡；使用者「掉分會不會是塊外？得等各項的換塊」）
# 基準＝lab full44_best 裁成 b1／b4 與相鄰塊（tools/crop_merged_neighbors.py => outputs/local/full44_best_nb14），去 Adam；
# 候選＝本機 outputs/local 的各臂終點 ckpt（自動找步數最大的）；換塊規則與 lab 的 task_full44.sh swap 相同。
# 用法：bash scripts/local/swap_local_1114.sh <塊>     輸出 logs/swap_local_b<塊>_<時間>.log
cd "$(dirname "$0")/../.." || exit 1
B=${1:?塊}
R=np16tricap0p5cmflattrimasklowresgeo8log0
S=best1maskdistrfed10b4
L=(本機新底3=g44tn_s20g70${R}_$S 重跑=rep2_g44tn_s20g70${R}_$S 17k=g44tn_s17g70${R}_$S 15k增生70=g44tn_s15g70${R}_$S 15k增生80=g44tn_s15g70${R}_$S
   兩輪名額85=g44tn_s20g70${R}r2r2c85_$S 完整路徑每16步=g44tn_s20g70${R}fp16_$S C區不跑=g44tn_s20g70${R}cskip_$S
   稠密起點=g44tn_s20g70${R}dense_$S 稠密起點不增生=g44tn_s20g0${R}dense_$S 點精靈刪除=g44tn_s20g70${R}pxdel_$S 點精靈吸收=g44tn_s20g70${R}pxabs_$S
   海面接續對照=seadose_b${B}_ctl 往下沉6px=seadose_b${B}_p6 往下沉24px=seadose_b${B}_p24 往下沉48px=seadose_b${B}_p48 沉48px保護H200=seadose_b${B}_p48h200)
L[4]=15k增生80=g44tn_s15g80${R}_$S
# ★ 11-14 SWAP_SET=sprite2：點精靈公平版（全解析度才刪／吸收）
[ "${SWAP_SET:-}" = sprite2 ] && L=(本機新底3=g44tn_s20g70${R}_$S 點精靈刪除=g44tn_s20g70${R}pxdel_$S 點精靈吸收=g44tn_s20g70${R}pxabs_$S
   全解析度才刪=g44tn_s20g70${R}pxdelf_$S 全解析度才吸收=g44tn_s20g70${R}pxabsf_$S)
C=(); N=()
for kv in "${L[@]}"; do
  d=outputs/local/${kv#*=}/blocks/block_$B/checkpoints
  c=$(ls "$d"/*step=*.ckpt 2>/dev/null | grep -v xyz_rgb | awk -F'step=' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2)
  [ -n "$c" ] && { C+=("$c"); N+=("${kv%%=*}"); } || echo "⚠ 缺 ${kv%%=*}（$d）"
done
echo "候選 ${#C[@]} 個：${N[*]}"
LOG=logs/swap_local_b${B}${SWAP_SET:+_$SWAP_SET}_$(date +%m%d_%H%M).log
conda run -n gspl --no-capture-output python -u tools/swap_block_eval.py --merged_run outputs/local/full44_best_nb14 --block "$B" --load_cpu \
  --cand "${C[@]}" --names "${N[@]}" 2>&1 | grep -v "pkg_resources\|declare_namespace" | tee "$LOG"
exit "${PIPESTATUS[0]}"
