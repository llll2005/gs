#!/usr/bin/env bash
# 在**本機**用已驗證的映射重建 `input/`，命名改成 **4 位數 = SfM 相機名**。
#
# 為什麼（研究總覽 §16.14）：現行 `input/00000k.png` 裝的是「該 block 的第 k 個**檔案**」，
# 而相機 k 的姿態屬於「串接後第 k **幀**」，兩者在 **86.6%** 的影像上不同
# （b12 的 284 台只有 73 台正確、位移中位 4.39 場景單位 > AABB 2.86x2.59）。
# 4 位數的另一個好處：`colmap_dataparser.py:350` 是**名稱優先**，
# 檔名 == 相機名 ⇒ 位置對應那條 fallback（藏住這個 bug 的路徑）永遠不會觸發。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial
BA="$D/train/block_all"
MAP="$D/global_frame_map_correct.json"
step(){ echo; echo "=========== $* ==========="; date; }

step "0 前置"
[ -s "$MAP" ] || { echo "⛔ 缺 $MAP（tools/build_global_frame_map.py 產的）"; exit 2; }
for n in $(seq 1 10); do
  [ -s "$D/train/block_$n/transforms.json" ] || { echo "⛔ 缺 block_$n/transforms.json"; exit 2; }
done

step "1 檢查來源影像齊不齊（缺的從本機 tar 重解）"
python - <<'PYEOF'
import json, os, subprocess
D="data/matrix_city/aerial"
for n in range(1, 11):
    d=f"{D}/train/block_{n}/input"
    fr=json.load(open(f"{D}/train/block_{n}/transforms.json"))["frames"]
    need={f"{f['frame_index']:04d}.png" for f in fr}
    have=set(os.listdir(d)) if os.path.isdir(d) else set()
    miss=need-have
    print(f"  block_{n:<2} 需要 {len(need):>5,} 張／現有 {len(have):>5,} 張／缺 {len(miss):>4}", end="")
    if miss:
        tar=f"{D}/train/block_{n}.tar"
        if os.path.exists(tar):
            print(f"  => 從 {os.path.basename(tar)} 重解")
            os.makedirs(d, exist_ok=True)
            subprocess.run(["tar","-xf",tar,"-C",f"{D}/train"], check=False)
            src=f"{D}/train/block_{n}"
            for f in os.listdir(src):
                if f.endswith(".png") and os.path.isfile(os.path.join(src,f)):
                    os.replace(os.path.join(src,f), os.path.join(d,f))
            have=set(os.listdir(d)); miss=need-have
            print(f"     重解後仍缺 {len(miss)}")
        else:
            print(f"  ⛔ 也沒有 {tar}")
    else:
        print("  ✓")
PYEOF

step "2 建新的 input/（4 位數、hardlink）"
NEW="$BA/input_new4"
rm -rf "$NEW"; mkdir -p "$NEW"
python - <<'PYEOF'
import json, os
D="data/matrix_city/aerial"; NEW=f"{D}/train/block_all/input_new4"
m=json.load(open(f"{D}/global_frame_map_correct.json"))
keys=sorted(m)          # 6 位數的全域檔名，順序 == 相機名順序
made=miss=0
for k,g in enumerate(keys):
    v=m[g]
    src=f"{D}/train/block_{v['block']}/input/{v['frame_index']:04d}.png"
    dst=f"{NEW}/{k:04d}.png"
    if not os.path.exists(src):
        miss+=1
        if miss<=5: print(f"  ⚠ 缺來源 {src}（給 {k:04d}.png）")
        continue
    os.link(src,dst); made+=1
print(f"  建了 {made:,} 張／缺來源 {miss:,}")
PYEOF
echo "  input_new4/ $(ls "$NEW" | wc -l) 張，前三：$(ls "$NEW" | head -3 | tr '\n' ' ')"

step "3 換上去（舊的搬到 input_broken_20260913）"
[ "$(ls "$NEW" | wc -l)" -eq 5621 ] || { echo "⛔ 新的不是 5,621 張，不換"; exit 3; }
[ -d "$BA/input" ] && mv "$BA/input" "$BA/input_broken_20260913"
mv "$NEW" "$BA/input"
cat > "$BA/input_broken_20260913/README.txt" <<'TXT'
⛔ 這是 2026-09-13 之前的 input/，**影像與姿態錯開**（研究總覽 §16.14）。
   內容 = 該 block 的第 k 個檔案；而相機 k 的姿態屬於串接後第 k 幀，
   86.6% 的影像因此配到錯的姿態（b12 位移中位 4.39 場景單位 > AABB 2.86x2.59）。
   所有用它訓練出來的 outputs/ 全部作廢。確認新資料沒問題後可以整個刪掉（約 21GB）。
TXT
echo "  現在 input/ $(ls "$BA/input" | wc -l) 張；舊的在 input_broken_20260913/"

step "4 ★ 配對驗證（C 段沒過就不要訓練）"
conda run -n gspl python tools/verify_pairing_geometric.py --data "$BA" --skip-epipolar \
  || { echo "⛔⛔ 沒過，停在這裡"; exit 9; }

step "5 partition 重算（相機->塊的分派會變）"
[ -d "$BA/partition" ] && mv "$BA/partition" "$BA/partition_broken_20260913"
conda run -n gspl --no-capture-output python utils/partition_from_colmap.py "$BA" \
  --block_dim 5 5 --content_threshold 0.08 --force 2>&1 | tail -4
P="$BA/partition/partitions-dim_5_5_visibility_0.08"
echo "  產出 $(ls "$P"/*.txt 2>/dev/null | wc -l) 塊"
for b in 6 12 13; do
  f="$P/$(printf '%03d_%03d' $((b%5)) $((b/5))).txt"
  [ -f "$f" ] && echo "    block_$b: $(wc -l < "$f") 台相機"
done

step "6 標記衍生資料已作廢"
for d in depth_init depth_init_fix depth_init_fix2 depth_init_o05 estimated_depths; do
  [ -d "$BA/$d" ] && cat > "$BA/$d/README_INVALID.txt" <<'TXT'
⛔ 2026-09-13：本目錄是用**錯開的**影像/姿態產生的（研究總覽 §16.14），全部作廢。
   depth_init：深度圖 x 姿態反投影 => 姿態錯了，點位就錯。
   estimated_depths：單張影像的深度本身沒錯，但檔名索引對應的是舊的 6 位數版面。
   要用就重生（配方 depth_loss_weight=0 時訓練本身不需要它們）。
TXT
done
echo "  已標記"

step "完成"
du -sh "$BA/input" "$BA/input_broken_20260913" 2>/dev/null
echo "下一步：本機佇列排 speed3（跑次名不帶塊編號）"
