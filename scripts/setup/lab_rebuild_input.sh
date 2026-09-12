#!/usr/bin/env bash
# 用**已驗證的映射**從官方 tar 重建 `input/`（位元相同），而不是上傳 20 GB。
#
# 依據：`tools/build_global_frame_map.py` 把本機已驗證的 5,621 幀對回 10 個 block 的
# 原始幀，位置與朝向殘差**中位與最大都是 0.00e+00**（逐位元相同）=> 映射是精確的。
# 官方沒有出 `transforms_train.json`（HF 的 pose/block_all 只有 test 那份），
# 所以「照官方腳本重跑」本來就重建不出同一組命名 —— 必須用這份映射。
#
# ⚠ 用 **hardlink** 而不是 cp：同一個檔案系統 => 不多佔 20 GB，內容保證位元相同。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial
MAP="$D/global_frame_map.json"
PY() { conda run -n gspl --no-capture-output python "$@"; }
step() { echo; echo "=========== $* ==========="; date; }

step "0 前置"
[ -s "$MAP" ] || { echo "⛔ 缺 $MAP —— 從本機上傳（tools/build_global_frame_map.py 產的）"; exit 2; }
PY -c "import json;m=json.load(open('$MAP'));print(f'  映射 {len(m):,} 筆')" || exit 2

step "1 解開 10 個 block 的 tar"
( cd "$D/train" && for n in $(seq 1 10); do
    c=$(ls "block_$n/input" 2>/dev/null | wc -l)
    if [ "$c" -gt 0 ]; then echo "  block_$n 已解（$c 張）"; continue; fi
    [ -s "block_$n.tar" ] || { echo "  ⚠ block_$n.tar 不在"; continue; }
    mkdir -p "block_$n/input"
    tar -xf "block_$n.tar" -C . && mv block_$n/*.png "block_$n/input/" 2>/dev/null
    echo "  block_$n -> $(ls "block_$n/input" | wc -l) 張"
  done )

step "2 依映射建 input/（hardlink）"
mkdir -p "$D/train/block_all/input"
PY - <<'PYEOF'
import json, os
D = "data/matrix_city/aerial"
m = json.load(open(f"{D}/global_frame_map.json"))
dst = f"{D}/train/block_all/input"
made = miss = have = 0
for gname, v in sorted(m.items()):
    t = os.path.join(dst, gname)
    if os.path.exists(t):
        have += 1; continue
    # per-block 影像是 4 位數、從 0000 起算；frame_index 從 1 起算
    src = os.path.join(D, "train", f"block_{v['block']}", "input", f"{v['frame_index']-1:04d}.png")
    if not os.path.exists(src):
        miss += 1
        if miss <= 5:
            print(f"  ⚠ 缺來源 {src}（給 {gname}）")
        continue
    os.link(src, t); made += 1
print(f"  新建 {made:,} ／ 已存在 {have:,} ／ 缺來源 {miss:,}")
PYEOF
echo "  input/ 現有 $(ls "$D/train/block_all/input" 2>/dev/null | wc -l) 張"

step "3 sparse（預算 COLMAP）"
if [ -d "$D/train/block_all/sparse/0" ]; then
  echo "  已存在"
else
  mkdir -p data/colmap_results
  if [ ! -f data/colmap_results.zip ]; then
    conda run -n gspl gdown "1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7" -O data/colmap_results.zip 2>&1 | tail -4
  fi
  if [ -s data/colmap_results.zip ]; then
    unzip -q -o data/colmap_results.zip -d data/ && echo "  解開完成"
    find data -maxdepth 4 -type d -name sparse | head -5
    S=$(find data/colmap_results -type d -path "*matrix_city_aerial/train/sparse" | head -1)
    [ -n "$S" ] && mkdir -p "$D/train/block_all" && cp -r "$S" "$D/train/block_all/" && echo "  sparse 就位"
  else
    echo "  ⛔ gdown 沒抓到（Drive 可能要人工確認）=> 從本機上傳 sparse/0（1.8GB）"
  fi
fi

step "4 transforms.json（映射的來源，本機已驗證那份）"
[ -s "$D/train/block_all/transforms.json" ] && echo "  已存在" || echo "  ⛔ 缺 —— 從本機上傳"

step "5 ★ 配對驗證（模型無關；沒過就不要訓練）"
PY tools/verify_pairing_geometric.py --data "$D/train/block_all" --n-epipolar 3 \
  || { echo "⛔⛔ 沒過，停在這裡"; exit 9; }

step "6 partition"
[ -d "$D/train/block_all/partition" ] && echo "  已存在" || \
  PY utils/partition_from_colmap.py "$D/train/block_all" --block_dim 5 5 --content_threshold 0.08 --force

step "完成"
du -sh "$D/train/block_all/input" "$D/train/block_all/sparse" 2>/dev/null
