#!/bin/bash
# 補齊官方 test 的 741 幀影像，並用 **md5 驗證**串接順序 —— 不猜。
#
# ## 為什麼需要這支
# `data/matrix_city/aerial/test/block_all_test/` 在 lab 上只有 transforms.json 與 sparse/0，
# **input/ 是空的** => `tools/eval_official_test.py` 必 `FileNotFoundError: .../images_1.2`
# （2026-09-12 已經這樣失敗過一次）。而 held-out 是**唯一跨年代可比的數字**。
#
# ## 為什麼不能直接下載了就用
# HF 上的官方 test 是**分塊的 10 個 tar**（block_1_test.tar .. block_10_test.tar），
# 而我方的 `block_all_test`（741 幀、六位數連號）是**自己串起來的**。
# 那個串接順序決定**相機↔影像的配對** —— 猜錯就是 2026-09-13 §16.14 那個廢掉整個專案的
# 同一類錯（86.64% 的影像對到別的姿態，而所有指標看起來都「正常」）。
# ⇒ 這支腳本用本機那份已在用的 input/ 的 md5 當基準，**逐檔比對建立映射**，
#   每一幀都必須唯一命中，否則整支中止。
#
# ## 命名（使用者 2026-09-13 決定）：**四位數**
#   input/0001.png .. 0741.png（與 train 集的四位數一致）
#   images_1.2 -> input（別名；eval_official_test.py 預設讀 images_1.2，而它本來就期待
#                        0001.png..0741.png，所以四位數剛好對上）
# ⚠ 1.2 倍是 dataparser 載入時算的，**不是**預先縮小的檔案（見 CLAUDE.md）。
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/test
T=$D/block_all_test
EXP=$T/EXPECTED_md5.txt
STAGE=$D/_stage_official
HF=https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial/test

[ -s "$EXP" ] || { echo "⛔ 缺基準 $EXP（本機產生後用 lab.py put 送上來）"; exit 1; }
echo "基準 $(wc -l < "$EXP") 筆"
command -v wget >/dev/null || { echo "⛔ 沒有 wget"; exit 2; }

echo "=== 1 下載 10 個 block_N_test.tar ==="
mkdir -p "$STAGE"
for n in 1 2 3 4 5 6 7 8 9 10; do
  f="$STAGE/block_${n}_test.tar"
  if [ -s "$f" ]; then echo "  block_${n}_test.tar 已存在 $(du -h "$f" | cut -f1)"; continue; fi
  echo "  抓 block_${n}_test.tar …"
  wget -q -O "$f" "$HF/block_${n}_test.tar" || { echo "  ⛔ 抓不到"; rm -f "$f"; exit 1; }
done
du -sh "$STAGE" | sed 's/^/  合計 /'

echo "=== 2 解開 ==="
for n in 1 2 3 4 5 6 7 8 9 10; do
  d="$STAGE/x_block_${n}"
  [ -d "$d" ] && continue
  mkdir -p "$d"
  tar xf "$STAGE/block_${n}_test.tar" -C "$d" || { echo "⛔ block_${n} 解不開"; exit 1; }
done
find "$STAGE" -name '*.png' | wc -l | sed 's/^/  解出 png /'

echo "=== 3 md5 全部解出來的影像（這是驗證的依據）==="
find "$STAGE" -name '*.png' -print0 | sort -z | xargs -0 md5sum > "$STAGE/official_md5.txt"
wc -l < "$STAGE/official_md5.txt" | sed 's/^/  /'

echo "=== 4 用 md5 建映射並產生四位數的 input/（唯一命中才算）==="
python3 - "$EXP" "$STAGE/official_md5.txt" "$T" <<'PY' || exit 1
import os, sys, collections
exp_path, off_path, T = sys.argv[1], sys.argv[2], sys.argv[3]
exp = {}
for ln in open(exp_path):
    h, n = ln.split(None, 1)
    exp[h] = n.strip()                      # md5 -> 我方的六位數檔名
off = collections.defaultdict(list)
for ln in open(off_path):
    h, p = ln.split(None, 1)
    off[h].append(p.strip())                # md5 -> 官方解出來的路徑（可能多個）

miss  = [h for h in exp if h not in off]
multi = [h for h in exp if len(off.get(h, [])) > 1]
print(f"  基準 {len(exp)} 筆／官方 {len(off)} 個不同 md5")
print(f"  沒命中 {len(miss)}／命中多個 {len(multi)}")
if miss:
    print("  ⛔ 有幀在官方檔案裡找不到 —— 不要繼續，映射不完整")
    for h in miss[:5]: print("     ", h, exp[h])
    sys.exit(1)
if multi:
    print("  ⚠ 有 md5 對到多個官方檔（內容重複）=> 任選其一不影響位元組正確性")
os.makedirs(os.path.join(T, "input"), exist_ok=True)
made = 0
for h, ours in exp.items():
    idx = int(os.path.splitext(ours)[0])     # 000001.png -> 1
    dst = os.path.join(T, "input", f"{idx:04d}.png")
    src = os.path.abspath(off[h][0])
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)                    # 硬連結，不多佔空間
    except OSError:
        import shutil; shutil.copy2(src, dst)
    made += 1
print(f"  ✔ 產生 {made} 個四位數影像（硬連結）")
PY

echo "=== 5 images_1.2 別名 ==="
ln -sfn input "$T/images_1.2"
echo "  images_1.2 -> $(readlink "$T/images_1.2")"

echo "=== 6 關卡：逐位元覆核 ==="
python3 - "$EXP" "$T" <<'PY' || exit 1
import hashlib, os, sys
exp_path, T = sys.argv[1], sys.argv[2]
bad = 0; n = 0
for ln in open(exp_path):
    h, ours = ln.split(None, 1)
    idx = int(os.path.splitext(ours.strip())[0])
    p = os.path.join(T, "images_1.2", f"{idx:04d}.png")
    if not os.path.exists(p):
        print(f"  ⛔ 缺 {p}"); bad += 1; continue
    got = hashlib.md5(open(p, "rb").read()).hexdigest()
    n += 1
    if got != h:
        print(f"  ⛔ md5 不符 {p}"); bad += 1
print(f"  覆核 {n} 檔，錯 {bad}")
sys.exit(1 if bad else 0)
PY
echo "✅ 完成。可以移除暫存：rm -rf $STAGE/*.tar（解出來的目錄是硬連結的來源，不要刪）"
