#!/bin/bash
# ★★★★★ trimvpc 到底剪掉了什麼？—— 驗證「很貴但幾乎沒貢獻」這個機制解釋（先量再解釋）
#
# 已知（b6 / 21,920 步 / 同 N = 1,850,094 / PSNR 平手）：離線 Load 中位 -67%、forward+backward -27%。
# 我在報告裡寫「被換掉的是很貴但幾乎沒貢獻的那批」—— **沒有量過**。這支量兩件可以推翻它的事：
#   ① 投影半徑分布（tools/footprint_dist.py）：整個分布左移？還是只剪掉右尾的大足跡？
#   ② 長寬比分布（tools/elongation_ceiling.py）：換掉的是不是拉長的退化高斯？
#      binning 用較長軸撐出正方形外接盒，長寬比 k 有 1-1/k 是空白 —— 若 trimvpc 主要砍這批，
#      「Load 大降而品質不掉」就說得通；若不是，機制解釋要改寫。
# 兩者都只讀 ckpt、純 CPU。
set -u
cd "$(dirname "$0")/.." || exit 1
R=logs/trimvpc_mechanism_$(date +%m%d_%H%M).log
{
  bad=0
  echo "════════ ① 投影半徑分布（base vs trimvpc @21920）════════"
  conda run -n gspl --no-capture-output python tools/footprint_dist.py lab/cs_base lab/cs_trimvpc --step 21920 || bad=1
  echo "════════ ② 長寬比：拉長的退化高斯（base vs trimvpc @21920）════════"
  conda run -n gspl --no-capture-output python tools/elongation_ceiling.py --runs lab/cs_base lab/cs_trimvpc --step 21920 || bad=1
  exit "$bad"
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|找不到深度圖|caching images' | tee "$R"
# ⚠ 管線結束碼是 tee 的；取大括號那段自己的，否則崩潰時台帳照樣記 DONE
st=${PIPESTATUS[0]}
echo "報告：$R"
[ "$st" -eq 0 ] || echo "⛔ 至少一段失敗（rc=$st）—— 台帳的 DONE 不代表成功，看報告"
exit "$st"
