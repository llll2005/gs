"""把 depth-init PLY 的 opacity 改成指定值，其餘完全不動。純 CPU、幾秒。

為什麼需要（研究總覽 §11.2）：depth_init 給**每一顆**都設 opacity 0.99，而那層殼比表面厚
20 倍。起始 trim（step 1）用 `sum T*alpha` 算貢獻度，前層在 0.99 之下吸收掉幾乎所有
transmittance => 後面的點貢獻度歸零 => 被砍。實測 b12：**1,211,537 顆裡只有 157 顆真的
在視錐外，卻有 873,504 顆（72.1%）被砍** => 砍掉的是「看得見但被前層遮住」的點，
其中包含正確的表面。錯的前層在任何最佳化之前就被鎖死。

半透明的殼讓後層也拿得到 transmittance 與梯度，最佳化才有機會自己選出正確的那層。

用法：python tools/reopacity_ply.py <in.ply> <out.ply> --opacity 0.5
"""
import argparse, math
import numpy as np
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--opacity", type=float, required=True, help="sigmoid 後的目標值，例 0.5")
a = ap.parse_args()
p = PlyData.read(a.src); v = p["vertex"]
raw = math.log(a.opacity / (1 - a.opacity))
old = np.array(v["opacity"])
d = v.data.copy(); d["opacity"] = np.float32(raw)
PlyData([PlyElement.describe(d, "vertex")], text=False).write(a.dst)
sig = lambda x: 1 / (1 + np.exp(-x))
print("%s -> %s" % (a.src, a.dst))
print("  %s 顆   opacity %.4f -> %.4f (raw %.3f -> %.3f)"
      % (format(len(d), ","), sig(np.median(old)), a.opacity, np.median(old), raw))
print("  單層後殘餘 transmittance %.4f -> %.4f；五層後 %.2e -> %.2e"
      % (1 - sig(np.median(old)), 1 - a.opacity,
         (1 - sig(np.median(old))) ** 5, (1 - a.opacity) ** 5))
