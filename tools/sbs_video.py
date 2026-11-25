"""兩支同路線影片左右並排（各加標籤），給對照展示用。2026-11-25 新增。純 CPU。

python tools/sbs_video.py --left outputs/lab/videos_1125/citygs_street.mp4 --right outputs/lab/videos_1125/ours_street.mp4 \
    --left_label CityGaussianV2 --right_label 我們 --out outputs/lab/videos_1125/sbs_street.mp4
"""
import argparse

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"


def label(frame, text, font):
    im = Image.fromarray(frame)
    ImageDraw.Draw(im).text((24, 16), text, font=font, fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--left_label", default="CityGaussianV2")
    ap.add_argument("--right_label", default="我們")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=int, default=16, help="中間白線寬（px；16 讓總寬是 16 的倍數，編碼器不會再縮放）")
    a = ap.parse_args()
    ra, rb = imageio.get_reader(a.left), imageio.get_reader(a.right)
    fps = ra.get_meta_data()["fps"]
    try:
        font = ImageFont.truetype(FONT, 44)
    except OSError:
        font = ImageFont.load_default()
    w = imageio.get_writer(a.out, fps=fps)   # 預設壓縮（同 render_traj／render_paths 的輸出）
    n = 0
    for fa, fb in zip(ra, rb):
        assert fa.shape == fb.shape, (fa.shape, fb.shape)
        gap = np.full((fa.shape[0], a.gap, 3), 255, np.uint8)
        w.append_data(np.hstack([label(fa, a.left_label, font), gap, label(fb, a.right_label, font)]))
        n += 1
    w.close()
    print(f"{n} 幀 -> {a.out}")


if __name__ == "__main__":
    main()
