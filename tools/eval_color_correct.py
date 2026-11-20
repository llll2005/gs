"""★ 2026-11-20 色彩校正（CC）版評測（使用者「什麼色彩校正 我們也能作嗎」）。

有些論文（DOGS、Momentum-GS camera-ready、VastGaussian、BlockGaussian）評測前對每張渲染圖做 Mip-NeRF 360（multinerf）的
color_correct：用該張 GT 擬合一個逐色版的二次色彩轉換（6 個二次項＋3 個一次項＋常數，5 輪、排除飽和像素），套上後再算指標。
同一個模型做 CC 會高約 1 dB（Momentum-GS 28.01 → 29.11）。本工具讓我方與 CityGS 也能各報一組 CC 數字，和那一群並列。

輸入：eval_official_test.py --save_dir 存的「左 GT｜右渲染」寬圖目錄（可多個）。
輸出：每個目錄各一列：不做 CC／做 CC 的 PSNR、SSIM、LPIPS（SSIM＝internal.utils.ssim、LPIPS＝torchmetrics alex，與 eval_official_test 相同）。
⚠ 從 8-bit PNG 算，不做 CC 的數字會和 eval_official_test（float 渲染）差約 0.005 dB；CC 的增量用同一批 PNG 算，彼此一致。
用法：python tools/eval_color_correct.py 名稱=<目錄> [名稱=<目錄> ...] [--nproc 12]
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
    """multinerf internal/image.py color_correct 的 numpy 版（逐行對照）：把 img 的顏色對齊到 ref。"""
    nc = img.shape[-1]
    img_mat = img.reshape([-1, nc]).astype(np.float64)
    ref_mat = ref.reshape([-1, nc]).astype(np.float64)
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))
    mask0 = is_unclipped(img_mat)
    for _ in range(num_iters):
        a_mat = []
        for c in range(nc):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])   # 二次項
        a_mat.append(img_mat)                                       # 一次項
        a_mat.append(np.ones_like(img_mat[:, :1]))                  # 常數
        a_mat = np.concatenate(a_mat, axis=-1)
        warp = []
        for c in range(nc):
            b = ref_mat[:, c]
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = np.where(mask[:, None], a_mat, 0)
            mb = np.where(mask, b, 0)
            # ★ 正規方程（10×10）：與 multinerf 對整個 ma_mat 做 lstsq 同一個最小平方解，快約 50 倍（float64；已在單張上核對一致）
            w = np.linalg.lstsq(ma_mat.T @ ma_mat, ma_mat.T @ mb, rcond=-1)[0]
            assert np.all(np.isfinite(w))
            warp.append(w)
        warp = np.stack(warp, axis=-1)
        img_mat = np.clip(a_mat @ warp, 0, 1)
    return img_mat.reshape(img.shape).astype(np.float32)


def load_pair(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    W = a.shape[1] // 2
    return a[:, :W], a[:, W:]          # GT、渲染


def one(path):
    gt, out = load_pair(path)
    cc = color_correct(out, gt)
    return os.path.basename(path), cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--nproc", type=int, default=12)
    a = ap.parse_args()
    import torch
    from internal.utils.ssim import ssim as ssim_fn
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex").to(dev)
    T = lambda x: torch.from_numpy(x).permute(2, 0, 1).contiguous().to(dev)
    rows = []
    for spec in a.dirs:
        nm, d = spec.split("=", 1)
        fs = sorted(glob.glob(os.path.join(d, "*_gt_vs_render.png")))
        M = {"raw": [[], [], []], "cc": [[], [], []]}
        with ProcessPoolExecutor(a.nproc) as ex:
            for f, cc in ex.map(one, fs, chunksize=4):
                gt, out = load_pair(os.path.join(d, f))
                with torch.no_grad():
                    g = T(gt)
                    for k, im in (("raw", out), ("cc", cc)):
                        o = T(im)
                        M[k][0].append(float(-10 * torch.log10(((o - g) ** 2).mean().clamp_min(1e-12))))
                        M[k][1].append(float(ssim_fn(o, g)))
                        M[k][2].append(float(lp(o.unsqueeze(0), g.unsqueeze(0))))
        r = {k: [float(np.mean(v)) for v in M[k]] for k in M}
        rows.append((nm, len(fs), r))
        print(f"{nm}（{len(fs)} 張）不做 CC {r['raw'][0]:.3f}／{r['raw'][1]:.4f}／{r['raw'][2]:.4f}｜做 CC {r['cc'][0]:.3f}／{r['cc'][1]:.4f}／{r['cc'][2]:.4f}"
              f"｜CC 增量 {r['cc'][0] - r['raw'][0]:+.3f}／{r['cc'][1] - r['raw'][1]:+.4f}／{r['cc'][2] - r['raw'][2]:+.4f}", flush=True)
    print("DONE eval_color_correct")


if __name__ == "__main__":
    main()
