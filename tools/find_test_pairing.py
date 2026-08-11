"""Which image does each official test camera actually see? Brute-force, no offset assumption.

eval_official_test.py searches offsets -1/0/+1 only, and on the merged 3x3 model all three scored
16.4-16.6 -- a 0.03 dB spread. That was read twice as something other than what it was; the user
checked a side-by-side and the two halves show different places. All three are wrong.

The naming explains why a small offset was never justified:
    cameras (sparse/0)    0000.png .. 0740.png   4-digit, 0-based
    images_1.2/ symlinks  0001.png .. 0741.png   4-digit, 1-based -> input/000001.png
The camera names do not exist in the image directory, so the pairing is inferred from ordering.

Render each probe camera ONCE, then score it against every image. If a constant offset exists,
every probe reports the same delta and the margin over the runner-up is large. If the deltas
disagree, the ordering assumption is dead and pairing has to come from geometry, not filenames.
"""
import argparse, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test")
ap.add_argument("--probes", type=int, default=6)
ap.add_argument("--down", type=int, default=8, help="score at 1/down resolution; only the ranking matters")
a = ap.parse_args()

dev = "cuda"
names, cams = load_test_cameras(a.test_dir, 1.2)
img_dir = os.path.join(a.test_dir, "images_1.2")
files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg")))
print(f"[相機] {len(names)} 台 {names[0]}..{names[-1]}   [影像] {len(files)} 張 {files[0]}..{files[-1]}")

model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, dev, eval_mode=True)
print(f"[模型] N={model.get_xyz.shape[0]:,}")
W, H = int(cams.width[0]), int(cams.height[0])
w, h = W // a.down, H // a.down

# every image once, at low resolution, on the CPU side to keep VRAM for the model
gts = torch.stack([
    torch.from_numpy(np.array(Image.open(os.path.join(img_dir, f)).convert("RGB").resize((w, h), Image.LANCZOS), np.uint8))
    for f in files]).float().permute(0, 3, 1, 2) / 255.

probe = np.linspace(5, len(names) - 6, a.probes).astype(int)
deltas = []
for i in probe:
    with torch.no_grad():
        r = rend(cams[i].to_device(dev), model, bg_color=torch.zeros(3, device=dev))["render"].clamp(0, 1)
    rd = torch.nn.functional.interpolate(r[None], size=(h, w), mode="bilinear", align_corners=False)[0].cpu()
    mse = ((gts - rd[None]) ** 2).mean(dim=(1, 2, 3))
    k = int(mse.argmin())
    psnr = float(-10 * torch.log10(mse[k].clamp_min(1e-12)))
    second = float(-10 * torch.log10(mse.topk(2, largest=False).values[1].clamp_min(1e-12)))
    cam_idx = int(names[i].rsplit(".", 1)[0])
    deltas.append(k - cam_idx)
    print(f"  相機 {names[i]} (idx {cam_idx:>3}) → 最佳影像 {files[k]} (idx {k:>3})"
          f"   delta {k - cam_idx:+d}   PSNR {psnr:.2f} (次佳 {second:.2f}, 領先 {psnr - second:+.2f})")
    del r, rd, mse
    torch.cuda.empty_cache()

u, c = np.unique(deltas, return_counts=True)
print(f"\n[delta 分布] " + "  ".join(f"{int(x):+d}×{int(y)}" for x, y in zip(u, c)))
if len(u) == 1:
    print(f"  ⇒ 固定偏移 {int(u[0]):+d}，用 --offset {int(u[0])} 重跑評測")
else:
    print("  ⇒ delta 不一致 ⇒ 相機順序與檔案順序不對應，配對不能靠檔名，要改用姿態比對")
