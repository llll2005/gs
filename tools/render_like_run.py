"""用任一模型 ckpt 在「某跑次某塊的 test 視角」上渲染，存成與 `main.py test --save_val` 相同格式的圖（2026-10-15，使用者：compare_server 預設顯示官方）。

用途：官方合併模型（release／論文設定）只有官方 741 幀的渲染，與我方各塊的 held-out test 視角（`3095.png` 這類）對不上，compare_server 無法並排。
  => 用我方渲染器（官方光柵器有 8~13% 暗幀，`實驗分析/09` §0.1；與 tools/eval_official_test.py 相同做法）在同一批相機上重新渲染。
做法：
  相機  ＝ like 跑次 60k ckpt 裡的 dataparser（同一個 block parser、同 down_sample）的 test_set
  GT    ＝ like 跑次已存好的寬圖（GT｜渲染）的左半 => 與我方圖逐位元相同的 GT
  輸出  ＝ <out_root>/<out_run>/blocks/block_<B>/test/epoch=0-step=60000/<影像名>.png（GT｜渲染，同 save_images）
用法：python tools/render_like_run.py --ckpt <模型 ckpt> --like outputs/lab/g44_best1/blocks/block_6 --out_run official_release [--out_root outputs/lab]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TRAIN_FLAGS = ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="要渲染的模型（例：官方合併 ckpt）")
    ap.add_argument("--like", required=True, help="提供相機與 GT 的跑次塊目錄（要有 checkpoints/*step=60000.ckpt 與 test/epoch=*/）")
    ap.add_argument("--out_run", required=True)
    ap.add_argument("--out_root", default="outputs/lab")
    a = ap.parse_args()
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    dev = torch.device("cuda")
    blk = os.path.basename(os.path.normpath(a.like))                       # block_6
    like_ck = sorted(glob.glob(os.path.join(a.like, "checkpoints", "*step=60000.ckpt")))
    tdirs = sorted(glob.glob(os.path.join(a.like, "test", "epoch=*")), key=lambda p: int(p.rsplit("=", 1)[1]))
    assert like_ck and tdirs, f"{a.like} 缺 60k ckpt 或 test 圖"
    tdir = tdirs[-1]
    ck = torch.load(like_ck[0], map_location="cpu")
    dmh = ck["datamodule_hyper_parameters"]; pc = dmh["parser"]; del ck
    if getattr(pc, "points_from", None) == "ply":
        pc.points_from = "random"
    ts = pc.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().test_set
    name2i = {n: i for i, n in enumerate(ts.image_names)}
    pngs = sorted(glob.glob(os.path.join(tdir, "*.png")))
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, dev, eval_mode=True)
    for f in TRAIN_FLAGS:
        if hasattr(renderer, f):
            setattr(renderer, f, False)
    out = os.path.join(a.out_root, a.out_run, "blocks", blk, "test", "epoch=0-step=60000")
    os.makedirs(out, exist_ok=True)
    print(f"[模型] {a.ckpt}  N={model.get_xyz.shape[0]:,}\n[視角] {a.like} 的 test：{len(pngs)} 張（{tdir}）-> {out}", flush=True)
    bg = torch.zeros(3, device=dev)
    done = miss = 0
    with torch.no_grad():
        for p in pngs:
            nm = os.path.basename(p)[:-4]                                     # "3095.png.png" -> "3095.png"
            if nm not in name2i:
                miss += 1; continue
            wide = np.asarray(Image.open(p).convert("RGB"))
            W = wide.shape[1] // 2
            gt = wide[:, :W]
            r = renderer(ts.cameras[name2i[nm]].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
            r = (r.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            assert r.shape[:2] == gt.shape[:2], f"{nm}：渲染 {r.shape[:2]} vs GT {gt.shape[:2]}"
            Image.fromarray(np.concatenate([gt, r], 1)).save(os.path.join(out, os.path.basename(p)))
            done += 1
    print(f"完成 {done} 張（對不上相機 {miss}）", flush=True)
    assert done > 0 and miss == 0


if __name__ == "__main__":
    main()
