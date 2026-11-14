"""★ 2026-11-14 本機換塊評測用：把合併模型裁成「指定塊＋周圍一圈」並去掉優化器狀態（使用者「lab 被佔、得等各項的換塊」）。
為什麼：tools/swap_block_eval.py 會把整個換塊後的合併模型搬上 GPU（full44_best 17.6M 顆約 4.2 GB），本機 6 GB 卡（viewer 再佔約 2 GB）放不下。
裁法：塊歸屬照 merge 規則（swap_block_eval.block_mask：rotation_transform → contract → 分區盒）；留下格子座標與任一指定塊相鄰（Chebyshev <= 1）的塊。
遠處的塊拿掉只影響畫面裡很遠的內容，而且所有候選共用同一個裁過的基準 => 候選之間的差（尤其本塊區域／幾何區域 PSNR）仍有效；絕對值和 lab 不同。
用法：python tools/crop_merged_neighbors.py --src outputs/lab/full44_best --dst outputs/local/full44_best_nb14 --blocks 1 4 --parser_from <任一 4x4 候選 ckpt>
"""
import argparse, os, sys, glob
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swap_block_eval import block_mask

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--blocks", type=int, nargs="+", required=True)
ap.add_argument("--parser_from", required=True, help="提供分區設定（datamodule_hyper_parameters.parser）的 4x4 候選 ckpt")
a = ap.parse_args()
pc = torch.load(a.parser_from, map_location="cpu", weights_only=False)["datamodule_hyper_parameters"]["parser"]
parts = torch.load(os.path.join(os.path.dirname(pc.image_list), "partitions.pt"), weights_only=False)
ids = parts["partition_coordinates"]["id"]
keep_b = sorted({j for j in range(len(ids)) for b in a.blocks if int((ids[j] - ids[b]).abs().max()) <= 1})
ck = torch.load(os.path.join(a.src, "checkpoints", "merged.ckpt"), map_location="cpu", weights_only=False)
sd = ck["state_dict"]; pre = "gaussian_model.gaussians."
X = sd[pre + "means"]
m = block_mask(X, pc)                                   # [塊數, N]
keep = m[keep_b].any(0)
n0 = X.shape[0]
for k in list(sd):
    if k.startswith(pre) and sd[k].shape[:1] == (n0,):
        sd[k] = sd[k][keep].clone()
out = {k: ck[k] for k in ck if k not in ("optimizer_states", "lr_schedulers", "loops", "callbacks")}
out["state_dict"] = sd
os.makedirs(os.path.join(a.dst, "checkpoints"), exist_ok=True)
torch.save(out, os.path.join(a.dst, "checkpoints", "merged.ckpt"))
print(f"留下塊 {keep_b}（指定 {a.blocks} 與相鄰）｜{n0:,} -> {int(keep.sum()):,} 顆｜{os.path.join(a.dst, 'checkpoints', 'merged.ckpt')} "
      f"{os.path.getsize(os.path.join(a.dst, 'checkpoints', 'merged.ckpt')) / 2**30:.2f} GB")
