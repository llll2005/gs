"""官方合併模型的「剪枝後品質」曲線用：依 opacity 由低到高剪掉 X%，各存成一個 ckpt（2026-09-29）。

用途：回答「官方模型裡有多少顆是多餘的」—— 本專案命題是用更少顆／更低成本達到同品質，
      這條曲線是官方那一側的對照。**評分交給官方自己的 `main.py test`**（同 config、同 test 集），
      所以剪枝後的分數與官方 held-out 分數完全同一條量測路徑，本工具只負責產生 ckpt。

做法（只讀不改官方原始碼；在 cityGS_origin 目錄、官方環境下執行，才能 unpickle 官方的 hyper_parameters）：
  - `gaussian_model.gaussians.*`（第一維 = N 的逐顆參數）依同一個遮罩剪
  - `density_controller.*` **原樣保留**（官方 merge 本來就是不經遮罩直接串接，與 N 不一致；test 不用它）
  - `optimizer_states` **清空**：test 只載 state_dict（on_load_checkpoint 依 means 的顆數重建模型），
    而它佔合併 ckpt 約 2/3（41 GB 裡約 27 GB）=> 保留會讓 RAM 放不下多個版本
  - opacity 以 sigmoid 後的值排序，並列時依索引（穩定排序）=> 可重現

用法（cwd = cityGS_origin）：
  python <gs>/tools/official_prune_ckpt.py <合併 ckpt> <輸出根目錄名前綴> 10 25 50 75
  => outputs/<前綴>_p10/checkpoints/<同檔名> ...
"""
import gc
import os
import sys

import torch


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    ratios = [int(x) for x in sys.argv[3:]] or [10, 25, 50, 75]
    print(f"載入 {src}（{os.path.getsize(src) / 2**30:.1f} GiB）…", flush=True)
    ck = torch.load(src, map_location="cpu")
    n_opt = sum(t.numel() for o in ck.get("optimizer_states", []) for st in o.get("state", {}).values()
                for t in st.values() if torch.is_tensor(t))
    ck["optimizer_states"] = []          # 見檔頭
    gc.collect()
    sd = ck["state_dict"]
    pre = "gaussian_model.gaussians."
    N = sd[pre + "means"].shape[0]
    keys = [k for k in sd if k.startswith(pre)]
    bad = [k for k in keys if sd[k].shape[0] != N]
    assert not bad, f"這些逐顆參數第一維不是 N={N}：{bad}"
    op = torch.sigmoid(sd[pre + "opacities"].reshape(N, -1)[:, 0].float())
    order = torch.argsort(op, stable=True)            # 由低到高
    print(f"N = {N:,}；已清掉 optimizer 狀態 {n_opt * 4 / 2**30:.1f} GiB（float32 估）", flush=True)
    orig = {k: sd[k] for k in keys}
    name = os.path.basename(src)
    for r in ratios:
        k_drop = int(round(N * r / 100))
        keep = torch.ones(N, dtype=torch.bool)
        keep[order[:k_drop]] = False
        thr = op[order[k_drop - 1]].item() if k_drop > 0 else float("nan")
        for k in keys:
            sd[k] = orig[k][keep].clone()
        out_dir = os.path.join("outputs", f"{prefix}_p{r:02d}", "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, name)
        torch.save(ck, out)
        print(f"  剪 {r:2d}%：剪掉 {k_drop:,} 顆（opacity <= {thr:.5f}），剩 {int(keep.sum()):,} 顆 -> {out}"
              f"（{os.path.getsize(out) / 2**30:.2f} GiB）", flush=True)
        for k in keys:
            sd[k] = None
        gc.collect()
    print("✅ 完成")


if __name__ == "__main__":
    main()
