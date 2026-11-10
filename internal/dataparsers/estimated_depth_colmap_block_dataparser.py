import os
import json
import numpy as np
import torch
from dataclasses import dataclass
from .colmap_block_dataparser import ColmapBlock, ColmapBlockDataParser
from internal.dataparsers import DataParserOutputs

@dataclass
class EstimatedDepthBlockColmap(ColmapBlock):
    depth_dir: str = "estimated_depths"

    depth_rescaling: bool = True

    depth_scale_name: str = "estimated_depth_scales"

    depth_scale_lower_bound: float = 0.2

    depth_scale_upper_bound: float = 5.

    depth_rel_dir: str = ""
    """★ 2026-11-11 可靠度分級圖的目錄（`tools/make_tri_depths.py` 產生，與深度圖同名、同大小的 uint8：斜率級 + 8 x 大小級）。
    空字串＝不用（行為與之前逐位元相同）。有給時，不可靠的像素深度設成 0 => 使用端（深度支撐閘門、補洞放點、單側不能看穿）
    本來就把 <= 0 當無效、不看那些像素。"""
    depth_rel_slope_max: int = 1
    """可靠＝斜率級 <= 這個值。斜率級 0／1／2／3／4＝三角形的無因次斜率 < 1／< 3／< 8／>= 8／凸包外
    （斜率＝|∇ log z| x fx＝表面對視線的 cot；跨高低落差的三角形斜率大）。"""
    depth_rel_size_max: int = 3
    """且 大小級 <= 這個值。大小級 0／1／2／3＝三角形最長邊 < 16／< 64／< 256／>= 256 px。3＝不限大小。"""

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "EstimatedDepthBlockColmapDataParser":
        return EstimatedDepthBlockColmapDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class EstimatedDepthBlockColmapDataParser(ColmapBlockDataParser):
    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        if self.params.depth_rescaling is True:
            depth_scale_json = os.path.join(self.path, self.params.depth_scale_name + ".json")
            if not os.path.exists(depth_scale_json):
                # The test set has no estimated depths / depth-scale json. Depth is only used for
                # the TRAINING loss; eval/test needs none. Without this guard, `main.py test` on
                # the test set crashes with FileNotFoundError here. Skip depth and return the base
                # (camera + image) outputs, matching the "No depth maps found" path below.
                print(f"[INFO] depth-scale json not found ({depth_scale_json}); depth supervision "
                      f"disabled (expected for test/eval).")
                return dataparser_outputs
            with open(depth_scale_json, "r") as f:
                depth_scales = json.load(f)

            median_scale = np.median(np.asarray([i["scale"] for i in depth_scales.values()]))

        loaded_depth_count = 0
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                # Depth maps are named after the IMAGE FILE, so resolve them from the path the
                # image dataparser actually picked -- not from the COLMAP name via zfill. That
                # zfill was off by one on this capture (COLMAP is 0-based, the files are 1-based),
                # which silently fed every image the neighbouring frame's depth (2026-08-12).
                actual = os.path.basename(image_set.image_paths[idx]) \
                    if getattr(image_set, "image_paths", None) is not None else image_name
                depth_file_path = os.path.join(self.path, self.params.depth_dir, f"{actual}.npy")
                if os.path.exists(depth_file_path) is False:
                    # ⛔ 這裡曾經有一個 zfill(6) fallback，會在位置解析失敗時**靜默**改用鄰幀的
                    # 深度圖（同一個 off-by-one 在本專案出現過三處：dataparser 2026-08-12、
                    # depth_init_blocks 2026-08-22、get_depth_scales 2026-08-23）。
                    # 靜默退回是這個 bug 活了六週的原因 => 改成大聲失敗，不要猜。
                    print("[WARNING] {} 找不到深度圖 {} —— 位置對應失敗，"
                          "**不做補零猜測**（那會餵鄰幀的深度）".format(image_name, depth_file_path))
                    continue

                depth_scale = {
                    "scale": 1.,
                    "offset": 0.,
                }
                if self.params.depth_rescaling is True:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is None:
                        print("[WARNING {} does not have a depth scale]".format(image_name))
                        continue
                    if depth_scale["scale"] < self.params.depth_scale_lower_bound * median_scale or depth_scale["scale"] > self.params.depth_scale_upper_bound * median_scale:
                        print("[WARNING depth scale of {} out of bound]".format(image_name))
                        continue
                
                image_set.extra_data[idx] = (depth_file_path, depth_scale)
                if self.params.depth_rel_dir:
                    rel_file_path = os.path.join(self.path, self.params.depth_rel_dir, f"{actual}.npy")
                    if not os.path.exists(rel_file_path):
                        raise FileNotFoundError(f"depth_rel_dir 有給但缺可靠度圖 {rel_file_path}（先跑 tools/make_tri_depths.py）")
                    image_set.extra_data[idx] = (depth_file_path, depth_scale, rel_file_path,
                                                 int(self.params.depth_rel_slope_max), int(self.params.depth_rel_size_max))
                loaded_depth_count += 1
            image_set.extra_data_processor = self.load_depth

        if loaded_depth_count == 0:
            print("[WARNING] No depth maps found, depth supervision disabled for this run")
            return dataparser_outputs
        print("found {} depth maps".format(loaded_depth_count))
        if self.params.depth_rel_dir:
            print(f"[depth-rel] ✅ 深度只留可靠像素：{self.params.depth_rel_dir}、斜率級 <= {self.params.depth_rel_slope_max}、"
                  f"大小級 <= {self.params.depth_rel_size_max}（其餘設 0＝無效）", flush=True)

        return dataparser_outputs

    @staticmethod
    def load_depth(depth_info):
        if depth_info is None:
            return None

        depth_file_path, depth_scale = depth_info[0], depth_info[1]
        depth = np.load(depth_file_path) * depth_scale["scale"] + depth_scale["offset"]
        if len(depth_info) > 2:                 # ★ 11-11 可靠度分級：不可靠的像素設 0（使用端當無效）
            rel = np.load(depth_info[2])
            if rel.shape != depth.shape[-2:]:
                raise ValueError(f"可靠度圖 {depth_info[2]} 大小 {rel.shape} != 深度圖 {depth.shape[-2:]}")
            ok = ((rel & 7) <= depth_info[3]) & ((rel >> 3) <= depth_info[4])
            depth = np.where(ok, depth, 0.0).astype(depth.dtype)

        # ★ 2026-10-14 ADMM P2（tools/admm_consensus.py）：共識深度圖在沒打到表面的像素存 NaN
        #   => 回傳 (depth, mask)，metric 端（citygsv2_metrics.get_inverse_depth_metric）本來就支援遮罩。
        #   單目深度檔沒有 NaN => 走原本那一行，行為逐位元不變。
        nan = np.isnan(depth)
        if nan.any():
            return (torch.tensor(np.nan_to_num(depth, nan=0.0), dtype=torch.float), torch.tensor(~nan, dtype=torch.float))
        return torch.tensor(depth, dtype=torch.float)
