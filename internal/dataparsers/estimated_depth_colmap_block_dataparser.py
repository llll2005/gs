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
                    base, ext = image_name.rsplit('.', 1)
                    padded_path = os.path.join(self.path, self.params.depth_dir, f"{base.zfill(6)}.{ext}.npy")
                    if os.path.exists(padded_path):
                        depth_file_path = padded_path
                    else:
                        print("[WARNING] {} does not have a depth file".format(image_name))
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
                loaded_depth_count += 1
            image_set.extra_data_processor = self.load_depth

        if loaded_depth_count == 0:
            print("[WARNING] No depth maps found, depth supervision disabled for this run")
            return dataparser_outputs
        print("found {} depth maps".format(loaded_depth_count))

        return dataparser_outputs

    @staticmethod
    def load_depth(depth_info):
        if depth_info is None:
            return None

        depth_file_path, depth_scale = depth_info
        depth = np.load(depth_file_path) * depth_scale["scale"] + depth_scale["offset"]

        return torch.tensor(depth, dtype=torch.float)
