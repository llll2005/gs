import os
import sys
import add_pypath
import argparse
import torch
import numpy as np
import logging
from tqdm import tqdm
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to the model output directory")
parser.add_argument("--enlarge", type=float, default=0., help="Enlarge AABB by this fraction when filtering Gaussians per block")
args = parser.parse_args()

checkpoint_dir = os.path.join(args.path, "blocks")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(os.path.dirname(checkpoint_dir), "merge.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# find max iteration
logger.info("Searching checkpoint files...")
max_iteration = -1
checkpoint_files = []
for i in os.listdir(checkpoint_dir):
    block_path = os.path.join(checkpoint_dir, i)
    loadable_file = GaussianModelLoader.search_load_file(block_path)
    if loadable_file.endswith(".ckpt") is False:
        continue
    try:
        step = int(loadable_file[loadable_file.index("step=") + 5:loadable_file.index(".ckpt")])
        if step > max_iteration:
            max_iteration = step
            checkpoint_files = []
        if step == max_iteration:
            checkpoint_files.append(loadable_file)
    except:
        pass

checkpoint_files = sorted(checkpoint_files)
assert len(checkpoint_files) > 0

logger.info(checkpoint_files)

import add_pypath
import torch
from internal.models.gaussian import Gaussian

is_new_model = True
param_list_key_by_name = {}
extra_param_list_key_by_name = {}
optimizer_state_exp_avg_list_key_by_index = {}
optimizer_state_exp_avg_sq_list_key_by_index = {}
density_controller_state_list_key_by_name = {}
number_of_gaussians = []

ckpt = torch.load(checkpoint_files[0], map_location="cpu")
dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
partitions = torch.load(os.path.join(os.path.dirname(dataparser_config.image_list), "partitions.pt"))
partition_coordinates = PartitionCoordinates(
    id=partitions['partition_coordinates']['id'],
    xy=partitions['partition_coordinates']['xy'],
)
partition_bounding_boxes = partition_coordinates.get_bounding_boxes(partitions['scene_config']['partition_size'], enlarge=args.enlarge)

del ckpt, dataparser_config

for i in tqdm(checkpoint_files, desc="Loading checkpoints"):
    ckpt = torch.load(i, map_location="cpu")
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    xyz_gs = ckpt['state_dict']['gaussian_model.gaussians.means'] @ partitions['extra_data']['rotation_transform'][:3, :3].T

    if partitions['scene_config']['contract']:
        xyz_gs = CityGSPartitioning.contract_to_unisphere(xyz_gs[:, :2], partitions['scene_config']['aabb'], ord=torch.inf)

    mask_preserved = CityGSPartitioning.is_in_bounding_boxes(
        bounding_boxes=partition_bounding_boxes,
        coordinates=xyz_gs[:, :2],
    )[dataparser_config.block_id]
    
    property_names = []
    gaussian_property_dict_key_prefix = "gaussian_model.gaussians."
    density_controller_state_dict_key_prefix = "density_controller."
    # extract gaussian properties
    for key, value in ckpt["state_dict"].items():
        if key.startswith(gaussian_property_dict_key_prefix):
            param_list_key_by_name.setdefault(key, []).append(value[mask_preserved])
            property_names.append(key[len(gaussian_property_dict_key_prefix):])
        elif key.startswith(density_controller_state_dict_key_prefix):
            # Density-controller buffers that are PER-PRIMITIVE must take the same block mask as
            # the gaussian properties. `VanillaDensityController` (and `CityGSV2DensityController`
            # through it) registers `max_radii2D`, `xyz_gradient_accum` and `denom` with
            # persistent=True, so they land here at full block length while the gaussians are
            # already cropped -- concatenating both gives two different totals and the merged
            # checkpoint is silently inconsistent.
            #
            # MCMC has no persistent per-primitive buffer (`binoms` is persistent=False), so our
            # own merges never hit this. The self-run CityGSV2 baseline for the 25-block north
            # star does.
            if isinstance(value, torch.Tensor) and value.shape[:1] == mask_preserved.shape[:1]:
                param_list_key_by_name.setdefault(key, []).append(value[mask_preserved])
            else:
                param_list_key_by_name.setdefault(key, []).append(value)

    # extract optimizer states, assume meet gaussian optimizers first
    for optimizer_idx, optimizer in enumerate(ckpt["optimizer_states"]):
        for param_group_idx, param_group in enumerate(optimizer["param_groups"]):
            if param_group["name"] not in property_names:
                continue

            property_names.remove(param_group["name"])

            # A parameter that never received a gradient has no Adam moments, so it is absent
            # from optimizer["state"] even though its param_group exists. Gaussian2DSB keeps
            # `shs_rest` allocated but unused -- colour comes from shs_dc + sb_params -- so
            # optimizer 1 has groups 0..5 while state holds only keys [0, 2, 3, 4, 5]. Indexing
            # blindly raised KeyError: 1 and killed the 2026-08-03 five-block merge after those
            # blocks had already cost ~40 GPU-hours. There is nothing to merge for such a
            # parameter; its VALUES are concatenated by the name-keyed state_dict loop above.
            state = optimizer["state"].get(param_group_idx)
            if state is None:
                logger.info(f"  '{param_group['name']}': 無 optimizer state（從未收過梯度），跳過動量合併")
                continue

            # Apply the same per-block cell mask as the gaussian properties: exp_avg/exp_avg_sq
            # are row-aligned with the property tensor. Without this the merged ckpt carries
            # optimizer states sized as the SUM OF FULL block models, which crashes any
            # resume-fit from the merged checkpoint (Adam shape mismatch at the first step).
            # [optimizer_idx][param_group_idx] = [state, ...]
            optimizer_state_exp_avg_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(state["exp_avg"][mask_preserved])
            optimizer_state_exp_avg_sq_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(state["exp_avg_sq"][mask_preserved])

        if len(property_names) == 0:
            break

    number_of_gaussians.append(mask_preserved.sum().item())

logger.info("Merging Gaussians and density controller states...")
initialize_from = ckpt['hyper_parameters'].get('initialize_from')
if initialize_from is not None and str(initialize_from).endswith('.ckpt'):
    ckpt["datamodule_hyper_parameters"]["parser"] = torch.load(initialize_from, map_location="cpu")["datamodule_hyper_parameters"]["parser"]
else:
    # depth-init (PLY) or no init: clear block_id so merged model renders full scene
    ckpt["datamodule_hyper_parameters"]["parser"].block_id = None
for i in param_list_key_by_name:
    ckpt["state_dict"][i] = torch.concat(param_list_key_by_name[i], dim=0)
if is_new_model is True:
    logger.info("Merging optimizers...")
    for optimizer_idx in optimizer_state_exp_avg_list_key_by_index.keys():
        for param_group_idx in optimizer_state_exp_avg_list_key_by_index[optimizer_idx].keys():
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg"] = torch.concat(
                optimizer_state_exp_avg_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg_sq"] = torch.concat(
                optimizer_state_exp_avg_sq_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
else:
    for i in extra_param_list_key_by_name:
        ckpt["gaussian_model_extra_state_dict"][i] = torch.concat(extra_param_list_key_by_name[i], dim=0)
    logger.info("Merging optimizers...")
    for i in optimizer_state_exp_avg_list_key_by_index.keys():
        ckpt["optimizer_states"][0]["state"][i]["exp_avg"] = torch.concat(optimizer_state_exp_avg_list_key_by_index[i], dim=0)
        ckpt["optimizer_states"][0]["state"][i]["exp_avg_sq"] = torch.concat(optimizer_state_exp_avg_sq_list_key_by_index[i], dim=0)


def rename_ddp_appearance_states():
    gaussian_property_dict_key_prefix = "renderer.appearance_model.module."
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith(gaussian_property_dict_key_prefix) is False:
            continue
        new_key = "renderer.model.{}".format(i[len(gaussian_property_dict_key_prefix):])
        ckpt["state_dict"][new_key] = ckpt["state_dict"][i]
        del ckpt["state_dict"][i]

# TODO: align ckpt["hyper_parameters"]["renderer"] if appearance model is used

logger.info("number_of_gaussians=sum({})={}".format(number_of_gaussians, sum(number_of_gaussians)))
if not os.path.exists(os.path.join(os.path.dirname(checkpoint_dir), "checkpoints")):
    os.makedirs(os.path.join(os.path.dirname(checkpoint_dir), "checkpoints"))
initialize_from = ckpt["hyper_parameters"].get("initialize_from")
merged_name = initialize_from.split('/')[-1] if (initialize_from is not None and str(initialize_from).endswith('.ckpt')) else "merged.ckpt"
output_path = os.path.join(os.path.dirname(checkpoint_dir), "checkpoints", merged_name)
logger.info("Saving...")
torch.save(ckpt, output_path)
logger.info(f"Saved to '{output_path}'")
