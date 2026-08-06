## Prepare Environment and Data

First, please follow steps A-D of [Installation](../doc/installation.md) document to install basic environment. Then, apply

```bash
pip install -r requirements/gsplat.txt
```

to install gsplat for MCMC-3DGS rendering. Then you can follow the instruction of [VGGT-X](https://github.com/Linketic/VGGT-X) to download data from MipNeRF360, Tanks and Temple, and CO3Dv2, and then feedforwardly generate 3D key attributes in COLMAP format. The output folder would be like:

``` 
data/
  ├── CO3Dv2/
  ├── CO3Dv2_vggt_x/
  ├── MipNeRF360/
  ├── MipNeRF360_vggt_x/
  └── TNT_GOF/
      ├── TrainingSet/
      └── TrainingSet_vggt_x/
```

## Run Joint Optimization
The detailed setting of each step on CO3Dv2, MipNeRF360, and Tanks and Temple can be found in `./scripts/vggt-x/run_mcmc_pose_opt_SCENE.sh` and `./scripts/vggt-x/pose_eval.sh`.

### A. Data Downsampling (Optional)
```bash
# $ds_factor is the desired downsample factor.
python utils/image_downsample.py $SCENE_DIR_vggt_x/
/images --factor $ds_factor &
```

### B. Joint Optimization
```bash
python main.py fit \
    --config configs/colmap_pose_opt_mcmc.yaml \
    --data.path $SCENE_DIR_vggt_x \
    --data.parser.init_args.down_sample_factor $ds_factor \
    --data.parser.init_args.down_sample_rounding_mode $ds_rounding_mode \
    --model.density.init_args.cap_max $cap_max \
    -n $(basename $SCENE_DIR_vggt_x)$post_fix \
```
`$SCENE_DIR_vggt_x` is the path to specific scene of the dataset, e.g., "data/MipNeRF360/bicycle". `$ds_factor` is the downsample factor of input images. `$ds_rounding_mode` is the rounding mode when downsampling, can be "floor", "round", "round_half_up", or "ceil". `$cap_max` is the max number of Gaussians for MCMC-3DGS. `$post_fix` is the postfix for the output folder name.

### C. Test View Alignment
Since the poses of test views are not optimized during training, we have to align it to optimized training poses for objective evaluation. The following script optimize the test view poses with frozen Gaussians:
```bash
python main.py fit \
    --config configs/colmap_pose_opt_mcmc_test.yaml \
    --data.path $SCENE_DIR_vggt_x \
    --model.density.init_args.cap_max $cap_max \
    --data.parser.init_args.down_sample_factor $ds_factor \
    --data.parser.init_args.down_sample_rounding_mode $ds_rounding_mode \
    --model.initialize_from outputs/$(basename $SCENE_DIR_vggt_x)$post_fix \
    -n $(basename $SCENE_DIR_vggt_x)${post_fix}_test \
```

### D. Evaluation
```bash
python main.py test \
    --config outputs/$(basename $SCENE_DIR_vggt_x)${post_fix}_test/config.yaml \
    --save_val --val_train
```

It test rendering quality and also applies for the training view evaluation after step B. If you want to test the pose accuracy after optimization, please use:

```bash
python tools/eval_pose.py \
    --result_path outputs/$(basename $SCENE_DIR_vggt_x)${post_fix}_test \
    --ref_folder $SCENE_DIR &
```

`$SCENE_DIR` is the directory containing groundtruth COLMAP result. Please make sure `$SCENE_DIR/sparse/0` exists.