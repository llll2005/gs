get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

PROJECT=VGGT
dir="data/TNT_GOF/TrainingSet_vggt_x"
ds_rounding_mode="round"
ds_factor=2
post_fix="_pose_opt_mcmc"
config_path="configs/colmap_pose_opt_mcmc.yaml"
test_config="configs/colmap_pose_opt_mcmc_test.yaml"

declare -a scenes=(
    "$dir/Barn"
    "$dir/Caterpillar"
    "$dir/Ignatius"
    "$dir/Meetingroom"
    "$dir/Truck"
    "$dir/Courthouse"
)

declare -a cap_maxs=(
    882_712
    1_267_267
    3_404_402
    1_161_601
    3_019_416
    551_910
)

# for data_path in $dir/*; do
#     python utils/image_downsample.py $data_path/images --factor 2 &
# done

# Training
for i in "${!scenes[@]}"; do
    data_path=${scenes[$i]}
    cap_max=${cap_maxs[$i]}
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running GS on '$data_path'"
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                    --config $config_path \
                    --data.path $data_path \
                    --model.density.init_args.cap_max $cap_max \
                    --data.parser.init_args.down_sample_factor $ds_factor \
                    --data.parser.init_args.down_sample_rounding_mode $ds_rounding_mode \
                    -n $(basename $data_path)$post_fix \
                    --output outputs/$(basename $dir)$post_fix \
                    --logger wandb \
                    --project $PROJECT &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
wait

# Test view alignment, it can be regarded as another independant training stage
for i in "${!scenes[@]}"; do
    data_path=${scenes[$i]}
    cap_max=${cap_maxs[$i]}
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start running GS on '$data_path'"
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                            --config $test_config \
                            --data.path $data_path \
                            --model.density.init_args.cap_max $cap_max \
                            --data.parser.init_args.down_sample_factor $ds_factor \
                            --data.parser.init_args.down_sample_rounding_mode $ds_rounding_mode \
                            --model.initialize_from outputs/$(basename $dir)${post_fix}/$(basename $data_path)$post_fix \
                            -n $(basename $data_path)${post_fix} \
                            --output outputs/$(basename $dir)${post_fix}_test \
                            --logger wandb \
                            --project $PROJECT &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 60
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 60
        fi
    done
done
wait

rm -rf outputs/$(basename $dir)${post_fix}_test/*/test
for data_path in "${scenes[@]}"; do
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start evaluating GS on '$data_path'"
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
                            --config outputs/$(basename $dir)${post_fix}_test/$(basename $data_path)$post_fix/config.yaml \
                            --save_val --val_train &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 30
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 30
        fi
    done
done
wait

python tools/gather_results.py outputs/$(basename $dir)${post_fix}_test --format_float