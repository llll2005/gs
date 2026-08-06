get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

dir="outputs/TrainingSet_vggt_x_pose_opt_mcmc_test"
ref_dir="data/TNT_GOF/TrainingSet"

for data_path in $dir/*; do
    while [ -d "$data_path" ]; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Start evaluation on '$data_path'"
            CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_pose.py \
                --result_path $data_path \
                --ref_folder $ref_dir &
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 15
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 15
        fi
    done
done
wait

python tools/gather_results.py $dir --result_filename corrected_pose_results.txt --format_float 