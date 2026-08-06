#!/bin/bash

# ==========================================
# 1. 中斷保護機制 (Trap)
# 捕捉 Ctrl+C (SIGINT)，確保按下時徹底終止整個腳本，而不會跳到下一個 Block
# ==========================================
trap "echo -e '\n=========停止訓練========='; exit 1" SIGINT SIGTERM

# ==========================================
# 2. 系統效能與防護設定
# ==========================================
export TORCH_FLOAT32_MATMUL_PRECISION=high  # 啟用 RTX 4050 Tensor Cores 加速
export WANDB_MODE=disabled                  # 禁用 wandb 雲端同步，節省網路與效能
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.8

BASE_DIR="/home/LnoArch/Projects/專題/CityGaussian/outputs/citygsv2_mc_aerial_sh2_trim/blocks"

# ==========================================
# 3. 智能續傳迴圈 (Block 0 到 24)
# ==========================================
for i in {24..24}; do
    BLOCK_DIR="$BASE_DIR/block_$i"

    # 【智能跳過】：檢查這個 Block 是否已經擁有 60000 步的最終權重
    if ls $BLOCK_DIR/checkpoints/*step=24000*.ckpt 1> /dev/null 2>&1; then
        echo "✅ Block $i 已經訓練完成，自動跳過..."
        continue
    fi
    if ls $BLOCK_DIR/checkpoints/*step=30000*.ckpt 1> /dev/null 2>&1; then
        echo "✅ Block $i 已經訓練完成，自動跳過..."
        continue
    fi
    if ls $BLOCK_DIR/checkpoints/*step=60000*.ckpt 1> /dev/null 2>&1; then
        echo "✅ Block $i 已經訓練完成，自動跳過..."
        continue
    fi

    echo "======================================================"
    echo " 🚀 開始 5x5 高畫質精雕區塊: Block $i / 24 (1.2x + SH=2)"
    echo "======================================================"

    # 【乾淨重啟】：如果這個 Block 之前跑到一半被中斷，清空它的殘骸，確保從頭乾淨訓練
    rm -rf $BLOCK_DIR/*

    python main.py fit \
        --config ./configs/citygsv2_mc_aerial_sh2_trim24.yaml \
        --data.parser.block_id $i \
        -n=citygsv2_mc_aerial_sh2_trim \
        --project aerial_fine_5x5_sh2

    # 【錯誤捕捉】：檢查 python 執行結果。如果因為 OOM 報錯或中斷，腳本會自動停止，不會盲目往下跑
    if [ $? -ne 0 ]; then
        echo "❌ Block $i 訓練失敗或被手動中斷！腳本已停止。"
        exit 1
    fi
done

echo "🎉 恭喜！所有 25 個區塊高畫質訓練完畢！"