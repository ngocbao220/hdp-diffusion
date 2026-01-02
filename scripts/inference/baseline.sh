#!/bin/bash

# Inference script for GSM8K baseline checkpoint
# Test sampling/generation from trained model

echo "=========================================="
echo "GSM8K Baseline Inference Test"
echo "=========================================="

# Find the latest checkpoint
# /workspace/hdp-diffusion/outputs/gsm8k/2026.01.02/055652/outputs/gsm8k_bd3lm_h200_bs16/checkpoints/best.ckpt
CHECKPOINT_DIR="/workspace/hdp-diffusion/outputs/gsm8k/2026.01.02/055652/outputs/gsm8k_bd3lm_h200_bs16"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoints/last.ckpt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    find /workspace/hdp-diffusion/outputs -name "*.ckpt" -type f
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "=========================================="

# Run inference
python -u main.py \
    mode=sample_eval \
    model=small \
    model.length=512 \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.T=1000 \
    algo.backbone=dit \
    block_size=16 \
    data=gsm8k_baseline \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    eval.disable_ema=false \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=5 \
    sampling.logdir=outputs/gsm8k_samples \
    wandb=null

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Inference successful!"
    echo "Samples saved to: outputs/gsm8k_samples"
else
    echo "❌ Inference failed with exit code: $EXIT_CODE"
fi
echo "=========================================="
