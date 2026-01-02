#!/bin/bash

# Inference script for HDP-Diffusion checkpoint
# Test hierarchical reasoning generation

echo "=========================================="
echo "HDP-Diffusion Inference Test"
echo "=========================================="

# Find the latest HDP checkpoint
# /content/hdp-diffusion/outputs/hdp_diffusion/2026.01.02/095301/outputs/hdp_diffusion_h200_bs4/checkpoints/best.ckpt
CHECKPOINT_DIR="/content/hdp-diffusion/outputs/hdp_diffusion/2026.01.02/095301/outputs/hdp_diffusion_h200_bs4"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoints/last.ckpt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    find /workspace/hdp-diffusion/outputs -name "*.ckpt" -type f
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Block structure: Q(128) + P(128) + E(256) = 512"
echo "=========================================="

# Run inference with HDP structure
python -u main.py \
    mode=sample_eval \
    model=tiny \
    model.length=512 \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.T=1000 \
    algo.backbone=dit \
    block_size=4 \
    data=hdp_overfit \
    data.hdp.use_hdp_attention=true \
    data.hdp.question_len=128 \
    data.hdp.plan_len=128 \
    data.hdp.exec_len=256 \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    eval.disable_ema=false \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=1 \
    sampling.logdir=outputs/hdp_samples \
    wandb=null

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ HDP Inference successful!"
    echo "Samples saved to: outputs/hdp_samples"
    echo ""
    echo "Check the generated samples to verify:"
    echo "  - Question block quality"
    echo "  - Plan coherence"
    echo "  - Execution correctness"
else
    echo "❌ Inference failed with exit code: $EXIT_CODE"
fi
echo "=========================================="
