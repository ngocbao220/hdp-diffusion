#!/bin/bash

# Inference script for HDP-Diffusion checkpoint
# Test hierarchical reasoning generation with CONDITIONAL sampling

echo "=========================================="
echo "HDP-Diffusion Conditional Inference Test"
echo "=========================================="

# Find the latest HDP checkpoint
CHECKPOINT_DIR="/workspace/hdp-diffusion/outputs/hdp_diffusion/2026.01.02/010620/outputs/hdp_diffusion_h200_bs16/"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoints/last.ckpt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found:  $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    find /workspace/hdp-diffusion/outputs -name "*.ckpt" -type f
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Block structure: Q(128) + P(128) + E(256) = 512"
echo "Mode:  Conditional generation (Question → Plan + Execution)"
echo "=========================================="

# Run inference with HDP structure
python -u main.py \
    mode=sample_eval \
    model=small \
    model.length=512 \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.T=1000 \
    algo.backbone=dit \
    block_size=16 \
    data=hdp_diffusion \
    data.test_path=data/gsm8k_hierarchical_test.json \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    eval.disable_ema=false \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=5 \
    sampling.logdir=outputs/hdp_samples \
    +hdp.enabled=true \
    +hdp.question_len=128 \
    +hdp.plan_len=128 \
    +hdp.exec_len=256 \
    +hdp.use_hdp_attention=true \
    wandb=null

EXIT_CODE=$? 

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ HDP Conditional Inference successful!"
    echo "Samples saved to: outputs/hdp_samples"
    echo ""
    echo "Check the generated samples to verify:"
    echo "  - Question block matches input"
    echo "  - Plan coherence with question"
    echo "  - Execution correctness"
else
    echo "❌ Inference failed with exit code:  $EXIT_CODE"
fi
echo "=========================================="