#!/bin/bash

# Quick test BD3-LM baseline on GSM8K with tiny model
# For rapid iteration and debugging

echo "=========================================="
echo "BD3-LM Baseline Quick Test on GSM8K"
echo "=========================================="

# Model settings (tiny for fast testing)
MODEL_SIZE=tiny
SEQ_LEN=512
BLOCK_SIZE=16

# Training settings (quick test)
MAX_STEPS=100     # Just 100 steps for testing
WARMUP_STEPS=10
BATCH_SIZE=4      # Small batch for quick test
GRAD_ACCUM=1
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

# Learning rate
LR=3e-4

# Output
OUTPUT_DIR="outputs/gsm8k_baseline_test_tiny"
mkdir -p ${OUTPUT_DIR}

echo "Model: ${MODEL_SIZE}"
echo "Max Steps: ${MAX_STEPS} (test only)"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Run quick test
python -u main.py \
    mode=train \
    model=${MODEL_SIZE} \
    model.length=${SEQ_LEN} \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    data=gsm8k \
    noise=loglinear \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=4 \
    loader.num_workers=4 \
    optim.lr=${LR} \
    training.ema=0.9999 \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${GRAD_ACCUM} \
    trainer.val_check_interval=50 \
    trainer.log_every_n_steps=10 \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    wandb.mode=disabled \
    +experiment_name=gsm8k_baseline_test \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Test successful! Ready for full training."
    echo ""
    echo "Run full training:"
    echo "bash scripts/train/train_gsm8k_baseline_bd3lm.sh"
else
    echo "❌ Test failed!"
fi
echo "=========================================="
