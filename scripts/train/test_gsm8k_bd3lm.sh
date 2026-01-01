#!/bin/bash

# Quick test script for GSM8K baseline training (no SLURM)
# For testing/debugging before running full training

BLOCK_SIZE=16
SEQ_LENGTH=512

echo "=========================================="
echo "GSM8K Baseline BD3-LM Quick Test"
echo "=========================================="
echo "Block Size: ${BLOCK_SIZE}"
echo "Sequence Length: ${SEQ_LENGTH}"
echo "=========================================="

source $(conda info --base)/etc/profile.d/conda.sh
conda activate bd3lm310

python -u main.py \
    loader.global_batch_size=8 \
    loader.eval_global_batch_size=8 \
    loader.batch_size=2 \
    loader.eval_batch_size=2 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=gsm8k_baseline \
    model.length=${SEQ_LENGTH} \
    block_size=${BLOCK_SIZE} \
    wandb=null \
    mode=train \
    model.attn_backend=sdpa \
    training.resample=True \
    training.from_pretrained=null \
    trainer.max_steps=100 \
    trainer.val_check_interval=50 \
    trainer.devices=1 \
    checkpointing.save_dir=outputs/gsm8k_bd3lm_test_bs${BLOCK_SIZE}

echo "=========================================="
echo "Test completed!"
echo "=========================================="
