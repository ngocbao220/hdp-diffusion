#!/bin/bash

# Training script for Hierarchical Block Diffusion (Plan-then-Generate)
# Based on train_owt_bd3lm.sh but adapted for hierarchical reasoning

#SBATCH --job-name=hier_bd3lm
#SBATCH --output=logs/hier_bd3lm_%j.out
#SBATCH --error=logs/hier_bd3lm_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

# Hierarchical structure settings
QUESTION_LEN=256
PLAN_LEN=256
EXEC_LEN=512
TOTAL_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))  # 1024

# Block size for diffusion (can be 4, 8, 16, 32, etc.)
BLOCK_SIZE=16

# Training steps
MAX_STEPS=100000
WARMUP_STEPS=10000
BATCH_SIZE=64
GRAD_ACCUM=1

# Learning rate
LR=5e-4

# Model size (tiny, small, medium)
MODEL_SIZE=small

# Output directory
OUTPUT_DIR="outputs/hierarchical_bd3lm_bs${BLOCK_SIZE}"

echo "=========================================="
echo "Hierarchical Block Diffusion Training"
echo "=========================================="
echo "Question Length: $QUESTION_LEN"
echo "Plan Length: $PLAN_LEN"
echo "Execution Length: $EXEC_LEN"
echo "Total Length: $TOTAL_LEN"
echo "Block Size: $BLOCK_SIZE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run training
python -u main.py \
    mode=train \
    model=${MODEL_SIZE} \
    model.length=${TOTAL_LEN} \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    data=openwebtext \
    noise=loglinear \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=8 \
    loader.num_workers=8 \
    optim.lr=${LR} \
    training.max_steps=${MAX_STEPS} \
    training.warmup_steps=${WARMUP_STEPS} \
    training.ema=0.9999 \
    training.gradient_accumulation=${GRAD_ACCUM} \
    training.val_check_interval=2000 \
    training.save_interval=10000 \
    training.hierarchical.enabled=true \
    training.hierarchical.question_len=${QUESTION_LEN} \
    training.hierarchical.plan_len=${PLAN_LEN} \
    training.hierarchical.exec_len=${EXEC_LEN} \
    training.hierarchical.mode=full \
    callbacks.checkpoint_monitor.filename="hierarchical-{step}-{valid_nelbo:.4f}" \
    +experiment_name="hierarchical_bd3lm_bs${BLOCK_SIZE}" \
    +output_dir=${OUTPUT_DIR}

echo "Training completed!"
