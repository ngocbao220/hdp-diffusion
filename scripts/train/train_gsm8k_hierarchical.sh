#!/bin/bash

# Training script for Hierarchical BD3-LM on GSM8K
# Adapted for math reasoning with [Question, Plan, Execution] structure

#SBATCH --job-name=gsm8k_hier_bd3lm
#SBATCH --output=logs/gsm8k_hier_bd3lm_%j.out
#SBATCH --error=logs/gsm8k_hier_bd3lm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

# GSM8K-specific hierarchical structure
# Based on analysis: questions are short, plans are concise, executions have calculations
QUESTION_LEN=128   # Math problems are typically short
PLAN_LEN=128       # High-level reasoning steps
EXEC_LEN=256       # Detailed calculations and intermediate steps
TOTAL_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))  # 512

# Block size for diffusion
BLOCK_SIZE=16  # Good balance for math reasoning

# Training settings
MAX_STEPS=50000    # GSM8K train is 7.5k examples
WARMUP_STEPS=5000
BATCH_SIZE=32      # Smaller batch for math reasoning
GRAD_ACCUM=2       # Effective batch = 32 * 2 = 64
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))  # 64
EVAL_INTERVAL=1000
SAVE_INTERVAL=5000

# Learning rate
LR=3e-4

# Model size
MODEL_SIZE=small

# Data paths
TRAIN_DATA="data/gsm8k/gsm8k_hierarchical_train.json"
TEST_DATA="data/gsm8k/gsm8k_hierarchical_test.json"

# Output directory
OUTPUT_DIR="outputs/gsm8k_hierarchical_bd3lm_bs${BLOCK_SIZE}"

echo "=========================================="
echo "Hierarchical BD3-LM Training on GSM8K"
echo "=========================================="
echo "Question Length: $QUESTION_LEN"
echo "Plan Length: $PLAN_LEN"
echo "Execution Length: $EXEC_LEN"
echo "Total Length: $TOTAL_LEN"
echo "Block Size: $BLOCK_SIZE"
echo "Train Data: $TRAIN_DATA"
echo "Test Data: $TEST_DATA"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo "Please run: sbatch scripts/data_prep/run_gsm8k_plan_generation.sh"
    exit 1
fi

# Run training
python -u main.py \
    mode=train \
    model=${MODEL_SIZE} \
    model.length=${TOTAL_LEN} \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    data=gsm8k \
    noise=loglinear \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=8 \
    loader.num_workers=8 \
    optim.lr=${LR} \
    training.ema=0.9999 \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${GRAD_ACCUM} \
    trainer.val_check_interval=${EVAL_INTERVAL} \
    trainer.log_every_n_steps=100 \
    trainer.devices=4 \
    trainer.num_nodes=1 \
    +experiment_name=gsm8k_hierarchical_bs${BLOCK_SIZE} \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training successful!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
else
    echo "❌ Training failed!"
    echo "Check logs for details"
fi

echo "=========================================="
