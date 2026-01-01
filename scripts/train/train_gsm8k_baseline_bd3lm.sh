#!/bin/bash

# Baseline BD3-LM Training on GSM8K (No Hierarchical)
# This is the standard block diffusion model for comparison

#SBATCH --job-name=gsm8k_bd3lm_baseline
#SBATCH --output=logs/gsm8k_baseline_%j.out
#SBATCH --error=logs/gsm8k_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

echo "=========================================="
echo "BD3-LM Baseline Training on GSM8K"
echo "=========================================="

# Model settings
MODEL_SIZE=small
SEQ_LEN=512       # Full sequence length for GSM8K
BLOCK_SIZE=16     # Standard block size for BD3-LM

# Training settings
MAX_STEPS=50000
WARMUP_STEPS=5000
BATCH_SIZE=32
GRAD_ACCUM=2
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))  # 64
EVAL_INTERVAL=1000

# Learning rate
LR=3e-4

# Output
OUTPUT_DIR="outputs/gsm8k_baseline_bd3lm_bs${BLOCK_SIZE}"
mkdir -p ${OUTPUT_DIR}

echo "Model: ${MODEL_SIZE}"
echo "Sequence Length: ${SEQ_LEN}"
echo "Block Size: ${BLOCK_SIZE}"
echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Max Steps: ${MAX_STEPS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Run standard BD3-LM training (no hierarchical)
python -u main.py \
    mode=train \
    model=${MODEL_SIZE} \
    model.length=${SEQ_LEN} \
    model.attn_backend=flash_attn \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    data=gsm8k \
    noise=loglinear \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=${GLOBAL_BATCH_SIZE} \
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
    wandb.mode=offline \
    +experiment_name=gsm8k_baseline_bd3lm_bs${BLOCK_SIZE} \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training successful!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
    echo ""
    echo "To evaluate:"
    echo "python main.py mode=sample_eval \\"
    echo "    eval.checkpoint_path=${OUTPUT_DIR}/checkpoints/last.ckpt \\"
    echo "    model=${MODEL_SIZE} \\"
    echo "    algo=bd3lm \\"
    echo "    data=gsm8k"
else
    echo "❌ Training failed!"
    echo "Check logs for details"
fi

echo "=========================================="
