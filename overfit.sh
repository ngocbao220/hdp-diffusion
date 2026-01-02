#!/bin/bash
#SBATCH -J train_hdp_h200              # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=256G                    # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="h200"           # H200 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # 1x GPU
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

echo "=========================================="
echo "HDP-Diffusion Training (Single GPU)"
echo "Hierarchical Dual-Process Diffusion"
echo "=========================================="

# HDP-Diffusion specific settings
QUESTION_LEN=128
PLAN_LEN=128
EXEC_LEN=256
SEQ_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))  # 512

# Block diffusion settings
BLOCK_SIZE=4  # Can try 4, 8, 16

# H200 Settings (optimized for dual training)
BATCH_SIZE=8         # Increased to use more VRAM
EVAL_BATCH_SIZE=4    # Increased proportionally
GLOBAL_BATCH_SIZE=16    # 48 * 1 GPU * 6 = 288
GRAD_ACCUM=2             # Balanced for speed and memory

# Training hyperparameters (optimized for 5 hours)
MAX_STEPS=1000       # Reduced from 50000 for 5-hour training
WARMUP_STEPS=50     # Reduced proportionally
VAL_EVERY_N_EPOCH=100     # Validate every 2 epochs (less frequent)
LOG_INTERVAL=100

# Learning rate (increased for larger batch)
LR=1e-3  # Increased for larger batch size

# Optional: Start from pretrained checkpoint
PRETRAIN_CKPT=null

# Output directory
OUTPUT_DIR="outputs/hdp_diffusion_h200_bs${BLOCK_SIZE}"
mkdir -p ${OUTPUT_DIR}

echo "H200 Training Configuration (Dual Training Optimized):"
echo "  GPU: 1x H200 (shared with baseline)"
echo "  Hierarchical Structure:"
echo "    Question: ${QUESTION_LEN} tokens"
echo "    Plan: ${PLAN_LEN} tokens"
echo "    Execution: ${EXEC_LEN} tokens"
echo "    Total: ${SEQ_LEN} tokens"
echo "  Diffusion Block Size: ${BLOCK_SIZE}"
echo "  Batch Size: ${BATCH_SIZE} per GPU"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "  Gradient Accumulation: ${GRAD_ACCUM}"
echo "  Learning Rate: ${LR}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Memory: ~55-60GB VRAM (optimized)"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hdp

# Run HDP-Diffusion training with H200 optimizations
python -u main.py \
    mode=train \
    model=tiny \
    data=hdp_overfit \
    model.length=${SEQ_LEN} \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    noise=loglinear \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=256 \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    loader.num_workers=16 \
    optim.lr=${LR} \
    training.ema=0.9999 \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${GRAD_ACCUM} \
    trainer.val_check_interval=null \
    +trainer.check_val_every_n_epoch=${VAL_EVERY_N_EPOCH} \
    trainer.log_every_n_steps=${LOG_INTERVAL} \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    +trainer.strategy=ddp \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=1.0 \
    wandb.name=hdp-diffusion-h200-bs${BLOCK_SIZE}-$(date +%Y%m%d-%H%M%S) \
    wandb.project=hdp-diffusion-h200 \
    wandb.tags=[hdp,gsm8k,hierarchical,h200,bs${BLOCK_SIZE}] \
    +experiment_name=hdp_diffusion_h200_bs${BLOCK_SIZE} \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ HDP-Diffusion training successful!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Fast Training on H200:"
    echo "  Batch size 128 with 4x gradient accumulation"
    echo "  Effective batch size: 512"
    echo "  Completed 10K steps in ~5 hours"
    echo ""
    echo "To evaluate:"
    echo "python main.py mode=sample_eval \\"
    echo "    eval.checkpoint_path=${OUTPUT_DIR}/checkpoints/last.ckpt \\"
    echo "    model=small \\"
    echo "    algo=bd3lm \\"
    echo "    data=hdp_diffusion"
else
    echo "❌ Training failed!"
    echo "Check logs: watch_folder/"
fi

echo "=========================================="
