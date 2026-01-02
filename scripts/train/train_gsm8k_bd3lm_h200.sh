#!/bin/bash
#SBATCH -J train_gsm8k_bd3lm_h200     # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=256G                    # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="h200"           # H200 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Single GPU
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# GSM8K Baseline BD3-LM Training - Single GPU

echo "=========================================="
echo "GSM8K Baseline Training (Single GPU)"
echo "=========================================="

BLOCK_SIZE=16  # Standard block size for BD3-LM (can try 4, 8, 16)
SEQ_LENGTH=512 # Context length for GSM8K problems

# H200 Settings (optimized for dual training)
BATCH_SIZE=48          # Increased to use more VRAM
EVAL_BATCH_SIZE=24     # Increased proportionally
GLOBAL_BATCH_SIZE=288  # 48 * 1 GPU * 6 = 288
GRAD_ACCUM=6           # Balanced for speed and memory

MAX_STEPS=10000       # Reduced from 50000 for 5-hour training
WARMUP_STEPS=1000     # Reduced proportionally
VAL_EVERY_N_EPOCH=2      # Validate every 2 epochs (less frequent)
LOG_INTERVAL=50          # More frequent logging

# Learning rate (increased for larger batch)
LR=1e-3  # Increased for larger batch size

# Optional: Start from a pretrained checkpoint
PRETRAIN_CKPT=null  # or path to pretrained checkpoint

# Output directory
OUTPUT_DIR="outputs/gsm8k_bd3lm_h200_bs${BLOCK_SIZE}"
mkdir -p ${OUTPUT_DIR}

echo "H200 Training Configuration (Dual Training Optimized):"
echo "  GPU: 1x H200 (shared with HDP)"
echo "  Batch Size: ${BATCH_SIZE} per GPU"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "  Gradient Accumulation: ${GRAD_ACCUM}"
echo "  Learning Rate: ${LR}"
echo "  Sequence Length: ${SEQ_LENGTH}"
echo "  Block Size: ${BLOCK_SIZE}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Memory: ~55-60GB VRAM (optimized)"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="



# Run training with H200 optimizations
python -u main.py \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=256 \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    loader.num_workers=16 \
    model=small \
    algo=bd3lm \
    data=gsm8k_baseline \
    model.length=${SEQ_LENGTH} \
    model.attn_backend=sdpa \
    block_size=${BLOCK_SIZE} \
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
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=1.0 \
    wandb.name=bd3lm-gsm8k-h200-bs${BLOCK_SIZE}-$(date +%Y%m%d-%H%M%S) \
    wandb.project=gsm8k-bd3lm-h200 \
    wandb.tags=[gsm8k,baseline,bd3lm,h200,bs${BLOCK_SIZE}] \
    +experiment_name=gsm8k_bd3lm_h200_bs${BLOCK_SIZE} \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training successful!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Fast Training on H200:"
    echo "  Batch size 128 with 4x gradient accumulation"
    echo "  Effective batch size: 512"
    echo "  Completed 10K steps in ~5 hours"
else
    echo "❌ Training failed!"
    echo "Check logs for details"
fi

echo "=========================================="

