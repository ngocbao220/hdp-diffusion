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
#SBATCH --gres=gpu:1                  # Single H200 GPU
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# GSM8K Baseline BD3-LM Training - OPTIMIZED FOR H200
# H200: 141GB VRAM - Can handle much larger batch sizes!

echo "=========================================="
echo "GSM8K Baseline Training on H200"
echo "=========================================="

BLOCK_SIZE=16  # Standard block size for BD3-LM (can try 4, 8, 16)
SEQ_LENGTH=512 # Context length for GSM8K problems

# 2xH200 Optimized Settings (141GB VRAM each)
BATCH_SIZE=64           # Per GPU: 128 x 2 = 256 per step
EVAL_BATCH_SIZE=32      # Larger eval batch
GLOBAL_BATCH_SIZE=256   # Same global batch
GRAD_ACCUM=2             # Reduced from 4 (256 x 2 = 512)

MAX_STEPS=50000
WARMUP_STEPS=5000
VAL_EVERY_N_EPOCH=1      # Validate every epoch (18 batches)
LOG_INTERVAL=50          # More frequent logging

# Learning rate (can increase slightly with larger batch)
LR=5e-4  # Increased from 3e-4 for larger batch

# Optional: Start from a pretrained checkpoint
PRETRAIN_CKPT=null  # or path to pretrained checkpoint

# Output directory
OUTPUT_DIR="outputs/gsm8k_bd3lm_h200_bs${BLOCK_SIZE}"
mkdir -p ${OUTPUT_DIR}

echo "2xH200 Configuration:"
echo "  GPUs: 2x H200 (141GB VRAM each)"
echo "  Batch Size: ${BATCH_SIZE} per GPU (256 total per step)"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "  Gradient Accumulation: ${GRAD_ACCUM}"
echo "  Learning Rate: ${LR}"
echo "  Sequence Length: ${SEQ_LENGTH}"
echo "  Block Size: ${BLOCK_SIZE}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Strategy: DDP (Distributed Data Parallel)"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bd3lm310

# Run training with H200 optimizations
python -u main.py \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=256 \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    loader.num_workers=16 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
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
    trainer.devices=2 \
    trainer.num_nodes=1 \
    +trainer.strategy=ddp \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=1.0 \
    wandb.name=bd3lm-gsm8k-h200-bs${BLOCK_SIZE}-$(date +%Y%m%d) \
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
    echo "Training Speed Estimate:"
    echo "  With 2xH200 (batch_size=128 per GPU):"
    echo "    ~1.8x faster than 1xH200"
    echo "    ~3.5-5x faster than 4xA100"
    echo "    Expected time: ~35-38 hours for 50K steps (~1.5 days)"
else
    echo "❌ Training failed!"
    echo "Check logs for details"
fi

echo "=========================================="
