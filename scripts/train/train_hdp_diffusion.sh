#!/bin/bash
#SBATCH -J train_hdp_diffusion        # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64G                     # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

echo "=========================================="
echo "HDP-Diffusion Training on GSM8K"
echo "Hierarchical Dual-Process Diffusion"
echo "=========================================="

# HDP-Diffusion specific settings
QUESTION_LEN=128
PLAN_LEN=128
EXEC_LEN=256
SEQ_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))  # 512

# Block diffusion settings
BLOCK_SIZE=16  # Can try 4, 8, 16

# Training hyperparameters optimized for H200 (141GB VRAM)
MAX_STEPS=50000
WARMUP_STEPS=5000
BATCH_SIZE=128  # Increased for H200
GRAD_ACCUM=1    # No gradient accumulation needed
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))  # 128

# Learning rate
LR=3e-4

# Optional: Start from pretrained checkpoint
PRETRAIN_CKPT=null  # or path to checkpoint

# Output directory
OUTPUT_DIR="outputs/hdp_diffusion_gsm8k_bs${BLOCK_SIZE}"
mkdir -p ${OUTPUT_DIR}

echo "Configuration:"
echo "  Question Block: ${QUESTION_LEN} tokens"
echo "  Plan Block: ${PLAN_LEN} tokens"
echo "  Execution Block: ${EXEC_LEN} tokens"
echo "  Total Seq Length: ${SEQ_LEN} tokens"
echo "  Diffusion Block Size: ${BLOCK_SIZE}"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bd3lm310

# Run HDP-Diffusion training
python -u main.py \
    mode=train \
    model=small \
    model.length=${SEQ_LEN} \
    model.attn_backend=flash_attn \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    data=hdp_diffusion \
    noise=loglinear \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=8 \
    loader.num_workers=8 \
    optim.lr=${LR} \
    training.ema=0.9999 \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${GRAD_ACCUM} \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=100 \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.precision=bf16-mixed \
    wandb.name=hdp-diffusion-gsm8k-bs${BLOCK_SIZE} \
    wandb.project=hdp-diffusion \
    wandb.tags=[hdp,gsm8k,hierarchical,bs${BLOCK_SIZE}] \
    +experiment_name=hdp_diffusion_gsm8k_bs${BLOCK_SIZE} \
    +hdp.enabled=true \
    +hdp.question_len=${QUESTION_LEN} \
    +hdp.plan_len=${PLAN_LEN} \
    +hdp.exec_len=${EXEC_LEN} \
    +hdp.use_hdp_attention=true \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ HDP-Diffusion training successful!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
    echo ""
    echo "To evaluate:"
    echo "python main.py mode=sample_eval \\"
    echo "    eval.checkpoint_path=${OUTPUT_DIR}/checkpoints/last.ckpt \\"
    echo "    model=small \\"
    echo "    algo=bd3lm \\"
    echo "    data=hdp_diffusion"
else
    echo "❌ Training failed!"
    echo "Check logs for details: watch_folder/"
fi

echo "=========================================="
