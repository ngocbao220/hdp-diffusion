#!/bin/bash
#SBATCH -J train_hdp_overfit              # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=256G                    # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="overfit"           # overfit GPU
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # 1x GPU
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

echo "=========================================="
echo "HDP-Diffusion Training (Single GPU)"
echo "Hierarchical Dual-Process Diffusion"
echo "=========================================="

# ============================================
# EXPERIMENT CONFIGURATION
# ============================================

# Experiment Mode
EXP_NAME="hdp_overfit"           # Tên experiment
DATA_CONFIG="hdp_overfit"        # Data config: hdp_overfit, hdp_base
MODE="train"                     # train, sample, evaluate

# Model Architecture
MODEL_SIZE="tiny"                # tiny, small, base, large
ATTN_BACKEND="sdpa"              # sdpa, flash_attn, flex

# HDP Settings
USE_HDP_ATTENTION=true           # true = HDP hierarchical attention, false = standard
USE_SPECIAL_FORMAT=true          # true = [PLAN]/[EXECUTION] markers
CAUSAL_WITHIN_BLOCK=true         # true = causal mask trong từng block

# Diffusion Algorithm
ALGO="bd3lm"                     # bd3lm, ar, ddpm
SAMPLER="ddpm"                   # ddpm (analytic), semi_ar (block-wise)
BACKBONE="dit"                   # dit, transformer
NOISE_SCHEDULE="loglinear"       # loglinear, linear, cosine

# HDP-Diffusion specific settings
QUESTION_LEN=128
PLAN_LEN=128
EXEC_LEN=256
SEQ_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))  # 512

# Block diffusion settings
BLOCK_SIZE=4  # Can try 4, 8, 16

# Training Hyperparameters
BATCH_SIZE=1         # Small batch for overfitting
EVAL_BATCH_SIZE=1    
VAL_EVERY_N_EPOCH=10      # Validate every 10 epochs
GLOBAL_BATCH_SIZE=1  # No gradient accumulation needed
GRAD_ACCUM=1         

# Training Hyperparameters
MAX_STEPS=700                    # Total training steps
WARMUP_STEPS=10                  # Warmup steps
LOG_INTERVAL=10                  # Log every N steps
# NOTE: Validation disabled for overfit test (causes CUDA crash)
LR=1e-4                          # Learning rate (1e-4 stable, 3e-4 faster)
EMA=0.9999                       # EMA decay rate
RESAMPLE=True                    # Resample during training
GRAD_CLIP=1.0                    # Gradient clipping value

# Hardware Settings
DEVICES=1                        # Number of GPUs
NUM_NODES=1                      # Number of nodes
PRECISION="bf16-mixed"           # bf16-mixed, fp16, fp32
STRATEGY="auto"                  # auto for single GPU (ddp causes issues)

# Optional: Start from pretrained checkpoint
PRETRAIN_CKPT=null

# Output directory
OUTPUT_DIR="outputs/hdp_overfit_test"
mkdir -p ${OUTPUT_DIR}

echo "HDP overfit test:"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

# Run HDP-Diffusion training with configurable parameters
python -u main.py \
    mode=train \
    model=tiny \
    data=hdp_overfit \
    model.length=${SEQ_LEN} \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.backbone=dit \
    +algo.cross_attn=false \
    block_size=${BLOCK_SIZE} \
    noise=loglinear \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=256 \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    loader.num_workers=2 \
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
    +trainer.strategy=auto \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=1.0 \
    +sampling.disable_val_sampling=true \
    wandb.name=hdp-diffusion-overfit-bs${BLOCK_SIZE}-$(date +%Y%m%d-%H%M%S) \
    wandb.project=hdp-diffusion-overfit \
    wandb.tags=[hdp,gsm8k,hierarchical,overfit,bs${BLOCK_SIZE}] \
    +experiment_name=hdp_diffusion_overfit_bs${BLOCK_SIZE} \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"