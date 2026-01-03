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

# H200 Settings (optimized for OVERFITTING TEST)
BATCH_SIZE=1         # Small batch for overfitting
EVAL_BATCH_SIZE=1    
GLOBAL_BATCH_SIZE=1  # No gradient accumulation needed
GRAD_ACCUM=1         

# Training Hyperparameters
MAX_STEPS=500                    # Total training steps
WARMUP_STEPS=10                  # Warmup steps
VAL_EVERY_N_EPOCH=100             # Validate every N epochs
LOG_INTERVAL=10                  # Log every N steps
LR=1e-4                          # Learning rate (1e-4 stable, 3e-4 faster)
EMA=0.9999                       # EMA decay rate
RESAMPLE=True                    # Resample during training
GRAD_CLIP=1.0                    # Gradient clipping value

# Hardware Settings
DEVICES=1                        # Number of GPUs
NUM_NODES=1                      # Number of nodes
PRECISION="bf16-mixed"           # bf16-mixed, fp16, fp32
STRATEGY="ddp"                   # ddp, deepspeed, fsdp

# Optional: Start from pretrained checkpoint
PRETRAIN_CKPT=null

# Output directory
OUTPUT_DIR="outputs/hdp_overfit_test"
mkdir -p ${OUTPUT_DIR}

echo "HDP overfit test:"
echo "  Dataset: ${DATA_CONFIG} (1 sample)"
echo "  Goal: Verify model can memorize 1 example"
echo "  Hierarchical Structure:"
echo "    Question: ${QUESTION_LEN} tokens"
echo "    Plan: ${PLAN_LEN} tokens"
echo "    Execution: ${EXEC_LEN} tokens"
echo "    Total: ${SEQ_LEN} tokens"
echo "  Diffusion Block Size: ${BLOCK_SIZE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LR}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Expected: Loss should drop to near 0"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

# Run HDP-Diffusion training with configurable parameters
python -u main.py \
    mode=${MODE} \
    model=${MODEL_SIZE} \
    data=${DATA_CONFIG} \
    model.length=${SEQ_LEN} \
    model.attn_backend=${ATTN_BACKEND} \
    algo=${ALGO} \
    algo.sampler=${SAMPLER} \
    algo.backbone=${BACKBONE} \
    block_size=${BLOCK_SIZE} \
    noise=${NOISE_SCHEDULE} \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=256 \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    loader.num_workers=16 \
    data.hdp.use_hdp_attention=${USE_HDP_ATTENTION} \
    data.hdp.use_special_format=${USE_SPECIAL_FORMAT} \
    data.hdp.causal_within_block=${CAUSAL_WITHIN_BLOCK} \
    data.hdp.question_len=${QUESTION_LEN} \
    data.hdp.plan_len=${PLAN_LEN} \
    data.hdp.exec_len=${EXEC_LEN} \
    optim.lr=${LR} \
    training.ema=${EMA} \
    training.resample=${RESAMPLE} \
    training.from_pretrained=$PRETRAIN_CKPT \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${GRAD_ACCUM} \
    trainer.val_check_interval=null \
    +trainer.check_val_every_n_epoch=${VAL_EVERY_N_EPOCH} \
    trainer.log_every_n_steps=${LOG_INTERVAL} \
    trainer.devices=${DEVICES} \
    trainer.num_nodes=${NUM_NODES} \
    +trainer.strategy=${STRATEGY} \
    trainer.precision=${PRECISION} \
    trainer.gradient_clip_val=${GRAD_CLIP} \
    wandb.name=${EXP_NAME}-${SAMPLER}-bs${BLOCK_SIZE}-$(date +%Y%m%d-%H%M%S) \
    wandb.project=hdp-diffusion-experiments \
    wandb.tags=[hdp,gsm8k,${SAMPLER},bs${BLOCK_SIZE}] \
    +experiment_name=${EXP_NAME} \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Overfitting test completed!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
else
    echo "❌ Training failed!"
    echo "Check logs: watch_folder/"
fi

echo "=========================================="
