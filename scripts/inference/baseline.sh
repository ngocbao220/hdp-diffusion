#!/bin/bash

# Inference script for GSM8K baseline checkpoint
# Test sampling/generation from trained model

echo "=========================================="
echo "GSM8K Baseline Inference Test"
echo "=========================================="

# ============================================
# INFERENCE CONFIGURATION (Chỉnh ở đây!)
# ============================================

# Checkpoint Settings
CHECKPOINT_DIR="/workspace/hdp-diffusion/outputs/gsm8k/2026.01.02/055652/outputs/gsm8k_bd3lm_h200_bs16"
CHECKPOINT_NAME="last.ckpt"          # best.ckpt, last.ckpt, hoặc step_XXX.ckpt
USE_EMA=false                         # true = use EMA weights, false = use regular weights

# Model Architecture (phải khớp với training)
MODEL_SIZE="small"                   # tiny, small, base, large
SEQ_LEN=512                           # Total sequence length
ATTN_BACKEND="sdpa"                  # sdpa, flash_attn, flex
BACKBONE="dit"                       # dit, transformer

# Sampling Settings
SAMPLER="semi_ar"                    # ddpm (analytic), semi_ar (block-wise) - baseline thường dùng semi_ar
NUM_SAMPLING_STEPS=1000              # Number of denoising steps (ignored for semi_ar)
BLOCK_SIZE=16                        # Block size (phải khớp với training)
TOTAL_DIFFUSION_STEPS=1000           # algo.T value

# Generation Settings
NUM_SAMPLE_BATCHES=5                 # Number of batches to generate
BATCH_SIZE=1                         # Samples per batch
TEMPERATURE=1.0                      # Sampling temperature (0.8-1.2)

# Data Config
DATA_CONFIG="gsm8k_baseline"         # gsm8k_baseline

# Output Settings
OUTPUT_DIR="outputs/gsm8k_samples"   # Where to save samples
LOG_TO_WANDB=false                   # true = log to wandb, false = no logging

# ============================================

CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoints/${CHECKPOINT_NAME}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    find /workspace/hdp-diffusion/outputs -name "*.ckpt" -type f
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Model: ${MODEL_SIZE}, Sampler: ${SAMPLER}"
echo "Sequence Length: ${SEQ_LEN}, Block Size: ${BLOCK_SIZE}"
echo "=========================================="

# Prepare wandb setting
if [ "$LOG_TO_WANDB" = true ]; then
    WANDB_CONFIG="wandb.project=baseline-inference wandb.name=baseline-infer-$(date +%Y%m%d-%H%M%S)"
else
    WANDB_CONFIG="wandb=null"
fi

# Run inference with configurable parameters
python -u main.py \
    mode=sample_eval \
    model=${MODEL_SIZE} \
    model.length=${SEQ_LEN} \
    model.attn_backend=${ATTN_BACKEND} \
    algo=bd3lm \
    algo.T=${TOTAL_DIFFUSION_STEPS} \
    algo.backbone=${BACKBONE} \
    algo.sampler=${SAMPLER} \
    block_size=${BLOCK_SIZE} \
    data=${DATA_CONFIG} \
    eval.checkpoint_path=${CHECKPOINT_PATH} \
    eval.disable_ema=${USE_EMA} \
    loader.eval_batch_size=${BATCH_SIZE} \
    sampling.num_steps=${NUM_SAMPLING_STEPS} \
    sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    sampling.temperature=${TEMPERATURE} \
    sampling.logdir=${OUTPUT_DIR} \
    ${WANDB_CONFIG}

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Baseline Inference successful!"
    echo "Generated ${NUM_SAMPLE_BATCHES} batches"
    echo "Samples saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Check the generated samples for:"
    echo "  - Question understanding"
    echo "  - Solution quality"
    echo "  - Answer correctness"
else
    echo "❌ Inference failed with exit code: $EXIT_CODE"
    echo "Check:"
    echo "  - Checkpoint path: ${CHECKPOINT_PATH}"
    echo "  - Model size matches training: ${MODEL_SIZE}"
    echo "  - Block size matches training: ${BLOCK_SIZE}"
    echo "  - Sampler setting: ${SAMPLER}"
fi
echo "=========================================="
