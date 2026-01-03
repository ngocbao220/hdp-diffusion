#!/bin/bash

# Inference script for HDP-Diffusion checkpoint
# Test hierarchical reasoning generation

echo "=========================================="
echo "HDP-Diffusion Inference Test"
echo "=========================================="

# ============================================
# INFERENCE CONFIGURATION (Chỉnh ở đây!)
# ============================================

# Checkpoint Settings
CHECKPOINT_DIR="/content/hdp-diffusion/outputs/hdp_diffusion/2026.01.02/095301/outputs/hdp_diffusion_h200_bs4"
CHECKPOINT_NAME="best.ckpt"          # best.ckpt, last.ckpt, hoặc step_XXX.ckpt
USE_EMA=false                         # true = use EMA weights, false = use regular weights

# Model Architecture (phải khớp với training)
MODEL_SIZE="tiny"                    # tiny, small, base, large
SEQ_LEN=512                           # Total sequence length
ATTN_BACKEND="sdpa"                  # sdpa, flash_attn, flex
BACKBONE="dit"                       # dit, transformer

# HDP Structure (phải khớp với training)
QUESTION_LEN=128
PLAN_LEN=128
EXEC_LEN=256
USE_HDP_ATTENTION=true               # true = hierarchical attention
USE_SPECIAL_FORMAT=true              # true = [PLAN]/[EXECUTION] markers
CAUSAL_WITHIN_BLOCK=true

# Sampling Settings
SAMPLER="ddpm"                       # ddpm (analytic), semi_ar (block-wise)
NUM_SAMPLING_STEPS=1000              # Number of denoising steps (100-1000)
BLOCK_SIZE=4                         # Block size (phải khớp với training)
TOTAL_DIFFUSION_STEPS=1000           # algo.T value

# Generation Settings
NUM_SAMPLE_BATCHES=1                 # Number of batches to generate
BATCH_SIZE=1                         # Samples per batch
TEMPERATURE=1.0                      # Sampling temperature (0.8-1.2)

# Data Config
DATA_CONFIG="hdp_overfit"            # hdp_overfit, hdp_base

# Output Settings
OUTPUT_DIR="outputs/hdp_samples"     # Where to save samples
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
echo "Sampler: ${SAMPLER}, Steps: ${NUM_SAMPLING_STEPS}"
echo "Block structure: Q(${QUESTION_LEN}) + P(${PLAN_LEN}) + E(${EXEC_LEN}) = ${SEQ_LEN}"
echo "HDP Attention: ${USE_HDP_ATTENTION}"
echo "=========================================="

# Prepare wandb setting
if [ "$LOG_TO_WANDB" = true ]; then
    WANDB_CONFIG="wandb.project=hdp-inference wandb.name=hdp-infer-$(date +%Y%m%d-%H%M%S)"
else
    WANDB_CONFIG="wandb=null"
fi

# Run inference with configurable HDP structure
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
    data.hdp.use_hdp_attention=${USE_HDP_ATTENTION} \
    data.hdp.use_special_format=${USE_SPECIAL_FORMAT} \
    data.hdp.causal_within_block=${CAUSAL_WITHIN_BLOCK} \
    data.hdp.question_len=${QUESTION_LEN} \
    data.hdp.plan_len=${PLAN_LEN} \
    data.hdp.exec_len=${EXEC_LEN} \
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
    echo "✅ HDP Inference successful!"
else
    echo "❌ Inference failed with exit code: $EXIT_CODE"
fi
echo "=========================================="
