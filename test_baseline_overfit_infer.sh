#!/bin/bash
# Baseline Overfit Inference Test
# Test inference on the same 1 sample used for training

set -e

echo "=========================================="
echo "🧪 BASELINE OVERFIT INFERENCE TEST"
echo "=========================================="

# ============================================
# INFERENCE CONFIGURATION
# ============================================

# Checkpoint from overfit training
CHECKPOINT_DIR="outputs/baseline_overfit_test/gsm8k_1sample"
CHECKPOINT_NAME="last.ckpt"          # or best.ckpt if saved
USE_EMA=false                         # Usually false for small overfit tests

# Model Architecture (must match training)
MODEL_SIZE="tiny"                    # Same as training
SEQ_LEN=512                          # Same as training
BACKBONE="dit"                       # Same as training

# Sampling Settings
SAMPLER="semi_ar"                    # Same as training
NUM_SAMPLING_STEPS=100               # Start with 100, adjust if needed
BLOCK_SIZE=16                        # Same as training
TOTAL_DIFFUSION_STEPS=0              # T=0 for continuous time (BD3-LM)

# Generation Settings
NUM_SAMPLE_BATCHES=3                 # Generate 3 samples to check consistency
BATCH_SIZE=1                         
TEMPERATURE=1.0                      # Standard temperature

# Data Config
DATA_CONFIG="gsm8k_baseline_overfit" # Same 1 sample as training

# Output Settings
OUTPUT_DIR="outputs/baseline_overfit_samples"
LOG_TO_WANDB=false

# ============================================

CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoints/${CHECKPOINT_NAME}"

echo "📍 Checkpoint: ${CHECKPOINT_PATH}"
echo "🎯 Expected: Should regenerate training sample perfectly"
echo "   (Janet's ducks problem → answer: 18)"
echo "=========================================="

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo ""
    echo "Please train first using:"
    echo "  ./test_baseline_overfit.sh"
    echo ""
    echo "Or check available checkpoints:"
    ls -la outputs/baseline_overfit_test/*/checkpoints/*.ckpt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Run inference
python -u main.py \
    mode=sample_eval \
    model=${MODEL_SIZE} \
    model.length=${SEQ_LEN} \
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
    wandb=null

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Baseline Overfit Inference Complete!"
    echo ""
    echo "📊 Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "🔍 Check generated samples for:"
    echo "  ✓ Question: Janet's ducks lay 16 eggs..."
    echo "  ✓ Solution: Correct calculation (16-3-4=9)"
    echo "  ✓ Answer: 18 dollars"
    echo ""
    echo "If model overfitted correctly:"
    echo "  → Should regenerate training data perfectly"
    echo "  → All 3 samples should be nearly identical"
    echo "  → Answer should always be 18"
    echo ""
    echo "To view samples:"
    echo "  cat ${OUTPUT_DIR}/samples_*.txt"
else
    echo "❌ Inference failed with exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Checkpoint not found (train first!)"
    echo "  - Model config mismatch"
    echo "  - Sampling steps too low/high"
fi
echo "=========================================="
