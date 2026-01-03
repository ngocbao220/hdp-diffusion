#!/bin/bash
# Evaluate BD3-LM Baseline Model on Test Set
# Usage: bash scripts/eval_baseline.sh [checkpoint_path] [output_file]

echo "=========================================="
echo "BD3-LM Baseline Evaluation"
echo "=========================================="

# ============================================
# CONFIGURATION
# ============================================

# Checkpoint path (can override from command line)
CHECKPOINT=${1:-"outputs/bd_baseline_overfit_test/checkpoints/last.ckpt"}

# Output file for results
OUTPUT_FILE=${2:-"outputs/bd_baseline_eval_results.txt"}

# Test set configuration
TEST_SPLIT="test"           # test, validation, or train
NUM_SAMPLES=100             # Number of samples to evaluate (use -1 for all)
BATCH_SIZE=1                # Batch size for inference

# Model configuration (must match training)
MODEL_SIZE="tiny"
SEQ_LEN=512
BLOCK_SIZE=4
ATTN_BACKEND="sdpa"

# Sampling configuration
SAMPLING_MODE="bd3lm"       # bd3lm (baseline sampling)
NUM_STEPS=1000              # Number of denoising steps
TEMPERATURE=1.0             # Temperature for sampling

echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Test Split: ${TEST_SPLIT}"
echo "  Num Samples: ${NUM_SAMPLES}"
echo "  Sampling Mode: ${SAMPLING_MODE}"
echo "  Num Steps: ${NUM_STEPS}"
echo "  Output: ${OUTPUT_FILE}"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "❌ Checkpoint not found: ${CHECKPOINT}"
    echo "Available checkpoints:"
    find outputs/ -name "*.ckpt" 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hdp

# Create output directory
mkdir -p $(dirname ${OUTPUT_FILE})

echo ""
echo "🚀 Starting evaluation..."
echo ""

# Run evaluation
python -u main.py \
    mode=sample_eval \
    model=${MODEL_SIZE} \
    data=gsm8k_baseline \
    model.length=${SEQ_LEN} \
    model.attn_backend=${ATTN_BACKEND} \
    algo=bd3lm \
    algo.sampler=ddpm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    noise=loglinear \
    eval.checkpoint_path=${CHECKPOINT} \
    +sampling.sampling_mode=${SAMPLING_MODE} \
    +sampling.num_steps=${NUM_STEPS} \
    sampling.num_sample_batches=1 \
    +sampling.temperature=${TEMPERATURE} \
    loader.eval_batch_size=${BATCH_SIZE} \
    loader.num_workers=8 \
    +eval.split=${TEST_SPLIT} \
    +eval.num_samples=${NUM_SAMPLES} \
    +eval.save_predictions=true \
    +eval.output_file=${OUTPUT_FILE}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Evaluation completed!"
    echo ""
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""
    
    # Try to parse and display results
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "📊 Summary:"
        echo "----------------------------------------"
        grep -E "(Accuracy|Total|Correct)" ${OUTPUT_FILE} 2>/dev/null || echo "  (Check ${OUTPUT_FILE} for details)"
        echo "----------------------------------------"
    fi
    
    echo ""
    echo "📝 Next Steps:"
    echo "1. Check full results: cat ${OUTPUT_FILE}"
    echo "2. Analyze errors: grep 'WRONG' ${OUTPUT_FILE}"
    echo "3. Compare with ground truth in data/gsm8k/"
else
    echo "❌ Evaluation failed with exit code: ${EXIT_CODE}"
    echo "Check the error messages above"
fi
echo "=========================================="

exit ${EXIT_CODE}
