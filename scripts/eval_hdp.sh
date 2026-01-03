#!/bin/bash
# Evaluate HDP-Diffusion Model on Test Set
# Usage: bash scripts/eval_hdp.sh [checkpoint_path] [output_file] [sampling_mode]

echo "=========================================="
echo "HDP-Diffusion Evaluation"
echo "=========================================="

# ============================================
# CONFIGURATION
# ============================================

# Checkpoint path (can override from command line)
CHECKPOINT=${1:-"outputs/hdp_overfit_test/checkpoints/last.ckpt"}

# Output file for results
OUTPUT_FILE=${2:-"outputs/hdp_eval_results.txt"}

# Sampling mode: hdp (fair), hdp_oracle (ablation), bd3lm (baseline comparison)
SAMPLING_MODE=${3:-"hdp"}

# Test set configuration
TEST_SPLIT="test"           # test, validation, or train
NUM_SAMPLES=100             # Number of samples to evaluate (use -1 for all)
BATCH_SIZE=1                # Batch size for inference

# Model configuration (must match training)
MODEL_SIZE="tiny"
ATTN_BACKEND="sdpa"

# HDP configuration
USE_HDP_ATTENTION=true
USE_SPECIAL_FORMAT=true
QUESTION_LEN=128
PLAN_LEN=128
EXEC_LEN=256
SEQ_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))  # 512

# Sampling configuration
NUM_STEPS=1000              # Number of denoising steps
TEMPERATURE=1.0             # Temperature for sampling

echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Test Split: ${TEST_SPLIT}"
echo "  Num Samples: ${NUM_SAMPLES}"
echo "  Sampling Mode: ${SAMPLING_MODE}"
echo "  Num Steps: ${NUM_STEPS}"
echo "  HDP Structure: Q=${QUESTION_LEN}, P=${PLAN_LEN}, E=${EXEC_LEN}"
echo "  Output: ${OUTPUT_FILE}"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "❌ Checkpoint not found: ${CHECKPOINT}"
    echo "Available checkpoints:"
    find outputs/ -name "*.ckpt" 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Validate sampling mode
if [[ ! "${SAMPLING_MODE}" =~ ^(hdp|hdp_oracle|bd3lm)$ ]]; then
    echo "❌ Invalid sampling mode: ${SAMPLING_MODE}"
    echo "Valid modes: hdp, hdp_oracle, bd3lm"
    exit 1
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hdp

# Create output directory
mkdir -p $(dirname ${OUTPUT_FILE})

echo ""
echo "🚀 Starting evaluation..."
echo "   Mode: ${SAMPLING_MODE}"
case ${SAMPLING_MODE} in
    "hdp")
        echo "   Strategy: Categorical sampling + HDP structure (fair comparison)"
        ;;
    "hdp_oracle")
        echo "   Strategy: Argmax + confidence gate (ablation study)"
        ;;
    "bd3lm")
        echo "   Strategy: Standard BD3-LM (baseline comparison)"
        ;;
esac
echo ""

# Run evaluation
python -u main.py \
    mode=sample_eval \
    model=${MODEL_SIZE} \
    data=hdp_base \
    model.length=${SEQ_LEN} \
    model.attn_backend=${ATTN_BACKEND} \
    algo=bd3lm \
    algo.sampler=ddpm \
    algo.backbone=dit \
    noise=loglinear \
    data.hdp.use_hdp_attention=${USE_HDP_ATTENTION} \
    data.hdp.use_special_format=${USE_SPECIAL_FORMAT} \
    data.hdp.question_len=${QUESTION_LEN} \
    data.hdp.plan_len=${PLAN_LEN} \
    data.hdp.exec_len=${EXEC_LEN} \
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
        grep -E "(Accuracy|Total|Correct|Plan|Execution)" ${OUTPUT_FILE} 2>/dev/null || echo "  (Check ${OUTPUT_FILE} for details)"
        echo "----------------------------------------"
    fi
    
    echo ""
    echo "📝 Next Steps:"
    echo "1. Check full results: cat ${OUTPUT_FILE}"
    echo "2. Compare sampling modes:"
    echo "   bash scripts/eval_hdp.sh ${CHECKPOINT} outputs/hdp_eval_hdp.txt hdp"
    echo "   bash scripts/eval_hdp.sh ${CHECKPOINT} outputs/hdp_eval_oracle.txt hdp_oracle"
    echo "   bash scripts/eval_hdp.sh ${CHECKPOINT} outputs/hdp_eval_bd3lm.txt bd3lm"
    echo "3. Analyze hierarchical structure: grep 'PLAN\\|EXECUTION' ${OUTPUT_FILE}"
else
    echo "❌ Evaluation failed with exit code: ${EXIT_CODE}"
    echo "Check the error messages above"
fi
echo "=========================================="

exit ${EXIT_CODE}
