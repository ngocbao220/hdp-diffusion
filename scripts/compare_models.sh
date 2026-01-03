#!/bin/bash
# Compare BD3-LM Baseline vs HDP-Diffusion
# Usage: bash scripts/compare_models.sh [baseline_ckpt] [hdp_ckpt] [num_samples]

echo "=========================================="
echo "Model Comparison: BD3-LM vs HDP-Diffusion"
echo "=========================================="

# Checkpoints
BASELINE_CKPT=${1:-"outputs/bd_baseline_overfit_test/checkpoints/last.ckpt"}
HDP_CKPT=${2:-"outputs/hdp_overfit_test/checkpoints/last.ckpt"}
NUM_SAMPLES=${3:-100}

# Output directory
OUTPUT_DIR="outputs/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

echo "Configuration:"
echo "  Baseline Checkpoint: ${BASELINE_CKPT}"
echo "  HDP Checkpoint: ${HDP_CKPT}"
echo "  Test Samples: ${NUM_SAMPLES}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Check checkpoints
for ckpt in "${BASELINE_CKPT}" "${HDP_CKPT}"; do
    if [ ! -f "${ckpt}" ]; then
        echo "❌ Checkpoint not found: ${ckpt}"
        exit 1
    fi
done

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hdp

echo ""
echo "1️⃣  Evaluating BD3-LM Baseline..."
echo "=========================================="
bash scripts/eval_baseline.sh \
    ${BASELINE_CKPT} \
    ${OUTPUT_DIR}/baseline_bd3lm.txt \
    ${NUM_SAMPLES}

echo ""
echo "2️⃣  Evaluating HDP-Diffusion (Fair - Categorical Sampling)..."
echo "=========================================="
bash scripts/eval_hdp.sh \
    ${HDP_CKPT} \
    ${OUTPUT_DIR}/hdp_fair.txt \
    hdp \
    ${NUM_SAMPLES}

echo ""
echo "3️⃣  Evaluating HDP-Diffusion (Oracle - Argmax)..."
echo "=========================================="
bash scripts/eval_hdp.sh \
    ${HDP_CKPT} \
    ${OUTPUT_DIR}/hdp_oracle.txt \
    hdp_oracle \
    ${NUM_SAMPLES}

echo ""
echo "4️⃣  Evaluating HDP as BD3-LM (for comparison)..."
echo "=========================================="
bash scripts/eval_hdp.sh \
    ${HDP_CKPT} \
    ${OUTPUT_DIR}/hdp_as_bd3lm.txt \
    bd3lm \
    ${NUM_SAMPLES}

echo ""
echo "=========================================="
echo "✅ All evaluations completed!"
echo "=========================================="
echo ""
echo "📊 Results Summary:"
echo "----------------------------------------"
echo "1. BD3-LM Baseline:"
grep "Accuracy" ${OUTPUT_DIR}/baseline_bd3lm.txt 2>/dev/null || echo "   (See ${OUTPUT_DIR}/baseline_bd3lm.txt)"

echo ""
echo "2. HDP-Diffusion (Fair):"
grep "Accuracy" ${OUTPUT_DIR}/hdp_fair.txt 2>/dev/null || echo "   (See ${OUTPUT_DIR}/hdp_fair.txt)"

echo ""
echo "3. HDP-Diffusion (Oracle):"
grep "Accuracy" ${OUTPUT_DIR}/hdp_oracle.txt 2>/dev/null || echo "   (See ${OUTPUT_DIR}/hdp_oracle.txt)"

echo ""
echo "4. HDP as BD3-LM:"
grep "Accuracy" ${OUTPUT_DIR}/hdp_as_bd3lm.txt 2>/dev/null || echo "   (See ${OUTPUT_DIR}/hdp_as_bd3lm.txt)"

echo "----------------------------------------"
echo ""
echo "📁 All results saved to: ${OUTPUT_DIR}/"
echo ""
echo "📝 Analysis Commands:"
echo "  # View all results"
echo "  ls -lh ${OUTPUT_DIR}/"
echo ""
echo "  # Compare accuracies"
echo "  grep 'Accuracy' ${OUTPUT_DIR}/*.txt"
echo ""
echo "  # Analyze errors"
echo "  grep 'WRONG' ${OUTPUT_DIR}/baseline_bd3lm.txt"
echo "  grep 'WRONG' ${OUTPUT_DIR}/hdp_fair.txt"
echo ""
echo "  # Check hierarchical structure (HDP only)"
echo "  grep 'PLAN\\|EXECUTION' ${OUTPUT_DIR}/hdp_fair.txt | head -20"
echo ""
echo "=========================================="
