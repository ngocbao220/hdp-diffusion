#!/bin/bash

# Script to generate hierarchical GSM8K dataset on H200
# This runs vLLM with Llama-3-8B-Instruct to extract plans

#SBATCH --job-name=gsm8k_plan_gen
#SBATCH --output=logs/gsm8k_plan_gen_%j.out
#SBATCH --error=logs/gsm8k_plan_gen_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=h200  # Adjust to your H200 partition name

echo "=========================================="
echo "GSM8K Plan Generation with vLLM"
echo "=========================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="

# Model settings
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
SPLIT="train"  # or "test"
BATCH_SIZE=256  # Large batch for H200
MAX_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9
TENSOR_PARALLEL=1  # Use 1 GPU, increase if using multiple

# Output
OUTPUT_DIR="data/gsm8k"
mkdir -p ${OUTPUT_DIR}
OUTPUT_FILE="${OUTPUT_DIR}/gsm8k_hierarchical_${SPLIT}.json"

# For testing with small subset, uncomment:
# NUM_EXAMPLES="--num_examples 100"
NUM_EXAMPLES=""

echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Split: ${SPLIT}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Output: ${OUTPUT_FILE}"
echo "=========================================="

# Install vLLM if not already installed
pip show vllm > /dev/null 2>&1 || {
    echo "Installing vLLM..."
    pip install vllm
    # Fix numpy version conflict with pandas
    echo "Fixing numpy version compatibility..."
    pip install 'numpy<2,>=1.22.4' --force-reinstall --no-deps
}

# Run generation
python scripts/data_prep/generate_gsm8k_plans.py \
    --model ${MODEL} \
    --split ${SPLIT} \
    --batch_size ${BATCH_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --tensor_parallel_size ${TENSOR_PARALLEL} \
    --output_path ${OUTPUT_FILE} \
    ${NUM_EXAMPLES}

EXIT_CODE=$?

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Plan generation completed successfully!"
    echo "Output file: ${OUTPUT_FILE}"
    
    # Show statistics
    if [ -f ${OUTPUT_FILE} ]; then
        echo ""
        echo "File size: $(du -h ${OUTPUT_FILE} | cut -f1)"
        echo "Number of examples: $(jq length ${OUTPUT_FILE})"
        echo ""
        echo "Sample (first example):"
        jq '.[0]' ${OUTPUT_FILE}
    fi
else
    echo "❌ Plan generation failed with exit code ${EXIT_CODE}"
    echo "Check logs for details"
fi

echo "=========================================="
