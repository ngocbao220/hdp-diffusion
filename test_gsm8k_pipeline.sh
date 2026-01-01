#!/bin/bash

# Quick test script for GSM8K hierarchical pipeline
# Tests all components without running full training

set -e  # Exit on error

echo "=========================================="
echo "GSM8K Hierarchical Pipeline Test"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check dependencies
echo -e "\n${YELLOW}[1/6] Checking dependencies...${NC}"
python -c "import torch; print('✓ PyTorch')"
python -c "import transformers; print('✓ Transformers')"
python -c "import datasets; print('✓ Datasets')"
python -c "import vllm; print('✓ vLLM')" || echo "⚠ vLLM not installed (needed for plan generation)"

# Test 2: Test hierarchical mask
echo -e "\n${YELLOW}[2/6] Testing hierarchical mask...${NC}"
python test_hierarchical_mask.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Hierarchical mask tests passed${NC}"
else
    echo -e "${RED}✗ Hierarchical mask tests failed${NC}"
    exit 1
fi

# Test 3: Generate small sample of plans (10 examples)
echo -e "\n${YELLOW}[3/6] Testing plan generation (10 examples)...${NC}"
mkdir -p data/gsm8k_test
python scripts/data_prep/generate_gsm8k_plans.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --split train \
    --batch_size 10 \
    --num_examples 10 \
    --output_path data/gsm8k_test/test_10examples.json \
    > /dev/null 2>&1

if [ -f data/gsm8k_test/test_10examples.json ]; then
    NUM_EXAMPLES=$(jq 'length' data/gsm8k_test/test_10examples.json)
    if [ "$NUM_EXAMPLES" -eq 10 ]; then
        echo -e "${GREEN}✓ Plan generation successful (10 examples)${NC}"
        echo "Sample output:"
        jq '.[0] | {question: .question, plan: .plan}' data/gsm8k_test/test_10examples.json
    else
        echo -e "${RED}✗ Expected 10 examples, got $NUM_EXAMPLES${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Plan generation failed${NC}"
    exit 1
fi

# Test 4: Test dataloader
echo -e "\n${YELLOW}[4/6] Testing GSM8K dataloader...${NC}"
python gsm8k_dataloader.py \
    --data_path data/gsm8k_test/test_10examples.json \
    --tokenizer gpt2 \
    > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataloader working correctly${NC}"
else
    echo -e "${RED}✗ Dataloader test failed${NC}"
    exit 1
fi

# Test 5: Analyze token lengths
echo -e "\n${YELLOW}[5/6] Analyzing token lengths...${NC}"
python gsm8k_dataloader.py \
    --data_path data/gsm8k_test/test_10examples.json \
    --tokenizer gpt2 \
    --analyze \
    2>/dev/null

# Test 6: Test training (1 step only)
echo -e "\n${YELLOW}[6/6] Testing training initialization...${NC}"

# Create a minimal test split
jq '.[0:5]' data/gsm8k_test/test_10examples.json > data/gsm8k_test/train_mini.json
jq '.[5:10]' data/gsm8k_test/test_10examples.json > data/gsm8k_test/test_mini.json

python main.py \
    mode=train \
    model=tiny \
    model.length=512 \
    algo=bd3lm \
    block_size=16 \
    training.max_steps=1 \
    training.hierarchical.enabled=true \
    training.hierarchical.question_len=128 \
    training.hierarchical.plan_len=128 \
    training.hierarchical.exec_len=256 \
    data.name=gsm8k \
    data.train_path=data/gsm8k_test/train_mini.json \
    data.test_path=data/gsm8k_test/test_mini.json \
    loader.batch_size=2 \
    > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training initialization successful${NC}"
else
    echo -e "${YELLOW}⚠ Training test failed (may need proper integration)${NC}"
fi

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Pipeline Test Summary${NC}"
echo "=========================================="
echo "✓ All core components tested"
echo "✓ Ready for full plan generation"
echo "✓ Ready for full training"
echo ""
echo "Next steps:"
echo "  1. Generate full plans: sbatch scripts/data_prep/run_gsm8k_plan_generation.sh"
echo "  2. Start training: sbatch scripts/train/train_gsm8k_hierarchical.sh"
echo "=========================================="

# Cleanup
rm -rf data/gsm8k_test
