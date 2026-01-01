#!/bin/bash

# Quick test cho BASELINE GSM8K training
# Kiểm tra xem baseline có train đúng không

echo "=========================================="
echo "🧪 BASELINE GSM8K Training Test"
echo "Format: Simple 'Question: ... Answer: ...'"
echo "=========================================="

BLOCK_SIZE=16
SEQ_LENGTH=512

echo ""
echo "📊 Configuration:"
echo "  Format: Baseline (no hierarchy)"
echo "  Sequence length: ${SEQ_LENGTH} tokens"
echo "  Block size: ${BLOCK_SIZE}"
echo "  Training steps: 50 (quick test)"
echo "  Batch size: 4"
echo "=========================================="

# Check if conda env exists
if conda env list | grep -q "^hdp "; then
    echo "✅ Found conda env: hdp"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate hdp
else
    echo "⚠️  Conda env 'hdp' not found, using current environment"
fi

echo ""
echo "🔥 Starting baseline training..."
echo ""

# Run baseline training
python -u main.py \
    mode=train \
    model=small \
    model.length=${SEQ_LENGTH} \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    data=gsm8k_baseline \
    noise=loglinear \
    loader.global_batch_size=4 \
    loader.eval_global_batch_size=4 \
    loader.batch_size=4 \
    loader.eval_batch_size=4 \
    loader.num_workers=2 \
    training.resample=True \
    training.from_pretrained=null \
    trainer.max_steps=50 \
    trainer.val_check_interval=25 \
    trainer.log_every_n_steps=10 \
    trainer.devices=1 \
    trainer.accumulate_grad_batches=1 \
    wandb=null \
    checkpointing.save_dir=outputs/baseline_test

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Baseline training test completed!"
    echo ""
    echo "📂 Checkpoint: outputs/baseline_test"
    echo ""
    echo "🔍 Compare with HDP training:"
    echo "   Baseline: Simple concatenation"
    echo "   HDP: [PLAN] [EXECUTION] [ANSWER] structure"
else
    echo "❌ Training failed with exit code: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
