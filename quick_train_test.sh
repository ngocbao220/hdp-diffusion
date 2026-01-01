#!/bin/bash

# Quick training test for HDP-Diffusion with new [PLAN] [EXECUTION] [ANSWER] format
# Runs 100 steps để verify training works

echo "=========================================="
echo "🚀 HDP-Diffusion Quick Training Test"
echo "Format: [PLAN] ... [EXECUTION] ... [ANSWER]"
echo "=========================================="

# Block structure
QUESTION_LEN=128
PLAN_LEN=128
EXEC_LEN=256
SEQ_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))
BLOCK_SIZE=16

echo ""
echo "📊 Configuration:"
echo "  Block structure:"
echo "    • Question: ${QUESTION_LEN} tokens"
echo "    • Plan: ${PLAN_LEN} tokens (contains [PLAN])"
echo "    • Execution: ${EXEC_LEN} tokens (contains [EXECUTION] ... [ANSWER])"
echo "  Total sequence: ${SEQ_LEN} tokens"
echo "  Block size: ${BLOCK_SIZE}"
echo "  Training steps: 100"
echo "  Batch size: 2"
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
echo "🔥 Starting training..."
echo ""

# Run training with new format
python -u main.py \
    mode=train \
    model=small \
    model.length=${SEQ_LEN} \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    data=hdp_diffusion \
    noise=loglinear \
    loader.global_batch_size=4 \
    loader.eval_global_batch_size=4 \
    loader.batch_size=4 \
    loader.eval_batch_size=4 \
    loader.num_workers=2 \
    training.resample=True \
    training.from_pretrained=null \
    trainer.max_steps=100 \
    trainer.val_check_interval=50 \
    trainer.log_every_n_steps=10 \
    trainer.devices=1 \
    trainer.accumulate_grad_batches=1 \
    wandb=null \
    +hdp.enabled=true \
    +hdp.question_len=${QUESTION_LEN} \
    +hdp.plan_len=${PLAN_LEN} \
    +hdp.exec_len=${EXEC_LEN} \
    +hdp.use_hdp_attention=true \
    +hdp.use_special_format=true \
    checkpointing.save_dir=outputs/hdp_test_new_format

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training test completed successfully!"
    echo ""
    echo "📂 Checkpoint saved to: outputs/hdp_test_new_format"
    echo ""
    echo "🔍 To verify model generates correct format, run:"
    echo "   python test_inference.py"
else
    echo "❌ Training failed with exit code: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
