#!/bin/bash
# Inference test: Baseline BD3-LM
# Standard block diffusion without HDP

set -e

CKPT_PATH=${1:-"outputs/test_baseline_bd3lm/checkpoints/last.ckpt"}

if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    echo "💡 Run training first: bash test_train_baseline.sh"
    exit 1
fi

echo "=========================================="
echo "🔍 Inference: Baseline BD3-LM"
echo "=========================================="
echo "📦 Checkpoint: $CKPT_PATH"
echo "=========================================="

python main.py \
    mode=sample_eval \
    model=tiny \
    data=gsm8k_baseline \
    algo=bd3lm \
    algo.sampler=semi_ar \
    block_size=16 \
    eval.checkpoint_path=$CKPT_PATH \
    eval.disable_ema=true \
    sampling.num_sample_batches=1

echo ""
echo "✅ Inference complete!"
