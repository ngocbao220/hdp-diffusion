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
    +experiment=baseline_bd3lm \
    mode=sample_eval \
    eval.checkpoint_path=$CKPT_PATH \
    eval.disable_ema=true \
    sampling.num_sample_batches=1 \
    data.test_path=data/gsm8k/gsm8k_overfit.json

echo ""
echo "✅ Inference complete!"
