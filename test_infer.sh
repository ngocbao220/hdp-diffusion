#!/bin/bash
# Quick inference test: HDP + Attention + Analytic
# Test on trained checkpoint

set -e

CKPT_PATH=${1:-"outputs/test_hdp_analytic_att/checkpoints/last.ckpt"}

if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    echo "💡 Run training first: bash test_train.sh"
    exit 1
fi

echo "=========================================="
echo "🔍 Inference: HDP + Attention + Analytic"
echo "=========================================="
echo "📦 Checkpoint: $CKPT_PATH"
echo "=========================================="

python main.py \
    mode=sample_eval \
    data=hdp_overfit \
    algo.sampler=ddpm \
    data.hdp.enabled=true \
    data.hdp.use_hdp_attention=true \
    eval.checkpoint_path=$CKPT_PATH \
    eval.disable_ema=true \
    sampling.num_sample_batches=1 \
    +sampling.num_steps=1000 \
    data.test_path=data/gsm8k/gsm8k_overfit.json

echo ""
echo "✅ Inference complete!"
