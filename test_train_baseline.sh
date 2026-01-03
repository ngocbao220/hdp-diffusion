#!/bin/bash
# Test baseline: Pure BD3-LM (original algorithm)
# No HDP, just standard block diffusion

set -e

echo "=========================================="
echo "🧪 Baseline: Pure BD3-LM (Block Diffusion)"
echo "=========================================="
echo "📊 Data: gsm8k_overfit.json"
echo "🔥 Training 500 steps from scratch"
echo "⚙️  Standard semi-AR sampler, 16 tokens/block"
echo "=========================================="

python main.py \
    mode=train \
    data=gsm8k_baseline \
    algo.sampler=semi_ar \
    trainer.max_steps=500 \
    loader.global_batch_size=1 \
    block_size=16 \
    sampling.num_steps=64 \
    sampling.first_hitting=true \
    checkpointing.save_dir=outputs/test_baseline_bd3lm \
    checkpointing.every_n_train_steps=100

echo ""
echo "✅ Training complete!"
echo "📁 Checkpoints: outputs/test_baseline_bd3lm/checkpoints/"
echo ""
echo "🔍 To test inference:"
echo "   bash test_infer_baseline.sh"
