#!/bin/bash
# Quick test: BD3-LM Baseline (no HDP)
# Train from scratch on gsm8k_baseline.json

set -e

echo "=========================================="
echo "🧪 Test: BD3-LM Baseline (No HDP)"
echo "=========================================="
echo "📊 Data: gsm8k_baseline.json"
echo "🔥 Training 500 steps from scratch"
echo "⚙️  Standard semi-AR sampler, 16 tokens/block"
echo "=========================================="

python main.py \
    mode=train \
    model=tiny \
    data=gsm8k_baseline \
    model.length=512 \
    algo=bd3lm \
    algo.backbone=dit \
    algo.sampler=semi_ar \
    block_size=16 \
    loader.global_batch_size=1 \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    optim.lr=1e-4 \
    trainer.max_steps=500 \
    trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=10 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
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
