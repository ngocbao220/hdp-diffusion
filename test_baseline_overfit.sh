#!/bin/bash
# Baseline Overfit Test: 1 sample only
# Test if baseline model can overfit a single GSM8K example

set -e

echo "=========================================="
echo "🧪 BASELINE OVERFIT TEST (1 Sample)"
echo "=========================================="
echo "📊 Data: gsm8k_baseline_overfit.json (1 sample)"
echo "🎯 Goal: Loss → 0, Accuracy → 100%"
echo "🔥 Training 1000 steps"
echo "⚙️  Semi-AR sampler, batch_size=1"
echo "=========================================="

python main.py \
    mode=train \
    model=tiny \
    data=gsm8k_baseline_overfit \
    model.length=512 \
    algo=bd3lm \
    algo.backbone=dit \
    algo.sampler=semi_ar \
    block_size=16 \
    loader.global_batch_size=1 \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    optim.lr=1e-4 \
    trainer.max_steps=1000 \
    trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=10 \
    trainer.val_check_interval=100 \
    project_name=baseline_overfit_test \
    experiment_name=gsm8k_1sample \
    wandb.mode=disabled

echo ""
echo "=========================================="
echo "✅ Training complete!"
echo "Check logs above for:"
echo "  - Loss should drop close to 0"
echo "  - Accuracy should reach ~100%"
echo "  - Debug logs every 50 steps show predictions"
echo "=========================================="
