#!/bin/bash
# Quick test: HDP + Attention Mask + Analytic Sampler
# Train from scratch on gsm8k_overfit.json

set -e

echo "=========================================="
echo "🧪 Test: HDP + Attention + Analytic"
echo "=========================================="
echo "📊 Data: gsm8k_overfit.json"
echo "🔥 Training 500 steps from scratch"
echo "=========================================="

python main.py \
    mode=train \
    model=tiny \
    data=hdp_overfit \
    model.length=512 \
    algo=bd3lm \
    algo.backbone=dit \
    algo.sampler=ddpm \
    loader.global_batch_size=1 \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    optim.lr=1e-4 \
    trainer.max_steps=500 \
    trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=10 \
    trainer.val_check_interval=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    sampling.num_steps=1000 \
    checkpointing.save_dir=outputs/test_hdp_analytic_att \
    checkpointing.every_n_train_steps=100

echo ""
echo "✅ Training complete!"
echo "📁 Checkpoints: outputs/test_hdp_analytic_att/checkpoints/"
echo ""
echo "🔍 To test inference:"
echo "   bash test_infer.sh"
