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
    data=hdp_overfit \
    algo.sampler=ddpm \
    data.hdp.enabled=true \
    data.hdp.use_hdp_attention=true \
    data.hdp.use_special_format=true \
    data.hdp.question_len=128 \
    data.hdp.plan_len=128 \
    data.hdp.exec_len=256 \
    model.length=512 \
    trainer.max_steps=500 \
    loader.global_batch_size=1 \
    data.train_path=data/gsm8k/gsm8k_overfit.json \
    data.test_path=data/gsm8k/gsm8k_overfit.json \
    checkpointing.save_dir=outputs/test_hdp_analytic_att \
    checkpointing.every_n_train_steps=100

echo ""
echo "✅ Training complete!"
echo "📁 Checkpoints: outputs/test_hdp_analytic_att/checkpoints/"
echo ""
echo "🔍 To test inference:"
echo "   bash test_infer.sh"
