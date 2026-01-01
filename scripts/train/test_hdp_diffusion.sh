#!/bin/bash

# Quick test for HDP-Diffusion (no SLURM)
# Tests hierarchical attention and training pipeline

echo "=========================================="
echo "HDP-Diffusion Quick Test"
echo "Hierarchical Dual-Process Diffusion"
echo "=========================================="

# Block structure
QUESTION_LEN=128
PLAN_LEN=128
EXEC_LEN=256
SEQ_LEN=$((QUESTION_LEN + PLAN_LEN + EXEC_LEN))
BLOCK_SIZE=16

echo "Block structure:"
echo "  Question: ${QUESTION_LEN} tokens"
echo "  Plan: ${PLAN_LEN} tokens"
echo "  Execution: ${EXEC_LEN} tokens"
echo "  Total: ${SEQ_LEN} tokens"
echo "=========================================="

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bd3lm310

# Run short test
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
    loader.global_batch_size=8 \
    loader.eval_global_batch_size=8 \
    loader.batch_size=2 \
    loader.eval_batch_size=2 \
    loader.num_workers=4 \
    training.resample=True \
    training.from_pretrained=null \
    trainer.max_steps=50 \
    trainer.val_check_interval=25 \
    trainer.log_every_n_steps=10 \
    trainer.devices=1 \
    wandb=null \
    +hdp.enabled=true \
    +hdp.question_len=${QUESTION_LEN} \
    +hdp.plan_len=${PLAN_LEN} \
    +hdp.exec_len=${EXEC_LEN} \
    +hdp.use_hdp_attention=true \
    checkpointing.save_dir=outputs/hdp_diffusion_test

EXIT_CODE=$?

echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ HDP-Diffusion test passed!"
else
    echo "❌ Test failed with exit code: ${EXIT_CODE}"
fi
echo "=========================================="
