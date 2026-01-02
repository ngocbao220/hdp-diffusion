#!/bin/bash

# Test inference WITHOUT HDP attention mask
# If this works, the HDP mask is the problem

echo "=========================================="
echo "TEST: Inference WITHOUT HDP Attention Mask"
echo "=========================================="

CHECKPOINT_PATH="/content/hdp-diffusion/outputs/hdp_diffusion/2026.01.02/160202/outputs/hdp_overfit_test/checkpoints/best.ckpt"

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Config: HDP attention DISABLED (use_hdp_attention=false)"
echo "Expected: If output is correct, HDP mask is the bug"
echo "=========================================="

python -u main.py \
    mode=sample_eval \
    model=tiny \
    model.length=512 \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.T=1000 \
    algo.backbone=dit \
    algo.sampler=ddpm \
    block_size=4 \
    data=hdp_overfit \
    data.hdp.use_hdp_attention=false \
    data.hdp.question_len=128 \
    data.hdp.plan_len=128 \
    data.hdp.exec_len=256 \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    eval.disable_ema=false \
    loader.eval_batch_size=1 \
    sampling.num_steps=100 \
    sampling.num_sample_batches=1 \
    wandb=null

EXIT_CODE=$?

echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Test completed!"
    echo ""
    echo "Compare output with HDP-enabled inference:"
    echo "  - If output CORRECT without mask → HDP mask is broken"
    echo "  - If output STILL wrong → Deeper bug (not mask)"
else
    echo "❌ Test failed!"
fi
echo "=========================================="
