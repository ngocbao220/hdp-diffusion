#!/bin/bash
echo "🔬 SINGLE BATCH OVERFIT TEST"
echo "================================"

# Create single sample file
cat > /tmp/single_sample.json << 'EOF'
[
  {
    "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? ",
    "plan": "Identify the initial quantity of blue fiber.  Calculate the initial quantity of white fiber using the given ratio. Sum the quantities of both fibers to find the total.",
    "execution": "It takes 2/2=1 bolt of white fiber So the total amount of fabric is 2+1=3 bolts of fabric",
    "answer": "3"
  }
]
EOF

echo "✅ Created single sample file"
echo ""
echo "🚀 Starting overfit test (500 steps on 1 sample)..."
echo "   Expected: Loss should drop to < 0.1"
echo "   If loss stays high:  CODE HAS BUG!"
echo ""

# Run training on single batch
python -u main.py \
    mode=train \
    seed=42 \
    model=small \
    model.length=512 \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.backbone=dit \
    algo.var_min=false \
    block_size=4 \
    data=hdp_diffusion \
    data.train_path=/tmp/single_sample.json \
    +data.valid_path=/tmp/single_sample.json \
    data.hdp.use_hdp_attention=true \
    data.hdp.question_len=128 \
    data.hdp.plan_len=128 \
    data.hdp.exec_len=256 \
    data.hdp.use_special_format=true \
    noise=loglinear \
    loader.batch_size=1 \
    loader.num_workers=0 \
    loader.global_batch_size=1 \
    optim.lr=1e-3 \
    optim.weight_decay=0 \
    training.ema=0 \
    training.resample=true \
    training.from_pretrained=null \
    training.antithetic_sampling=false \
    trainer.max_steps=500 \
    trainer.log_every_n_steps=50 \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=0 \
    trainer.val_check_interval=null \
    +trainer.check_val_every_n_epoch=null \
    trainer.num_sanity_val_steps=0 \
    trainer.accumulate_grad_batches=1 \
    trainer.gradient_clip_val=1.0 \
    checkpointing.save_dir=/tmp/overfit_test \
    wandb=null

echo ""
echo "================================"
echo "🏁 Test complete!"
echo ""
echo "📊 ANALYSIS:"
echo "  Check the loss curve above."
echo ""
echo "  ✅ SUCCESS if:"
echo "     - Loss drops from ~8.0 to < 0.1"
echo "     - Final loss < 0.5"
echo "     → Model CAN learn!  Code is correct."
echo ""
echo "  ❌ FAILURE if:"
echo "     - Loss stays > 5.0 after 500 steps"
echo "     - Loss doesn't decrease at all"
echo "     → BUG in HDP attention or loss masking!"
echo ""
echo "================================"