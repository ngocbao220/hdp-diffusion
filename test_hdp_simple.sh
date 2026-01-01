#!/bin/bash
# Simple HDP training test - direct execution
cd /workspace/hdp-diffusion

python -u main.py \
  mode=train \
  data=hdp_diffusion \
  model=small \
  model.length=512 \
  model.attn_backend=sdpa \
  algo=bd3lm \
  algo.backbone=dit \
  block_size=16 \
  noise=loglinear \
  loader.global_batch_size=4 \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  loader.num_workers=2 \
  trainer.max_steps=100 \
  trainer.val_check_interval=50 \
  trainer.log_every_n_steps=10 \
  trainer.accumulate_grad_batches=1 \
  trainer.devices=2 \
  wandb=null
