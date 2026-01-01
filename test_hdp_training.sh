#!/bin/bash
# Quick training test with HDP attention enabled - 100 steps
cd /workspace/hdp-diffusion

conda run -n hdp python main.py \
  mode=train \
  data=hdp_diffusion \
  model=small \
  model.length=512 \
  model.attn_backend=sdpa \
  algo=bd3lm \
  algo.backbone=dit \
  noise=loglinear \
  loader.global_batch_size=4 \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  loader.num_workers=4 \
  trainer.max_steps=100 \
  trainer.val_check_interval=50 \
  trainer.log_every_n_steps=10 \
  trainer.accumulate_grad_batches=1 \
  trainer.devices=2 \
  wandb=null \
  +get_eval_metric=none \
  +work_dir=./test_hdp_attention_output
