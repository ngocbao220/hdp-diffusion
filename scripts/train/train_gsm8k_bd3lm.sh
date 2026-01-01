#!/bin/bash
#SBATCH -J train_gsm8k_bd3lm          # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64G                     # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# GSM8K Baseline BD3-LM Training
# Simple Q&A format (no hierarchical structure)

BLOCK_SIZE=16  # Standard block size for BD3-LM (can try 4, 8, 16)
SEQ_LENGTH=512 # Context length for GSM8K problems

# Optional: Start from a pretrained checkpoint (set to null to train from scratch)
PRETRAIN_CKPT=null  # or path to pretrained checkpoint

python -u main.py \
    loader.global_batch_size=128 \
    loader.eval_global_batch_size=128 \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=gsm8k_baseline \
    model.length=${SEQ_LENGTH} \
    block_size=${BLOCK_SIZE} \
    wandb.name=bd3lm-gsm8k-bs${BLOCK_SIZE} \
    wandb.project=gsm8k-bd3lm \
    wandb.tags=[gsm8k,baseline,bd3lm,bs${BLOCK_SIZE}] \
    mode=train \
    model.attn_backend=flash_attn \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT \
    trainer.max_steps=50000 \
    trainer.val_check_interval=1000 \
    checkpointing.save_dir=outputs/gsm8k_bd3lm_bs${BLOCK_SIZE}
