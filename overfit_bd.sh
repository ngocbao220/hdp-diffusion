#!/bin/bash
#SBATCH -J overfit_bd_baseline        # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=128G                    # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="h200"           # H200 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # 1x GPU
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

echo "=========================================="
echo "BD3-LM BASELINE Overfitting Test"
echo "Simple Question + Answer Format"
echo "=========================================="

# Baseline BD3-LM settings (no hierarchical structure)
SEQ_LEN=512  # Full sequence length

# Block diffusion settings
BLOCK_SIZE=4  # Can try 4, 8, 16

# Overfitting test settings
BATCH_SIZE=1         # Single sample batch
EVAL_BATCH_SIZE=1    
GLOBAL_BATCH_SIZE=1  
GRAD_ACCUM=1         

# Training hyperparameters (OVERFITTING TEST)
MAX_STEPS=500        # 500 steps should memorize 1 sample
WARMUP_STEPS=10      # Short warmup
VAL_EVERY_N_EPOCH=1  # Validate every epoch
LOG_INTERVAL=10      # Log frequently

# Learning rate
LR=3e-4  # Higher LR to memorize faster

# Optional: Start from pretrained checkpoint
PRETRAIN_CKPT=null

# Output directory
OUTPUT_DIR="outputs/bd_baseline_overfit_test"
mkdir -p ${OUTPUT_DIR}

echo "BD3-LM BASELINE OVERFITTING TEST:"
echo "  GPU: 1x H200"
echo "  Dataset: 1 SAMPLE (gsm8k_overfit.json)"
echo "  Format: BASELINE (Question: ... Answer: ...)"
echo "  Goal: Verify model can memorize 1 example"
echo "  Sequence Length: ${SEQ_LEN} tokens"
echo "  Diffusion Block Size: ${BLOCK_SIZE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LR}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Expected: Loss should drop to near 0"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hdp

# Run BD3-LM BASELINE training (no HDP attention)
python -u main.py \
    mode=train \
    model=tiny \
    data=gsm8k_baseline \
    model.length=${SEQ_LEN} \
    model.attn_backend=sdpa \
    algo=bd3lm \
    algo.backbone=dit \
    block_size=${BLOCK_SIZE} \
    noise=loglinear \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=256 \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    loader.num_workers=16 \
    optim.lr=${LR} \
    training.ema=0.9999 \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${GRAD_ACCUM} \
    trainer.val_check_interval=null \
    +trainer.check_val_every_n_epoch=${VAL_EVERY_N_EPOCH} \
    trainer.log_every_n_steps=${LOG_INTERVAL} \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    +trainer.strategy=ddp \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=1.0 \
    wandb.name=bd-baseline-overfit-$(date +%Y%m%d-%H%M%S) \
    wandb.project=bd3lm-baseline-test \
    wandb.tags=[baseline,gsm8k,overfit,h200] \
    +experiment_name=bd_baseline_overfit_test \
    checkpointing.save_dir=${OUTPUT_DIR}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Baseline overfitting test completed!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Next Steps:"
    echo "1. Check loss curve - should drop to near 0"
    echo "2. Run inference to verify memorization:"
    echo ""
    echo "bash scripts/inference/baseline.sh"
    echo ""
    echo "Or manually:"
    echo "python main.py mode=sample_eval \\"
    echo "    eval.checkpoint_path=${OUTPUT_DIR}/checkpoints/last.ckpt \\"
    echo "    model=tiny \\"
    echo "    model.length=512 \\"
    echo "    algo=bd3lm \\"
    echo "    algo.sampler=ddpm \\"
    echo "    data=gsm8k_baseline \\"
    echo "    sampling.num_steps=100 \\"
    echo "    loader.eval_batch_size=1"
    echo ""
    echo "Expected: Model should generate exact answer:"
    echo "  'Question: Janet's ducks lay 16 eggs per day...'"
    echo "  'Answer: Janet sells 16 - 3 - 4 = 9...'"
else
    echo "❌ Training failed!"
    echo "Check logs: watch_folder/"
fi

echo "=========================================="
