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

# ============================================
# EXPERIMENT CONFIGURATION (Chỉnh ở đây!)
# ============================================

# Experiment Mode
EXP_NAME="bd_baseline_overfit"   # Tên experiment
DATA_CONFIG="gsm8k_baseline"     # Data config: gsm8k_baseline
MODE="train"                     # train, sample, evaluate

# Model Architecture
MODEL_SIZE="tiny"                # tiny, small, base, large
ATTN_BACKEND="sdpa"              # sdpa, flash_attn, flex

# Diffusion Algorithm
ALGO="bd3lm"                     # bd3lm, ar, ddpm
SAMPLER="semi_ar"                # ddpm (analytic), semi_ar (block-wise)
BACKBONE="dit"                   # dit, transformer
NOISE_SCHEDULE="loglinear"       # loglinear, linear, cosine

# Baseline BD3-LM settings (no hierarchical structure)
SEQ_LEN=512  # Full sequence length

# Block diffusion settings
BLOCK_SIZE=4  # Can try 4, 8, 16

# Overfitting test settings
BATCH_SIZE=1         # Single sample batch
EVAL_BATCH_SIZE=1    
GLOBAL_BATCH_SIZE=1  
GRAD_ACCUM=1         

# Training Hyperparameters
MAX_STEPS=500                    # Total training steps
WARMUP_STEPS=10                  # Warmup steps
LOG_INTERVAL=10                  # Log every N steps
# NOTE: Validation disabled for overfit test (causes CUDA crash)
# Use eval scripts after training: bash scripts/eval_baseline.sh
LR=1e-4                          # Learning rate (reduced to prevent NaN)
EMA=0.9999                       # EMA decay rate
RESAMPLE=True                    # Resample during training
GRAD_CLIP=1.0                    # Gradient clipping value

# Hardware Settings
DEVICES=1                        # Number of GPUs
NUM_NODES=1                      # Number of nodes
PRECISION="bf16-mixed"           # bf16-mixed, fp16, fp32
STRATEGY="auto"                  # auto for single GPU (ddp causes issues)

# Optional: Start from pretrained checkpoint
PRETRAIN_CKPT=null

# Output directory
OUTPUT_DIR="outputs/bd_baseline_overfit_test"
mkdir -p ${OUTPUT_DIR}

echo "BD3-LM BASELINE OVERFITTING TEST"
echo "  Dataset: ${DATA_CONFIG} (1 sample)"
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

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "🐛 CUDA Debugging enabled:"
echo "   CUDA_LAUNCH_BLOCKING=1"
echo "   TORCH_USE_CUDA_DSA=1"
echo ""

echo "⚙️  Configuration Check:"
echo "   Data: ${DATA_CONFIG}"
echo "   Model: ${MODEL_SIZE}"
echo "   Seq Length: ${SEQ_LEN}"
echo "   Block Size: ${BLOCK_SIZE}"
echo "   Algo: ${ALGO}"
echo ""

# Run BD3-LM BASELINE training with configurable parameters
echo "🚀 Starting training..."
python -u main.py \
    mode=${MODE} \
    model=${MODEL_SIZE} \
    data=${DATA_CONFIG} \
    model.length=${SEQ_LEN} \
    model.attn_backend=${ATTN_BACKEND} \
    algo=${ALGO} \
    algo.sampler=${SAMPLER} \
    algo.backbone=${BACKBONE} \
    block_size=${BLOCK_SIZE} \
    noise=${NOISE_SCHEDULE} \
    loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
    loader.eval_global_batch_size=256 \
    loader.batch_size=${BATCH_SIZE} \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    loader.num_workers=16 \
    optim.lr=${LR} \
    training.ema=${EMA} \
    training.resample=${RESAMPLE} \
    training.from_pretrained=$PRETRAIN_CKPT \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${GRAD_ACCUM} \
    trainer.val_check_interval=null \
    +trainer.check_val_every_n_epoch=999999 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    trainer.log_every_n_steps=${LOG_INTERVAL} \
    trainer.devices=${DEVICES} \
    trainer.num_nodes=${NUM_NODES} \
    +trainer.strategy=${STRATEGY} \
    trainer.precision=${PRECISION} \
    trainer.gradient_clip_val=${GRAD_CLIP} \
    +sampling.disable_val_sampling=true \
    wandb.name=${EXP_NAME}-${SAMPLER}-bs${BLOCK_SIZE}-$(date +%Y%m%d-%H%M%S) \
    wandb.project=bd3lm-baseline-experiments \
    wandb.tags=[baseline,gsm8k,${SAMPLER},bs${BLOCK_SIZE}] \
    +experiment_name=${EXP_NAME} \
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


# ============================================
# Test trained model quality (not just logic)
# ============================================
echo ""
echo "To test TRAINED MODEL (not just logic), use main.py:"
echo ""
echo "python main.py mode=sample_eval \\"
echo "  model=tiny \\"
echo "  model.length=512 \\"
echo "  algo=bd3lm \\"
echo "  data=hdp_overfit \\"
echo "  data.hdp.use_hdp_attention=true \\"
echo "  eval.checkpoint_path=/content/hdp-diffusion/outputs/hdp_diffusion/2026.01.03/161254/outputs/hdp_overfit_test/checkpoints/best.ckpt \\"
echo "  +sampling.sampling_mode=hdp \\"
echo "  sampling.num_steps=100 \\"
echo "  loader.eval_batch_size=1"
echo ""