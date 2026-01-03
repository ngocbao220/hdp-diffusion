#!/bin/bash
# Run HDP Experiments
# Usage: ./run_experiment.sh [experiment_name] [mode] [checkpoint_step]
#   experiment_name: hdp_analytic_att | hdp_analytic_noatt | hdp_semiar_att | hdp_semiar_noatt
#   mode: train | infer
#   checkpoint_step: (optional, for inference) e.g., 800

set -e

EXPERIMENT=$1
MODE=$2
CKPT_STEP=${3:-"last"}

if [ -z "$EXPERIMENT" ] || [ -z "$MODE" ]; then
    echo "Usage: $0 [experiment_name] [mode] [checkpoint_step]"
    echo ""
    echo "Experiments:"
    echo "  hdp_analytic_att    - Analytic sampler WITH HDP attention"
    echo "  hdp_analytic_noatt  - Analytic sampler WITHOUT HDP attention"
    echo "  hdp_semiar_att      - Semi-AR sampler WITH HDP attention"
    echo "  hdp_semiar_noatt    - Semi-AR sampler WITHOUT HDP attention"
    echo ""
    echo "Modes:"
    echo "  train  - Training mode"
    echo "  infer  - Inference mode"
    echo ""
    echo "Example:"
    echo "  $0 hdp_analytic_att train"
    echo "  $0 hdp_analytic_att infer 800"
    exit 1
fi

# Validate experiment name
case $EXPERIMENT in
    hdp_analytic_att|hdp_analytic_noatt|hdp_semiar_att|hdp_semiar_noatt)
        ;;
    *)
        echo "❌ Invalid experiment: $EXPERIMENT"
        exit 1
        ;;
esac

echo "=========================================="
echo "🚀 Running Experiment: $EXPERIMENT"
echo "📋 Mode: $MODE"
echo "=========================================="

if [ "$MODE" = "train" ]; then
    echo "🔥 Training..."
    python main.py \
        +experiment=$EXPERIMENT \
        mode=train \
        trainer.max_steps=500 \
        loader.global_batch_size=1
        
elif [ "$MODE" = "infer" ]; then
    echo "🔍 Inference..."
    
    # Determine checkpoint path
    if [ "$CKPT_STEP" = "last" ]; then
        CKPT_PATH="outputs/${EXPERIMENT}/checkpoints/last.ckpt"
    else
        CKPT_PATH="outputs/${EXPERIMENT}/checkpoints/step_${CKPT_STEP}.ckpt"
    fi
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "❌ Checkpoint not found: $CKPT_PATH"
        exit 1
    fi
    
    echo "📦 Using checkpoint: $CKPT_PATH"
    
    python main.py \
        +experiment=$EXPERIMENT \
        mode=sample_eval \
        eval.checkpoint_path=$CKPT_PATH \
        eval.disable_ema=true \
        sampling.num_sample_batches=1
else
    echo "❌ Invalid mode: $MODE (use 'train' or 'infer')"
    exit 1
fi

echo "✅ Done!"
