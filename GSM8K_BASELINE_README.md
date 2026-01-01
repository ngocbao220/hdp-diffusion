# GSM8K Baseline Training with BD3-LM

This directory contains scripts for training block diffusion models on the GSM8K dataset using the standard Q&A format (no hierarchical structure).

## Dataset

GSM8K (Grade School Math 8K) is a dataset of 8.5K high-quality linguistically diverse grade school math word problems.

- **Train**: 7,473 problems
- **Test**: 1,319 problems
- **Format**: `Question: [problem]\nAnswer: [solution with reasoning]`

The dataset is automatically downloaded from Hugging Face datasets.

## Quick Start

### 1. Test the Setup (Quick)

Run a quick test to verify everything works:

```bash
bash scripts/train/test_gsm8k_bd3lm.sh
```

This will train for 100 steps on 1 GPU to verify the pipeline works.

### 2. Full Training

For full training on SLURM cluster:

```bash
sbatch scripts/train/train_gsm8k_bd3lm.sh
```

Or locally without SLURM:

```bash
# Edit the script to remove SBATCH directives
bash scripts/train/train_gsm8k_bd3lm.sh
```

## Configuration

Key hyperparameters (can be adjusted in the training script):

- **Block Size**: 4, 8, or 16 (default: 16)
  - Smaller blocks → more autoregressive
  - Larger blocks → more diffusion-like
- **Sequence Length**: 512 (default for GSM8K)
- **Batch Size**: 128 global batch size (32 per GPU × 4 GPUs)
- **Training Steps**: 50,000 steps
- **Learning Rate**: 3e-4 (default from config)

## File Structure

```
configs/data/gsm8k_baseline.yaml    # Data configuration
scripts/train/
  ├── train_gsm8k_bd3lm.sh          # Full training script (SLURM)
  └── test_gsm8k_bd3lm.sh           # Quick test script
```

## Training Options

### Different Block Sizes

Try different block sizes to find the best trade-off:

```bash
# Block size 4 (more AR-like)
BLOCK_SIZE=4 bash scripts/train/train_gsm8k_bd3lm.sh

# Block size 8 (balanced)
BLOCK_SIZE=8 bash scripts/train/train_gsm8k_bd3lm.sh

# Block size 16 (more diffusion-like)
BLOCK_SIZE=16 bash scripts/train/train_gsm8k_bd3lm.sh
```

### Start from Pretrained Checkpoint

To start from a pretrained checkpoint (e.g., from OpenWebText):

```bash
# Edit train_gsm8k_bd3lm.sh and set:
PRETRAIN_CKPT=kuleshov-group/bd3lm-owt-block_size16
# or local path:
PRETRAIN_CKPT=/path/to/checkpoint.ckpt
```

## Evaluation

After training, evaluate perplexity on test set:

```bash
python main.py \
    mode=ppl_eval \
    model=small \
    algo=bd3lm \
    data=gsm8k_baseline \
    model.length=512 \
    block_size=16 \
    eval.checkpoint_path=outputs/gsm8k_bd3lm_bs16/checkpoints/last.ckpt
```

## Expected Results

Training metrics to monitor:
- **Training loss**: Should decrease steadily
- **Validation perplexity**: Should decrease (lower is better)
- **Convergence**: Typically around 30-40K steps

## Notes

- This is the **baseline** approach using standard BD3-LM
- For hierarchical/structured approaches, see the hierarchical configs
- GSM8K problems vary in length; the model learns to generate full solutions
- The wrapped format allows the model to learn across multiple problems in one sequence

## Troubleshooting

If you encounter OOM errors:
- Reduce `loader.batch_size`
- Reduce `model.length`
- Use `model.attn_backend=sdpa` instead of `flash_attn`

If training is slow:
- Increase number of GPUs: `trainer.devices=8`
- Increase batch size: `loader.global_batch_size=256`
- Use gradient accumulation automatically handled by PyTorch Lightning
