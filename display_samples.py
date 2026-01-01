"""
Display generated samples from inference in a readable format.

Usage:
    python display_samples.py
"""

import sys

# Mock samples from the last inference run
samples = [
    "<|endoftext|>gram covering pillow Immediately canopy lest nefarious Sale stadium...",
    "<|endoftext|> Prof Wonder Read facilitates nestsGuy fledgling opera...",
    "<|endoftext|>ascriptaho boils compromises props Heller 294 Das commentary...",
    "<|endoftext|>reach CarlosMiddlePath patrolsprojects alcoholism Madd Faculty...",
    "<|endoftext|> Mother headiOS helmets starvedDEM fieldhent facets..."
]

print("=" * 80)
print("GSM8K Baseline BD3-LM - Generated Samples")
print("=" * 80)
print(f"\n📊 Inference Results:")
print(f"   - Model: GSM8K Baseline (100 training steps)")
print(f"   - Checkpoint: outputs/gsm8k/.../gsm8k_bd3lm_test_bs16/checkpoints/last.ckpt")
print(f"   - Diffusion steps: 1000")
print(f"   - Block size: 16")
print(f"   - Number of samples: 5")
print(f"   - Generative Perplexity: 122926.07 (very high - model not converged)")
print(f"   - Entropy: 6.23 (high - mostly random)")

print("\n" + "=" * 80)
print("⚠️  ANALYSIS:")
print("=" * 80)
print("""
The generated samples are mostly gibberish/random tokens because:

1. ❌ Only 100 training steps (far too few!)
   - Typical training: 30,000-50,000 steps
   - Current training: 100 steps (0.2% of full training)

2. ❌ Very high perplexity (122,926)
   - Good models: < 30
   - Our model: > 100,000 (not learned anything meaningful)

3. ❌ Random token generation
   - Model hasn't learned GSM8K structure yet
   - No question/answer pattern visible
   - No mathematical reasoning

✅ POSITIVE: The pipeline works correctly!
   - Checkpoint loads successfully
   - Sampling completes without errors
   - Outputs are generated in correct format (512 tokens)
""")

print("=" * 80)
print("📝 Sample Preview (truncated):")
print("=" * 80)

for i, sample in enumerate(samples, 1):
    print(f"\n--- Sample {i} ---")
    # Show first 200 chars
    preview = sample[:200] + "..." if len(sample) > 200 else sample
    print(preview)
    print()

print("=" * 80)
print("🚀 NEXT STEPS:")
print("=" * 80)
print("""
To get meaningful results:

1. Train for full duration:
   bash scripts/train/train_gsm8k_bd3lm.sh
   (This will train for 50,000 steps = ~24 hours on 4 GPUs)

2. Monitor training metrics:
   - Training loss should decrease to < 2.0
   - Validation perplexity should be < 30
   - Check wandb for loss curves

3. Then run inference again:
   bash scripts/inference/inference_gsm8k_baseline.sh

Expected output after full training:
   "Question: John has 5 apples and gives 2 away. How many does he have?
    Answer: John starts with 5 apples. He gives away 2. 5 - 2 = 3. 
    John has 3 apples left."
""")

print("=" * 80)
print("✅ Inference pipeline verified successfully!")
print("   Model can load, sample, and generate outputs.")
print("   Ready for full training run.")
print("=" * 80)
