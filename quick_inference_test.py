"""
Quick inference test script for GSM8K checkpoints.
Loads a checkpoint and generates a few samples to verify it works.

Usage:
    python quick_inference_test.py --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import os
import sys
import torch
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
import diffusion
import utils


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create minimal config with all required fields
    config = OmegaConf.create({
        'model': {
            'length': 512,
            'attn_backend': 'sdpa'
        },
        'algo': {
            'name': 'bd3lm',
            'backbone': 'dit',
            'T': 1000,
            'sampler': 'ddpm',  # Add required sampler field
            'clip_search_widths': [0.5, 0.6, 0.7, 0.8, 0.9]
        },
        'data': {
            'tokenizer_name_or_path': 'gpt2'
        },
        'eval': {
            'checkpoint_path': checkpoint_path,
            'disable_ema': False
        },
        'block_size': 16,
        'sampling': {
            'num_sample_batches': 1
        }
    })
    
    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(config)
    
    # Load model
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False,
        weights_only=False
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"  Device: {device}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    return model, tokenizer, config


def generate_samples(model, tokenizer, num_samples=5, max_length=512):
    """Generate samples from the model."""
    print(f"\nGenerating {num_samples} samples...")
    
    with torch.no_grad():
        # Use model's sampling method
        samples = model.restore_model_and_sample(num_steps=model.config.algo.T)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Quick inference test')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        os.system("find /workspace/hdp-diffusion/outputs -name '*.ckpt' -type f")
        return
    
    print("=" * 60)
    print("Quick Inference Test")
    print("=" * 60)
    
    # Load model
    try:
        model, tokenizer, config = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate samples
    try:
        samples = generate_samples(model, tokenizer, args.num_samples)
        
        print("\n" + "=" * 60)
        print("Generated Samples:")
        print("=" * 60)
        
        if isinstance(samples, list):
            for i, sample in enumerate(samples[:5], 1):
                print(f"\n--- Sample {i} ---")
                print(sample)
                print("-" * 40)
        else:
            print(samples)
        
        print("\n✅ Inference test completed successfully!")
        
    except Exception as e:
        print(f"❌ Failed to generate samples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
