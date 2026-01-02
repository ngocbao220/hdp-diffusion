"""
Test HDP inference with proper configuration.
"""

import argparse
import os
import sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
import diffusion


def load_hdp_model(checkpoint_path, device='cuda'):
    """Load HDP model from checkpoint."""
    print(f"Loading HDP checkpoint: {checkpoint_path}")
    
    # Create config with HDP settings
    config = OmegaConf.create({
        'model': {
            'length': 512,
            'attn_backend': 'sdpa'
        },
        'algo': {
            'name': 'bd3lm',
            'backbone': 'dit',
            'T': 50,  # Fewer steps for testing
            'sampler': 'ddpm',
            'clip_search_widths': [0.5, 0.6, 0.7, 0.8, 0.9],
            'cross_attn': False,
            'parameterization': 'subs'
        },
        'data': {
            'tokenizer_name_or_path': 'gpt2',
            'hdp': {
                'use_hdp_attention': True,
                'question_len': 128,
                'plan_len': 128,
                'exec_len': 256
            }
        },
        'eval': {
            'checkpoint_path': checkpoint_path,
            'disable_ema': False
        },
        'block_size': 16,
        'sampling': {
            'num_sample_batches': 1,
            'nucleus_p': 1.0
        },
        'loader': {
            'eval_batch_size': 2
        },
        'training': {
            'antithetic_sampling': False
        }
    })
    
    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(config)
    
    # Load model
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ HDP Model loaded!")
    print(f"  use_hdp_attention: {model.use_hdp_attention}")
    if hasattr(model, 'hdp_block_sizes'):
        print(f"  hdp_block_sizes: {model.hdp_block_sizes}")
    
    return model, tokenizer, config


def test_hdp_generation(model, tokenizer, question=None):
    """Test HDP generation."""
    print(f"\n{'='*70}")
    print("Testing HDP Generation")
    print(f"{'='*70}")
    
    if question is None:
        question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning."
    
    print(f"Question: {question}")
    
    with torch.no_grad():
        # Clear debug flags
        if hasattr(model, '_forward_debug_printed'):
            delattr(model, '_forward_debug_printed')
        if hasattr(model, '_forward_debug_printed2'):
            delattr(model, '_forward_debug_printed2')
        
        # Use HDP sampling
        output = model.sample_hdp_conditional(
            question_text=question,
            num_steps=50,  # Use fewer steps for testing
            eps=1e-5
        )
    
    print(f"\n{'='*70}")
    print("Generated Output:")
    print(f"{'='*70}")
    print(output)
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--question', type=str, default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Load model
    try:
        model, tokenizer, config = load_hdp_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test generation
    try:
        output = test_hdp_generation(model, tokenizer, args.question)
        print(f"\n✅ HDP inference test completed!")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
