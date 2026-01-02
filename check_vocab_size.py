"""Check vocab_size and mask_index in checkpoint vs current model."""

import torch
from transformers import GPT2Tokenizer

# Load checkpoint
ckpt_path = "/content/hdp-diffusion/outputs/hdp_diffusion/2026.01.02/181001/outputs/hdp_overfit_test/checkpoints/best.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')

print("="*80)
print("🔍 CHECKPOINT INSPECTION")
print("="*80)

# Check state dict keys
if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    print(f"State dict keys: {len(state_dict.keys())} keys")
    
    # Look for vocab_size related info
    for key in state_dict.keys():
        if 'embed' in key.lower() or 'output' in key.lower():
            tensor = state_dict[key]
            print(f"  {key}: shape={tensor.shape}")

# Check hyper_parameters
if 'hyper_parameters' in ckpt:
    hp = ckpt['hyper_parameters']
    print(f"\n📋 Hyper-parameters:")
    if 'vocab_size' in hp:
        print(f"  vocab_size: {hp['vocab_size']}")
    if 'mask_index' in hp:
        print(f"  mask_index: {hp['mask_index']}")

# Check current tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(f"\n🔤 Current Tokenizer:")
print(f"  vocab_size: {tokenizer.vocab_size}")
print(f"  len(tokenizer): {len(tokenizer)}")
print(f"  eos_token_id: {tokenizer.eos_token_id}")
print(f"  pad_token_id: {tokenizer.pad_token_id}")
print(f"  mask_token_id: {tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 'N/A'}")

# Expected mask_index
expected_mask_index = len(tokenizer) if not hasattr(tokenizer, 'mask_token') else tokenizer.mask_token_id
print(f"\n✅ Expected mask_index: {expected_mask_index}")

print("="*80)
