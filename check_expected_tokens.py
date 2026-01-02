"""
Test if HDP model can learn ANYTHING by checking predicted tokens.
"""

import torch
from transformers import GPT2Tokenizer
from hdp_dataset import HDPDataset
from torch.utils.data import DataLoader
from hdp_dataset import collate_hdp_batch

# Load tokenizer and dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

dataset = HDPDataset(
    data_path='data/gsm8k/gsm8k_overfit.json',
    tokenizer=tokenizer,
    block_sizes=(128, 128, 256),
    use_special_format=True
)

print("="*80)
print("🔍 EXPECTED TOKENS FROM TRAINING DATA")
print("="*80)

# Get first sample
sample = dataset[0]
input_ids = sample['input_ids']

# Decode blocks
q_ids = input_ids[:128]
p_ids = input_ids[128:256]
e_ids = input_ids[256:512]

# Print expected tokens for each block
print("\n📝 Question Block (0-128):")
print(f"First 10 token IDs: {q_ids[:10].tolist()}")
print(f"Decoded: {tokenizer.decode(q_ids[:10])}")

print("\n📝 Plan Block (128-256):")
print(f"First 10 token IDs: {p_ids[:10].tolist()}")
print(f"Decoded: {tokenizer.decode(p_ids[:10])}")
print(f"Last 5 (should be PAD={tokenizer.pad_token_id}): {p_ids[-5:].tolist()}")

print("\n📝 Execution Block (256-512):")
print(f"First 10 token IDs: {e_ids[:10].tolist()}")
print(f"Decoded: {tokenizer.decode(e_ids[:10])}")
print(f"Last 5 (should be PAD={tokenizer.pad_token_id}): {e_ids[-5:].tolist()}")

# Check for exclamation marks
exclamation_id = tokenizer.encode("!", add_special_tokens=False)[0]
print(f"\n🔍 '!' token ID: {exclamation_id}")
print(f"   Count in Question: {(q_ids == exclamation_id).sum().item()}")
print(f"   Count in Plan: {(p_ids == exclamation_id).sum().item()}")  
print(f"   Count in Exec: {(e_ids == exclamation_id).sum().item()}")

print("\n" + "="*80)
print("✅ If the model outputs these tokens during inference, it learned correctly")
print("❌ If the model outputs different tokens (all PAD or all !), training is broken")
print("="*80)
