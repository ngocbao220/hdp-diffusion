#!/usr/bin/env python3
"""Verify xem training data có đúng format không"""

import torch
from transformers import AutoTokenizer
from hdp_dataset import HDPDataset

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load dataset như training sẽ load
dataset = HDPDataset(
    data_path='data/gsm8k/gsm8k_hierarchical_train.json',
    tokenizer=tokenizer,
    block_sizes=(128, 128, 256),
    add_special_tokens=True,
    return_block_indices=True,
    use_special_format=True
)

print(f"Dataset size: {len(dataset)}")
print(f"\n{'='*80}")

# Get first batch
sample = dataset[0]
print("Sample keys:", list(sample.keys()))
print("Input shape:", sample['input_ids'].shape)

# Decode
decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

print(f"\n{'='*80}")
print("First 600 chars:")
print(decoded[:600])

print(f"\n{'='*80}")
# Check tokens
has_plan = '[PLAN]' in decoded
has_exec = '[EXECUTION]' in decoded
has_answer = '[ANSWER]' in decoded

print(f"✅ Format check:")
print(f"  [PLAN]: {has_plan}")
print(f"  [EXECUTION]: {has_exec}")
print(f"  [ANSWER]: {has_answer}")

if has_plan and has_exec and has_answer:
    print("\n🎉 Dataset has correct format!")
else:
    print("\n❌ Dataset format is incorrect!")
    print("\nThis means the dataloader is NOT using HDPDataset.")
