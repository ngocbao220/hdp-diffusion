#!/usr/bin/env python3
"""Test HDPDataset with new [PLAN] [EXECUTION] [ANSWER] format"""

from hdp_dataset import HDPDataset
from transformers import AutoTokenizer
import torch

print('Testing HDPDataset with new [PLAN] [EXECUTION] [ANSWER] format...\n')

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Test with special format enabled
dataset = HDPDataset(
    data_path='data/gsm8k/gsm8k_hierarchical_train.json',
    tokenizer=tokenizer,
    block_sizes=(128, 128, 256),
    use_special_format=True
)

print(f'Dataset loaded: {len(dataset)} samples\n')

# Get first sample
sample = dataset[0]
print('Sample keys:', list(sample.keys()))
print('Input IDs shape:', sample['input_ids'].shape)
print('Block indices shape:', sample['block_indices'].shape)
print('Block distribution:', torch.bincount(sample['block_indices']).tolist())

# Decode to check format
decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
print('\n=== Decoded output (first 800 chars) ===')
print(decoded[:800])

# Check if [PLAN], [EXECUTION], [ANSWER] appear
has_plan = '[PLAN]' in decoded
has_exec = '[EXECUTION]' in decoded or '[EXEC]' in decoded
has_answer = '[ANSWER]' in decoded

print(f'\n✅ Format check:')
print(f'  [PLAN] present: {has_plan}')
print(f'  [EXECUTION]/[EXEC] present: {has_exec}')
print(f'  [ANSWER] present: {has_answer}')

if has_plan and has_exec and has_answer:
    print('\n🎉 Dataset format is correct!')
else:
    print('\n⚠️ Warning: Some tokens are missing')
