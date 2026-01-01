#!/usr/bin/env python3
"""Test HDPDataset - detailed view"""

from hdp_dataset import HDPDataset
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Test with special format enabled
dataset = HDPDataset(
    data_path='data/gsm8k/gsm8k_hierarchical_train.json',
    tokenizer=tokenizer,
    block_sizes=(128, 128, 256),
    use_special_format=True
)

print(f'✅ Dataset loaded: {len(dataset)} samples\n')
print('='*80)

# Get first sample
sample = dataset[0]

# Decode each block separately
q_ids = sample['input_ids'][:128]
p_ids = sample['input_ids'][128:256]
e_ids = sample['input_ids'][256:512]

q_text = tokenizer.decode(q_ids, skip_special_tokens=False)
p_text = tokenizer.decode(p_ids, skip_special_tokens=False)
e_text = tokenizer.decode(e_ids, skip_special_tokens=False)

print('📋 BLOCK 0 - QUESTION (128 tokens):')
print(q_text)
print('\n' + '='*80)

print('\n📝 BLOCK 1 - PLAN (128 tokens):')
print(p_text)
print('\n' + '='*80)

print('\n⚙️  BLOCK 2 - EXECUTION (256 tokens):')
print(e_text)
print('\n' + '='*80)

print('\n🔍 Full sequence:')
full_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
print(full_text[:1000])
