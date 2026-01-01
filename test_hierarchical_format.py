#!/usr/bin/env python3
"""Test hierarchical_gsm8k_dataset.py với format mới"""

from hierarchical_gsm8k_dataset import GSM8KHierarchicalDataset
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Test with special format enabled (default)
dataset = GSM8KHierarchicalDataset(
    data_path='data/gsm8k/gsm8k_hierarchical_train.json',
    tokenizer=tokenizer,
    question_len=128,
    plan_len=128,
    exec_len=256,
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
print(q_text[:200])
print('\n' + '='*80)

print('\n📝 BLOCK 1 - PLAN (128 tokens):')
print(p_text[:200])
print('\n' + '='*80)

print('\n⚙️  BLOCK 2 - EXECUTION (256 tokens):')
print(e_text[:300])
print('\n' + '='*80)

# Check format
full_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
has_plan = '[PLAN]' in full_text
has_exec = '[EXECUTION]' in full_text or '[EXEC]' in full_text
has_answer = '[ANSWER]' in full_text

print(f'\n✅ Format verification:')
print(f'  [PLAN]: {has_plan}')
print(f'  [EXECUTION]/[EXEC]: {has_exec}')
print(f'  [ANSWER]: {has_answer}')

if has_plan and has_exec and has_answer:
    print('\n🎉 hierarchical_gsm8k_dataset.py works perfectly!')
