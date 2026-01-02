#!/usr/bin/env python3
"""
Debug tại sao model sinh ra toàn dấu !
"""

from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print("="*80)
print("🔍 DEBUG EXCLAMATION MARK BUG")
print("="*80)

# Check token IDs
exclamation = tokenizer.encode('!', add_special_tokens=False)
print(f"\n1️⃣ Token IDs:")
print(f"   '!' token ID: {exclamation}")
print(f"   EOS token ID: {tokenizer.eos_token_id}")
print(f"   PAD token ID: {tokenizer.pad_token_id}")
print(f"   BOS token ID: {tokenizer.bos_token_id}")

# Decode a sequence of exclamations
repeated_exclaim = [exclamation[0]] * 20
decoded = tokenizer.decode(repeated_exclaim)
print(f"\n2️⃣ Decoding 20x '!' tokens:")
print(f"   Result: '{decoded}'")

# Check common tokens in training data
print(f"\n3️⃣ Check common math tokens:")
math_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
               '+', '-', '*', '/', '=', '.', ',', ' ', '\n']
for token in math_tokens:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    print(f"   '{token}' → {token_ids}")

# Load and check training data
print(f"\n{'='*80}")
print("4️⃣ CHECK TRAINING DATA:")
print(f"{'='*80}")

import json
with open('data/gsm8k/gsm8k_overfit.json', 'r') as f:
    data = json.load(f)

sample = data[0]
print(f"\nQuestion: {sample['question'][:100]}")
print(f"Plan: {sample['plan'][:100]}")
print(f"Execution: {sample['execution'][:100]}")

# Check if data contains many exclamation marks
full_text = sample.get('full_text', f"{sample['plan']} {sample['execution']}")
exclaim_count = full_text.count('!')
print(f"\n❗ Exclamation marks in training data: {exclaim_count}")
print(f"   Total chars: {len(full_text)}")
print(f"   Percentage: {100 * exclaim_count / len(full_text):.2f}%")

# Check tokenized length
print(f"\n{'='*80}")
print("5️⃣ TOKEN LENGTH CHECK:")
print(f"{'='*80}")

q_tokens = tokenizer(sample['question'], max_length=128, padding='max_length', truncation=True)
p_tokens = tokenizer(sample['plan'], max_length=128, padding='max_length', truncation=True)
e_tokens = tokenizer(sample['execution'], max_length=256, padding='max_length', truncation=True)

q_real = sum(1 for t in q_tokens['input_ids'] if t != tokenizer.pad_token_id)
p_real = sum(1 for t in p_tokens['input_ids'] if t != tokenizer.pad_token_id)
e_real = sum(1 for t in e_tokens['input_ids'] if t != tokenizer.pad_token_id)

print(f"Question: {q_real}/128 real tokens (padding: {128-q_real})")
print(f"Plan: {p_real}/128 real tokens (padding: {128-p_real})")
print(f"Execution: {e_real}/256 real tokens (padding: {256-e_real})")

total_padding = (128-q_real) + (128-p_real) + (256-e_real)
print(f"\n⚠️  TOTAL PADDING TOKENS: {total_padding}/512 ({100*total_padding/512:.1f}%)")

if total_padding > 256:
    print(f"\n❌ WARNING: Too much padding! Model might be learning to predict PAD tokens!")
    print(f"   PAD token ID: {tokenizer.pad_token_id}")
    print(f"   This could cause model collapse")

# Decode padded sequences
print(f"\n{'='*80}")
print("6️⃣ INSPECT PADDED SEQUENCES:")
print(f"{'='*80}")

print(f"\nQuestion tokens (last 20):")
print(f"   {q_tokens['input_ids'][-20:]}")
print(f"   Decoded: '{tokenizer.decode(q_tokens['input_ids'][-20:])}'")

print(f"\nPlan tokens (last 20):")
print(f"   {p_tokens['input_ids'][-20:]}")
print(f"   Decoded: '{tokenizer.decode(p_tokens['input_ids'][-20:])}'")

print(f"\n{'='*80}")
print("7️⃣ POSSIBLE CAUSES:")
print(f"{'='*80}")

print(f"\n1. Model learning PAD tokens instead of content")
print(f"2. Attention mask not properly masking padding")
print(f"3. Loss function including padding positions")
print(f"4. Temperature/sampling causing mode collapse")
print(f"5. HDP mask interfering with proper attention")

print(f"\n{'='*80}")
print("8️⃣ RECOMMENDATIONS:")
print(f"{'='*80}")

print(f"\n1. Check loss mask in training:")
print(f"   - Ensure padding tokens are masked in loss")
print(f"   - loss_mask = (input_ids != pad_token_id)")

print(f"\n2. Check attention mask:")
print(f"   - Should be 0 for padding, 1 for real tokens")
print(f"   - Currently: {q_tokens['attention_mask'][-10:]}")

print(f"\n3. Inspect model outputs during training:")
print(f"   - Add debug prints to see predicted token IDs")
print(f"   - Check if loss is actually decreasing on REAL tokens")

print(f"\n4. Try different sampling strategy:")
print(f"   - Add temperature parameter")
print(f"   - Try top-k or top-p sampling")

print(f"\n5. Check training data format:")
print(f"   - Ensure special tokens are correct")
print(f"   - Verify block boundaries")
