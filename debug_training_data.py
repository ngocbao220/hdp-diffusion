"""
Debug script to check if training data is correctly formatted.
"""

import json
import torch
from transformers import GPT2Tokenizer
from hdp_dataset import HDPDataset

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = HDPDataset(
    data_path='data/gsm8k/gsm8k_overfit.json',
    tokenizer=tokenizer,
    block_sizes=(128, 128, 256),
    use_special_format=True
)

print("="*80)
print("🔍 TRAINING DATA DEBUG")
print("="*80)

# Get first sample
sample = dataset[0]

input_ids = sample['input_ids']
attention_mask = sample['attention_mask']
block_indices = sample['block_indices']

print(f"\n📊 Sample shape: {input_ids.shape}")
print(f"   Block sizes: Q=128, P=128, E=256, Total=512")

# Decode each block
q_ids = input_ids[:128]
p_ids = input_ids[128:256]
e_ids = input_ids[256:512]

q_text = tokenizer.decode(q_ids, skip_special_tokens=False)
p_text = tokenizer.decode(p_ids, skip_special_tokens=False)
e_text = tokenizer.decode(e_ids, skip_special_tokens=False)

print("\n" + "="*80)
print("📝 QUESTION BLOCK (0-128)")
print("="*80)
print(q_text)
print(f"Non-padding tokens: {(q_ids != tokenizer.pad_token_id).sum().item()}")

print("\n" + "="*80)
print("📝 PLAN BLOCK (128-256)")
print("="*80)
print(p_text)
print(f"Non-padding tokens: {(p_ids != tokenizer.pad_token_id).sum().item()}")

print("\n" + "="*80)
print("📝 EXECUTION BLOCK (256-512)")
print("="*80)
print(e_text)
print(f"Non-padding tokens: {(e_ids != tokenizer.pad_token_id).sum().item()}")

# Check for "!!!!" tokens
exclamation_token_id = tokenizer.encode("!", add_special_tokens=False)[0]
print(f"\n🔍 Checking for '!' token (id={exclamation_token_id}):")
print(f"   Question block: {(q_ids == exclamation_token_id).sum().item()} occurrences")
print(f"   Plan block: {(p_ids == exclamation_token_id).sum().item()} occurrences")
print(f"   Execution block: {(e_ids == exclamation_token_id).sum().item()} occurrences")

# Check attention mask
print(f"\n🔍 Attention Mask Analysis:")
q_mask = attention_mask[:128]
p_mask = attention_mask[128:256]
e_mask = attention_mask[256:512]
print(f"   Question: {q_mask.sum().item()}/128 tokens active")
print(f"   Plan: {p_mask.sum().item()}/128 tokens active")
print(f"   Execution: {e_mask.sum().item()}/256 tokens active")

# Load raw JSON to compare
with open('data/gsm8k/gsm8k_overfit.json', 'r') as f:
    raw_data = json.load(f)[0]

print("\n" + "="*80)
print("📄 RAW JSON DATA (Expected)")
print("="*80)
print(f"Question: {raw_data['question']}")
print(f"Plan: {raw_data['plan']}")
print(f"Execution: {raw_data['execution']}")
print(f"Answer: {raw_data.get('answer', 'N/A')}")

print("\n" + "="*80)
print("✅ If Plan/Exec blocks have real content above, training data is OK")
print("❌ If Plan/Exec are all [PAD] or '!', the data loading is broken")
print("="*80)
