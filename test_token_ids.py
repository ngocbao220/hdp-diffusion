#!/usr/bin/env python3
"""
Test script to verify special token IDs are unique
"""

from transformers import AutoTokenizer

print("="*80)
print("🧪 Testing Special Token IDs")
print("="*80)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print(f"\n📊 Original vocab size: {len(tokenizer)}")

# Add PAD token
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
print(f"📊 After adding PAD: {len(tokenizer)}")
print(f"   <|pad|> ID: {tokenizer.pad_token_id}")

# Add HDP special tokens
special_tokens_dict = {'additional_special_tokens': ['<|plan|>', '<|execution|>', '<|answer|>']}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"\n📊 After adding HDP tokens: {len(tokenizer)} (+{num_added} tokens)")

# Print all IDs
print(f"\n✅ Special Token IDs:")
print(f"   <|pad|>: {tokenizer.pad_token_id}")
print(f"   <|plan|>: {tokenizer.additional_special_tokens_ids[0]}")
print(f"   <|execution|>: {tokenizer.additional_special_tokens_ids[1]}")
print(f"   <|answer|>: {tokenizer.additional_special_tokens_ids[2]}")

# Verify uniqueness
ids = [tokenizer.pad_token_id] + tokenizer.additional_special_tokens_ids
print(f"\n🔍 Uniqueness check:")
if len(ids) == len(set(ids)):
    print("   ✅ All token IDs are UNIQUE!")
else:
    print("   ❌ ERROR: Duplicate token IDs found!")
    print(f"   IDs: {ids}")
    print(f"   Unique IDs: {set(ids)}")

# Test tokenization
print(f"\n🧪 Tokenization test:")
test_text = "<|plan|> This is a plan <|execution|> This is execution <|answer|> 42"
tokens = tokenizer.encode(test_text)
print(f"   Input: {test_text}")
print(f"   Token IDs: {tokens}")
print(f"   Decoded: {tokenizer.decode(tokens)}")

print("\n" + "="*80)
