#!/usr/bin/env python3
"""
Kiểm tra chi tiết baseline GSM8K training data format
"""

from transformers import AutoTokenizer
import datasets

print("="*80)
print("🔍 KIỂM TRA BASELINE GSM8K DATA FORMAT")
print("="*80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load GSM8K dataset từ HuggingFace
print("\n1️⃣ Loading GSM8K từ HuggingFace...")
dataset = datasets.load_dataset('gsm8k', 'main', split='train[:5]')

print(f"\n📊 Dataset info:")
print(f"   Số samples: {len(dataset)}")
print(f"   Keys: {list(dataset[0].keys())}")

# Xem sample đầu tiên
print(f"\n{'='*80}")
print("📝 SAMPLE 1 - RAW DATA:")
print(f"{'='*80}")
sample = dataset[0]
print(f"\n❓ Question:")
print(f"   {sample['question']}")
print(f"\n✅ Answer:")
print(f"   {sample['answer']}")

# Xem baseline format trong dataloader.py
print(f"\n{'='*80}")
print("🔧 BASELINE FORMAT (từ dataloader.py):")
print(f"{'='*80}")
baseline_format = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
print(baseline_format)

# Tokenize để xem length
print(f"\n{'='*80}")
print("📏 TOKEN LENGTH CHECK:")
print(f"{'='*80}")
tokens = tokenizer(baseline_format, return_tensors='pt')
print(f"   Token length: {len(tokens['input_ids'][0])}")
print(f"   First 50 tokens:")
print(f"   {tokenizer.decode(tokens['input_ids'][0][:50])}")

# So sánh với HDP format
print(f"\n{'='*80}")
print("🆚 SO SÁNH VỚI HDP FORMAT:")
print(f"{'='*80}")

# Load hierarchical data
import json
with open('/workspace/hdp-diffusion/data/gsm8k/gsm8k_hierarchical_train.json', 'r') as f:
    hdp_data = json.load(f)

hdp_sample = hdp_data[0]
print(f"\n📝 HDP Sample:")
print(f"   Question: {hdp_sample['question'][:100]}...")
print(f"   Plan: {hdp_sample['plan'][:100]}...")
print(f"   Execution: {hdp_sample['execution'][:100]}...")
print(f"   Answer: {hdp_sample['answer']}")
print(f"\n   Full text (model output):")
print(f"   {hdp_sample['full_text'][:200]}...")

# Token counts
baseline_tokens = len(tokenizer(baseline_format)['input_ids'])
hdp_question = len(tokenizer(hdp_sample['question'])['input_ids'])
hdp_plan = len(tokenizer(hdp_sample['plan'])['input_ids'])
hdp_exec = len(tokenizer(hdp_sample['execution'])['input_ids'])

print(f"\n{'='*80}")
print("📊 TOKEN COUNT COMPARISON:")
print(f"{'='*80}")
print(f"   Baseline total: ~{baseline_tokens} tokens")
print(f"   HDP Question: ~{hdp_question} tokens")
print(f"   HDP Plan: ~{hdp_plan} tokens")
print(f"   HDP Execution: ~{hdp_exec} tokens")
print(f"   HDP Total: ~{hdp_question + hdp_plan + hdp_exec} tokens")

print(f"\n{'='*80}")
print("✅ KẾT LUẬN:")
print(f"{'='*80}")
print("""
BASELINE FORMAT:
  - Simple concatenation: "Question: ... \\nAnswer: ..."
  - Không có phân chia structure
  - Token length biến đổi tùy sample
  
HDP FORMAT:
  - 3 blocks rõ ràng: [Question | Plan | Execution]
  - Có special tokens: [PLAN], [EXECUTION], [ANSWER]
  - Fixed length mỗi block: 128 + 128 + 256 = 512 tokens
  - Hierarchical attention: Plan không nhìn thấy Execution
""")
