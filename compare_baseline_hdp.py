#!/usr/bin/env python3
"""
So sánh chi tiết BASELINE vs HDP training
"""

from transformers import AutoTokenizer
from hdp_dataset import HDPDataset
import torch

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print("="*80)
print("⚖️  SO SÁNH CHI TIẾT: BASELINE vs HDP")
print("="*80)

# === 1. BASELINE FORMAT ===
print("\n" + "="*80)
print("1️⃣  BASELINE GSM8K FORMAT")
print("="*80)

import datasets
baseline_ds = datasets.load_dataset('gsm8k', 'main', split='train[:1]', cache_dir='.cache')
sample = baseline_ds[0]

baseline_text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
baseline_tokens = tokenizer(baseline_text, max_length=512, padding='max_length', 
                            truncation=True, return_tensors='pt')

print(f"\n📝 Sample text:")
print(baseline_text[:300] + "...")
print(f"\n📊 Format:")
print(f"   Structure: Simple concatenation")
print(f"   Prefix: 'Question: ' and 'Answer: '")
print(f"   No special tokens")
print(f"   Token count: {(baseline_tokens['input_ids'] != tokenizer.pad_token_id).sum().item()} real tokens")
print(f"   Padding: {(baseline_tokens['input_ids'] == tokenizer.pad_token_id).sum().item()} pad tokens")

# Decode để xem
decoded_baseline = tokenizer.decode(baseline_tokens['input_ids'][0], skip_special_tokens=False)
print(f"\n🔤 Decoded (first 400 chars):")
print(decoded_baseline[:400])

# === 2. HDP FORMAT ===
print("\n" + "="*80)
print("2️⃣  HDP FORMAT (Hierarchical Dual-Process)")
print("="*80)

hdp_dataset = HDPDataset(
    data_path='/workspace/hdp-diffusion/data/gsm8k/gsm8k_hierarchical_train.json',
    tokenizer=tokenizer,
    block_sizes=(128, 128, 256),
    use_special_format=True
)

hdp_sample = hdp_dataset[0]
decoded_hdp = tokenizer.decode(hdp_sample['input_ids'], skip_special_tokens=False)

print(f"\n📝 Structure:")
print(f"   Block 0 (Question): 128 tokens")
print(f"   Block 1 (Plan): 128 tokens → [PLAN] prefix")
print(f"   Block 2 (Execution): 256 tokens → [EXECUTION] ... [ANSWER]")
print(f"   Total: 512 tokens (fixed)")

# Tách từng block
q_block = tokenizer.decode(hdp_sample['input_ids'][:128], skip_special_tokens=False)
p_block = tokenizer.decode(hdp_sample['input_ids'][128:256], skip_special_tokens=False)
e_block = tokenizer.decode(hdp_sample['input_ids'][256:512], skip_special_tokens=False)

print(f"\n🔤 Block 0 - Question (first 150 chars):")
print(q_block[:150] + "...")

print(f"\n🔤 Block 1 - Plan (first 150 chars):")
print(p_block[:150] + "...")

print(f"\n🔤 Block 2 - Execution (first 200 chars):")
print(e_block[:200] + "...")

# Check special tokens
has_plan = '[PLAN]' in decoded_hdp
has_exec = '[EXECUTION]' in decoded_hdp or '[EXEC]' in decoded_hdp
has_answer = '[ANSWER]' in decoded_hdp

print(f"\n✅ Special tokens:")
print(f"   [PLAN]: {'✓' if has_plan else '✗'}")
print(f"   [EXECUTION]: {'✓' if has_exec else '✗'}")
print(f"   [ANSWER]: {'✓' if has_answer else '✗'}")

# === 3. KEY DIFFERENCES ===
print("\n" + "="*80)
print("3️⃣  KEY DIFFERENCES")
print("="*80)

print(f"""
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Aspect              │ BASELINE             │ HDP                  │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Format              │ Simple Q&A           │ Hierarchical 3-block │
│ Special Tokens      │ None                 │ [PLAN][EXEC][ANSWER] │
│ Sequence Length     │ Variable (padded)    │ Fixed 512 tokens     │
│ Attention           │ Full bidirectional   │ Hierarchical mask    │
│ Planning            │ Implicit             │ Explicit [PLAN] step │
│ Answer Separation   │ Inline with steps    │ [ANSWER] token       │
└─────────────────────┴──────────────────────┴──────────────────────┘

🎯 ADVANTAGES OF HDP:

1. **Explicit Reasoning Structure**
   - Baseline: Question và answer trộn lẫn
   - HDP: Tách biệt question → plan → execution → answer
   
2. **Hierarchical Attention**
   - Baseline: Plan và execution nhìn thấy tất cả
   - HDP: Plan KHÔNG nhìn thấy execution (causal reasoning)
   
3. **Easier to Parse**
   - Baseline: Cần parse "#### 42" để lấy answer
   - HDP: [ANSWER] token rõ ràng

4. **Fixed Block Sizes**
   - Baseline: Variable length → khó optimize
   - HDP: Fixed 128+128+256 → better batching

5. **Training Signal**
   - Baseline: Model học cả question + answer cùng lúc
   - HDP: Model học phân tầng: think (plan) → solve (exec) → conclude (answer)
""")

print("\n" + "="*80)
print("✅ RECOMMENDATION")
print("="*80)
print("""
Để huấn luyện mô hình GSM8K tốt nhất:

1. **Baseline**: Dùng để establish performance floor
   - Đơn giản, dễ implement
   - Không có inductive bias về reasoning structure

2. **HDP**: Dùng để improve reasoning capability
   - Hierarchical structure matches human reasoning
   - Explicit planning step
   - Better interpretability

💡 Nên train cả 2 để so sánh performance!
""")
