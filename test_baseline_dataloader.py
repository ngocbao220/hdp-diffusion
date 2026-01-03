"""Quick test to verify gsm8k_baseline dataloader works"""

import torch
from transformers import GPT2Tokenizer
from hdp_dataset import SimpleGSM8KDataset

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = SimpleGSM8KDataset(
    data_path='data/gsm8k/gsm8k_baseline.json',
    tokenizer=tokenizer,
    max_length=512,
    add_special_tokens=True
)

print(f"Dataset size: {len(dataset)}")

# Check first sample
sample = dataset[0]
print(f"\nSample 0 keys: {sample.keys()}")
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"attention_mask shape: {sample['attention_mask'].shape}")

# Decode to verify
decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
print(f"\nDecoded text (first 200 chars):")
print(decoded[:200])

# Test dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

batch = next(iter(dataloader))
print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")

print("\n✅ GSM8K Baseline dataloader works!")
