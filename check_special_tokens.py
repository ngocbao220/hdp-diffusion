#!/usr/bin/env python3
"""
Check what special tokens were added to reach vocab_size=50262
"""
import torch
from transformers import AutoTokenizer

# Load checkpoint
ckpt_path = "outputs/hdp_diffusion_h200_bs16/checkpoints/last.ckpt"
print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Get embedding layer
embedding = checkpoint['state_dict']['backbone.vocab_embed.embedding']
print(f"\n📊  Checkpoint vocab_size: {embedding.shape[0]}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"📊  Base tokenizer vocab_size: {len(tokenizer)}")
print(f"📊  Difference: {embedding.shape[0] - len(tokenizer)} tokens")

# Check tokenizer special tokens
print(f"\n🔍  Current tokenizer special tokens:")
print(f"   eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"   bos_token: {tokenizer.bos_token}")
print(f"   pad_token: {tokenizer.pad_token}")
print(f"   unk_token: {tokenizer.unk_token}")
print(f"   mask_token: {tokenizer.mask_token}")

# Check if checkpoint has tokenizer info
if 'tokenizer' in checkpoint:
    print("\n✅  Checkpoint contains tokenizer info!")
    print(f"   Keys: {checkpoint['tokenizer'].keys()}")
elif 'hyper_parameters' in checkpoint:
    print("\n🔍  Checking hyper_parameters for tokenizer info...")
    hp = checkpoint['hyper_parameters']
    if 'tokenizer' in hp:
        print(f"   Found tokenizer: {hp['tokenizer']}")

# Check for additional_special_tokens in checkpoint config
print("\n🔍  Searching for special tokens in checkpoint...")
for key in checkpoint.keys():
    if 'token' in key.lower():
        print(f"   {key}: {checkpoint[key]}")

# Look in state_dict for clues
state_dict = checkpoint['state_dict']
print("\n🔍  Embedding layer analysis:")
print(f"   Last 10 token embeddings (50252-50261):")
for i in range(50252, min(50262, embedding.shape[0])):
    emb_norm = torch.norm(embedding[i]).item()
    print(f"   Token {i}: norm={emb_norm:.4f}")
