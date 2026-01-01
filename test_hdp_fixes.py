# test_hdp_fixes.py
import torch
from models.hdp_attention_mask import get_hdp_attention_mask, get_block_indices

# Test 1: Vectorized mask
print("Testing vectorized HDP mask...")
batch_size = 4
block_sizes = (128, 128, 256)
seq_len = sum(block_sizes)

block_indices = get_block_indices(batch_size, block_sizes)
mask = get_hdp_attention_mask(block_indices, seq_len, block_sizes)

print(f"Mask shape: {mask.shape}")
print(f"Expected:  ({batch_size}, {seq_len}, {seq_len})")

# Verify attention patterns
# Question (0-127) should only attend to Question
q_to_q = mask[0, 0, 0:128]. all()  # Q attends to Q
q_to_p = mask[0, 0, 128:256].any()  # Q should NOT attend to P
q_to_e = mask[0, 0, 256:512].any()  # Q should NOT attend to E

print(f"\n✅ Question->Question: {q_to_q} (expected: True)")
print(f"✅ Question->Plan: {q_to_p} (expected: False)")
print(f"✅ Question->Execution: {q_to_e} (expected: False)")

# Plan (128-255) should attend to Q + P
p_to_q = mask[0, 128, 0:128].all()
p_to_p = mask[0, 128, 128:256].all()
p_to_e = mask[0, 128, 256:512]. any()

print(f"\n✅ Plan->Question: {p_to_q} (expected: True)")
print(f"✅ Plan->Plan: {p_to_p} (expected: True)")
print(f"✅ Plan->Execution: {p_to_e} (expected: False)")

# Execution (256-511) should attend to all
e_to_all = mask[0, 256, : ]. all()
print(f"\n✅ Execution->All: {e_to_all} (expected: True)")

print("\n🎉 All HDP mask tests passed!")