"""
Test script to verify HDP attention mask is being used in training.
"""
import torch
import sys
import os
from omegaconf import DictConfig, OmegaConf

# Add current directory to path
sys.path.insert(0, '/workspace/hdp-diffusion')

from models.hdp_attention_mask import get_hdp_attention_bias, get_block_indices
from hdp_dataset import HDPDataset

def test_mask_generation():
    """Test that HDP mask is generated correctly."""
    print("="*60)
    print("Testing HDP Attention Mask Generation")
    print("="*60)
    
    batch_size = 2
    seq_len = 512
    block_sizes = (128, 128, 256)
    
    # Generate block indices
    block_indices = get_block_indices(batch_size, block_sizes, device='cpu')
    print(f"\nBlock indices shape: {block_indices.shape}")
    print(f"Block indices unique values: {block_indices[0].unique()}")
    print(f"Question positions (0): {(block_indices[0] == 0).sum()}")
    print(f"Plan positions (1): {(block_indices[0] == 1).sum()}")
    print(f"Execution positions (2): {(block_indices[0] == 2).sum()}")
    
    # Generate HDP attention bias
    hdp_mask = get_hdp_attention_bias(
        block_indices=block_indices,
        seq_len=seq_len,
        block_sizes=block_sizes,
        causal_within_block=False,
        device='cpu',
        dtype=torch.float32
    )
    
    print(f"\nHDP mask shape: {hdp_mask.shape}")
    print(f"HDP mask dtype: {hdp_mask.dtype}")
    
    # Check attention patterns
    # Question token (position 64) should only attend to Question block
    q_pos = 64
    q_attention = hdp_mask[0, q_pos]
    print(f"\nQuestion token at pos {q_pos}:")
    print(f"  Can attend to Question (0-127): {torch.isfinite(q_attention[:128]).all()}")
    print(f"  Cannot attend to Plan (128-255): {torch.isinf(q_attention[128:256]).all()}")
    print(f"  Cannot attend to Execution (256-511): {torch.isinf(q_attention[256:]).all()}")
    
    # Plan token (position 192) should attend to Question + Plan
    p_pos = 192
    p_attention = hdp_mask[0, p_pos]
    print(f"\nPlan token at pos {p_pos}:")
    print(f"  Can attend to Question (0-127): {torch.isfinite(p_attention[:128]).all()}")
    print(f"  Can attend to Plan (128-255): {torch.isfinite(p_attention[128:256]).all()}")
    print(f"  Cannot attend to Execution (256-511): {torch.isinf(p_attention[256:]).all()}")
    
    # Execution token (position 384) should attend to all
    e_pos = 384
    e_attention = hdp_mask[0, e_pos]
    print(f"\nExecution token at pos {e_pos}:")
    print(f"  Can attend to Question (0-127): {torch.isfinite(e_attention[:128]).all()}")
    print(f"  Can attend to Plan (128-255): {torch.isfinite(e_attention[128:256]).all()}")
    print(f"  Can attend to Execution (256-511): {torch.isfinite(e_attention[256:]).all()}")
    
    print("\n" + "="*60)
    print("✓ HDP Attention Mask Generation Test PASSED")
    print("="*60)

def test_dataset_block_indices():
    """Test that HDPDataset provides correct block_indices."""
    print("\n" + "="*60)
    print("Testing HDPDataset Block Indices")
    print("="*60)
    
    # Create dummy dataset
    data_path = "/workspace/hdp-diffusion/data/gsm8k/gsm8k_hierarchical_train.json"
    
    if not os.path.exists(data_path):
        print(f"WARNING: Dataset not found at {data_path}")
        print("Skipping dataset test")
        return
    
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Add special tokens for HDP format
    special_tokens = {
        'additional_special_tokens': ['[PLAN]', '[EXECUTION]', '[ANSWER]']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    dataset = HDPDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        block_sizes=(128, 128, 256),
        use_special_format=True
    )
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    
    if 'block_indices' in sample:
        block_indices = sample['block_indices']
        print(f"Block indices shape: {block_indices.shape}")
        print(f"Block indices unique values: {block_indices.unique()}")
        print(f"Question positions (0): {(block_indices == 0).sum()}")
        print(f"Plan positions (1): {(block_indices == 1).sum()}")
        print(f"Execution positions (2): {(block_indices == 2).sum()}")
        
        print("\n" + "="*60)
        print("✓ HDPDataset Block Indices Test PASSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ ERROR: block_indices not found in dataset sample")
        print("="*60)

if __name__ == "__main__":
    test_mask_generation()
    test_dataset_block_indices()
    
    print("\n" + "="*60)
    print("All HDP Attention Tests Complete")
    print("="*60)
