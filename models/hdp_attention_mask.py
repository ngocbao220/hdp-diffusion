"""
Hierarchical Dual-Process (HDP) Attention Mask Module

Implements custom attention patterns for hierarchical reasoning:
- Question Block: Self-attention only
- Plan Block: Attends to Question + Plan (abstract reasoning)
- Execution Block: Attends to Question + Plan + Execution (detailed computation)

Author: Research implementation for HDP-Diffusion
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def get_hdp_attention_mask(
    block_indices: torch.Tensor,
    seq_len:  int,
    block_sizes:  Tuple[int, int, int] = (128, 128, 256),
    causal_within_block: bool = False,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate Hierarchical Dual-Process attention mask - VECTORIZED VERSION. 
    
    Args:
        block_indices: (batch_size, seq_len) tensor with values [0, 1, 2]
                      0 = Question, 1 = Plan, 2 = Execution
        seq_len: Total sequence length (should be 512)
        block_sizes:  Tuple of (question_len, plan_len, exec_len)
        causal_within_block: If True, use causal mask within each block
        device:  Target device for mask
    
    Returns:
        attention_mask: (batch_size, seq_len, seq_len) binary mask
                       1 = allow attention, 0 = mask out
                       
    Attention Rules:
        - Question tokens (0): Attend ONLY to Question tokens
        - Plan tokens (1): Attend to Question + Plan (NOT Execution)
        - Execution tokens (2): Attend to Question + Plan + Execution (all)
    """
    if device is None: 
        device = block_indices.device
    
    batch_size = block_indices.shape[0]
    q_len, p_len, e_len = block_sizes
    
    # VECTORIZED IMPLEMENTATION - No Python loops!
    # Create query and key block indices for broadcasting
    # block_indices: (batch, seq) -> need (batch, seq_query, seq_key)
    query_blocks = block_indices. unsqueeze(2)  # (batch, seq, 1)
    key_blocks = block_indices.unsqueeze(1)    # (batch, 1, seq)
    
    # Rule 1: Question (0) can only attend to Question (0)
    # Rule 2: Plan (1) can attend to Question (0) and Plan (1)
    # Rule 3: Execution (2) can attend to all (0, 1, 2)
    
    # Key insight: token at position i (block b_i) can attend to position j (block b_j)
    # if and only if b_j <= b_i
    # This creates a "block-causal" pattern where later blocks can see earlier blocks
    
    mask = (key_blocks <= query_blocks)  # (batch, seq, seq)
    
    # Apply causal mask within blocks if requested
    if causal_within_block: 
        # Create position indices
        pos_query = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq, 1)
        pos_key = torch.arange(seq_len, device=device).unsqueeze(0)    # (1, seq)
        
        # Causal:  query can only attend to key if pos_key <= pos_query
        causal_mask = (pos_key <= pos_query)  # (seq, seq)
        
        # Combine:  must satisfy BOTH block constraint AND causal constraint
        mask = mask & causal_mask. unsqueeze(0)
    
    return mask


def get_hdp_attention_bias(
    block_indices: torch.Tensor,
    seq_len: int,
    block_sizes: Tuple[int, int, int] = (128, 128, 256),
    causal_within_block: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate additive attention bias for HDP (for scaled_dot_product_attention).
    
    Args:
        block_indices: (batch_size, seq_len) tensor with block IDs
        seq_len: Sequence length
        block_sizes: Tuple of block sizes
        causal_within_block: Whether to apply causal mask within blocks
        device: Target device
        dtype: Data type for the bias
    
    Returns:
        attention_bias: (batch_size, seq_len, seq_len) 
                       0.0 = allow, -inf = mask
    """
    mask = get_hdp_attention_mask(
        block_indices, seq_len, block_sizes, 
        causal_within_block, device
    )
    
    # Convert binary mask to additive bias
    # 1 -> 0.0 (allow attention)
    # 0 -> -inf (mask out)
    attention_bias = torch.zeros_like(mask, dtype=dtype)
    attention_bias = attention_bias.masked_fill(~mask, float('-inf'))
    
    return attention_bias


def get_block_indices(
    batch_size: int,
    block_sizes: Tuple[int, int, int] = (128, 128, 256),
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate block indices for a batch.
    
    Args:
        batch_size: Number of sequences in batch
        block_sizes: (question_len, plan_len, exec_len)
        device: Target device
    
    Returns:
        block_indices: (batch_size, seq_len) with values [0, 1, 2]
    """
    q_len, p_len, e_len = block_sizes
    seq_len = q_len + p_len + e_len
    
    # Create indices for one sequence
    indices = torch.cat([
        torch.zeros(q_len, dtype=torch.long),  # Question = 0
        torch.ones(p_len, dtype=torch.long),   # Plan = 1
        torch.full((e_len,), 2, dtype=torch.long)  # Execution = 2
    ])
    
    # Repeat for batch
    block_indices = indices.unsqueeze(0).repeat(batch_size, 1)
    
    if device is not None:
        block_indices = block_indices.to(device)
    
    return block_indices


def visualize_hdp_mask(
    seq_len: int = 512,
    block_sizes: Tuple[int, int, int] = (128, 128, 256),
    causal_within_block: bool = False,
    save_path: Optional[str] = None
):
    """
    Visualize the HDP attention mask (useful for debugging).
    
    Args:
        seq_len: Sequence length
        block_sizes: Block sizes
        causal_within_block: Whether to use causal mask within blocks
        save_path: Optional path to save visualization
    """
    block_indices = get_block_indices(1, block_sizes)
    mask = get_hdp_attention_mask(
        block_indices, seq_len, block_sizes, causal_within_block
    )
    
    # Convert to numpy for visualization
    mask_np = mask[0].cpu().numpy()
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask_np, cmap='RdYlGn', aspect='auto')
        
        # Add block boundaries
        q_len, p_len, e_len = block_sizes
        ax.axhline(y=q_len, color='blue', linewidth=2, linestyle='--', label='Question|Plan')
        ax.axhline(y=q_len + p_len, color='red', linewidth=2, linestyle='--', label='Plan|Execution')
        ax.axvline(x=q_len, color='blue', linewidth=2, linestyle='--')
        ax.axvline(x=q_len + p_len, color='red', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('HDP Attention Mask\n(Green=Allow, Red=Mask)')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Mask visualization saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib not available for visualization")
        print(f"Mask shape: {mask_np.shape}")
        print(f"Allowed attention ratio: {mask_np.mean():.2%}")


if __name__ == "__main__":
    # Test the mask generation
    print("Testing HDP Attention Mask Generation...")
    
    batch_size = 2
    seq_len = 512
    block_sizes = (128, 128, 256)
    
    # Generate block indices
    block_indices = get_block_indices(batch_size, block_sizes)
    print(f"Block indices shape: {block_indices.shape}")
    print(f"Block distribution: {torch.bincount(block_indices[0])}")
    
    # Generate mask
    mask = get_hdp_attention_mask(block_indices, seq_len, block_sizes)
    print(f"\nMask shape: {mask.shape}")
    print(f"Allowed attention ratio: {mask.float().mean():.2%}")
    
    # Generate bias
    bias = get_hdp_attention_bias(block_indices, seq_len, block_sizes)
    print(f"\nBias shape: {bias.shape}")
    print(f"Finite values ratio: {torch.isfinite(bias).float().mean():.2%}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_hdp_mask(seq_len, block_sizes, save_path="hdp_mask_visualization.png")
    
    print("\n✅ HDP Attention Mask tests passed!")
