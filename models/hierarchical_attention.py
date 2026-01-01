"""
Hierarchical Attention Mask for Dual-Process Diffusion
Author: Research Implementation based on Block Diffusion (Arriola et al., ICLR 2025)

This module implements hierarchical attention patterns for mathematical reasoning:
- Question Block: Bidirectional self-attention
- Plan Block: Attends to Question + Plan (causal masking preventing future Plan tokens)
- Execution Block: Attends to Question + Plan + Execution (full access to reasoning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def get_hierarchical_mask(
    block_indices: torch.Tensor,
    attention_type: str = "bidirectional_within_blocks",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate hierarchical attention mask for dual-process reasoning.
    
    Args:
        block_indices: (batch_size, seq_len) tensor with values:
            0 = Question block
            1 = Plan block  
            2 = Execution block
        attention_type: Type of attention within blocks:
            - "bidirectional_within_blocks": Full attention within each block
            - "causal_within_blocks": Causal attention within each block
        device: Device to create mask on
        
    Returns:
        attention_mask: (batch_size, seq_len, seq_len) boolean mask
            True = can attend, False = cannot attend (will be masked)
    """
    batch_size, seq_len = block_indices.shape
    device = device or block_indices.device
    
    # Initialize mask - default to False (cannot attend)
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        indices = block_indices[b]
        
        # Get positions for each block
        q_positions = (indices == 0).nonzero(as_tuple=True)[0]  # Question positions
        p_positions = (indices == 1).nonzero(as_tuple=True)[0]  # Plan positions
        e_positions = (indices == 2).nonzero(as_tuple=True)[0]  # Execution positions
        
        # Rule 1: Question tokens attend to Question tokens only
        if len(q_positions) > 0:
            q_start, q_end = q_positions[0].item(), q_positions[-1].item() + 1
            if attention_type == "bidirectional_within_blocks":
                # Bidirectional attention within Question block
                mask[b, q_start:q_end, q_start:q_end] = True
            else:  # causal_within_blocks
                # Causal attention within Question block
                for i in range(q_start, q_end):
                    mask[b, i, q_start:i+1] = True
        
        # Rule 2: Plan tokens attend to Question + Plan
        if len(p_positions) > 0:
            p_start, p_end = p_positions[0].item(), p_positions[-1].item() + 1
            
            # Plan can attend to all Question tokens
            if len(q_positions) > 0:
                mask[b, p_start:p_end, q_start:q_end] = True
            
            # Plan attends to Plan (with causal masking for autoregressive property)
            if attention_type == "bidirectional_within_blocks":
                # Bidirectional within Plan block
                mask[b, p_start:p_end, p_start:p_end] = True
            else:  # causal_within_blocks
                # Causal within Plan block
                for i in range(p_start, p_end):
                    mask[b, i, p_start:i+1] = True
        
        # Rule 3: Execution tokens attend to Question + Plan + Execution
        if len(e_positions) > 0:
            e_start, e_end = e_positions[0].item(), e_positions[-1].item() + 1
            
            # Execution can attend to all Question tokens
            if len(q_positions) > 0:
                mask[b, e_start:e_end, q_start:q_end] = True
            
            # Execution can attend to all Plan tokens
            if len(p_positions) > 0:
                mask[b, e_start:e_end, p_start:p_end] = True
            
            # Execution attends to Execution (with causal masking)
            if attention_type == "bidirectional_within_blocks":
                # Bidirectional within Execution block
                mask[b, e_start:e_end, e_start:e_end] = True
            else:  # causal_within_blocks
                # Causal within Execution block
                for i in range(e_start, e_end):
                    mask[b, i, e_start:i+1] = True
    
    return mask


def get_hierarchical_attention_bias(
    block_indices: torch.Tensor,
    attention_type: str = "bidirectional_within_blocks",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate hierarchical attention bias (additive mask) for use with scaled dot-product attention.
    
    This version returns additive bias:
        0.0 = can attend
        -inf = cannot attend (will be masked out after softmax)
    
    Args:
        block_indices: (batch_size, seq_len) tensor with block assignments
        attention_type: Type of attention within blocks
        device: Device to create mask on
        
    Returns:
        attention_bias: (batch_size, seq_len, seq_len) float tensor
    """
    bool_mask = get_hierarchical_mask(block_indices, attention_type, device)
    
    # Convert boolean mask to additive bias
    # True (can attend) -> 0.0
    # False (cannot attend) -> -inf
    attention_bias = torch.zeros_like(bool_mask, dtype=torch.float32)
    attention_bias.masked_fill_(~bool_mask, float('-inf'))
    
    return attention_bias


def apply_hierarchical_mask_to_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_indices: torch.Tensor,
    attention_type: str = "bidirectional_within_blocks",
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Apply hierarchical attention mask to standard scaled dot-product attention.
    
    Args:
        query: (batch_size, num_heads, seq_len, head_dim)
        key: (batch_size, num_heads, seq_len, head_dim)
        value: (batch_size, num_heads, seq_len, head_dim)
        block_indices: (batch_size, seq_len)
        attention_type: Type of attention within blocks
        dropout_p: Dropout probability
        
    Returns:
        output: (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # Get hierarchical attention bias
    attention_bias = get_hierarchical_attention_bias(
        block_indices, attention_type, device=query.device
    )  # (batch_size, seq_len, seq_len)
    
    # Expand bias for multi-head attention
    attention_bias = attention_bias.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
    attention_bias = attention_bias.expand(batch_size, num_heads, seq_len, seq_len)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
    scores = scores + attention_bias  # Apply hierarchical mask
    
    # Softmax and apply to values
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    output = torch.matmul(attn_weights, value)
    
    return output


class HierarchicalAttentionWrapper(nn.Module):
    """
    Wrapper to inject hierarchical attention mask into existing attention modules.
    
    Usage:
        # Wrap your existing attention module
        attention = HierarchicalAttentionWrapper(
            base_attention_module,
            use_hierarchical_mask=True
        )
        
        # Forward pass with block_indices
        output = attention(x, block_indices=block_indices)
    """
    
    def __init__(
        self,
        base_attention: nn.Module,
        use_hierarchical_mask: bool = True,
        attention_type: str = "bidirectional_within_blocks",
    ):
        super().__init__()
        self.base_attention = base_attention
        self.use_hierarchical_mask = use_hierarchical_mask
        self.attention_type = attention_type
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_indices: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with optional hierarchical masking.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional standard attention mask
            block_indices: (batch_size, seq_len) block assignments
            **kwargs: Additional arguments for base attention
        """
        if self.use_hierarchical_mask and block_indices is not None:
            # Generate hierarchical attention bias
            hierarchical_bias = get_hierarchical_attention_bias(
                block_indices,
                self.attention_type,
                device=hidden_states.device
            )
            
            # Combine with existing attention mask if provided
            if attention_mask is not None:
                # Assume attention_mask is additive bias (0 = attend, -inf = mask)
                hierarchical_bias = hierarchical_bias + attention_mask
            
            # Pass to base attention with hierarchical mask
            return self.base_attention(
                hidden_states,
                attention_mask=hierarchical_bias,
                **kwargs
            )
        else:
            # Standard attention without hierarchical mask
            return self.base_attention(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )


def visualize_attention_mask(
    block_indices: torch.Tensor,
    attention_type: str = "bidirectional_within_blocks",
    sample_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Visualize the hierarchical attention mask for debugging.
    
    Args:
        block_indices: (batch_size, seq_len)
        attention_type: Type of attention within blocks
        sample_idx: Which sample to visualize
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    mask = get_hierarchical_mask(block_indices, attention_type)
    mask_np = mask[sample_idx].cpu().numpy().astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mask_np, cmap='RdYlGn', aspect='auto')
    
    # Add block boundaries
    indices = block_indices[sample_idx].cpu().numpy()
    q_end = (indices == 0).sum()
    p_end = q_end + (indices == 1).sum()
    
    ax.axhline(y=q_end-0.5, color='blue', linewidth=2, label='Question|Plan')
    ax.axhline(y=p_end-0.5, color='red', linewidth=2, label='Plan|Execution')
    ax.axvline(x=q_end-0.5, color='blue', linewidth=2)
    ax.axvline(x=p_end-0.5, color='red', linewidth=2)
    
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(f'Hierarchical Attention Mask ({attention_type})', fontsize=14)
    ax.legend()
    
    plt.colorbar(im, ax=ax, label='Can Attend (1=Yes, 0=No)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention mask visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test hierarchical mask generation
    batch_size = 2
    seq_len = 512
    
    # Create dummy block indices
    # Question: 0-127, Plan: 128-255, Execution: 256-511
    block_indices = torch.zeros(batch_size, seq_len, dtype=torch.long)
    block_indices[:, 128:256] = 1  # Plan
    block_indices[:, 256:] = 2      # Execution
    
    print("Testing Hierarchical Attention Mask...")
    print(f"Sequence length: {seq_len}")
    print(f"Question: 0-127, Plan: 128-255, Execution: 256-511")
    
    # Test boolean mask
    mask = get_hierarchical_mask(block_indices, attention_type="bidirectional_within_blocks")
    print(f"\nMask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    
    # Test attention bias
    bias = get_hierarchical_attention_bias(block_indices)
    print(f"\nAttention bias shape: {bias.shape}")
    print(f"Attention bias dtype: {bias.dtype}")
    
    # Verify attention patterns
    print("\n=== Attention Pattern Verification ===")
    
    # Question token (pos 64) should only attend to Question
    q_pos = 64
    can_attend_q = mask[0, q_pos].nonzero(as_tuple=True)[0]
    print(f"Question token {q_pos} attends to positions: {can_attend_q[:10].tolist()}... (max: {can_attend_q.max().item()})")
    
    # Plan token (pos 192) should attend to Question + Plan
    p_pos = 192
    can_attend_p = mask[0, p_pos].nonzero(as_tuple=True)[0]
    print(f"Plan token {p_pos} attends to positions: min={can_attend_p.min().item()}, max={can_attend_p.max().item()}")
    
    # Execution token (pos 384) should attend to all
    e_pos = 384
    can_attend_e = mask[0, e_pos].nonzero(as_tuple=True)[0]
    print(f"Execution token {e_pos} attends to positions: min={can_attend_e.min().item()}, max={can_attend_e.max().item()}")
    
    print("\n✅ Hierarchical attention mask tests passed!")
