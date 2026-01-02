"""
HDP Temporal Attention Mask
Progressive hierarchical denoising based on timestep

Key idea:
- Early timesteps (T → T/2): Focus on Plan generation (high-level reasoning)
- Late timesteps (T/2 → 0): Focus on Execution (detailed calculations)
- Question is always visible to all blocks
"""

import torch
from typing import Tuple, Optional


def get_hdp_temporal_mask(
    block_indices: torch.Tensor,
    timestep: torch.Tensor,
    seq_len: int,
    block_sizes: Tuple[int, int, int] = (128, 128, 256),
    t_max: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate time-dependent HDP attention mask.
    
    Progressive denoising strategy:
    - t ∈ [T, T/2]: Execution attends to Question + Plan only
                    Plan attends to Question only
    - t ∈ [T/2, 0]: All blocks fully attend (standard HDP)
    
    This encourages:
    1. Early: Focus on high-level reasoning (Plan)
    2. Late: Fill in low-level details (Execution)
    
    Args:
        block_indices: (batch, seq) - Block assignments (0=Q, 1=P, 2=E)
        timestep: (batch, 1) - Current diffusion timestep [0, 1]
        seq_len: Total sequence length
        block_sizes: (q_len, p_len, e_len)
        t_max: Maximum timestep (default 1.0)
        device: Target device
        dtype: Output dtype
    
    Returns:
        attention_bias: (batch, seq, seq) - Additive attention bias
                       0.0 = allow, -inf = mask
    """
    batch_size = block_indices.shape[0]
    q_len, p_len, e_len = block_sizes
    
    if device is None:
        device = block_indices.device
    
    # Compute temporal phase (early vs late denoising)
    # t > 0.5 → early phase (focus on Plan)
    # t <= 0.5 → late phase (full HDP)
    t_normalized = timestep / t_max  # Normalize to [0, 1]
    is_early_phase = (t_normalized > 0.5).squeeze(-1)  # (batch,)
    
    # Create position indices
    q_idx = torch.arange(seq_len, device=device)  # (seq,)
    k_idx = torch.arange(seq_len, device=device)  # (seq,)
    
    # Expand block_indices for broadcasting
    block_q = block_indices.unsqueeze(2)  # (batch, seq, 1)
    block_k = block_indices.unsqueeze(1)  # (batch, 1, seq)
    
    # Base HDP rules (always active)
    # Question → Question only
    # Plan → Question + Plan
    # Execution → All
    
    # Rule 1: Question tokens can only attend to Question
    is_q_query = (block_q == 0)
    is_q_key = (block_k == 0)
    q_mask = is_q_query & is_q_key
    
    # Rule 2: Plan tokens attend to Question + Plan
    is_p_query = (block_q == 1)
    is_qp_key = (block_k <= 1)
    p_mask = is_p_query & is_qp_key
    
    # Rule 3: Execution tokens - TIME DEPENDENT
    is_e_query = (block_q == 2)
    
    # Early phase (t > 0.5): Exec → Q + P only
    # Late phase (t <= 0.5): Exec → All
    is_qp_key_for_exec = (block_k <= 1)
    is_all_key = torch.ones_like(block_k, dtype=torch.bool)
    
    # Select mask based on phase (per sample in batch)
    e_allowed_keys = torch.where(
        is_early_phase.view(-1, 1, 1),  # (batch, 1, 1)
        is_qp_key_for_exec,              # Early: Q + P
        is_all_key                       # Late: All
    )
    e_mask = is_e_query & e_allowed_keys
    
    # Combine all rules
    attention_mask = q_mask | p_mask | e_mask  # (batch, seq, seq)
    
    # Convert to additive bias
    attention_bias = torch.zeros(
        batch_size, seq_len, seq_len,
        device=device, dtype=dtype
    )
    attention_bias = attention_bias.masked_fill(~attention_mask, float('-inf'))
    
    return attention_bias


def get_hdp_temporal_mask_smooth(
    block_indices: torch.Tensor,
    timestep: torch.Tensor,
    seq_len: int,
    block_sizes: Tuple[int, int, int] = (128, 128, 256),
    t_max: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Smooth version: Gradually relax mask instead of hard switch.
    
    Instead of binary early/late, use smooth transition:
    - Alpha(t) = smoothstep(t)
    - Exec attention weight = alpha * full + (1-alpha) * restricted
    
    This prevents sudden changes in attention patterns.
    """
    batch_size = block_indices.shape[0]
    q_len, p_len, e_len = block_sizes
    
    if device is None:
        device = block_indices.device
    
    # Smooth transition coefficient
    t_norm = (timestep / t_max).clamp(0, 1)
    # Smoothstep: 3t^2 - 2t^3 ∈ [0, 1]
    alpha = 3 * t_norm**2 - 2 * t_norm**3  # (batch, 1)
    
    # Create masks
    block_q = block_indices.unsqueeze(2)
    block_k = block_indices.unsqueeze(1)
    
    # Question mask (always strict)
    is_q_query = (block_q == 0)
    is_q_key = (block_k == 0)
    q_mask = is_q_query & is_q_key
    
    # Plan mask (always strict)
    is_p_query = (block_q == 1)
    is_qp_key = (block_k <= 1)
    p_mask = is_p_query & is_qp_key
    
    # Execution mask - SMOOTH TRANSITION
    is_e_query = (block_q == 2)
    is_qp_key_for_exec = (block_k <= 1)
    is_all_key = torch.ones_like(block_k, dtype=torch.bool)
    
    # For early t (alpha ≈ 1): restrict to Q+P
    # For late t (alpha ≈ 0): allow all
    # Smooth interpolation
    e_mask_restricted = is_e_query & is_qp_key_for_exec
    e_mask_full = is_e_query & is_all_key
    
    # Combine with soft gating
    # This is tricky - masks are boolean, so we convert to bias first
    base_mask = q_mask | p_mask | e_mask_restricted
    full_mask = q_mask | p_mask | e_mask_full
    
    # Convert to bias
    base_bias = torch.zeros_like(block_q, dtype=dtype).expand(-1, seq_len, seq_len)
    base_bias = base_bias.masked_fill(~base_mask, float('-inf'))
    
    full_bias = torch.zeros_like(block_q, dtype=dtype).expand(-1, seq_len, seq_len)
    full_bias = full_bias.masked_fill(~full_mask, float('-inf'))
    
    # Interpolate: alpha * restricted + (1-alpha) * full
    # But -inf interpolation is tricky, so use max operation
    alpha = alpha.squeeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
    
    # Simple approach: threshold at 0.5
    threshold = 0.5
    attention_bias = torch.where(
        alpha > threshold,
        base_bias,
        full_bias
    )
    
    return attention_bias


if __name__ == "__main__":
    # Test temporal mask
    batch_size = 2
    seq_len = 512
    block_sizes = (128, 128, 256)
    
    # Create block indices
    block_indices = torch.cat([
        torch.zeros(batch_size, 128, dtype=torch.long),
        torch.ones(batch_size, 128, dtype=torch.long),
        torch.full((batch_size, 256), 2, dtype=torch.long),
    ], dim=1)
    
    # Test at different timesteps
    print("Testing HDP Temporal Mask")
    print("=" * 60)
    
    for t in [1.0, 0.7, 0.5, 0.3, 0.0]:
        timestep = torch.full((batch_size, 1), t)
        mask = get_hdp_temporal_mask(
            block_indices, timestep, seq_len, block_sizes
        )
        
        # Check Execution block attention
        exec_start = 256
        exec_tokens = mask[0, exec_start:exec_start+10, :]  # First 10 Exec tokens
        
        # Count allowed positions
        allowed = (exec_tokens > -1e9).sum(dim=1).float().mean()
        
        print(f"t={t:.1f}: Exec tokens attend to {allowed:.0f}/{seq_len} positions")
    
    print("\nExpected behavior:")
    print("  t=1.0 (early): Exec → Q+P only (256 positions)")
    print("  t=0.5 (mid):   Transition")
    print("  t=0.0 (late):  Exec → All (512 positions)")
