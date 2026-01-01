"""
Hierarchical Block Diffusion Mask for Plan-then-Generate Architecture

This implements the attention mask structure for hierarchical reasoning:
- Question tokens can attend to themselves
- Plan Block can attend to Question and itself
- Execution Block can attend to Question, Plan, and itself
- Plan Block CANNOT see Execution Block (causal constraint)

Structure: [Question | Plan_xt | Plan_x0 | Execution_xt | Execution_x0]
"""

import torch
from functools import partial

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False


def hierarchical_block_diff_mask(
    b, h, q_idx, kv_idx, 
    question_len=None,
    plan_len=None, 
    exec_len=None,
    block_size=None,
    n=None
):
    """
    Constructs hierarchical block diffusion attention mask for plan-then-generate.
    
    Sequence structure: [Question | Plan_xt | Plan_x0 | Exec_xt | Exec_x0]
    
    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        question_len: Length of question/context tokens.
        plan_len: Length of plan block (number of tokens).
        exec_len: Length of execution block (number of tokens).
        block_size: Defines the block structure for diffusion.
        n: Total sequence length (question + plan + execution).
        
    Returns:
        A boolean attention mask.
    """
    
    # Define boundaries
    question_end = question_len
    plan_xt_end = question_end + plan_len
    plan_x0_end = plan_xt_end + plan_len
    exec_xt_end = plan_x0_end + exec_len
    exec_x0_end = exec_xt_end + exec_len
    
    # Determine which region each token belongs to
    q_in_question = (q_idx < question_end)
    q_in_plan_xt = (q_idx >= question_end) & (q_idx < plan_xt_end)
    q_in_plan_x0 = (q_idx >= plan_xt_end) & (q_idx < plan_x0_end)
    q_in_exec_xt = (q_idx >= plan_x0_end) & (q_idx < exec_xt_end)
    q_in_exec_x0 = (q_idx >= exec_xt_end)
    
    kv_in_question = (kv_idx < question_end)
    kv_in_plan_xt = (kv_idx >= question_end) & (kv_idx < plan_xt_end)
    kv_in_plan_x0 = (kv_idx >= plan_xt_end) & (kv_idx < plan_x0_end)
    kv_in_exec_xt = (kv_idx >= plan_x0_end) & (kv_idx < exec_xt_end)
    kv_in_exec_x0 = (kv_idx >= exec_xt_end)
    
    # Compute block indices for plan and execution
    plan_block_q = torch.where(
        q_in_plan_x0,
        (q_idx - plan_xt_end) // block_size,
        torch.where(q_in_plan_xt, (q_idx - question_end) // block_size, -1)
    )
    
    plan_block_kv = torch.where(
        kv_in_plan_x0,
        (kv_idx - plan_xt_end) // block_size,
        torch.where(kv_in_plan_xt, (kv_idx - question_end) // block_size, -1)
    )
    
    exec_block_q = torch.where(
        q_in_exec_x0,
        (q_idx - exec_xt_end) // block_size,
        torch.where(q_in_exec_xt, (q_idx - plan_x0_end) // block_size, -1)
    )
    
    exec_block_kv = torch.where(
        kv_in_exec_x0,
        (kv_idx - exec_xt_end) // block_size,
        torch.where(kv_in_exec_xt, (kv_idx - plan_x0_end) // block_size, -1)
    )
    
    # MASK RULES:
    
    # 1. Question tokens can attend to all question tokens (full attention)
    question_mask = q_in_question & kv_in_question
    
    # 2. Plan tokens can attend to:
    #    - Question (all tokens)
    #    - Plan_xt block diagonal
    #    - Plan_x0 with offset block causal + block causal patterns
    plan_to_question = (q_in_plan_xt | q_in_plan_x0) & kv_in_question
    
    plan_block_diagonal = (
        (plan_block_q == plan_block_kv) & 
        (q_in_plan_xt & kv_in_plan_xt | q_in_plan_x0 & kv_in_plan_x0)
    )
    
    plan_offset_block_causal = (
        (plan_block_q > plan_block_kv) &
        kv_in_plan_x0 &
        q_in_plan_xt
    )
    
    plan_block_causal = (
        (plan_block_q >= plan_block_kv) &
        kv_in_plan_x0 &
        q_in_plan_x0
    )
    
    # 3. Execution tokens can attend to:
    #    - Question (all tokens)
    #    - Plan tokens (all - both xt and x0)
    #    - Exec_xt block diagonal
    #    - Exec_x0 with offset block causal + block causal patterns
    exec_to_question = (q_in_exec_xt | q_in_exec_x0) & kv_in_question
    exec_to_plan = (q_in_exec_xt | q_in_exec_x0) & (kv_in_plan_xt | kv_in_plan_x0)
    
    exec_block_diagonal = (
        (exec_block_q == exec_block_kv) &
        (q_in_exec_xt & kv_in_exec_xt | q_in_exec_x0 & kv_in_exec_x0)
    )
    
    exec_offset_block_causal = (
        (exec_block_q > exec_block_kv) &
        kv_in_exec_x0 &
        q_in_exec_xt
    )
    
    exec_block_causal = (
        (exec_block_q >= exec_block_kv) &
        kv_in_exec_x0 &
        q_in_exec_x0
    )
    
    # 4. CAUSAL CONSTRAINT: Plan cannot see Execution
    # (This is implicitly enforced by not adding any exec->plan rules)
    
    # Combine all masks
    final_mask = (
        question_mask |
        plan_to_question | plan_block_diagonal | plan_offset_block_causal | plan_block_causal |
        exec_to_question | exec_to_plan | exec_block_diagonal | exec_offset_block_causal | exec_block_causal
    )
    
    return final_mask


def create_hierarchical_mask(seqlen, block_size, question_len, plan_len, exec_len, attn_backend='sdpa'):
    """
    Helper function to create hierarchical attention mask.
    
    Args:
        seqlen: Total sequence length (question + plan + exec)
        block_size: Block size for diffusion
        question_len: Number of question tokens
        plan_len: Number of plan tokens  
        exec_len: Number of execution tokens
        attn_backend: 'sdpa' or 'flex'
        
    Returns:
        Attention mask tensor
    """
    total_len = question_len + plan_len * 2 + exec_len * 2  # *2 for xt and x0
    
    if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
        mask = create_block_mask(
            partial(
                hierarchical_block_diff_mask,
                question_len=question_len,
                plan_len=plan_len,
                exec_len=exec_len,
                block_size=block_size,
                n=seqlen
            ),
            B=None, H=None, Q_LEN=total_len, KV_LEN=total_len
        )
    elif attn_backend == 'sdpa':
        mask = hierarchical_block_diff_mask(
            b=None, h=None,
            q_idx=torch.arange(total_len)[:, None],
            kv_idx=torch.arange(total_len)[None, :],
            question_len=question_len,
            plan_len=plan_len,
            exec_len=exec_len,
            block_size=block_size,
            n=seqlen
        )
    else:
        raise ValueError(f'Unknown attention backend: {attn_backend}')
    
    return mask
