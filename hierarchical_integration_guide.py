"""
Integration Guide: Hierarchical Attention with Block Diffusion

This file demonstrates how to integrate the hierarchical attention mechanism
into the Block Diffusion (BD3-LM) architecture for mathematical reasoning.

Author: Research Implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from models.hierarchical_attention import (
    get_hierarchical_mask,
    get_hierarchical_attention_bias,
    HierarchicalAttentionWrapper,
)


# ============================================================================
# Example 1: Injecting Hierarchical Mask into DiT (Diffusion Transformer)
# ============================================================================

def modify_dit_attention_for_hierarchical(
    dit_model: nn.Module,
    use_hierarchical_mask: bool = True,
) -> nn.Module:
    """
    Modify a DiT model to use hierarchical attention masks.
    
    This wraps all attention layers in the DiT backbone with hierarchical masking.
    
    Args:
        dit_model: The DiT model (from models/dit.py)
        use_hierarchical_mask: Whether to enable hierarchical masking
        
    Returns:
        Modified DiT model with hierarchical attention
    """
    # Assuming DiT has transformer blocks with self-attention
    # We need to wrap each attention module
    
    for name, module in dit_model.named_modules():
        # Look for attention layers (adjust based on actual DiT implementation)
        if 'attn' in name.lower() or 'attention' in name.lower():
            # Wrap with hierarchical attention
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            parent = dit_model.get_submodule(parent_name) if parent_name else dit_model
            original_attn = getattr(parent, attr_name)
            
            wrapped_attn = HierarchicalAttentionWrapper(
                original_attn,
                use_hierarchical_mask=use_hierarchical_mask,
            )
            
            setattr(parent, attr_name, wrapped_attn)
            print(f"Wrapped attention layer: {name}")
    
    return dit_model


# ============================================================================
# Example 2: Custom Transformer Block with Hierarchical Attention
# ============================================================================

class HierarchicalTransformerBlock(nn.Module):
    """
    Custom transformer block with built-in hierarchical attention support.
    
    This can replace standard transformer blocks in BD3-LM.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_hierarchical_mask: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_hierarchical_mask = use_hierarchical_mask
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        block_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional hierarchical masking.
        
        Args:
            x: (batch_size, seq_len, hidden_size)
            block_indices: (batch_size, seq_len) - block assignments
            attention_mask: Optional standard attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        # Generate hierarchical mask if enabled
        if self.use_hierarchical_mask and block_indices is not None:
            hierarchical_bias = get_hierarchical_attention_bias(
                block_indices,
                attention_type="bidirectional_within_blocks",
                device=x.device,
            )  # (batch_size, seq_len, seq_len)
            
            # nn.MultiheadAttention with batch_first=True expects:
            # 2D mask: (seq_len, seq_len) broadcast across batch
            # 3D mask: (batch * num_heads, seq_len, seq_len)
            # We'll use 2D mask which is simpler
            # Take first sample's mask (they should all be the same for hierarchical structure)
            attn_mask = hierarchical_bias[0]  # (seq_len, seq_len)
            
            # Combine with existing mask if provided
            if attention_mask is not None:
                attn_mask = attn_mask + attention_mask
        else:
            attn_mask = attention_mask
        
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


# ============================================================================
# Example 3: Patching BD3-LM Forward Pass
# ============================================================================

def hierarchical_bd3lm_forward_wrapper(
    original_forward,
    use_hierarchical_mask: bool = True,
):
    """
    Wrapper for BD3-LM forward pass to inject hierarchical masking.
    
    Usage:
        model.forward = hierarchical_bd3lm_forward_wrapper(
            model.forward,
            use_hierarchical_mask=True
        )
    """
    def forward(
        x: torch.Tensor,
        block_indices: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Inject hierarchical mask into kwargs
        if use_hierarchical_mask and block_indices is not None:
            hierarchical_bias = get_hierarchical_attention_bias(
                block_indices,
                attention_type="bidirectional_within_blocks",
                device=x.device,
            )
            
            # Add to attention_mask in kwargs
            if 'attention_mask' in kwargs:
                kwargs['attention_mask'] = kwargs['attention_mask'] + hierarchical_bias
            else:
                kwargs['attention_mask'] = hierarchical_bias
        
        # Call original forward
        return original_forward(x, **kwargs)
    
    return forward


# ============================================================================
# Example 4: Training Loop Integration
# ============================================================================

def train_step_with_hierarchical_attention(
    model: nn.Module,
    batch: dict,
    criterion: nn.Module,
    use_hierarchical_mask: bool = True,
) -> torch.Tensor:
    """
    Example training step with hierarchical attention.
    
    Args:
        model: BD3-LM model
        batch: Batch from hierarchical dataloader containing:
            - input_ids: (batch_size, seq_len)
            - attention_mask: (batch_size, seq_len)
            - block_indices: (batch_size, seq_len)
        criterion: Loss function
        use_hierarchical_mask: Enable hierarchical masking
        
    Returns:
        Loss tensor
    """
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    block_indices = batch['block_indices']
    
    # Forward pass with hierarchical mask
    if use_hierarchical_mask:
        # Generate hierarchical attention bias
        hierarchical_bias = get_hierarchical_attention_bias(
            block_indices,
            attention_type="bidirectional_within_blocks",
            device=input_ids.device,
        )
        
        # Combine with standard attention mask
        combined_mask = attention_mask.unsqueeze(1) + hierarchical_bias
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=combined_mask,
        )
    else:
        # Standard forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    
    # Compute loss
    loss = criterion(outputs, input_ids)
    
    return loss


# ============================================================================
# Example 5: Complete Training Script Template
# ============================================================================

def train_hierarchical_bd3lm():
    """
    Complete example of training BD3-LM with hierarchical attention on GSM8K.
    """
    import transformers
    from torch.utils.data import DataLoader
    from hierarchical_gsm8k_dataset import GSM8KHierarchicalDataset, collate_hierarchical_batch
    
    # Configuration
    config = {
        'model_name': 'gpt2',
        'hidden_size': 768,
        'num_heads': 12,
        'num_layers': 12,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'num_epochs': 10,
        'question_len': 128,
        'plan_len': 128,
        'exec_len': 256,
        'use_hierarchical_mask': True,
    }
    
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader
    train_dataset = GSM8KHierarchicalDataset(
        data_path='data/gsm8k/gsm8k_hierarchical_train.json',
        tokenizer=tokenizer,
        question_len=config['question_len'],
        plan_len=config['plan_len'],
        exec_len=config['exec_len'],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_hierarchical_batch,
        num_workers=4,
    )
    
    # Initialize model (replace with actual BD3-LM model)
    # model = BD3LM(config)
    # model = modify_dit_attention_for_hierarchical(model, use_hierarchical_mask=True)
    
    # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("Starting training with hierarchical attention...")
    
    # for epoch in range(config['num_epochs']):
    #     for batch_idx, batch in enumerate(train_loader):
    #         # Move to device
    #         input_ids = batch['input_ids'].to(device)
    #         block_indices = batch['block_indices'].to(device)
    #         
    #         # Training step
    #         loss = train_step_with_hierarchical_attention(
    #             model, batch, criterion,
    #             use_hierarchical_mask=config['use_hierarchical_mask']
    #         )
    #         
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         
    #         if batch_idx % 100 == 0:
    #             print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print("Training template created. Uncomment and adapt to your model.")


# ============================================================================
# Example 6: Inference with Hierarchical Attention
# ============================================================================

@torch.no_grad()
def generate_with_hierarchical_blocks(
    model: nn.Module,
    question_ids: torch.Tensor,
    tokenizer,
    question_len: int = 128,
    plan_len: int = 128,
    exec_len: int = 256,
    num_diffusion_steps: int = 1000,
) -> dict:
    """
    Generate plan and execution using hierarchical block diffusion.
    
    This demonstrates a 2-stage generation process:
    1. Generate Plan block conditioned on Question
    2. Generate Execution block conditioned on Question + Plan
    
    Args:
        model: Trained BD3-LM model
        question_ids: (batch_size, question_len) tokenized question
        tokenizer: Tokenizer for decoding
        question_len: Length of question block
        plan_len: Length of plan block
        exec_len: Length of execution block
        num_diffusion_steps: Number of diffusion sampling steps
        
    Returns:
        Dictionary with generated text
    """
    batch_size = question_ids.shape[0]
    device = question_ids.device
    seq_len = question_len + plan_len + exec_len
    
    # Initialize full sequence with question + random noise for plan and execution
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    input_ids[:, :question_len] = question_ids
    
    # Random initialization for plan and execution
    vocab_size = len(tokenizer)
    input_ids[:, question_len:] = torch.randint(0, vocab_size, (batch_size, plan_len + exec_len), device=device)
    
    # Create block indices
    block_indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    block_indices[:, question_len:question_len+plan_len] = 1  # Plan
    block_indices[:, question_len+plan_len:] = 2  # Execution
    
    # Stage 1: Generate Plan block using diffusion
    print("Generating Plan block...")
    for t in range(num_diffusion_steps, 0, -1):
        # Get hierarchical mask
        hierarchical_bias = get_hierarchical_attention_bias(
            block_indices,
            device=device,
        )
        
        # Diffusion denoising step (pseudo-code - adapt to actual BD3-LM)
        # outputs = model.denoise_step(input_ids, t, attention_mask=hierarchical_bias)
        # input_ids[:, question_len:question_len+plan_len] = outputs[:, question_len:question_len+plan_len]
        pass
    
    # Stage 2: Generate Execution block conditioned on Question + Plan
    print("Generating Execution block...")
    # Plan is now fixed, only denoise Execution block
    # Similar diffusion process but only updating execution tokens
    
    # Decode results
    results = {
        'question': tokenizer.decode(input_ids[0, :question_len]),
        'plan': tokenizer.decode(input_ids[0, question_len:question_len+plan_len]),
        'execution': tokenizer.decode(input_ids[0, question_len+plan_len:]),
        'full_sequence': tokenizer.decode(input_ids[0]),
    }
    
    return results


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Hierarchical Attention Integration Examples")
    print("=" * 80)
    
    # Test hierarchical transformer block
    print("\nTesting HierarchicalTransformerBlock...")
    batch_size, seq_len, hidden_size = 2, 512, 768
    
    block = HierarchicalTransformerBlock(
        hidden_size=hidden_size,
        num_heads=12,
        use_hierarchical_mask=True,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    block_indices = torch.zeros(batch_size, seq_len, dtype=torch.long)
    block_indices[:, 128:256] = 1  # Plan
    block_indices[:, 256:] = 2      # Execution
    
    output = block(x, block_indices=block_indices)
    print(f"✅ HierarchicalTransformerBlock output shape: {output.shape}")
    
    print("\n" + "=" * 80)
    print("Integration guide complete!")
    print("See function docstrings for usage examples.")
    print("=" * 80)
