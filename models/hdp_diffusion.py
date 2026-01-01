"""
HDP-Diffusion Model: Hierarchical Dual-Process Diffusion for Mathematical Reasoning

Integrates custom hierarchical attention masks with Block Diffusion framework.

Key Innovation:
- 3-block structure: Question (context) -> Plan (reasoning) -> Execution (computation)
- Custom attention: Plan cannot see Execution (preserves causal reasoning)
- Compatible with BD3-LM training pipeline

Author: Research implementation for HDP-Diffusion
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

from models.hdp_attention_mask import (
    get_hdp_attention_mask,
    get_hdp_attention_bias,
    get_block_indices
)

logger = logging.getLogger(__name__)


class HDPDiffusionWrapper(nn.Module):
    """
    Wrapper that adds HDP attention to any transformer backbone.
    
    This can wrap DiT, GPT, or other transformer models used in BD3-LM.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        block_sizes: Tuple[int, int, int] = (128, 128, 256),
        use_hdp_attention: bool = True,
        causal_within_block: bool = False
    ):
        """
        Args:
            backbone: Base transformer model (e.g., DiT, GPT)
            block_sizes: (question_len, plan_len, exec_len)
            use_hdp_attention: If False, use standard attention (for ablation)
            causal_within_block: Whether to use causal mask within blocks
        """
        super().__init__()
        self.backbone = backbone
        self.block_sizes = block_sizes
        self.use_hdp_attention = use_hdp_attention
        self.causal_within_block = causal_within_block
        
        self.seq_len = sum(block_sizes)
        
        logger.info(f"Initialized HDP-Diffusion wrapper:")
        logger.info(f"  Block sizes: {block_sizes}")
        logger.info(f"  Total seq len: {self.seq_len}")
        logger.info(f"  HDP attention: {use_hdp_attention}")
    
    def _prepare_hdp_mask(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate HDP attention bias for the batch."""
        # Generate block indices
        block_indices = get_block_indices(
            batch_size, 
            self.block_sizes, 
            device=device
        )
        
        # Generate attention bias
        attention_bias = get_hdp_attention_bias(
            block_indices,
            self.seq_len,
            self.block_sizes,
            self.causal_within_block,
            device=device,
            dtype=dtype
        )
        
        return attention_bias
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with HDP attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim) or (batch_size, seq_len)
            timesteps: Diffusion timesteps (batch_size,)
            **kwargs: Additional arguments for backbone
        
        Returns:
            Output tensor (same shape as input)
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype if x.dtype in [torch.float16, torch.float32] else torch.float32
        
        # Generate HDP attention mask if enabled
        if self.use_hdp_attention:
            attention_bias = self._prepare_hdp_mask(batch_size, device, dtype)
            
            # Add to kwargs for backbone
            # Different backbones may use different arg names
            if 'attention_mask' not in kwargs:
                kwargs['attention_mask'] = attention_bias
            if 'attn_bias' not in kwargs:
                kwargs['attn_bias'] = attention_bias
        
        # Forward through backbone
        output = self.backbone(x, timesteps=timesteps, **kwargs)
        
        return output


class HDPDiffusionConfig:
    """Configuration for HDP-Diffusion model."""
    
    def __init__(
        self,
        # Block structure
        question_len: int = 128,
        plan_len: int = 128,
        exec_len: int = 256,
        
        # Attention
        use_hdp_attention: bool = True,
        causal_within_block: bool = False,
        
        # Model
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        
        # Diffusion
        num_diffusion_steps: int = 1000,
        vocab_size: int = 50257,
        
        **kwargs
    ):
        self.question_len = question_len
        self.plan_len = plan_len
        self.exec_len = exec_len
        self.seq_len = question_len + plan_len + exec_len
        
        self.use_hdp_attention = use_hdp_attention
        self.causal_within_block = causal_within_block
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.num_diffusion_steps = num_diffusion_steps
        self.vocab_size = vocab_size
        
        # Store extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question_len': self.question_len,
            'plan_len': self.plan_len,
            'exec_len': self.exec_len,
            'seq_len': self.seq_len,
            'use_hdp_attention': self.use_hdp_attention,
            'causal_within_block': self.causal_within_block,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_diffusion_steps': self.num_diffusion_steps,
            'vocab_size': self.vocab_size
        }


def create_hdp_model_from_backbone(
    backbone: nn.Module,
    config: HDPDiffusionConfig
) -> HDPDiffusionWrapper:
    """
    Create HDP-Diffusion model from an existing backbone.
    
    Args:
        backbone: Base transformer (e.g., from BD3-LM)
        config: HDP configuration
    
    Returns:
        Wrapped model with HDP attention
    """
    model = HDPDiffusionWrapper(
        backbone=backbone,
        block_sizes=(config.question_len, config.plan_len, config.exec_len),
        use_hdp_attention=config.use_hdp_attention,
        causal_within_block=config.causal_within_block
    )
    
    return model


if __name__ == "__main__":
    print("Testing HDP-Diffusion Model...")
    
    # Create a dummy backbone (simple transformer)
    class DummyBackbone(nn.Module):
        def __init__(self, seq_len, hidden_size):
            super().__init__()
            self.embed = nn.Embedding(50257, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    batch_first=True
                ),
                num_layers=2
            )
            self.out = nn.Linear(hidden_size, 50257)
        
        def forward(self, x, timesteps=None, attention_mask=None, **kwargs):
            # x: (batch, seq_len) token ids
            if x.dim() == 2:
                x = self.embed(x)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert to format expected by nn.Transformer
                # (batch, seq_len, seq_len) -> (batch * num_heads, seq_len, seq_len)
                src_mask = attention_mask.repeat_interleave(8, dim=0)
            else:
                src_mask = None
            
            x = self.transformer(x, mask=src_mask)
            return self.out(x)
    
    # Create config
    config = HDPDiffusionConfig(
        question_len=128,
        plan_len=128,
        exec_len=256,
        hidden_size=256,
        use_hdp_attention=True
    )
    
    # Create backbone
    backbone = DummyBackbone(config.seq_len, config.hidden_size)
    
    # Wrap with HDP
    model = create_hdp_model_from_backbone(backbone, config)
    
    # Test forward pass
    batch_size = 2
    x = torch.randint(0, 50257, (batch_size, config.seq_len))
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    
    with torch.no_grad():
        output = model(x, timesteps)
    
    print(f"Output shape: {output.shape}")
    print(f"Output type: {output.dtype}")
    
    # Test ablation (no HDP attention)
    model_no_hdp = HDPDiffusionWrapper(
        backbone,
        block_sizes=(128, 128, 256),
        use_hdp_attention=False
    )
    
    with torch.no_grad():
        output_no_hdp = model_no_hdp(x, timesteps)
    
    print(f"\nAblation output shape: {output_no_hdp.shape}")
    
    print("\n✅ HDP-Diffusion model tests passed!")
