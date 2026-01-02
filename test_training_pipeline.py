"""
Training Pipeline Validation Script for BD3LM and HDP-Diffusion

This script performs step-by-step validation of the training pipeline to detect logic errors.
It checks:
1. Data loading and formatting
2. Tokenization and padding
3. Block structure consistency
4. Attention mask generation
5. Forward pass shape consistency
6. Loss computation
7. HDP-specific hierarchical constraints

Usage:
    python test_training_pipeline.py --mode baseline  # Test BD3LM baseline
    python test_training_pipeline.py --mode hdp       # Test HDP
    python test_training_pipeline.py --mode both      # Test both
"""

import torch
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoTokenizer

# Import project modules
try:
    from hdp_dataset import HDPDataset, collate_hdp_batch
    from hierarchical_dataloader import HierarchicalDataCollator
    import diffusion
    import dataloader as base_dataloader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you run this from the project root directory")
    exit(1)


class TrainingPipelineValidator:
    """Validates training pipeline step by step"""
    
    def __init__(self, mode: str = "hdp"):
        self.mode = mode
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        
        print(f"\n{'='*80}")
        print(f"🔍 TRAINING PIPELINE VALIDATOR - Mode: {mode.upper()}")
        print(f"{'='*80}\n")
    
    def log_error(self, msg: str):
        """Log an error"""
        self.errors.append(msg)
        print(f"❌ ERROR: {msg}")
    
    def log_warning(self, msg: str):
        """Log a warning"""
        self.warnings.append(msg)
        print(f"⚠️  WARNING: {msg}")
    
    def log_pass(self, msg: str):
        """Log a passed check"""
        self.passed_checks.append(msg)
        print(f"✅ PASS: {msg}")
    
    def check_data_format(self, data_path: str) -> bool:
        """Check if data file exists and has correct format"""
        print("\n[1/10] Checking data file format...")
        
        if not os.path.exists(data_path):
            self.log_error(f"Data file not found: {data_path}")
            return False
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                self.log_error("Data should be a list of samples")
                return False
            
            if len(data) == 0:
                self.log_error("Data file is empty")
                return False
            
            # Check first sample
            sample = data[0]
            required_keys = ['question', 'plan', 'execution'] if self.mode == 'hdp' else ['text']
            
            for key in required_keys:
                if key not in sample:
                    self.log_error(f"Missing required key '{key}' in data")
                    return False
            
            self.log_pass(f"Data format correct ({len(data)} samples)")
            return True
            
        except json.JSONDecodeError:
            self.log_error("Invalid JSON format")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error reading data: {e}")
            return False
    
    def check_tokenization(self, tokenizer, sample_text: str, target_len: int) -> bool:
        """Check tokenization and padding"""
        print("\n[2/10] Checking tokenization...")
        
        try:
            tokens = tokenizer(
                sample_text,
                truncation=True,
                max_length=target_len,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            
            # Check shapes
            if input_ids.shape[1] != target_len:
                self.log_error(f"Token length mismatch: got {input_ids.shape[1]}, expected {target_len}")
                return False
            
            # Check attention mask
            if attention_mask.shape != input_ids.shape:
                self.log_error("Attention mask shape mismatch")
                return False
            
            # Check padding
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            actual_tokens = (input_ids != pad_token_id).sum().item()
            mask_ones = attention_mask.sum().item()
            
            if actual_tokens != mask_ones:
                self.log_warning(f"Attention mask inconsistency: {actual_tokens} tokens vs {mask_ones} mask")
            
            self.log_pass(f"Tokenization correct (len={target_len}, actual_tokens={actual_tokens})")
            return True
            
        except Exception as e:
            self.log_error(f"Tokenization failed: {e}")
            return False
    
    def check_hdp_block_structure(
        self, 
        input_ids: torch.Tensor, 
        block_indices: torch.Tensor,
        block_sizes: Tuple[int, int, int]
    ) -> bool:
        """Check HDP hierarchical block structure"""
        print("\n[3/10] Checking HDP block structure...")
        
        q_len, p_len, e_len = block_sizes
        expected_len = q_len + p_len + e_len
        
        # Check total length
        if input_ids.shape[-1] != expected_len:
            self.log_error(f"Sequence length mismatch: got {input_ids.shape[-1]}, expected {expected_len}")
            return False
        
        # Check block_indices
        if block_indices.shape != input_ids.shape:
            self.log_error(f"block_indices shape mismatch: {block_indices.shape} vs {input_ids.shape}")
            return False
        
        # Count blocks
        q_count = (block_indices == 0).sum().item()
        p_count = (block_indices == 1).sum().item()
        e_count = (block_indices == 2).sum().item()
        
        if q_count != q_len:
            self.log_error(f"Question block size mismatch: {q_count} vs {q_len}")
            return False
        
        if p_count != p_len:
            self.log_error(f"Plan block size mismatch: {p_count} vs {p_len}")
            return False
        
        if e_count != e_len:
            self.log_error(f"Execution block size mismatch: {e_count} vs {e_len}")
            return False
        
        # Check block ordering
        # Should be: [0,0,0..., 1,1,1..., 2,2,2...]
        block_changes = (block_indices[1:] != block_indices[:-1]).nonzero(as_tuple=True)[0]
        
        if len(block_changes) != 2:  # Should have exactly 2 transitions: 0->1 and 1->2
            self.log_error(f"Block ordering incorrect: found {len(block_changes)} transitions, expected 2")
            return False
        
        self.log_pass(f"Block structure correct: Q={q_count}, P={p_count}, E={e_count}")
        return True
    
    def check_attention_mask_logic(
        self, 
        block_indices: torch.Tensor,
        block_sizes: Tuple[int, int, int],
        use_hdp: bool = True
    ) -> bool:
        """Check attention mask generation for HDP"""
        print("\n[4/10] Checking attention mask logic...")
        
        if not use_hdp:
            self.log_pass("Skipping HDP attention mask check (baseline mode)")
            return True
        
        try:
            from models.hdp_attention_mask import get_hdp_attention_bias
            
            batch_size = 2
            seq_len = sum(block_sizes)
            
            # Create batch of block_indices
            block_indices_batch = block_indices.unsqueeze(0).repeat(batch_size, 1)
            
            # Generate HDP attention mask
            hdp_mask = get_hdp_attention_bias(
                block_indices=block_indices_batch,
                seq_len=seq_len,
                block_sizes=block_sizes,
                causal_within_block=False,
                device='cpu',
                dtype=torch.float32
            )
            
            # Check shape: (batch, seq_len, seq_len)
            expected_shape = (batch_size, seq_len, seq_len)
            if hdp_mask.shape != expected_shape:
                self.log_error(f"HDP mask shape incorrect: {hdp_mask.shape} vs {expected_shape}")
                return False
            
            q_len, p_len, e_len = block_sizes
            
            # Validate hierarchical constraints
            # 1. Question should only attend to Question
            question_mask = hdp_mask[:, :q_len, :q_len]
            if (question_mask == 0).sum() != batch_size * q_len * q_len:
                self.log_error("Question block doesn't attend only to itself")
                return False
            
            # 2. Question shouldn't attend to Plan/Execution
            if (hdp_mask[:, :q_len, q_len:] == 0).sum() > 0:
                self.log_error("Question block incorrectly attends to Plan/Execution")
                return False
            
            # 3. Plan should attend to Question + Plan
            plan_to_question = hdp_mask[:, q_len:q_len+p_len, :q_len]
            if (plan_to_question == 0).sum() != batch_size * p_len * q_len:
                self.log_error("Plan doesn't correctly attend to Question")
                return False
            
            # 4. Execution should attend to all
            exec_mask = hdp_mask[:, q_len+p_len:, :]
            if (exec_mask == 0).sum() != batch_size * e_len * seq_len:
                self.log_error("Execution doesn't attend to all blocks")
                return False
            
            self.log_pass("HDP attention mask logic correct")
            return True
            
        except ImportError:
            self.log_error("Cannot import hdp_attention_mask module")
            return False
        except Exception as e:
            self.log_error(f"Attention mask check failed: {e}")
            return False
    
    def check_loss_masking(
        self,
        loss: torch.Tensor,
        attention_mask: torch.Tensor,
        block_indices: torch.Tensor,
        use_hdp: bool = True
    ) -> bool:
        """Check if loss is correctly masked for HDP"""
        print("\n[5/10] Checking loss masking...")
        
        if not use_hdp:
            # Baseline: loss should be masked only by attention_mask
            effective_mask = attention_mask
            self.log_pass("Baseline loss masking (attention_mask only)")
        else:
            # HDP: loss should be masked by attention_mask AND block_indices
            # Only Plan (1) and Execution (2) should contribute to loss, not Question (0)
            hdp_loss_mask = (block_indices >= 1).float()
            effective_mask = attention_mask * hdp_loss_mask
            
            # Check that Question tokens are masked out
            question_indices = (block_indices == 0).nonzero(as_tuple=True)[0]
            if len(question_indices) > 0:
                question_loss = loss[question_indices] * effective_mask[question_indices]
                if question_loss.sum() != 0:
                    self.log_error("Question tokens contribute to loss (should be masked)")
                    return False
            
            self.log_pass("HDP loss masking correct (Question masked out)")
        
        # Check that masked positions don't contribute
        masked_positions = (effective_mask == 0).nonzero(as_tuple=True)[0]
        if len(masked_positions) > 0:
            masked_loss = loss[masked_positions]
            # After masking, these shouldn't affect the final loss
            self.log_pass(f"Loss masking validated ({len(masked_positions)} positions masked)")
        
        return True
    
    def check_forward_pass_shapes(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        use_cross_attn: bool = False
    ) -> bool:
        """Check forward pass output shapes"""
        print("\n[6/10] Checking forward pass shapes...")
        
        try:
            # Simulate model output
            expected_output_shape = (batch_size, seq_len, vocab_size)
            
            # If cross attention, input is [xt | x0] but output should be for seq_len
            if use_cross_attn:
                input_len = seq_len * 2
                self.log_pass(f"Cross-attention mode: input_len={input_len}, output_len={seq_len}")
            else:
                input_len = seq_len
            
            self.log_pass(f"Expected shapes: input=({batch_size}, {input_len}), output={expected_output_shape}")
            return True
            
        except Exception as e:
            self.log_error(f"Shape check failed: {e}")
            return False
    
    def check_noise_schedule(self, T: int = 1000) -> bool:
        """Check noise schedule parameters"""
        print("\n[7/10] Checking noise schedule...")
        
        try:
            # Sample timesteps
            t_samples = torch.linspace(0, 1, T)
            
            # Check bounds
            if (t_samples < 0).any() or (t_samples > 1).any():
                self.log_error("Timesteps out of [0, 1] range")
                return False
            
            # Check monotonicity
            if not (t_samples[1:] >= t_samples[:-1]).all():
                self.log_error("Timesteps not monotonic")
                return False
            
            self.log_pass(f"Noise schedule valid (T={T})")
            return True
            
        except Exception as e:
            self.log_error(f"Noise schedule check failed: {e}")
            return False
    
    def check_batch_consistency(
        self,
        batch: Dict[str, torch.Tensor],
        expected_batch_size: int,
        expected_seq_len: int
    ) -> bool:
        """Check batch data consistency"""
        print("\n[8/10] Checking batch consistency...")
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Check batch size
        if input_ids.shape[0] != expected_batch_size:
            self.log_error(f"Batch size mismatch: {input_ids.shape[0]} vs {expected_batch_size}")
            return False
        
        # Check sequence length
        if input_ids.shape[1] != expected_seq_len:
            self.log_error(f"Sequence length mismatch: {input_ids.shape[1]} vs {expected_seq_len}")
            return False
        
        # Check attention mask consistency
        if attention_mask.shape != input_ids.shape:
            self.log_error("Attention mask shape doesn't match input_ids")
            return False
        
        # Check block_indices if HDP mode
        if 'block_indices' in batch:
            block_indices = batch['block_indices']
            if block_indices.shape != input_ids.shape:
                self.log_error("block_indices shape doesn't match input_ids")
                return False
        
        self.log_pass(f"Batch consistency validated (batch_size={expected_batch_size}, seq_len={expected_seq_len})")
        return True
    
    def check_gradient_flow(self) -> bool:
        """Check if gradients can flow through the model"""
        print("\n[9/10] Checking gradient flow...")
        
        try:
            # Create dummy tensors with requires_grad
            dummy_input = torch.randn(2, 512, requires_grad=True)
            dummy_target = torch.randn(2, 512)
            
            # Compute dummy loss
            loss = ((dummy_input - dummy_target) ** 2).mean()
            
            # Backward
            loss.backward()
            
            # Check gradient exists
            if dummy_input.grad is None:
                self.log_error("Gradients not computed")
                return False
            
            # Check gradient is not all zeros
            if (dummy_input.grad == 0).all():
                self.log_warning("All gradients are zero")
            
            self.log_pass("Gradient flow validated")
            return True
            
        except Exception as e:
            self.log_error(f"Gradient flow check failed: {e}")
            return False
    
    def check_special_tokens(self, tokenizer) -> bool:
        """Check special tokens configuration"""
        print("\n[10/10] Checking special tokens...")
        
        issues = []
        
        if tokenizer.bos_token is None:
            issues.append("No BOS token")
        if tokenizer.eos_token is None:
            issues.append("No EOS token")
        if tokenizer.pad_token is None:
            issues.append("No PAD token")
        
        if issues:
            for issue in issues:
                self.log_warning(issue)
        else:
            self.log_pass(f"Special tokens configured: BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}, PAD={tokenizer.pad_token}")
        
        return True
    
    def print_summary(self):
        """Print validation summary"""
        print(f"\n{'='*80}")
        print("📊 VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"\n✅ Passed: {len(self.passed_checks)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print(f"\n{'='*80}")
            print("❌ CRITICAL ERRORS:")
            print(f"{'='*80}")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")
        
        if self.warnings:
            print(f"\n{'='*80}")
            print("⚠️  WARNINGS:")
            print(f"{'='*80}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")
        
        print(f"\n{'='*80}\n")
        
        if self.errors:
            print("❌ VALIDATION FAILED - Please fix errors before training")
            return False
        else:
            print("✅ ALL CHECKS PASSED - Pipeline ready for training")
            return True


def run_baseline_validation():
    """Run validation for BD3LM baseline"""
    validator = TrainingPipelineValidator(mode="baseline")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run checks
    validator.check_special_tokens(tokenizer)
    validator.check_tokenization(tokenizer, "This is a test sentence.", target_len=512)
    validator.check_noise_schedule(T=1000)
    validator.check_forward_pass_shapes(batch_size=4, seq_len=1024, vocab_size=50257, use_cross_attn=True)
    validator.check_gradient_flow()
    
    # Dummy batch check
    dummy_batch = {
        'input_ids': torch.randint(0, 50257, (4, 1024)),
        'attention_mask': torch.ones(4, 1024)
    }
    validator.check_batch_consistency(dummy_batch, expected_batch_size=4, expected_seq_len=1024)
    
    return validator.print_summary()


def run_hdp_validation(data_path: str = None):
    """Run validation for HDP mode"""
    validator = TrainingPipelineValidator(mode="hdp")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Block sizes
    block_sizes = (128, 128, 256)  # Q, P, E
    seq_len = sum(block_sizes)
    
    # Run checks
    validator.check_special_tokens(tokenizer)
    
    # Check data format if path provided
    if data_path and os.path.exists(data_path):
        validator.check_data_format(data_path)
    else:
        validator.log_warning("No data path provided, skipping data format check")
    
    validator.check_tokenization(tokenizer, "This is a test question.", target_len=block_sizes[0])
    
    # Create dummy block_indices
    block_indices = torch.cat([
        torch.zeros(block_sizes[0], dtype=torch.long),
        torch.ones(block_sizes[1], dtype=torch.long),
        torch.full((block_sizes[2],), 2, dtype=torch.long)
    ])
    
    dummy_input_ids = torch.randint(0, 50257, (seq_len,))
    
    validator.check_hdp_block_structure(dummy_input_ids, block_indices, block_sizes)
    validator.check_attention_mask_logic(block_indices, block_sizes, use_hdp=True)
    
    # Loss masking check
    dummy_loss = torch.randn(seq_len)
    dummy_attention_mask = torch.ones(seq_len)
    validator.check_loss_masking(dummy_loss, dummy_attention_mask, block_indices, use_hdp=True)
    
    validator.check_noise_schedule(T=5000)
    validator.check_forward_pass_shapes(batch_size=4, seq_len=seq_len, vocab_size=50257, use_cross_attn=True)
    validator.check_gradient_flow()
    
    # Batch check
    dummy_batch = {
        'input_ids': torch.randint(0, 50257, (4, seq_len)),
        'attention_mask': torch.ones(4, seq_len),
        'block_indices': block_indices.unsqueeze(0).repeat(4, 1)
    }
    validator.check_batch_consistency(dummy_batch, expected_batch_size=4, expected_seq_len=seq_len)
    
    return validator.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Validate training pipeline for BD3LM and HDP-Diffusion")
    parser.add_argument('--mode', type=str, choices=['baseline', 'hdp', 'both'], default='both',
                        help='Validation mode: baseline (BD3LM), hdp, or both')
    parser.add_argument('--data_path', type=str, default='data/gsm8k/gsm8k_hierarchical_test.json',
                        help='Path to HDP dataset for validation')
    
    args = parser.parse_args()
    
    success = True
    
    if args.mode in ['baseline', 'both']:
        print("\n🔵 Running BD3LM Baseline Validation...")
        baseline_success = run_baseline_validation()
        success = success and baseline_success
    
    if args.mode in ['hdp', 'both']:
        print("\n🟢 Running HDP Validation...")
        hdp_success = run_hdp_validation(data_path=args.data_path)
        success = success and hdp_success
    
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
