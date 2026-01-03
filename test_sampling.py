"""
Test script to validate the sampling fix.
Run this BEFORE training to ensure logic is correct.
"""

import torch
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="configs", config_name="config")
def test_sampling_fix(config):
    print("="*80)
    print("🧪 TESTING SAMPLING FIX")
    print("="*80)
    
    # Load model
    from diffusion import Diffusion
    import transformers
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.tokenizer)
    model = Diffusion(config, tokenizer).cuda()
    model.eval()
    
    # Test 1: Cross-attention logic
    print("\n📝 Test 1: Cross-Attention Logic")
    print("-" * 80)
    
    B, L = 2, 512
    x = torch.randint(0, 50000, (B, L)).cuda()
    x[x < 256] = model.mask_index  # Simulate masking
    
    sigma = torch.tensor([[2.0]]).cuda()
    t = sigma * torch.ones(B, 1).cuda()
    
    if model.use_hdp_attention and model.hdp_block_sizes:
        q_len, p_len, e_len = model.hdp_block_sizes
        block_indices = torch.cat([
            torch.zeros(q_len, dtype=torch.long),
            torch.ones(p_len, dtype=torch.long),
            torch.full((e_len,), 2, dtype=torch.long)
        ]).unsqueeze(0).repeat(B, 1).cuda()
    else:
        block_indices = None
    
    # Test forward pass with different x0_pred
    if model.cross_attn:
        # Wrong way (old bug)
        x_wrong = torch.cat((x, x), dim=-1)
        out_wrong = model.forward(x_wrong, sigma, sample_mode=True, block_indices=block_indices)
        
        # Correct way (fixed)
        x0_pred = out_wrong.argmax(dim=-1)
        x_correct = torch.cat((x, x0_pred), dim=-1)
        out_correct = model.forward(x_correct, sigma, sample_mode=True, block_indices=block_indices)
        
        diff = (out_wrong - out_correct).abs().mean().item()
        print(f"   Output difference (wrong vs correct): {diff:.6f}")
        if diff > 0.1:
            print(f"   ✅ PASS: Outputs differ significantly (model sensitive to x0_pred)")
        else:
            print(f"   ⚠️  WARNING: Outputs too similar (model may not use x0_pred)")
    else:
        print(f"   ⏭️  SKIP: cross_attn=False")
    
    # Test 2: Sampling with x0_pred tracking
    print("\n📝 Test 2: Sampling with x0_pred Tracking")
    print("-" * 80)
    
    x_sample = model._sample_prior(B, L).cuda()
    if block_indices is not None:
        # Fill question
        x_sample[:, :q_len] = torch.randint(0, 50000, (B, q_len)).cuda()
    
    dt = torch.tensor([[0.1]]).cuda()
    
    # Test update
    x_new, x0_pred_new = model._analytic_update(
        x=x_sample,
        t=t,
        dt=dt,
        block_indices=block_indices,
        x0_pred=None
    )
    
    print(f"   x_sample.shape: {x_sample.shape}")
    print(f"   x_new.shape: {x_new.shape}")
    print(f"   x0_pred_new: {'None' if x0_pred_new is None else x0_pred_new.shape}")
    
    if model.cross_attn:
        if x0_pred_new is not None:
            print(f"   ✅ PASS: x0_pred tracking works")
        else:
            print(f"   ❌ FAIL: x0_pred is None (should be tensor)")
    else:
        if x0_pred_new is None:
            print(f"   ✅ PASS: x0_pred=None (expected for non-cross_attn)")
        else:
            print(f"   ⚠️  WARNING: x0_pred should be None for non-cross_attn")
    
    # Test 3: Full sampling
    print("\n📝 Test 3: Full Sampling Pipeline")
    print("-" * 80)
    
    try:
        question_text = "What is 2 + 2?"
        question_tokens = tokenizer(
            question_text,
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids'].cuda()
        
        samples = model._analytic_sampler(
            n_samples=1,
            num_steps=10,  # Quick test
            seqlen=L,
            eps=1e-5,
            question_tokens=question_tokens if model.use_hdp_attention else None
        )
        
        if samples is not None:
            print(f"   ✅ PASS: Sampling completed")
            print(f"   Sample shape: {samples.shape}")
            decoded = tokenizer.decode(samples[0], skip_special_tokens=False)
            print(f"   Sample preview: {decoded[:200]}...")
        else:
            print(f"   ⚠️  WARNING: Sampling returned None (stop condition)")
    
    except Exception as e:
        print(f"   ❌ FAIL: Sampling crashed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Block indices shape handling
    print("\n📝 Test 4: Block Indices Shape Handling")
    print("-" * 80)
    
    if block_indices is not None and model.cross_attn:
        try:
            # Test with original shape
            x_test = x.clone()
            out1 = model.forward(x_test, sigma, block_indices=block_indices)
            print(f"   Input shape: {x_test.shape}, block_indices: {block_indices.shape}")
            print(f"   ✅ PASS: Original shape works")
            
            # Test with cross_attn concatenation
            x_test_cat = torch.cat((x, x), dim=-1)
            out2 = model.forward(x_test_cat, sigma, block_indices=block_indices)
            print(f"   Input shape: {x_test_cat.shape}, block_indices: {block_indices.shape}")
            print(f"   ✅ PASS: Cross-attn shape works")
            
        except Exception as e:
            print(f"   ❌ FAIL: Shape handling error: {e}")
    else:
        print(f"   ⏭️  SKIP: No block_indices or cross_attn=False")
    
    print("\n" + "="*80)
    print("🎉 TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_sampling_fix()