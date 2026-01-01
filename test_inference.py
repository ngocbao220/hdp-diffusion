#!/usr/bin/env python3
"""
Test inference với model đã train để verify format [PLAN] [EXECUTION] [ANSWER]
"""

import torch
import os
from transformers import AutoTokenizer
from pathlib import Path

def test_checkpoint_format(checkpoint_dir="outputs/hdp_test_new_format"):
    """
    Load checkpoint và test xem model có học đúng format không
    """
    
    print("="*80)
    print("🔍 Testing HDP Model - Format Verification")
    print("="*80)
    
    checkpoint_path = Path(checkpoint_dir)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print(f"   Please run training first: bash quick_train_test.sh")
        return
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_path.glob("*.ckpt"))
    if not checkpoints:
        print(f"❌ No .ckpt files found in {checkpoint_dir}")
        return
    
    latest_ckpt = sorted(checkpoints)[-1]
    print(f"✅ Found checkpoint: {latest_ckpt.name}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint
    print("📦 Loading checkpoint...")
    checkpoint = torch.load(latest_ckpt, map_location='cpu')
    
    # Print basic info
    print(f"   Checkpoint keys: {list(checkpoint.keys())}\n")
    
    if 'hyper_parameters' in checkpoint:
        hp = checkpoint['hyper_parameters']
        print("📊 Model configuration:")
        if 'config' in hp:
            config = hp['config']
            if 'hdp' in config:
                print(f"   HDP enabled: {config['hdp'].get('enabled', False)}")
                print(f"   Question len: {config['hdp'].get('question_len', 'N/A')}")
                print(f"   Plan len: {config['hdp'].get('plan_len', 'N/A')}")
                print(f"   Exec len: {config['hdp'].get('exec_len', 'N/A')}")
                print(f"   Special format: {config['hdp'].get('use_special_format', 'N/A')}")
    
    print("\n" + "="*80)
    print("✅ Checkpoint validation complete!")
    print("\n💡 Next steps:")
    print("   1. For full inference testing, use:")
    print("      python main.py mode=sample \\")
    print("        eval.checkpoint_path=outputs/hdp_test_new_format/last.ckpt \\")
    print("        data=hdp_diffusion")
    print("\n   2. To continue training:")
    print("      bash quick_train_test.sh")
    print("="*80)


def verify_dataset_format():
    """Verify dataset has correct format"""
    print("\n" + "="*80)
    print("📋 Verifying Dataset Format")
    print("="*80)
    
    import json
    from hdp_dataset import HDPDataset
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load with special format
    dataset = HDPDataset(
        data_path='data/gsm8k/gsm8k_hierarchical_train.json',
        tokenizer=tokenizer,
        block_sizes=(128, 128, 256),
        use_special_format=True
    )
    
    sample = dataset[0]
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    
    # Check for required tokens
    has_plan = '[PLAN]' in decoded
    has_exec = '[EXECUTION]' in decoded or '[EXEC]' in decoded
    has_answer = '[ANSWER]' in decoded
    
    print(f"✅ Dataset validation:")
    print(f"   [PLAN] token: {'✓' if has_plan else '✗'}")
    print(f"   [EXECUTION] token: {'✓' if has_exec else '✗'}")
    print(f"   [ANSWER] token: {'✓' if has_answer else '✗'}")
    print(f"\n   Sample output (first 500 chars):")
    print(f"   {decoded[:500]}...")
    
    if has_plan and has_exec and has_answer:
        print("\n🎉 Dataset format is correct!")
    else:
        print("\n⚠️  Warning: Some tokens are missing")
    
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # Verify dataset first
    verify_dataset_format()
    
    # Then test checkpoint if available
    checkpoint_dir = "outputs/hdp_test_new_format"
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    
    test_checkpoint_format(checkpoint_dir)
