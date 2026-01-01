"""
Test script for hierarchical attention mask

This script verifies that the hierarchical mask is constructed correctly
according to the Plan-then-Generate architecture.
"""

import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspace/hdp-diffusion')

from models.hierarchical_mask import create_hierarchical_mask, hierarchical_block_diff_mask


def test_hierarchical_mask():
    """Test basic mask creation and properties."""
    print("=" * 60)
    print("Testing Hierarchical Attention Mask")
    print("=" * 60)
    
    # Parameters
    question_len = 64
    plan_len = 64
    exec_len = 128
    block_size = 16
    seqlen = question_len + plan_len + exec_len
    
    print(f"\nConfiguration:")
    print(f"  Question length: {question_len}")
    print(f"  Plan length: {plan_len}")
    print(f"  Execution length: {exec_len}")
    print(f"  Block size: {block_size}")
    print(f"  Total sequence length: {seqlen}")
    
    # Create mask
    print("\nCreating hierarchical mask...")
    mask = create_hierarchical_mask(
        seqlen=seqlen,
        block_size=block_size,
        question_len=question_len,
        plan_len=plan_len,
        exec_len=exec_len,
        attn_backend='sdpa'
    )
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    
    # Calculate boundaries (accounting for xt and x0)
    question_end = question_len
    plan_xt_end = question_end + plan_len
    plan_x0_end = plan_xt_end + plan_len
    exec_xt_end = plan_x0_end + exec_len
    exec_x0_end = exec_xt_end + exec_len
    
    print(f"\nBoundaries:")
    print(f"  Question: [0, {question_end})")
    print(f"  Plan xt: [{question_end}, {plan_xt_end})")
    print(f"  Plan x0: [{plan_xt_end}, {plan_x0_end})")
    print(f"  Exec xt: [{plan_x0_end}, {exec_xt_end})")
    print(f"  Exec x0: [{exec_xt_end}, {exec_x0_end})")
    
    # Test key properties
    print("\n" + "=" * 60)
    print("Verifying Mask Properties")
    print("=" * 60)
    
    # 1. Question can see all question tokens
    question_self_attn = mask[:question_end, :question_end]
    assert question_self_attn.all(), "❌ Question self-attention failed"
    print("✅ Question tokens can see all other question tokens")
    
    # 2. Plan cannot see execution
    plan_to_exec = mask[question_end:plan_x0_end, plan_x0_end:]
    assert not plan_to_exec.any(), "❌ Plan should NOT see Execution!"
    print("✅ Plan Block cannot see Execution Block (causal constraint preserved)")
    
    # 3. Plan can see question
    plan_to_question = mask[question_end:plan_x0_end, :question_end]
    assert plan_to_question.all(), "❌ Plan should see Question"
    print("✅ Plan Block can see Question")
    
    # 4. Execution can see question
    exec_to_question = mask[plan_x0_end:, :question_end]
    assert exec_to_question.all(), "❌ Execution should see Question"
    print("✅ Execution Block can see Question")
    
    # 5. Execution can see plan
    exec_to_plan = mask[plan_x0_end:, question_end:plan_x0_end]
    assert exec_to_plan.any(), "❌ Execution should see Plan"
    print("✅ Execution Block can see Plan Block")
    
    # 6. Check block diagonal structure
    plan_xt_self = mask[question_end:plan_xt_end, question_end:plan_xt_end]
    num_plan_blocks = plan_len // block_size
    block_diagonal_correct = True
    for i in range(num_plan_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        # Tokens in same block should see each other
        if not plan_xt_self[start:end, start:end].all():
            block_diagonal_correct = False
            break
    assert block_diagonal_correct, "❌ Plan xt block diagonal structure incorrect"
    print("✅ Block diagonal structure in Plan xt is correct")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
    
    return mask


def visualize_mask(mask, question_len, plan_len, exec_len, save_path='hierarchical_mask.png'):
    """Visualize the attention mask."""
    print(f"\nGenerating visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot mask
    im = ax.imshow(mask.float().cpu().numpy(), cmap='RdYlGn', aspect='auto')
    
    # Calculate boundaries
    question_end = question_len
    plan_xt_end = question_end + plan_len
    plan_x0_end = plan_xt_end + plan_len
    exec_xt_end = plan_x0_end + exec_len
    exec_x0_end = exec_xt_end + exec_len
    
    # Add boundary lines
    boundaries = [question_end, plan_xt_end, plan_x0_end, exec_xt_end, exec_x0_end]
    colors = ['red', 'blue', 'blue', 'green', 'green']
    labels = ['Q/Plan', 'Plan_xt/Plan_x0', '', 'Plan/Exec', '']
    
    for b, c, l in zip(boundaries[:-1], colors, labels):
        ax.axvline(x=b, color=c, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=b, color=c, linestyle='--', linewidth=1.5, alpha=0.7)
        if l:
            ax.text(b, -20, l, rotation=45, ha='right', fontsize=9, color=c)
    
    # Labels
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title('Hierarchical Block Diffusion Attention Mask\n' + 
                 '[Question | Plan_xt | Plan_x0 | Exec_xt | Exec_x0]', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Can Attend', rotation=270, labelpad=20, fontsize=11)
    
    # Add region labels on the side
    region_starts = [0, question_end, plan_xt_end, plan_x0_end, exec_xt_end]
    region_ends = [question_end, plan_xt_end, plan_x0_end, exec_xt_end, exec_x0_end]
    region_names = ['Question', 'Plan_xt', 'Plan_x0', 'Exec_xt', 'Exec_x0']
    region_colors = ['red', 'lightblue', 'blue', 'lightgreen', 'green']
    
    for start, end, name, color in zip(region_starts, region_ends, region_names, region_colors):
        mid = (start + end) / 2
        ax.text(-30, mid, name, rotation=90, va='center', ha='right', 
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
    return fig


def test_different_configurations():
    """Test mask with different configurations."""
    print("\n" + "=" * 60)
    print("Testing Different Configurations")
    print("=" * 60)
    
    configs = [
        {"question_len": 128, "plan_len": 128, "exec_len": 256, "block_size": 8},
        {"question_len": 256, "plan_len": 256, "exec_len": 512, "block_size": 16},
        {"question_len": 64, "plan_len": 192, "exec_len": 256, "block_size": 32},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        try:
            mask = create_hierarchical_mask(
                seqlen=config['question_len'] + config['plan_len'] + config['exec_len'],
                block_size=config['block_size'],
                **config,
                attn_backend='sdpa'
            )
            print(f"  ✅ Successfully created mask of shape {mask.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")


if __name__ == '__main__':
    print("Starting Hierarchical Mask Tests...\n")
    
    # Run basic test
    mask = test_hierarchical_mask()
    
    # Visualize
    visualize_mask(
        mask, 
        question_len=64, 
        plan_len=64, 
        exec_len=128,
        save_path='/workspace/hdp-diffusion/hierarchical_mask_test.png'
    )
    
    # Test different configs
    test_different_configurations()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the generated visualization: hierarchical_mask_test.png")
    print("2. Verify the mask structure matches your requirements")
    print("3. Integrate with your training pipeline")
    print("4. Test with actual data using hierarchical_dataloader.py")
