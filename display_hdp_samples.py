"""
Display HDP-Diffusion samples with question context for better understanding.

This script helps visualize generated samples by showing:
1. The original question (from validation set)
2. The generated plan
3. The generated execution
4. Comparison with ground truth
"""

import json
import torch
from transformers import AutoTokenizer
import sys

def parse_hdp_sample(sample_text, tokenizer):
    """
    Parse a sample into Question, Plan, and Execution blocks.
    
    Args:
        sample_text: Full decoded sample
        tokenizer: Tokenizer to detect special tokens
        
    Returns:
        dict with 'question', 'plan', 'execution'
    """
    # Find [PLAN] and [EXECUTION] markers
    plan_marker = "[PLAN]"
    exec_marker = "[EXECUTION]"
    answer_marker = "[ANSWER]"
    
    parts = {
        'question': '',
        'plan': '',
        'execution': ''
    }
    
    # Split by markers
    if plan_marker in sample_text:
        question_part, rest = sample_text.split(plan_marker, 1)
        parts['question'] = question_part.strip()
        
        if exec_marker in rest:
            plan_part, exec_part = rest.split(exec_marker, 1)
            parts['plan'] = plan_part.strip()
            parts['execution'] = exec_part.strip()
        else:
            parts['plan'] = rest.strip()
    else:
        # No markers found - might be plain text
        # Assume first 128 tokens = question, next 128 = plan, rest = execution
        parts['question'] = sample_text[:200] + "..."
        parts['plan'] = "[No [PLAN] marker found]"
        parts['execution'] = "[No [EXECUTION] marker found]"
    
    # Clean up padding tokens
    for key in parts:
        parts[key] = parts[key].replace(tokenizer.eos_token, '').strip()
    
    return parts


def display_sample_with_context(sample_idx, generated_text, validation_data, tokenizer):
    """
    Display a generated sample alongside its ground truth.
    
    Args:
        sample_idx: Index in validation set
        generated_text: Generated sample text
        validation_data: List of validation examples
        tokenizer: Tokenizer
    """
    # Get ground truth
    if sample_idx < len(validation_data):
        truth = validation_data[sample_idx]
    else:
        truth = None
    
    # Parse generated sample
    generated = parse_hdp_sample(generated_text, tokenizer)
    
    print("=" * 80)
    print(f"SAMPLE #{sample_idx}")
    print("=" * 80)
    
    # Display Question
    print("\n📋 QUESTION:")
    print("-" * 80)
    if truth:
        print(truth['question'])
    else:
        print(generated['question'])
    
    # Display Generated Plan
    print("\n🧠 GENERATED PLAN:")
    print("-" * 80)
    print(generated['plan'] if generated['plan'] else "[Empty or not found]")
    
    # Display Ground Truth Plan
    if truth:
        print("\n✅ GROUND TRUTH PLAN:")
        print("-" * 80)
        print(truth['plan'])
    
    # Display Generated Execution
    print("\n🔢 GENERATED EXECUTION:")
    print("-" * 80)
    print(generated['execution'] if generated['execution'] else "[Empty or not found]")
    
    # Display Ground Truth Execution
    if truth:
        print("\n✅ GROUND TRUTH EXECUTION:")
        print("-" * 80)
        print(f"{truth['execution']} [ANSWER] {truth.get('answer', 'N/A')}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """
    Main function to display HDP samples.
    
    Usage:
        python display_hdp_samples.py <samples_file> [validation_data_path]
    """
    if len(sys.argv) < 2:
        print("Usage: python display_hdp_samples.py <samples_file> [validation_data_path]")
        print("\nExample:")
        print("  python display_hdp_samples.py samples.txt")
        print("  python display_hdp_samples.py samples.txt data/gsm8k/gsm8k_hierarchical_test.json")
        sys.exit(1)
    
    samples_file = sys.argv[1]
    validation_path = sys.argv[2] if len(sys.argv) > 2 else "data/gsm8k/gsm8k_hierarchical_test.json"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation data
    validation_data = []
    try:
        with open(validation_path, 'r') as f:
            validation_data = json.load(f)
        print(f"✅ Loaded {len(validation_data)} validation examples from {validation_path}")
    except FileNotFoundError:
        print(f"⚠️  Validation data not found at {validation_path}")
        print("   Will display generated samples without ground truth comparison")
    
    # Read generated samples
    with open(samples_file, 'r') as f:
        samples = f.readlines()
    
    print(f"\n📊 Processing {len(samples)} generated samples...\n")
    
    # Display each sample
    for idx, sample_text in enumerate(samples):
        sample_text = sample_text.strip()
        if not sample_text:
            continue
        
        display_sample_with_context(idx, sample_text, validation_data, tokenizer)
        
        # Pause after every 5 samples
        if (idx + 1) % 5 == 0 and idx < len(samples) - 1:
            input("\nPress Enter to see next 5 samples...")
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
