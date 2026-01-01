"""
Analyze and visualize GSM8K hierarchical results

This script analyzes:
1. Plan abstraction (number density)
2. Execution detail (calculation patterns)
3. Quality metrics for paper
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import argparse


def count_numbers(text: str) -> int:
    """Count numbers in text."""
    return len(re.findall(r'\d+', text))


def count_operations(text: str) -> int:
    """Count math operations in text."""
    operations = ['+', '-', '*', '/', '=', '<', '>']
    return sum(text.count(op) for op in operations)


def analyze_abstraction(data: List[Dict]):
    """Analyze plan abstraction vs execution detail."""
    
    results = {
        'plan_numbers': [],
        'exec_numbers': [],
        'plan_operations': [],
        'exec_operations': [],
        'plan_lengths': [],
        'exec_lengths': [],
    }
    
    for item in data:
        plan = item.get('plan', '')
        execution = item.get('execution', '')
        
        results['plan_numbers'].append(count_numbers(plan))
        results['exec_numbers'].append(count_numbers(execution))
        results['plan_operations'].append(count_operations(plan))
        results['exec_operations'].append(count_operations(execution))
        results['plan_lengths'].append(len(plan.split()))
        results['exec_lengths'].append(len(execution.split()))
    
    return results


def print_statistics(results: Dict):
    """Print statistical summary."""
    
    print("\n" + "="*60)
    print("GSM8K Hierarchical Analysis")
    print("="*60)
    
    # Numbers
    print("\n📊 Number Density:")
    print(f"  Plan:")
    print(f"    Mean: {np.mean(results['plan_numbers']):.2f}")
    print(f"    Median: {np.median(results['plan_numbers']):.1f}")
    print(f"    % with 0 numbers: {(np.array(results['plan_numbers']) == 0).mean() * 100:.1f}%")
    print(f"    % with <3 numbers: {(np.array(results['plan_numbers']) < 3).mean() * 100:.1f}%")
    
    print(f"  Execution:")
    print(f"    Mean: {np.mean(results['exec_numbers']):.2f}")
    print(f"    Median: {np.median(results['exec_numbers']):.1f}")
    
    # Operations
    print("\n🔢 Math Operations:")
    print(f"  Plan:")
    print(f"    Mean: {np.mean(results['plan_operations']):.2f}")
    print(f"  Execution:")
    print(f"    Mean: {np.mean(results['exec_operations']):.2f}")
    
    # Lengths
    print("\n📏 Text Lengths (words):")
    print(f"  Plan:")
    print(f"    Mean: {np.mean(results['plan_lengths']):.1f}")
    print(f"    Median: {np.median(results['plan_lengths']):.1f}")
    print(f"  Execution:")
    print(f"    Mean: {np.mean(results['exec_lengths']):.1f}")
    print(f"    Median: {np.median(results['exec_lengths']):.1f}")
    
    # Ratio
    print("\n📈 Plan vs Execution Ratios:")
    ratio_numbers = np.mean(results['exec_numbers']) / (np.mean(results['plan_numbers']) + 1e-10)
    ratio_ops = np.mean(results['exec_operations']) / (np.mean(results['plan_operations']) + 1e-10)
    print(f"  Numbers ratio (Exec/Plan): {ratio_numbers:.2f}x")
    print(f"  Operations ratio (Exec/Plan): {ratio_ops:.2f}x")
    
    print("="*60)


def visualize_results(results: Dict, output_path: str = 'gsm8k_analysis.png'):
    """Create visualization for paper."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Number distribution
    ax = axes[0, 0]
    ax.hist([results['plan_numbers'], results['exec_numbers']], 
            bins=20, alpha=0.7, label=['Plan', 'Execution'])
    ax.set_xlabel('Number of Numbers in Text')
    ax.set_ylabel('Frequency')
    ax.set_title('Number Distribution: Plan vs Execution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Operation distribution
    ax = axes[0, 1]
    ax.hist([results['plan_operations'], results['exec_operations']], 
            bins=15, alpha=0.7, label=['Plan', 'Execution'])
    ax.set_xlabel('Number of Math Operations')
    ax.set_ylabel('Frequency')
    ax.set_title('Math Operation Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Length comparison
    ax = axes[1, 0]
    positions = [1, 2]
    data = [results['plan_lengths'], results['exec_lengths']]
    bp = ax.boxplot(data, positions=positions, widths=0.5, 
                    patch_artist=True, labels=['Plan', 'Execution'])
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax.set_ylabel('Text Length (words)')
    ax.set_title('Text Length Distribution')
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Scatter: Plan vs Execution numbers
    ax = axes[1, 1]
    ax.scatter(results['plan_numbers'], results['exec_numbers'], 
              alpha=0.3, s=20)
    ax.set_xlabel('Numbers in Plan')
    ax.set_ylabel('Numbers in Execution')
    ax.set_title('Plan Abstraction vs Execution Detail')
    
    # Add diagonal line
    max_val = max(max(results['plan_numbers']), max(results['exec_numbers']))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")


def sample_examples(data: List[Dict], n: int = 5):
    """Print sample examples for qualitative analysis."""
    
    print("\n" + "="*60)
    print(f"Sample Examples (first {n})")
    print("="*60)
    
    for i in range(min(n, len(data))):
        item = data[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {item['question'][:100]}...")
        print(f"\nPlan: {item['plan']}")
        print(f"  └─ Numbers: {count_numbers(item['plan'])}")
        print(f"  └─ Operations: {count_operations(item['plan'])}")
        print(f"\nExecution: {item['execution'][:150]}...")
        print(f"  └─ Numbers: {count_numbers(item['execution'])}")
        print(f"  └─ Operations: {count_operations(item['execution'])}")
        if 'answer_numerical' in item:
            print(f"\nAnswer: {item['answer_numerical']}")
    
    print("="*60)


def analyze_abstraction_quality(data: List[Dict]):
    """Specific analysis for plan abstraction quality."""
    
    print("\n" + "="*60)
    print("Plan Abstraction Quality Analysis")
    print("="*60)
    
    # Categorize plans
    perfect_abstract = []  # 0 numbers
    mostly_abstract = []   # 1-2 numbers
    somewhat_concrete = [] # 3-5 numbers
    too_concrete = []      # >5 numbers
    
    for item in data:
        num_count = count_numbers(item['plan'])
        if num_count == 0:
            perfect_abstract.append(item)
        elif num_count <= 2:
            mostly_abstract.append(item)
        elif num_count <= 5:
            somewhat_concrete.append(item)
        else:
            too_concrete.append(item)
    
    total = len(data)
    print(f"\nCategories:")
    print(f"  Perfect (0 numbers): {len(perfect_abstract)} ({len(perfect_abstract)/total*100:.1f}%)")
    print(f"  Mostly abstract (1-2): {len(mostly_abstract)} ({len(mostly_abstract)/total*100:.1f}%)")
    print(f"  Somewhat concrete (3-5): {len(somewhat_concrete)} ({len(somewhat_concrete)/total*100:.1f}%)")
    print(f"  Too concrete (>5): {len(too_concrete)} ({len(too_concrete)/total*100:.1f}%)")
    
    # Good abstraction percentage
    good_abstraction = len(perfect_abstract) + len(mostly_abstract)
    print(f"\n✨ Good abstraction rate: {good_abstraction/total*100:.1f}%")
    
    # Show examples of each category
    if too_concrete:
        print(f"\n⚠️  Example of too-concrete plan:")
        example = too_concrete[0]
        print(f"  Question: {example['question'][:80]}...")
        print(f"  Plan: {example['plan']}")
        print(f"  └─ Contains {count_numbers(example['plan'])} numbers (should be <3)")
    
    print("="*60)


def compare_with_baseline(hierarchical_path: str, baseline_path: str = None):
    """Compare hierarchical format with baseline (if available)."""
    
    if baseline_path is None:
        print("\n⚠️  Baseline comparison skipped (no baseline path provided)")
        return
    
    print("\n" + "="*60)
    print("Hierarchical vs Baseline Comparison")
    print("="*60)
    
    # TODO: Implement comparison logic
    print("TODO: Load baseline results and compare metrics")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GSM8K hierarchical results"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to GSM8K hierarchical JSON'
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default='gsm8k_analysis.png',
        help='Output path for visualization'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of sample examples to print'
    )
    parser.add_argument(
        '--baseline_path',
        type=str,
        default=None,
        help='Path to baseline results for comparison'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Analyze
    results = analyze_abstraction(data)
    
    # Print statistics
    print_statistics(results)
    
    # Analyze quality
    analyze_abstraction_quality(data)
    
    # Sample examples
    sample_examples(data, n=args.num_samples)
    
    # Visualize
    visualize_results(results, output_path=args.output_plot)
    
    # Compare with baseline
    if args.baseline_path:
        compare_with_baseline(args.data_path, args.baseline_path)
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
