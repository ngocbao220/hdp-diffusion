#!/usr/bin/env python3
"""
Convert HDP format to Baseline BD3-LM format
Merge plan + execution + answer into single solution field
"""

import json
import sys

def convert_hdp_to_baseline(hdp_file, output_file):
    """Convert HDP format (Q/P/E) to baseline format (Q/Solution)"""
    
    with open(hdp_file, 'r') as f:
        hdp_data = json.load(f)
    
    baseline_data = []
    
    for item in hdp_data:
        # Extract fields
        question = item['question']
        plan = item.get('plan', '')
        execution = item.get('execution', '')
        answer = item.get('answer', '')
        
        # Remove special markers
        plan = plan.replace('[PLAN]', '').replace('[EXECUTION]', '').replace('[ANSWER]', '').strip()
        execution = execution.replace('[PLAN]', '').replace('[EXECUTION]', '').replace('[ANSWER]', '').strip()
        
        # Merge into single solution
        solution_parts = []
        if plan:
            solution_parts.append(plan)
        if execution:
            solution_parts.append(execution)
        if answer:
            solution_parts.append(f"The answer is {answer}.")
        
        solution = ' '.join(solution_parts)
        
        baseline_item = {
            'question': question,
            'answer': solution  # Full solution in one field
        }
        
        baseline_data.append(baseline_item)
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"✅ Converted {len(baseline_data)} samples")
    print(f"   Input:  {hdp_file}")
    print(f"   Output: {output_file}")
    
    # Show example
    print(f"\n📝 Example baseline format:")
    print(f"   Question: {baseline_data[0]['question'][:60]}...")
    print(f"   Answer:   {baseline_data[0]['answer'][:80]}...")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create_baseline_data.py [hdp_file] [output_file]")
        print("Example: python create_baseline_data.py data/gsm8k/gsm8k_overfit.json data/gsm8k/gsm8k_baseline.json")
        sys.exit(1)
    
    hdp_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else hdp_file.replace('overfit', 'baseline')
    
    convert_hdp_to_baseline(hdp_file, output_file)
