# """
# Generate high-level plans for GSM8K dataset using vLLM + Llama-3-8B-Instruct.

# This script:
# 1. Loads GSM8K train set.
# 2. Uses vLLM for high-throughput batch inference on H200.
# 3. Extracts reasoning skeleton (Plan) using 1-shot English prompt.
# 4. Saves in hierarchical format [Question, Plan, Execution] suitable for Block Diffusion training.

# Usage:
#     python generate_gsm8k_plans.py \
#         --model meta-llama/Meta-Llama-3-8B-Instruct \
#         --batch_size 512 \
#         --output_path data/gsm8k_hierarchical_train.json
# """

import os
import json
import argparse
import re
from typing import List
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- IMPROVED PROMPT (ENGLISH) ---
# GSM8K is in English, so the few-shot example MUST be in English to avoid language mixing.
PLAN_EXTRACTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert reasoning engine. Your task is to extract a high-level "Plan" from a math problem and its detailed solution.

The Plan must be an abstract skeleton of the reasoning logic.
RULES:
1. Describe WHAT operations to perform, using abstract terms (e.g., "Calculate the total cost", "Find the difference").
2. Do NOT use specific numbers from the problem (use terms like "initial value", "given ratio", "the remaining amount").
3. Do NOT calculate the final answer.
4. Keep it concise (3-5 short sentences).

EXAMPLE:
Question: "Janet has 3 times as many apples as Bob. If Bob has 5 apples, how many do they have together?"
Solution: "Bob has 5. Janet has 3 * 5 = 15. Total = 5 + 15 = 20."
Plan: "Identify Bob's quantity. Calculate Janet's quantity using the given multiplier. Sum both quantities to find the total."

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Solution: {solution}

Extract the abstract reasoning plan:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def load_gsm8k_dataset(split='train'):
    """Load GSM8K dataset."""
    print(f"Loading GSM8K {split} set...")
    dataset = load_dataset('gsm8k', 'main', split=split)
    print(f"Loaded {len(dataset)} examples")
    return dataset

def create_prompts(dataset) -> List[str]:
    """Create formatted prompts."""
    prompts = []
    for example in dataset:
        prompt = PLAN_EXTRACTION_PROMPT.format(
            question=example['question'],
            solution=example['answer']
        )
        prompts.append(prompt)
    return prompts

def batch_generate_plans(
    model_name: str,
    prompts: List[str],
    batch_size: int = 256,
    tensor_parallel_size: int = 1,
) -> List[str]:
    """Run vLLM inference."""
    print(f"\nInitializing vLLM with model: {model_name}")
    
    # H200 supports bfloat16 efficiently.
    # gpu_memory_utilization=0.9 is usually safe.
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=4096, # Llama 3 supports up to 8k, but 4k is enough for GSM8K
        gpu_memory_utilization=0.9,
    )
    
    sampling_params = SamplingParams(
        temperature=0.6,    # Low temp for more deterministic/focused plans
        top_p=0.9,
        max_tokens=256,     # Plans are short
        stop=["<|eot_id|>", "Question:", "Solution:"],
    )
    
    print(f"Generating plans for {len(prompts)} examples...")
    
    # vLLM handles batching internally, passing all prompts at once is usually faster/easier
    # unless dataset is massive (millions). For 7.5k, passing all is fine.
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract text
    plans = [output.outputs[0].text.strip() for output in outputs]
    
    return plans

def clean_plan_text(plan: str) -> str:
    """ robust cleaning of generated text """
    # 1. Remove specific Llama 3 chat artifacts or common prefixes
    prefixes = [
        "Here is the abstract reasoning plan:",
        "Here is the plan:",
        "The plan is:",
        "Plan:",
        "Abstract reasoning plan:",
        "Here is the extracted plan:"
    ]
    
    for p in prefixes:
        if plan.lower().startswith(p.lower()):
            plan = plan[len(p):].strip()
            
    # 2. Remove quotes if the model wrapped the output
    plan = plan.strip('"').strip("'")
    
    # 3. Clean up newlines (flatten to one line usually looks better for training data)
    plan = plan.replace("\n", " ").strip()
    
    # 4. Fallback if empty (Rare)
    if not plan or len(plan) < 5:
        plan = "Analyze the problem statement. Perform necessary arithmetic operations. State the final result."
        
    return plan

def create_hierarchical_dataset(
    original_dataset,
    generated_plans: List[str],
    output_path: str
):
    hierarchical_data = []
    
    print("\nProcessing and merging data...")
    for idx, (example, raw_plan) in enumerate(tqdm(zip(original_dataset, generated_plans), total=len(original_dataset))):
        
        clean_plan = clean_plan_text(raw_plan)
        
        # GSM8K Answer format: "reasoning steps.... #### 1234"
        full_solution = example['answer']
        parts = full_solution.split('####')
        
        execution = parts[0].strip()
        answer_numerical = parts[1].strip() if len(parts) > 1 else ""
        
        hierarchical_data.append({
            "id": f"gsm8k_{idx}",
            "question": example['question'],
            "plan": clean_plan,
            "execution": execution,
            "answer_numerical": answer_numerical
        })
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchical_data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Saved {len(hierarchical_data)} examples.")
    
    # Preview
    print("\n" + "="*50)
    print("SAMPLE PREVIEW")
    print("="*50)
    sample = hierarchical_data[0]
    print(f"Q: {sample['question'][:100]}...")
    print(f"Plan: {sample['plan']}")
    print(f"Exec: {sample['execution'][:100]}...")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--batch_size', type=int, default=256) # vLLM dynamic batching handles this, mainly for user reference
    parser.add_argument('--output_path', type=str, default='data/gsm8k/gsm8k_hierarchical_train.json')
    parser.add_argument('--split', type=str, default='train')
    
    args = parser.parse_args()
    
    dataset = load_gsm8k_dataset(args.split)
    prompts = create_prompts(dataset)
    
    # On H200, tensor_parallel_size=1 is enough for 8B model. 
    # If using multiple GPUs, increase this.
    plans = batch_generate_plans(args.model, prompts)
    
    create_hierarchical_dataset(dataset, plans, args.output_path)

if __name__ == '__main__':
    main()