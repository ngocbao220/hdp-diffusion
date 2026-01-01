"""
GSM8K Dataset Loader for Hierarchical Training

Loads GSM8K dataset in hierarchical format [Question, Plan, Execution]
Integrates with the hierarchical_dataloader.py system.
"""

import json
import torch
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer
import datasets


class GSM8KHierarchicalDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for hierarchical GSM8K data.
    
    Expected JSON format:
    [
        {
            "id": "gsm8k_train_0",
            "question": "Janet's ducks lay 16 eggs per day...",
            "plan": "Calculate daily egg production. Determine consumption...",
            "execution": "Janet gets 16 eggs per day. She eats 3 for breakfast...",
            "answer_numerical": "18"
        },
        ...
    ]
    """
    
    def __init__(
        self,
        json_path: str,
        tokenizer: PreTrainedTokenizer,
        question_max_len: int = 128,
        plan_max_len: int = 128,
        exec_max_len: int = 256,
    ):
        """
        Args:
            json_path: Path to hierarchical GSM8K JSON file
            tokenizer: HuggingFace tokenizer
            question_max_len: Max tokens for question
            plan_max_len: Max tokens for plan
            exec_max_len: Max tokens for execution
        """
        self.tokenizer = tokenizer
        self.question_max_len = question_max_len
        self.plan_max_len = plan_max_len
        self.exec_max_len = exec_max_len
        
        # Load data
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize each component
        question_ids = self.tokenizer.encode(
            item['question'],
            add_special_tokens=False,
            max_length=self.question_max_len,
            truncation=True
        )
        
        plan_ids = self.tokenizer.encode(
            item['plan'],
            add_special_tokens=False,
            max_length=self.plan_max_len,
            truncation=True
        )
        
        execution_ids = self.tokenizer.encode(
            item['execution'],
            add_special_tokens=False,
            max_length=self.exec_max_len,
            truncation=True
        )
        
        # Concatenate: [BOS] question plan execution [EOS]
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        
        input_ids = [bos_id] + question_ids + plan_ids + execution_ids + [eos_id]
        
        # Pad or truncate to fixed length
        max_len = self.question_max_len + self.plan_max_len + self.exec_max_len + 2  # +2 for BOS/EOS
        if len(input_ids) < max_len:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_len - len(input_ids))
        else:
            input_ids = input_ids[:max_len]
        
        # Return format compatible with standard dataloader
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
        }


def load_gsm8k_hierarchical(
    mode: str,  # 'train', 'test', or 'validation'
    block_size: int,
    tokenizer: PreTrainedTokenizer,
    insert_eos: bool = True,
    **kwargs  # Absorb extra args from get_dataset
):
    """
    Load hierarchical GSM8K dataset compatible with dataloader.get_dataset().
    
    Args:
        mode: 'train', 'test', or 'validation'
        block_size: Max sequence length (should be 512 for GSM8K)
        tokenizer: HuggingFace tokenizer  
        insert_eos: Whether to insert EOS tokens
        
    Returns:
        GSM8KHierarchicalDataset instance
    """
    import os
    # Get absolute path - gsm8k_dataloader.py is in /workspace/hdp-diffusion/
    # So __file__ parent is /workspace/hdp-diffusion
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file)  # /workspace/hdp-diffusion
    
    # Map mode to file path
    if mode == 'train':
        json_path = os.path.join(project_root, 'data/gsm8k/gsm8k_hierarchical_train.json')
    elif mode in ['test', 'validation']:
        json_path = os.path.join(project_root, 'data/gsm8k/gsm8k_hierarchical_test.json')
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Parse hierarchical lengths from block_size
    # Default: 128 (question) + 128 (plan) + 256 (exec) = 512
    question_max_len = 128
    plan_max_len = 128
    exec_max_len = block_size - question_max_len - plan_max_len
    
    dataset = GSM8KHierarchicalDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        question_max_len=question_max_len,
        plan_max_len=plan_max_len,
        exec_max_len=exec_max_len,
    )
    
    return dataset


def analyze_gsm8k_lengths(json_path: str, tokenizer: PreTrainedTokenizer):
    """
    Analyze token lengths in GSM8K to determine optimal max_len settings.
    
    Usage:
        from gsm8k_dataloader import analyze_gsm8k_lengths
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        analyze_gsm8k_lengths('data/gsm8k/gsm8k_hierarchical_train.json', tokenizer)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    question_lens = []
    plan_lens = []
    exec_lens = []
    
    for item in data:
        q_tokens = tokenizer.encode(item['question'], add_special_tokens=False)
        p_tokens = tokenizer.encode(item['plan'], add_special_tokens=False)
        e_tokens = tokenizer.encode(item['execution'], add_special_tokens=False)
        
        question_lens.append(len(q_tokens))
        plan_lens.append(len(p_tokens))
        exec_lens.append(len(e_tokens))
    
    import numpy as np
    
    print("\n" + "="*60)
    print("GSM8K Token Length Analysis")
    print("="*60)
    print(f"Number of examples: {len(data)}")
    print()
    
    for name, lens in [
        ("Question", question_lens),
        ("Plan", plan_lens),
        ("Execution", exec_lens)
    ]:
        print(f"{name}:")
        print(f"  Mean: {np.mean(lens):.1f}")
        print(f"  Median: {np.median(lens):.1f}")
        print(f"  Min: {np.min(lens)}")
        print(f"  Max: {np.max(lens)}")
        print(f"  95th percentile: {np.percentile(lens, 95):.1f}")
        print(f"  99th percentile: {np.percentile(lens, 99):.1f}")
        print()
    
    # Suggest optimal lengths
    q_len = int(np.percentile(question_lens, 95))
    p_len = int(np.percentile(plan_lens, 95))
    e_len = int(np.percentile(exec_lens, 95))
    
    print("="*60)
    print("Recommended max_len settings (95th percentile):")
    print(f"  question_max_len: {q_len}")
    print(f"  plan_max_len: {p_len}")
    print(f"  exec_max_len: {e_len}")
    print(f"  total: {q_len + p_len + e_len}")
    print("="*60)


# Integration with hierarchical_dataloader.py
def create_gsm8k_collator(
    tokenizer: PreTrainedTokenizer,
    question_len: int = 128,
    plan_len: int = 128,
    exec_len: int = 256,
):
    """
    Create a collator compatible with GSM8K hierarchical format.
    Uses the HierarchicalDataCollator from hierarchical_dataloader.py
    """
    from hierarchical_dataloader import HierarchicalDataCollator
    
    return HierarchicalDataCollator(
        tokenizer=tokenizer,
        question_len=question_len,
        plan_len=plan_len,
        exec_len=exec_len,
    )


# Example usage
if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to GSM8K hierarchical JSON')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze token lengths')
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    if args.analyze:
        analyze_gsm8k_lengths(args.data_path, tokenizer)
    else:
        # Test loading
        dataset = GSM8KHierarchicalDataset(
            json_path=args.data_path,
            tokenizer=tokenizer,
            question_max_len=128,
            plan_max_len=128,
            exec_max_len=256,
        )
        
        print(f"\nDataset size: {len(dataset)}")
        print("\nFirst example:")
        example = dataset[0]
        print(f"  ID: {example['id']}")
        print(f"  Question tokens: {len(example['question'])}")
        print(f"  Plan tokens: {len(example['plan'])}")
        print(f"  Execution tokens: {len(example['execution'])}")
        print(f"  Answer: {example['answer_numerical']}")
