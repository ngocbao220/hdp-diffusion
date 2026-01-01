"""
Hierarchical Data Collator for Plan-then-Generate Structure

This module handles data preprocessing for hierarchical reasoning tasks
where input consists of [Question, Plan, Execution] triplets.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import datasets


class HierarchicalDataCollator:
    """
    Collates data for hierarchical plan-then-generate training.
    
    Expected input format:
    - question: Input question/context tokens
    - plan: High-level plan tokens
    - execution: Detailed execution tokens
    
    Output format: [Question | Plan | Execution] concatenated
    """
    
    def __init__(
        self,
        tokenizer,
        question_len: int = 256,
        plan_len: int = 256,
        exec_len: int = 512,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.question_len = question_len
        self.plan_len = plan_len
        self.exec_len = exec_len
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples into hierarchical format.
        
        Args:
            examples: List of dicts with keys 'question', 'plan', 'execution'
                     or 'text' which will be split into these components.
                     
        Returns:
            Dict with:
                - input_ids: [batch, question_len + plan_len + exec_len]
                - attention_mask: [batch, question_len + plan_len + exec_len]
                - hierarchical_info: Dict with boundaries
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        for example in examples:
            # If data is in the format with separate fields
            if 'question' in example and 'plan' in example and 'execution' in example:
                question_ids = example['question']
                plan_ids = example['plan']
                exec_ids = example['execution']
            # If data is a single text that needs to be split
            elif 'text' in example or 'input_ids' in example:
                # For now, split the input into three equal parts
                # TODO: Implement proper splitting logic based on your data format
                if 'input_ids' in example:
                    input_ids = example['input_ids']
                else:
                    input_ids = self.tokenizer.encode(example['text'])
                
                total_len = len(input_ids)
                # Simple split: first 1/4 is question, next 1/4 is plan, rest is execution
                q_end = min(self.question_len, total_len // 4)
                p_end = min(q_end + self.plan_len, q_end + total_len // 4)
                
                question_ids = input_ids[:q_end]
                plan_ids = input_ids[q_end:p_end]
                exec_ids = input_ids[p_end:]
            else:
                raise ValueError(
                    "Example must contain either ('question', 'plan', 'execution') "
                    "or 'text'/'input_ids' keys"
                )
            
            # Pad or truncate to fixed lengths
            question_ids = self._pad_or_truncate(question_ids, self.question_len)
            plan_ids = self._pad_or_truncate(plan_ids, self.plan_len)
            exec_ids = self._pad_or_truncate(exec_ids, self.exec_len)
            
            # Concatenate: [Question | Plan | Execution]
            input_ids = question_ids + plan_ids + exec_ids
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 
                            for token_id in input_ids]
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
        
        # Convert to tensors
        batch_dict = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.float),
            'hierarchical_info': {
                'question_len': self.question_len,
                'plan_len': self.plan_len,
                'exec_len': self.exec_len,
                'question_end': self.question_len,
                'plan_end': self.question_len + self.plan_len,
                'exec_end': self.question_len + self.plan_len + self.exec_len,
            }
        }
        
        return batch_dict
    
    def _pad_or_truncate(self, token_ids: List[int], target_len: int) -> List[int]:
        """Pad or truncate token sequence to target length."""
        if len(token_ids) >= target_len:
            return token_ids[:target_len]
        else:
            # Pad with pad_token_id
            padding = [self.tokenizer.pad_token_id] * (target_len - len(token_ids))
            return token_ids + padding


def create_hierarchical_dataset(
    dataset_name: str,
    tokenizer,
    split: str = 'train',
    question_len: int = 256,
    plan_len: int = 256,
    exec_len: int = 512,
    cache_dir: str = './data_cache',
    num_proc: int = 4,
):
    """
    Create a dataset formatted for hierarchical training.
    
    For demonstration, this creates synthetic hierarchical data.
    You should replace this with your actual data pipeline.
    
    Args:
        dataset_name: Name of base dataset
        tokenizer: Tokenizer to use
        split: Dataset split ('train', 'validation', 'test')
        question_len: Target length for question
        plan_len: Target length for plan
        exec_len: Target length for execution
        cache_dir: Cache directory
        num_proc: Number of processes for parallel processing
        
    Returns:
        Dataset with hierarchical structure
    """
    
    # Load base dataset (example: using existing openwebtext)
    if dataset_name == 'openwebtext':
        dataset = datasets.load_dataset(
            'openwebtext',
            split='train[:10000]' if split == 'train' else 'train[-1000:]',
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        def process_example(example):
            """
            Process a single example into hierarchical format.
            
            For now, this is a placeholder that splits text into sections.
            Replace with your actual logic for extracting question/plan/execution.
            """
            text = example['text']
            tokens = tokenizer.encode(text)
            
            # Simple splitting logic - replace with your domain-specific logic
            total = len(tokens)
            q_end = min(question_len, total // 4)
            p_end = min(q_end + plan_len, q_end + total // 4)
            
            return {
                'question': tokens[:q_end],
                'plan': tokens[q_end:p_end],
                'execution': tokens[p_end:p_end + exec_len],
                'original_text': text,
            }
        
        dataset = dataset.map(
            process_example,
            num_proc=num_proc,
            desc="Creating hierarchical structure"
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for hierarchical structure")
    
    return dataset


def load_reasoning_dataset(
    dataset_path: str,
    tokenizer,
    question_len: int = 256,
    plan_len: int = 256, 
    exec_len: int = 512,
):
    """
    Load a dataset specifically formatted for reasoning tasks.
    
    Expected JSON format:
    [
        {
            "question": "What is...",
            "plan": "First, I will... Then...",
            "execution": "Step 1: ... Step 2: ..."
        },
        ...
    ]
    
    Args:
        dataset_path: Path to JSON file or directory
        tokenizer: Tokenizer
        question_len, plan_len, exec_len: Target lengths
        
    Returns:
        Dataset with tokenized hierarchical structure
    """
    import json
    
    # Load JSON data
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Tokenize each component
    processed_data = []
    for item in data:
        processed_data.append({
            'question': tokenizer.encode(item['question'])[:question_len],
            'plan': tokenizer.encode(item['plan'])[:plan_len],
            'execution': tokenizer.encode(item['execution'])[:exec_len],
        })
    
    # Create HuggingFace dataset
    dataset = datasets.Dataset.from_list(processed_data)
    
    return dataset


# Example usage and testing
if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create collator
    collator = HierarchicalDataCollator(
        tokenizer=tokenizer,
        question_len=256,
        plan_len=256,
        exec_len=512,
    )
    
    # Test with dummy data
    dummy_examples = [
        {
            'text': 'This is a test question. ' * 10 + 
                   'This is the plan. ' * 10 + 
                   'This is the execution. ' * 20
        },
        {
            'text': 'Another question here. ' * 8 + 
                   'Another plan here. ' * 12 + 
                   'Another execution here. ' * 25
        }
    ]
    
    # Tokenize
    for ex in dummy_examples:
        ex['input_ids'] = tokenizer.encode(ex['text'])
    
    # Collate
    batch = collator(dummy_examples)
    
    print("Batch keys:", batch.keys())
    print("Input IDs shape:", batch['input_ids'].shape)
    print("Attention mask shape:", batch['attention_mask'].shape)
    print("Hierarchical info:", batch['hierarchical_info'])
