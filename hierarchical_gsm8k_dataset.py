"""
GSM8K Hierarchical Dataset for Dual-Process Diffusion

This module loads GSM8K data with hierarchical structure:
- Question: Math word problem (fixed length L_q = 128)
- Plan: Abstract reasoning steps (fixed length L_p = 128)  
- Execution: Detailed calculations (fixed length L_e = 256)

Total sequence length: 512 tokens
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path
import transformers


class GSM8KHierarchicalDataset(Dataset):
    """
    PyTorch Dataset for GSM8K with hierarchical structure.
    
    Each sample contains:
        - question: Math word problem
        - plan: High-level reasoning steps
        - execution: Detailed calculations and final answer
    
    The data is formatted into fixed-length blocks with special tokens.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        question_len: int = 128,
        plan_len: int = 128,
        exec_len: int = 256,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding: str = "max_length",
        use_special_format: bool = True,
    ):
        """
        Initialize GSM8K Hierarchical Dataset.
        
        Args:
            data_path: Path to JSON file with hierarchical GSM8K data
            tokenizer: HuggingFace tokenizer (e.g., GPT-2)
            question_len: Fixed length for question block (default: 128)
            plan_len: Fixed length for plan block (default: 128)
            exec_len: Fixed length for execution block (default: 256)
            add_special_tokens: Whether to add BOS/EOS tokens
            truncation: Whether to truncate sequences that are too long
            padding: Padding strategy ("max_length" or "longest")
            use_special_format: Whether to use [PLAN] [EXECUTION] [ANSWER] format
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.question_len = question_len
        self.plan_len = plan_len
        self.exec_len = exec_len
        self.seq_len = question_len + plan_len + exec_len
        self.add_special_tokens = add_special_tokens
        self.truncation = truncation
        self.padding = padding
        self.use_special_format = use_special_format
        
        # Load data from JSON
        self.data = self._load_data(data_path)
        
        # Define special tokens for each block
        self.q_start_token = "[Q]"  # Question start marker
        self.p_start_token = "[PLAN]"  # Plan start marker
        self.e_start_token = "[EXEC]"  # Execution start marker
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Sequence structure: Question({question_len}) + Plan({plan_len}) + Execution({exec_len}) = {self.seq_len}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load hierarchical GSM8K data from JSON file."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data format
        if isinstance(data, list) and len(data) > 0:
            required_keys = {'question', 'plan', 'execution'}
            sample_keys = set(data[0].keys())
            
            if not required_keys.issubset(sample_keys):
                raise ValueError(
                    f"Data must contain keys: {required_keys}. "
                    f"Found: {sample_keys}"
                )
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _tokenize_block(
        self,
        text: str,
        max_length: int,
        prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single block with fixed length.
        
        Args:
            text: Text to tokenize
            max_length: Maximum length for this block
            prefix: Optional prefix token (e.g., "[Q]", "[PLAN]")
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Add prefix if provided
        if prefix:
            text = f"{prefix} {text}"
        
        # Tokenize with fixed length
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=False,  # We'll handle special tokens manually
            return_tensors="pt",
        )
        
        # Squeeze batch dimension (we're processing one sample at a time)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with hierarchical structure.
        
        Returns:
            Dictionary containing:
                - input_ids: (seq_len,) concatenated token IDs
                - attention_mask: (seq_len,) attention mask
                - block_indices: (seq_len,) block assignments (0=Q, 1=P, 2=E)
                - question_text: Original question text
                - plan_text: Original plan text
                - execution_text: Original execution text
        """
        sample = self.data[idx]
        
        # Extract text fields
        question = sample['question']
        plan = sample.get('plan', '')
        execution = sample.get('execution', '')
        answer = sample.get('answer', '')
        
        # Format output based on use_special_format flag
        if self.use_special_format:
            # New format: [PLAN] ... [EXECUTION] ... [ANSWER] ...
            plan_text = f"[PLAN] {plan}"
            execution_text = f"[EXECUTION] {execution}"
            if answer:
                execution_text = f"{execution_text} [ANSWER] {answer}"
        else:
            # Original format with prefixes
            plan_text = plan
            execution_text = execution
        
        # Tokenize each block separately
        q_tokens = self._tokenize_block(question, self.question_len, prefix=self.q_start_token)
        p_tokens = self._tokenize_block(plan_text, self.plan_len, prefix="" if self.use_special_format else self.p_start_token)
        e_tokens = self._tokenize_block(execution_text, self.exec_len, prefix="" if self.use_special_format else self.e_start_token)
        
        # Concatenate blocks
        input_ids = torch.cat([
            q_tokens['input_ids'],
            p_tokens['input_ids'],
            e_tokens['input_ids'],
        ], dim=0)
        
        attention_mask = torch.cat([
            q_tokens['attention_mask'],
            p_tokens['attention_mask'],
            e_tokens['attention_mask'],
        ], dim=0)
        
        # Create block indices (0=Question, 1=Plan, 2=Execution)
        block_indices = torch.cat([
            torch.zeros(self.question_len, dtype=torch.long),  # Question
            torch.ones(self.plan_len, dtype=torch.long),       # Plan
            torch.full((self.exec_len,), 2, dtype=torch.long), # Execution
        ], dim=0)
        
        # Add BOS token at the beginning if requested
        if self.add_special_tokens:
            bos_id = self.tokenizer.bos_token_id
            if bos_id is not None:
                # Prepend BOS and adjust lengths
                input_ids = torch.cat([
                    torch.tensor([bos_id], dtype=torch.long),
                    input_ids[:-1]  # Remove last token to maintain length
                ])
                attention_mask = torch.cat([
                    torch.tensor([1], dtype=torch.long),
                    attention_mask[:-1]
                ])
                # Block indices stay the same (BOS belongs to Question block)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'block_indices': block_indices,
            'question_text': question,
            'plan_text': plan,
            'execution_text': execution,
            'sample_id': sample.get('id', f'sample_{idx}'),
        }
    
    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(input_ids, skip_special_tokens=False)
    
    def get_block_texts(self, input_ids: torch.Tensor, block_indices: torch.Tensor) -> Dict[str, str]:
        """
        Extract and decode text for each block.
        
        Args:
            input_ids: (seq_len,) token IDs
            block_indices: (seq_len,) block assignments
            
        Returns:
            Dictionary with decoded text for each block
        """
        q_tokens = input_ids[block_indices == 0]
        p_tokens = input_ids[block_indices == 1]
        e_tokens = input_ids[block_indices == 2]
        
        return {
            'question': self.tokenizer.decode(q_tokens, skip_special_tokens=True),
            'plan': self.tokenizer.decode(p_tokens, skip_special_tokens=True),
            'execution': self.tokenizer.decode(e_tokens, skip_special_tokens=True),
        }


def collate_hierarchical_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for hierarchical GSM8K data.
    
    Args:
        batch: List of samples from GSM8KHierarchicalDataset
        
    Returns:
        Batched tensors
    """
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'block_indices': torch.stack([x['block_indices'] for x in batch]),
    }


def create_gsm8k_hierarchical_dataloader(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    question_len: int = 128,
    plan_len: int = 128,
    exec_len: int = 256,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for hierarchical GSM8K dataset.
    
    Args:
        data_path: Path to JSON data file
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        question_len: Question block length
        plan_len: Plan block length
        exec_len: Execution block length
        **dataset_kwargs: Additional arguments for GSM8KHierarchicalDataset
        
    Returns:
        DataLoader instance
    """
    dataset = GSM8KHierarchicalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        question_len=question_len,
        plan_len=plan_len,
        exec_len=exec_len,
        **dataset_kwargs,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_hierarchical_batch,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    import transformers
    
    print("Testing GSM8K Hierarchical Dataset...")
    
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test with sample data file
    data_path = "data/gsm8k/gsm8k_hierarchical_train.json"
    
    try:
        dataset = GSM8KHierarchicalDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            question_len=128,
            plan_len=128,
            exec_len=256,
        )
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Block indices shape: {sample['block_indices'].shape}")
        print(f"Unique block values: {torch.unique(sample['block_indices']).tolist()}")
        
        # Decode and print
        print("\n=== Sample Content ===")
        print(f"Question: {sample['question_text'][:100]}...")
        print(f"Plan: {sample['plan_text'][:100]}...")
        print(f"Execution: {sample['execution_text'][:100]}...")
        
        # Test block extraction
        block_texts = dataset.get_block_texts(sample['input_ids'], sample['block_indices'])
        print(f"\n=== Decoded Blocks ===")
        print(f"Q: {block_texts['question'][:80]}...")
        print(f"P: {block_texts['plan'][:80]}...")
        print(f"E: {block_texts['execution'][:80]}...")
        
        # Test DataLoader
        dataloader = create_gsm8k_hierarchical_dataloader(
            data_path=data_path,
            tokenizer=tokenizer,
            batch_size=4,
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        print(f"\n✅ DataLoader test passed!")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch block_indices shape: {batch['block_indices'].shape}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Test file not found: {e}")
        print("Create sample data file to test the dataset.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
