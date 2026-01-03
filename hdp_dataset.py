"""
HDP-Diffusion Dataset Module

Handles data loading and formatting for Hierarchical Dual-Process reasoning.
Supports GSM8K format with Question -> Plan -> Execution structure.

Author: Research implementation for HDP-Diffusion
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class HDPDataset(Dataset):
    """
    Dataset for Hierarchical Dual-Process Diffusion.
    
    Loads and formats data with 3-block structure:
    - Block 0 (Question): Context/problem statement
    - Block 1 (Plan): High-level reasoning steps
    - Block 2 (Execution): Detailed calculations
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        block_sizes: Tuple[int, int, int] = (128, 128, 256),
        add_special_tokens: bool = True,
        return_block_indices: bool = True,
        use_special_format: bool = True
    ):
        """
        Args:
            data_path: Path to JSON file with {"question", "plan", "execution", "answer"}
            tokenizer: Hugging Face tokenizer
            block_sizes: (question_len, plan_len, exec_len) in tokens
            add_special_tokens: Whether to add BOS/EOS tokens
            return_block_indices: Whether to return block_indices tensor
            use_special_format: Whether to use [PLAN] [EXECUTION] [ANSWER] format
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.block_sizes = block_sizes
        self.add_special_tokens = add_special_tokens
        self.return_block_indices = return_block_indices
        self.use_special_format = use_special_format
        
        self.q_len, self.p_len, self.e_len = block_sizes
        self.seq_len = sum(block_sizes)
        
        # Load data
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Block sizes: Q={self.q_len}, P={self.p_len}, E={self.e_len}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Validate data format
        required_keys = ['question', 'plan', 'execution']
        for i, sample in enumerate(data):
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Sample {i} missing key '{key}'")
            # answer key is optional but recommended
            if 'answer' not in sample:
                logger.warning(f"Sample {i} missing optional 'answer' key")
        
        return data
    
    def _tokenize_and_pad(
        self, 
        text: str, 
        max_length: int,
        padding_side: str = 'right'
    ) -> torch.Tensor:
        """
        Tokenize and pad/truncate text to fixed length.
        
        Args:
            text: Input text
            max_length: Target length
            padding_side: 'right' or 'left'
        
        Returns:
            token_ids: (max_length,) tensor
        """
        # Tokenize
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        # Pad or truncate
        if len(tokens) < max_length:
            # Pad
            padding_length = max_length - len(tokens)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            
            if padding_side == 'right':
                tokens = torch.cat([
                    tokens,
                    torch.full((padding_length,), pad_token_id, dtype=torch.long)
                ])
            else:  # left
                tokens = torch.cat([
                    torch.full((padding_length,), pad_token_id, dtype=torch.long),
                    tokens
                ])
        else:
            tokens = tokens[:max_length]
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - input_ids: (seq_len,) concatenated [Q | P | E]
                - attention_mask: (seq_len,) 1 for real tokens, 0 for padding
                - block_indices: (seq_len,) 0=Q, 1=P, 2=E (optional)
        """
        sample = self.data[idx]
        
        # Build text based on format
        if self.use_special_format:
            # Format: [PLAN] plan_text [EXECUTION] execution_text [ANSWER] answer
            plan_text = f"[PLAN] {sample['plan']}"
            execution_text = f"[EXECUTION] {sample['execution']}"
            answer = sample.get('answer', '')
            if answer:
                execution_text = f"{execution_text} [ANSWER] {answer}"
        else:
            # Original format
            plan_text = sample['plan']
            execution_text = sample['execution']
        
        # Tokenize each block separately
        question_ids = self._tokenize_and_pad(
            sample['question'], 
            self.q_len, 
            padding_side='right'
        )
        
        plan_ids = self._tokenize_and_pad(
            plan_text, 
            self.p_len,
            padding_side='right'
        )
        
        execution_ids = self._tokenize_and_pad(
            execution_text, 
            self.e_len,
            padding_side='right'
        )
        
        # Concatenate blocks
        input_ids = torch.cat([question_ids, plan_ids, execution_ids])
        
        # Create attention mask (1 for non-padding tokens)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        attention_mask = (input_ids != pad_token_id).long()
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Add block indices if requested
        if self.return_block_indices:
            block_indices = torch.cat([
                torch.zeros(self.q_len, dtype=torch.long),
                torch.ones(self.p_len, dtype=torch.long),
                torch.full((self.e_len,), 2, dtype=torch.long)
            ])
            result['block_indices'] = block_indices
        
        return result


class HDPDatasetSimple(Dataset):
    """
    Simplified HDP Dataset for baseline experiments.
    
    Automatically splits a single text into Question + Solution format,
    then further splits Solution into Plan + Execution based on heuristics.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        block_sizes: Tuple[int, int, int] = (128, 128, 256),
        question_key: str = 'question',
        answer_key: str = 'answer'
    ):
        """
        Args:
            data_path: Path to JSON with question/answer pairs
            tokenizer: Tokenizer
            block_sizes: Block sizes
            question_key: Key for question in JSON
            answer_key: Key for answer in JSON
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.block_sizes = block_sizes
        self.question_key = question_key
        self.answer_key = answer_key
        
        self.q_len, self.p_len, self.e_len = block_sizes
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _split_answer_to_plan_exec(self, answer: str) -> Tuple[str, str]:
        """
        Heuristically split answer into plan and execution.
        
        This is a simple baseline - in practice you'd want better splitting logic.
        """
        # Simple heuristic: first 30% is plan, rest is execution
        sentences = answer.split('.')
        split_point = max(1, len(sentences) // 3)
        
        plan = '. '.join(sentences[:split_point]) + '.'
        execution = '. '.join(sentences[split_point:])
        
        return plan, execution
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        question = sample[self.question_key]
        answer = sample[self.answer_key]
        
        # Split answer into plan and execution
        plan, execution = self._split_answer_to_plan_exec(answer)
        
        # Tokenize
        dataset = HDPDataset.__new__(HDPDataset)
        dataset.tokenizer = self.tokenizer
        dataset.block_sizes = self.block_sizes
        dataset.q_len, dataset.p_len, dataset.e_len = self.block_sizes
        dataset.return_block_indices = True
        
        # Create temp sample
        temp_sample = {
            'question': question,
            'plan': plan,
            'execution': execution
        }
        
        # Use parent class tokenization
        dataset.data = [temp_sample]
        return dataset.__getitem__(0)


def collate_hdp_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from HDPDataset
    
    Returns:
        Batched tensors
    """
    # Stack all tensors
    batched = {}
    
    for key in batch[0].keys():
        batched[key] = torch.stack([sample[key] for sample in batch])
    
    return batched

if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer, GPT2Tokenizer
    from torch.utils.data import DataLoader
    
    # --- CẤU HÌNH ---
    # Thay đường dẫn này bằng đường dẫn tới file thật của bạn
    REAL_DATA_PATH = 'data/gsm8k/gsm8k_hierarchical_train.json'
    
    print(f"Testing HDP Dataset with real data at: {REAL_DATA_PATH}")
    
    # 1. Kiểm tra file
    if not os.path.exists(REAL_DATA_PATH):
        print(f"\n❌ ERROR: File not found at {REAL_DATA_PATH}")
        exit(1)

    # 2. Load Tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2') 
        tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Dataset
    print("\nAttempting to load HDPDataset...")
    try:
        # Thử load dataset chuẩn
        dataset = HDPDataset(
            data_path=REAL_DATA_PATH,
            tokenizer=tokenizer,
            block_sizes=(128, 128, 256) # Q=128, P=128, E=256
        )
    except ValueError:
        print("⚠️  Standard load failed (missing keys), trying Simple/Fallback mode...")
        # Fallback nếu file json chưa chia sẵn plan/execution
        dataset = HDPDatasetSimple(
            data_path=REAL_DATA_PATH,
            tokenizer=tokenizer,
            block_sizes=(128, 128, 256)
        )

    # 4. HÀM KIỂM TRA DECODE CHI TIẾT
    def debug_print_sample(sample_idx, dataset, tokenizer):
        sample = dataset[sample_idx]
        
        input_ids = sample['input_ids']
        block_indices = sample['block_indices']
        
        # Tách các token dựa trên block_indices
        # 0: Question, 1: Plan, 2: Execution
        q_tokens = input_ids[block_indices == 0]
        p_tokens = input_ids[block_indices == 1]
        e_tokens = input_ids[block_indices == 2]
        
        # Hàm lọc padding để in cho đẹp
        pad_id = tokenizer.pad_token_id
        def decode_clean(tokens):
            # Lọc bỏ pad token để dễ đọc nội dung
            valid_tokens = tokens[tokens != pad_id]
            text = tokenizer.decode(valid_tokens, skip_special_tokens=False)
            return text if text.strip() else "[EMPTY / PADDING ONLY]"

        print("\n" + "="*60)
        print(f"🔎 DETAILED STRUCTURE CHECK (Sample #{sample_idx})")
        
        print(f"\n[BLOCK 0: QUESTION] (Allocated: {len(q_tokens)} tokens)")
        print(f"   Shape check: {q_tokens.shape}")
        print(f"QUESTION TOKENS:", q_tokens[q_tokens != pad_id].tolist())
        print("QUESTION DECODE:", decode_clean(q_tokens))
        
        print(f"\n[BLOCK 1: PLAN] (Allocated: {len(p_tokens)} tokens)")
        print(f"   Shape check: {p_tokens.shape}")
        print("PLAN TOKENS:", p_tokens[p_tokens != pad_id].tolist())
        print("PLAN DECODE:", decode_clean(p_tokens))
        
        print(f"\n[BLOCK 2: EXECUTION] (Allocated: {len(e_tokens)} tokens)")
        print(f"   Shape check: {e_tokens.shape}")
        print("EXECUTION TOKENS:", e_tokens[e_tokens != pad_id].tolist())
        print("EXECUTION DECODE:", decode_clean(e_tokens))
        
        print("\n" + "="*60 + "\n")

    # 5. Thực hiện in kiểm tra mẫu đầu tiên
    print("Dataset size:", len(dataset))
    if len(dataset) > 0:
        debug_print_sample(0, dataset, tokenizer)
        
        # Kiểm tra thêm 1 mẫu nữa để chắc chắn (ví dụ mẫu số 1)
        if len(dataset) > 1:
            debug_print_sample(2, dataset, tokenizer)

    print("✅ Done verifying format.")


class SimpleGSM8KDataset(Dataset):
    """
    Simple dataset for GSM8K baseline format (Q&A only, no HDP structure).
    Loads from JSON with {"question": "...", "answer": "..."} format.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        add_special_tokens: bool = True
    ):
        """
        Args:
            data_path: Path to JSON file with {"question", "answer"}
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            add_special_tokens: Whether to add BOS/EOS tokens
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        item = self.data[idx]
        
        # Format: "Question: ... Answer: ..."
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }