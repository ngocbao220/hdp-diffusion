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
            use_special_format: Whether to use <|plan|> <|execution|> <|answer|> format
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
            # Format: <|plan|> plan_text <|execution|> execution_text <|answer|> answer
            question_text = f"<|question|> {sample['question']}"
            plan_text = f"<|plan|> {sample['plan']}"
            execution_text = f"<|execution|> {sample['execution']}"
            answer = sample.get('answer', '')
            if answer:
                execution_text = f"{execution_text} <|answer|> {answer}"
        else:
            # Original format
            plan_text = sample['plan']
            execution_text = sample['execution']
        
        # Tokenize each block separately
        question_ids = self._tokenize_and_pad(
            question_text, 
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
    
if __name__ == "__main__":
    import os
    import tempfile
    import sys
    from transformers import AutoTokenizer

    # Cấu hình logging ra màn hình để dễ nhìn
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    print("\n" + "="*50)
    print("🛠️  HDP DATASET DIAGNOSTIC TOOL")
    print("="*50 + "\n")

    # ---------------------------------------------------------
    # 1. TẠO DỮ LIỆU GIẢ (Dummy Data)
    # ---------------------------------------------------------
    # Để test mà không cần file thật bên ngoài
    dummy_data = [
        {
            "question": "John has 5 apples. He buys 3 more. How many apples?",
            "plan": "Identify initial count. Add bought count.",
            "execution": "5 + 3 = 8.",
            "answer": "8"
        },
        {
            "question": "Calculate area of a square with side 4.",
            "plan": "Use area formula. Substitute side length.",
            "execution": "Area = side * side = 4 * 4 = 16.",
            "answer": "16"
        }
    ]
    
    # Ghi dữ liệu giả vào file tạm
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(dummy_data, tmp_file)
        tmp_path = tmp_file.name
    
    logger.info(f"Created temporary dummy dataset at: {tmp_path}")

    try:
        # ---------------------------------------------------------
        # 2. KHỞI TẠO TOKENIZER
        # ---------------------------------------------------------
        # Sử dụng gpt2 làm ví dụ (nhanh và nhẹ)
        logger.info("Loading Tokenizer (gpt2)...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        # GPT2 không có pad_token mặc định, gán nó là eos_token
        tokenizer.pad_token = tokenizer.eos_token 

        # ---------------------------------------------------------
        # 3. KHỞI TẠO DATASET VÀ DATALOADER
        # ---------------------------------------------------------
        # Thiết lập block size nhỏ để dễ debug
        # Q=32, P=32, E=64 -> Tổng seq_len = 128
        BLOCK_SIZES = (32, 32, 64) 
        
        logger.info("Initializing HDPDataset...")
        dataset = HDPDataset(
            data_path=tmp_path,
            tokenizer=tokenizer,
            block_sizes=BLOCK_SIZES,
            use_special_format=True,
            return_block_indices=True
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=collate_hdp_batch
        )

        # ---------------------------------------------------------
        # 4. KIỂM TRA BATCH
        # ---------------------------------------------------------
        logger.info("Fetching one batch to inspect...")
        batch = next(iter(dataloader))
        
        input_ids = batch['input_ids']           # Shape: (B, Seq_Len)
        attn_mask = batch['attention_mask']      # Shape: (B, Seq_Len)
        block_idxs = batch['block_indices']      # Shape: (B, Seq_Len)

        print("\n--- Tensor Shapes ---")
        print(f"Batch Size:      {input_ids.shape[0]}")
        print(f"Sequence Length: {input_ids.shape[1]} (Target: {sum(BLOCK_SIZES)})")
        print(f"Input IDs:       {input_ids.shape}")
        print(f"Block Indices:   {block_idxs.shape}")

        # ---------------------------------------------------------
        # 5. GIẢI MÃ VÀ KIỂM TRA NỘI DUNG (DECODING CHECK)
        # ---------------------------------------------------------
        print("\n--- Content Decoding Check (Sample 0) ---")
        
        # Lấy sample đầu tiên trong batch
        sample_ids = input_ids[0]
        sample_blocks = block_idxs[0]
        
        # Tách các phần dựa trên block_indices
        q_mask = (sample_blocks == 0)
        p_mask = (sample_blocks == 1)
        e_mask = (sample_blocks == 2)
        
        # Hàm decode bỏ qua padding (eos_token trong trường hợp gpt2)
        def robust_decode(tokens):
            # Lọc bỏ pad token để nhìn cho sạch
            valid_tokens = tokens[tokens != tokenizer.pad_token_id]
            return tokenizer.decode(valid_tokens, skip_special_tokens=False)

        q_text = robust_decode(sample_ids[q_mask])
        p_text = robust_decode(sample_ids[p_mask])
        e_text = robust_decode(sample_ids[e_mask])

        print(f"\n[BLOCK 0 - QUESTION] ({BLOCK_SIZES[0]} tokens reserved)")
        print(f"Raw Content: \"{q_text}\"")
        
        print(f"\n[BLOCK 1 - PLAN] ({BLOCK_SIZES[1]} tokens reserved)")
        print(f"Raw Content: \"{p_text}\"")
        
        print(f"\n[BLOCK 2 - EXECUTION] ({BLOCK_SIZES[2]} tokens reserved)")
        print(f"Raw Content: \"{e_text}\"")

        print("\n" + "="*50)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*50)

    except Exception as e:
        logger.error(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Dọn dẹp file tạm
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Cleaned up temporary files.")