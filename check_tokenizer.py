import hydra
import torch
from dataloader import get_tokenizer
from omegaconf import OmegaConf

def check():
    # Tạo config giả
    cfg = OmegaConf.create({'data': {'tokenizer_name_or_path': 'gpt2'}})
    
    # Load tokenizer đã sửa
    tokenizer = get_tokenizer(cfg)
    
    text = "Question: 1+1? [PLAN] First add numbers [EXECUTION] 1+1=2 [ANSWER] 2"
    
    print(f"Vocab size (len): {len(tokenizer)}")
    
    # Tokenize
    tokens = tokenizer.encode(text)
    decoded_chunks = [tokenizer.decode([t]) for t in tokens]
    
    print("\n--- Token Analysis ---")
    print(f"Original Text: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Decoded chunks: {decoded_chunks}")
    
    # Kiểm tra xem [PLAN] có phải là 1 token không
    if '[PLAN]' in decoded_chunks:
        print("\n✅ SUCCESS: '[PLAN]' is a single token!")
    else:
        print("\n❌ FAIL: '[PLAN]' is still being split!")

if __name__ == "__main__":
    check()