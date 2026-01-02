import torch
import hydra
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback 
from omegaconf import OmegaConf

# --- Import modules của project ---
import diffusion
import utils
from transformers import AutoTokenizer, GPT2Tokenizer

# ==========================================
# 🚑 MONKEY PATCH: SỬA LỖI ROPE (FINAL VERSION)
# ==========================================
def fixed_apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    """
    Hàm vá lỗi RoPE mạnh mẽ: 
    1. Tự động nhân đôi cos/sin nếu lệch dimension (16 vs 32).
    2. Tự động unsqueeze để broadcast đúng chiều (Seq vs Heads).
    """
    # 1. Fix Dimension Mismatch (Head Dim)
    # Nếu cos là 16 mà qkv là 32 -> Nhân đôi cos
    if cos.shape[-1] != qkv.shape[-1]:
        # print(f"🔧 Patching RoPE HeadDim: {cos.shape[-1]} -> {qkv.shape[-1]}")
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

    # 2. Fix Broadcasting Mismatch (SeqLen vs Heads)
    # qkv shape chuẩn: [Batch, Seq, 3, Heads, HeadDim] (5 chiều)
    # cos shape gốc:   [Seq, HeadDim] hoặc [Batch, Seq, HeadDim]
    # Lỗi xảy ra khi PyTorch cố khớp Seq (256) của cos vào Heads (12) của qkv.
    
    if qkv.ndim == 5:
        # Nếu cos chỉ có 2 chiều [Seq, Dim] -> biến thành [1, Seq, 1, 1, Dim]
        if cos.ndim == 2:
            cos = cos[None, :, None, None, :]
            sin = sin[None, :, None, None, :]
        # Nếu cos có 3 chiều [Batch, Seq, Dim] -> biến thành [Batch, Seq, 1, 1, Dim]
        elif cos.ndim == 3:
            cos = cos[:, :, None, None, :]
            sin = sin[:, :, None, None, :]
            
    # Logic xoay cũ
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (qkv * cos) + (rotate_half(qkv) * sin)

# Áp dụng bản vá
try:
    import models.dit
    models.dit.apply_rotary_pos_emb_torchscript = fixed_apply_rotary_pos_emb_torchscript
    print("🚑 Applied RoPE Patch: Fixed Broadcasting & Dimensions.")
except Exception as e:
    print(f"⚠️ Could not apply RoPE patch: {e}")

# ==========================================
# 🛠️ FIX RESOLVER
# ==========================================
try:
    OmegaConf.register_new_resolver('cwd', os.getcwd)
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except ValueError:
    pass

# ==========================================
# ⚙️ CẤU HÌNH DEBUG
# ==========================================
OVERRIDES = [
    "algo=bd3lm",              
    "data=hdp_diffusion",      
    "model=small",
    "noise=loglinear",         
    "trainer.devices=1",
    "loader.batch_size=1",
    "loader.eval_batch_size=1",
    
    # Ép cấu hình khớp với RoPE mặc định
    "model.hidden_size=384", 
    "model.n_heads=12",       
]

CHECKPOINT_PATH = None 
CONFIG_PATH = "configs"
CONFIG_NAME = "config"

# ==========================================

def visualize_mask(mask, title="Attention Mask"):
    if mask is None:
        print(f"❌ {title} is None!")
        return
    mask_to_plot = mask.detach().cpu()
    while mask_to_plot.dim() > 2:
        mask_to_plot = mask_to_plot[0] 
    heatmap_data = torch.where(mask_to_plot == float('-inf'), torch.tensor(-1.0), mask_to_plot)
    heatmap_data = torch.where(heatmap_data == 0, torch.tensor(1.0), heatmap_data) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', vmin=-1, vmax=1)
    plt.title(title)
    filename = f"debug_{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"📸 Saved visualization to: {filename}")
    plt.close()

def debug_pipeline():
    print("\n🚀 STARTING HDP + BD3LM DEBUG PIPELINE...")
    
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path=CONFIG_PATH)
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=OVERRIDES)
        OmegaConf.resolve(cfg)
        print("✅ Config loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    print("\n📦 Loading Model...")
    try:
        tok_path = cfg.data.get('tokenizer_name_or_path', 'gpt2')
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            model = diffusion.Diffusion.load_from_checkpoint(CHECKPOINT_PATH, config=cfg, tokenizer=tokenizer)
        else:
            print("   ⚠️  Using Random Weights (Logic Check Mode)")
            model = diffusion.Diffusion(cfg, tokenizer=tokenizer)
            
        model.eval()
        model.cuda()
        print("✅ Model loaded to CUDA.")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        traceback.print_exc()
        return

    try:
        q_len = cfg.data.hdp.get('question_len', 128)
        p_len = cfg.data.hdp.get('plan_len', 128)
        e_len = cfg.data.hdp.get('exec_len', 256)
        total_len = q_len + p_len + e_len
        print(f"\n📏 Block Config: Q={q_len}, P={p_len}, E={e_len} (Total={total_len})")
    except:
        q_len, p_len, e_len = 128, 128, 256
        total_len = 512

    block_indices = torch.cat([
        torch.zeros(q_len, dtype=torch.long),
        torch.ones(p_len, dtype=torch.long),
        torch.full((e_len,), 2, dtype=torch.long)
    ]).unsqueeze(0).to('cuda')

    print("\n🔍 CHECK 1: Masking Logic")
    try:
        hdp_mask = model._create_hdp_bd3lm_mask(block_indices, total_len, device='cuda')
        print(f"   Mask Shape: {hdp_mask.shape}")
        if hdp_mask.dim() == 4:
            val = hdp_mask[0, 0, q_len, 0].item()
        else:
            val = hdp_mask[0, q_len, 0].item()
        print(f"   Connectivity check (Plan->Question): {val}")
        if val > -100:
            print("   ✅ PASS: Plan sees Question.")
        else:
            print("   ❌ FAIL: Plan is blind.")
    except Exception as e:
        print(f"❌ Failed check 1: {e}")

    print("\n🔍 CHECK 2: Forward Pass")
    try:
        x_input = torch.full((1, total_len), model.mask_index, dtype=torch.long).to('cuda')
        
        # t shape (1, 1) to match assert sigma.ndim == 2
        t = torch.ones(1, 1).to('cuda') * 0.99
        sigma = model._sigma_from_p(t)
        
        with torch.no_grad():
            logits = model.forward(x_input, sigma=sigma, block_indices=block_indices)
            
        print(f"   Logits Shape: {logits.shape}")
        if torch.isnan(logits).any():
            print("   ❌ FAIL: Logits contain NaN.")
        else:
            print("   ✅ PASS: Forward pass runs successfully.")
            
    except Exception as e:
        print(f"❌ Failed check 2: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_pipeline()