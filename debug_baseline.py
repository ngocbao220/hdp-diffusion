import torch
import hydra
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback 
from omegaconf import OmegaConf

# --- Import modules ---
import diffusion
import utils
from transformers import AutoTokenizer, GPT2Tokenizer

# ==========================================
# 🚑 MONKEY PATCH ROPE (Giữ lại để chạy trên máy không có FlashAttn)
# ==========================================
def fixed_apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    if cos.shape[-1] != qkv.shape[-1]:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    if qkv.ndim == 5: 
        if cos.ndim == 2:
            cos = cos[None, :, None, None, :]
            sin = sin[None, :, None, None, :]
        elif cos.ndim == 3:
            cos = cos[:, :, None, None, :]
            sin = sin[:, :, None, None, :]
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return (qkv * cos) + (rotate_half(qkv) * sin)

try:
    import models.dit
    models.dit.apply_rotary_pos_emb_torchscript = fixed_apply_rotary_pos_emb_torchscript
except:
    pass

# ==========================================
# ⚙️ CẤU HÌNH BASELINE
# ==========================================
try:
    OmegaConf.register_new_resolver('cwd', os.getcwd)
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except:
    pass

OVERRIDES = [
    "algo=bd3lm",              # 🟢 Thuật toán gốc
    "data=gsm8k_baseline",     # 🟢 Dataset gốc (Chỉ Q + A)
    "model=small",
    "noise=loglinear",
    
    # Tắt HDP explicit
    "+data.hdp.use_hdp_attention=False", 
    
    # Setup debug
    "trainer.devices=1",
    "loader.batch_size=1",
    "loader.eval_batch_size=1",
    
    # Fix Dimension cho RoPE patch
    "model.hidden_size=384", 
    "model.n_heads=12",
]

CONFIG_PATH = "configs"
CONFIG_NAME = "config"

# ==========================================

def visualize_mask(mask, title="Baseline_BD3LM_Mask"):
    if mask is None:
        print(f"⚠️ {title} is None (Flash Attention might handle this internally)")
        return
    
    mask_to_plot = mask.detach().cpu()
    while mask_to_plot.dim() > 2:
        mask_to_plot = mask_to_plot[0] 
        
    # Chuyển đổi: 1/True -> 1.0 (Sáng), 0/False -> -1.0 (Tối)
    if mask_to_plot.dtype == torch.bool:
        heatmap_data = torch.where(mask_to_plot, torch.tensor(1.0), torch.tensor(-1.0))
    else:
        heatmap_data = torch.where(mask_to_plot == float('-inf'), torch.tensor(-1.0), mask_to_plot)
        heatmap_data = torch.where(heatmap_data == 0, torch.tensor(1.0), heatmap_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', vmin=-1, vmax=1)
    plt.title(title)
    filename = f"debug_{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"📸 Saved visualization to: {filename}")
    plt.close()

def debug_baseline():
    print("\n🚀 STARTING BASELINE (BD3LM) DEBUG PIPELINE...")
    
    # 1. Load Config
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path=CONFIG_PATH)
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=OVERRIDES)
        OmegaConf.resolve(cfg)
        print("✅ Config loaded.")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # 2. Load Model
    print("\n📦 Loading Model...")
    try:
        tok_path = cfg.data.get('tokenizer_name_or_path', 'gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Init model
        model = diffusion.Diffusion(cfg, tokenizer=tokenizer)
        model.eval()
        model.cuda()
        print("✅ Model loaded.")
        
        # Check if HDP is strictly OFF
        if hasattr(model, 'use_hdp_attention'):
            print(f"ℹ️  HDP Mode Status: {model.use_hdp_attention}")
            if model.use_hdp_attention:
                print("⚠️  WARNING: HDP is ON! Baseline debug might be invalid.")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        traceback.print_exc()
        return

    # 3. Simulate Baseline Input (No Block Indices)
    seq_len = cfg.model.length
    print(f"\n📏 Model Length: {seq_len}")
    
    # Input giả lập: [Question ... Answer ...]
    # BD3LM thường cần sequence length * 2 (vì cross-attn xt và x0)
    # Nhưng input vào model forward chỉ là x_t (nếu cross_attn=False) hoặc x_t, x_0
    
    x_input = torch.randint(0, 1000, (1, seq_len)).to('cuda')
    
    # 4. Check Mask chuẩn của BD3LM
    print("\n🔍 CHECK 1: Standard BD3LM Mask")
    try:
        # Trong BD3LM, mask được tạo sẵn trong self.backbone.block_diff_mask
        if hasattr(model.backbone, 'block_diff_mask'):
            base_mask = model.backbone.block_diff_mask
            print(f"   Mask Found via 'block_diff_mask': {base_mask.shape}")
            visualize_mask(base_mask, title="Baseline_BD3LM_Structure")
            
            # Logic check: Diagonal phải sáng (Self-attention)
            # BD3LM Mask (cho training) thường là ma trận boolean [2*L, 2*L]
            # Vùng [0:L, 0:L] thường là block diagonal
            val_diag = base_mask[0, 0].item() if base_mask.dim() > 2 else base_mask[0, 0].item()
            print(f"   Diagonal check (0,0): {val_diag} (Should be True/1)")
        else:
            print("⚠️ No static mask found (maybe generated dynamically or using FlashAttn causal mode)")

    except Exception as e:
        print(f"❌ Error inspecting baseline mask: {e}")
        traceback.print_exc()

    # 5. Check Forward
    print("\n🔍 CHECK 2: Forward Pass (No hdp_mask arg)")
    try:
        t = torch.ones(1, 1).to('cuda') * 0.5
        sigma = model._sigma_from_p(t)
        
        with torch.no_grad():
            # Quan trọng: GỌI KHÔNG CÓ BLOCK_INDICES
            logits = model.forward(x_input, sigma=sigma, block_indices=None)
            
        print(f"   Logits Shape: {logits.shape}")
        if torch.isnan(logits).any():
            print("   ❌ FAIL: NaNs detected.")
        else:
            print("   ✅ PASS: Baseline forward works correctly.")
            
    except Exception as e:
        print(f"❌ Failed check 2: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_baseline()