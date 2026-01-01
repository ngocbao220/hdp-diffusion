"""
Test script for HDP Attention Mask Logic (Visual & Logic Check)

Generates a visual heatmap mimicking the block-level structure:
- Question (Q)
- Plan (P)
- Execution (E)
"""

import torch
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

# Giả lập import (Em thay bằng import thực tế từ project của em)
try:
    from models.hdp_attention_mask import get_hdp_attention_mask
except ImportError:
    # Dummy implementation for standalone testing if module is missing
    def get_hdp_attention_mask(block_indices, seq_len, block_sizes, causal_within_block=False):
        mask = torch.ones((1, seq_len, seq_len))
        # Basic logic simulation for demo
        for r in range(seq_len):
            for c in range(seq_len):
                q_blk = block_indices[0, r].item()
                k_blk = block_indices[0, c].item()
                # Logic: Q sees Q; P sees Q+P; E sees Q+P+E
                if q_blk == 0 and k_blk != 0: mask[0, r, c] = 0
                if q_blk == 1 and k_blk == 2: mask[0, r, c] = 0
                # Within block causal (optional simulation)
                # if causal_within_block and q_blk == k_blk and c > r: mask[0, r, c] = 0
        return mask

def plot_mask_heatmap(mask, block_indices, save_path="hdp_mask_visual.png"):
    """
    Vẽ heatmap đẹp như hình minh họa trong paper dùng Matplotlib.
    """
    mask_2d = mask[0].cpu().numpy()
    seq_len = mask_2d.shape[0]
    block_ids = block_indices[0].cpu().numpy()
    
    # Setup plot
    plt.figure(figsize=(8, 8))
    
    # Custom colormap: Red (Masked/0) -> Green (Allowed/1)
    # Hoặc style trắng/chéo như em muốn. Ở đây dùng màu cho dễ nhìn.
    cmap = ListedColormap(['#ffebee', '#e8f5e9']) # Light Red to Light Green
    
    # Vẽ heatmap
    ax = sns.heatmap(mask_2d, cmap=cmap, cbar=False, 
                     linewidths=1, linecolor='gray', square=True,
                     annot=True, fmt='g', annot_kws={"size": 8, "color": "black"})

    # Vẽ đường ranh giới giữa các khối (Block boundaries)
    # Tìm điểm chuyển giao giữa các block
    boundaries = [0]
    labels = []
    current_blk = block_ids[0]
    for i, blk in enumerate(block_ids):
        if blk != current_blk:
            boundaries.append(i)
            labels.append(current_blk)
            current_blk = blk
    boundaries.append(seq_len)
    labels.append(current_blk)
    
    # Mapping tên block
    name_map = {0: 'Question', 1: 'Plan', 2: 'Execution'}
    block_names = [name_map.get(l, str(l)) for l in labels]

    # Vẽ đường kẻ đậm chia block
    for b in boundaries[1:-1]:
        plt.axhline(b, color='black', linewidth=3)
        plt.axvline(b, color='black', linewidth=3)

    # Đặt nhãn trục (Căn giữa block)
    mid_points = [(boundaries[i] + boundaries[i+1])/2 for i in range(len(boundaries)-1)]
    plt.xticks(mid_points, block_names, fontsize=12, fontweight='bold')
    plt.yticks(mid_points, block_names, fontsize=12, fontweight='bold', rotation=0)
    
    # Trang trí
    plt.title("HDP Attention Mask Structure\n(1 = Attend, 0 = Mask)", fontsize=14, pad=20)
    plt.xlabel("Key (Source)", fontsize=12)
    plt.ylabel("Query (Target)", fontsize=12)
    
    # Thêm text chú thích vùng quan trọng
    # Vùng Plan -> Exec (Góc phải giữa): Phải là 0
    p_idx = boundaries[1]
    e_idx = boundaries[2] if len(boundaries) > 2 else seq_len
    
    # Highlight vùng cấm (Plan looking at Exec)
    if len(boundaries) >= 3:
        # Tọa độ vùng cấm: Rows [bound[1]:bound[2]], Cols [bound[2]:]
        rect = plt.Rectangle((boundaries[2], boundaries[1]), 
                             seq_len - boundaries[2], boundaries[2] - boundaries[1], 
                             fill=False, edgecolor='red', linewidth=4, linestyle='--')
        ax.add_patch(rect)
        plt.text(seq_len-0.5, boundaries[1]+0.5, "CRITICAL:\nMUST BE 0", 
                 color='red', fontsize=10, ha='right', va='top', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Visualization saved to: {save_path}")
    print("(Open this image to see the block structure clearly)")

def print_block_summary(mask, block_indices):
    """
    In ra ma trận tóm tắt 3x3 (Block-level summary) trên terminal
    """
    mask_2d = mask[0]
    block_ids = block_indices[0]
    
    # Lấy đại diện 1 phần tử từ mỗi vùng giao thoa block
    # Giả sử block indices đã sort tăng dần 0 -> 1 -> 2
    b0_idx = (block_ids == 0).nonzero(as_tuple=True)[0][0].item()
    b1_idx = (block_ids == 1).nonzero(as_tuple=True)[0][0].item()
    b2_idx = (block_ids == 2).nonzero(as_tuple=True)[0][0].item()
    
    idxs = [b0_idx, b1_idx, b2_idx]
    names = ["Q", "P", "E"]
    
    print("\n" + "="*40)
    print("BLOCK-LEVEL CONNECTION MATRIX (3x3)")
    print("="*40)
    print("      |  Q  |  P  |  E  |")
    print("-------------------------")
    
    for i, row_idx in enumerate(idxs):
        row_str = f"   {names[i]}  |"
        for col_idx in idxs:
            val = mask_2d[row_idx, col_idx].item()
            symbol = " 1 " if val == 1 else " . " # 1 là nhìn thấy, . là bị che
            row_str += f" {symbol} |"
        print(row_str)
        print("-------------------------")
    print("Legend: '1' = Connected, '.' = Masked")

def check_attention_rules(mask, block_indices):
    """Kiểm tra logic đúng sai tự động"""
    mask_2d = mask[0]
    b_ids = block_indices[0]
    seq_len = mask_2d.shape[0]
    
    all_passed = True
    print("\n[Auto-Check Rules]")
    
    # 1. Plan (1) NOT see Exec (2)
    p_mask = (b_ids == 1).unsqueeze(1) # Rows where Query is Plan
    e_mask = (b_ids == 2).unsqueeze(0) # Cols where Key is Exec
    bad_attention = (mask_2d * p_mask * e_mask).sum()
    
    if bad_attention == 0:
        print("✓ Rule: Plan cannot see Execution -> PASSED")
    else:
        print(f"✗ Rule: Plan cannot see Execution -> FAILED ({bad_attention} leaks found)")
        all_passed = False
        
    # 2. Exec (2) MUST see Plan (1)
    e_rows = (b_ids == 2)
    p_cols = (b_ids == 1)
    # Check if Exec rows attend to Plan cols (assuming bidirectional within blocks)
    # We take a sample check
    e_idx = e_rows.nonzero(as_tuple=True)[0][0]
    p_idx = p_cols.nonzero(as_tuple=True)[0][0]
    
    if mask_2d[e_idx, p_idx] == 1:
        print("✓ Rule: Execution can see Plan -> PASSED")
    else:
        print("✗ Rule: Execution can see Plan -> FAILED")
        all_passed = False
        
    return all_passed

def main():
    # 1. Setup Dummy Data (Tiny Example: 2 tokens per block)
    q_len, p_len, e_len = 2, 2, 2
    seq_len = q_len + p_len + e_len
    # Block indices: [0, 0, 1, 1, 2, 2]
    block_indices = torch.tensor([[0]*q_len + [1]*p_len + [2]*e_len], dtype=torch.long)
    
    print(f"Generating mask for sequence length {seq_len}...")
    
    # 2. Call your function
    # Lưu ý: Pass đúng tham số như hàm em đã viết
    mask = get_hdp_attention_mask(
        block_indices=block_indices,
        seq_len=seq_len,
        block_sizes=(q_len, p_len, e_len)
    )
    
    # 3. Print Summary Table (Dễ hiểu trên terminal)
    print_block_summary(mask, block_indices)
    
    # 4. Auto-verify rules
    passed = check_attention_rules(mask, block_indices)
    
    # 5. Generate Image (Vẽ như hình em muốn)
    plot_mask_heatmap(mask, block_indices)
    
    if passed:
        print("\nSUCCESS: Attention mask logic is correct!")
        sys.exit(0)
    else:
        print("\nFAILURE: Logic checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()