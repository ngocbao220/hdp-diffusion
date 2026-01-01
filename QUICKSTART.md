# QUICK START: Hierarchical Block Diffusion

## 🎯 Mục tiêu
Thêm hierarchical reasoning (Plan-then-Generate) vào BD3-LM để model có thể:
1. Đọc Question
2. Tạo Plan (high-level reasoning)
3. Thực thi Execution (detailed steps)

với constraint: **Plan KHÔNG thể nhìn thấy Execution** (causal)

## 📦 Files đã tạo

| File | Mục đích |
|------|---------|
| `models/hierarchical_mask.py` | Attention mask phân tầng |
| `hierarchical_dataloader.py` | Data collator cho [Q, P, E] format |
| `configs/algo/hierarchical.yaml` | Config hierarchical training |
| `scripts/train/train_hierarchical_bd3lm.sh` | Training script |
| `test_hierarchical_mask.py` | Test mask correctness |
| `HIERARCHICAL_README.md` | Full docs (Vietnamese) |
| `IMPLEMENTATION_SUMMARY.md` | Detailed summary |

## ⚡ Chạy thử ngay (3 bước)

### Bước 1: Test mask
```bash
cd /workspace/hdp-diffusion
python test_hierarchical_mask.py
```
Kết quả mong đợi:
- ✅ All tests passed
- 📊 File `hierarchical_mask_test.png` được tạo

### Bước 2: Chuẩn bị data mẫu
```python
# Tạo file test_data.json
[
  {
    "question": "What is machine learning?",
    "plan": "I will first define ML, then explain key concepts.",
    "execution": "Machine learning is a field of AI that enables computers to learn from data without being explicitly programmed. Key concepts include: 1) Training data..."
  }
]
```

### Bước 3: Train thử (với data nhỏ)
```bash
# Edit script để giảm số steps (test nhanh)
vim scripts/train/train_hierarchical_bd3lm.sh
# Set: MAX_STEPS=1000

# Run
sbatch scripts/train/train_hierarchical_bd3lm.sh
```

## 🔧 Tùy chỉnh nhanh

### Thay đổi độ dài blocks
```bash
# Trong training script:
QUESTION_LEN=128  # Giảm từ 256
PLAN_LEN=384      # Tăng từ 256  
EXEC_LEN=512      # Giữ nguyên
```

### Thay đổi block size (speed vs quality)
```bash
BLOCK_SIZE=8   # Nhanh hơn, chất lượng thấp hơn
BLOCK_SIZE=16  # Cân bằng (recommended)
BLOCK_SIZE=32  # Chậm hơn, chất lượng cao hơn
```

### Sửa logic tách data
File: `hierarchical_dataloader.py`, line ~90
```python
def process_example(example):
    text = example['text']
    
    # TODO: Thay bằng logic của bạn
    # Ví dụ: dùng regex, parse markdown, etc.
    question = extract_with_your_method(text)
    plan = extract_with_your_method(text)
    execution = extract_with_your_method(text)
    
    return {'question': ..., 'plan': ..., 'execution': ...}
```

## 🎨 Visualize mask

```python
from models.hierarchical_mask import create_hierarchical_mask
import matplotlib.pyplot as plt

mask = create_hierarchical_mask(
    seqlen=1024, block_size=16,
    question_len=256, plan_len=256, exec_len=512
)

plt.imshow(mask.float(), cmap='RdYlGn')
plt.savefig('my_mask.png')
```

Kiểm tra:
- ✅ Vùng Plan-to-Execution phải **TRẮNG** (không attend)
- ✅ Execution-to-Plan phải **XANH** (có attend)

## 📊 Cấu trúc Sequence

```
Training input:
┌─────────┬──────────┬──────────┬──────────┬──────────┐
│Question │ Plan_xt  │ Plan_x0  │ Exec_xt  │ Exec_x0  │
│  (256)  │  (256)   │  (256)   │  (512)   │  (512)   │
└─────────┴──────────┴──────────┴──────────┴──────────┘
           └─ Noisy ─┘ └─Clean─┘ └─ Noisy ─┘ └─Clean─┘

Total length: 256 + 256*2 + 512*2 = 1792 tokens
```

## 🐛 Debug nhanh

### Lỗi: "Mask shape mismatch"
```python
# Check dimensions:
total = question_len + plan_len*2 + exec_len*2
print(f"Expected mask shape: {total}x{total}")
print(f"Actual mask shape: {mask.shape}")
```

### Lỗi: "CUDA OOM"
```bash
# Trong training script:
BATCH_SIZE=16     # Giảm từ 64
MODEL_SIZE=tiny   # Thay vì small
```

### Model không học
1. Print data samples: Check format đúng chưa
2. Visualize attention: Check mask đúng chưa
3. Reduce LR: `LR=1e-4`
4. Increase warmup: `WARMUP_STEPS=20000`

## 📞 Getting Help

### Câu hỏi thường gặp:

**Q: Làm sao biết mask đúng?**
A: Run `python test_hierarchical_mask.py` - phải pass tất cả tests

**Q: Data của tôi không có format [Q, P, E]?**  
A: Sửa `process_example()` trong `hierarchical_dataloader.py`

**Q: Muốn thử với data có sẵn?**
A: Dùng OpenWebText, collator sẽ tự split (25% Q, 25% P, 50% E)

**Q: Training bao lâu?**
A: 
- Test (1K steps): ~30 phút (1 GPU)
- Small run (10K steps): ~5 giờ
- Full training (100K steps): ~48 giờ

**Q: Làm sao biết đang work?**
A:
- Loss giảm liên tục
- Valid NELBO giảm
- Check generated samples có structure

## 📚 Files để đọc thêm

1. **HIERARCHICAL_README.md**: Full documentation (Vietnamese)
2. **IMPLEMENTATION_SUMMARY.md**: Chi tiết implementation
3. **models/hierarchical_mask.py**: Code + comments chi tiết
4. **test_hierarchical_mask.py**: Examples + verification

## ✅ Checklist

Trước khi train:
- [ ] Test mask passed: `python test_hierarchical_mask.py`
- [ ] Visualization looks correct: Check `.png` file
- [ ] Data format verified: Print 3-5 samples
- [ ] Hyperparameters set: Check training script
- [ ] Output dir created: `mkdir outputs/test_run`

Sau 1000 steps đầu:
- [ ] Loss đang giảm (check tensorboard/logs)
- [ ] No errors/warnings
- [ ] Generated samples có structure (optional)

## 🚀 Production Checklist

Khi ready to scale:
- [ ] Data quality checked thoroughly
- [ ] Hyperparameters tuned (LR, warmup, etc.)
- [ ] Multiple seeds tested
- [ ] Evaluation metrics defined
- [ ] Comparison với baseline

---

**Bắt đầu từ đây:** Run `python test_hierarchical_mask.py` ✨
