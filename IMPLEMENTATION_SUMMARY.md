# TÓM TẮT THỰC HIỆN - HIERARCHICAL DISCRETE DIFFUSION

## ✅ CÁC BƯỚC ĐÃ HOÀN THÀNH

### Bước 1: ✅ Tìm và hiểu cấu trúc Mask

**Vị trí:** `models/dit.py` (dòng 30-75)

**Phát hiện:**
- Function `block_diff_mask()` định nghĩa mask gốc cho BD3-LM
- Function `gen_mask()` (dòng 706) tạo mask cho attention
- Mask được lưu trong `self.block_diff_mask`
- Hỗ trợ 2 backend: `sdpa` và `flex` (FlexAttention)

### Bước 2: ✅ Tạo Hierarchical Mask

**File mới:** `models/hierarchical_mask.py`

**Chức năng:**
```python
hierarchical_block_diff_mask(
    b, h, q_idx, kv_idx,
    question_len, plan_len, exec_len, 
    block_size, n
)
```

**Cấu trúc Mask (theo yêu cầu):**
- ✅ Question → Question: Full attention
- ✅ Plan → Question: Full attention  
- ✅ Plan → Plan: Block diffusion pattern
- ❌ Plan → Execution: **BLOCKED** (giữ tính nhân quả)
- ✅ Execution → Question: Full attention
- ✅ Execution → Plan: Full attention
- ✅ Execution → Execution: Block diffusion pattern

**Format sequence:**
```
[Question | Plan_xt | Plan_x0 | Exec_xt | Exec_x0]
   256    |   256    |   256   |   512   |   512   = 1792 tokens total
```

### Bước 3: ✅ Xử lý Input Data

**File mới:** `hierarchical_dataloader.py`

**Class chính:**

1. **HierarchicalDataCollator**
   - Nhận input: `{'question': ..., 'plan': ..., 'execution': ...}`
   - Hoặc: `{'text': ...}` (tự động split)
   - Output: Tensor shape `[batch, question_len + plan_len + exec_len]`

2. **create_hierarchical_dataset()**
   - Tạo dataset từ OpenWebText hoặc data khác
   - Tự động split thành 3 phần
   
3. **load_reasoning_dataset()**
   - Load từ JSON đã format sẵn
   - Format: `[{"question": ..., "plan": ..., "execution": ...}, ...]`

**Cách sử dụng:**
```python
from hierarchical_dataloader import HierarchicalDataCollator

collator = HierarchicalDataCollator(
    tokenizer=tokenizer,
    question_len=256,
    plan_len=256, 
    exec_len=512
)

batch = collator(examples)
# Output: {'input_ids': tensor, 'attention_mask': tensor, 'hierarchical_info': dict}
```

### Bước 4: ✅ Tích hợp vào Model

**Thay đổi trong:** `models/dit.py`

**Function mới:**
```python
def gen_mask(self, seqlen, block_size, attn_backend='sdpa', hierarchical_config=None):
    if hierarchical_config is not None:
        # Use hierarchical mask
        from models.hierarchical_mask import create_hierarchical_mask
        self.block_diff_mask = create_hierarchical_mask(...)
    else:
        # Use original BD3-LM mask
        ...
```

### Bước 5: ✅ Tắt/Đơn giản hóa tính năng không cần thiết

**Config:** `configs/algo/hierarchical.yaml`

```yaml
hierarchical:
  enabled: true
  question_len: 256
  plan_len: 256
  exec_len: 512

training:
  use_hierarchical_collator: true
  var_length_gen: false  # ❌ Tắt arbitrary-length

sampling:
  hierarchical_mode: 'full'
  var_length: false      # ❌ Tắt variable-length
  first_hitting: true    # ✅ Bật (nhanh hơn)
  kv_cache: true         # ✅ Bật (tăng tốc)
```

### Bước 6: ✅ Training Script

**File mới:** `scripts/train/train_hierarchical_bd3lm.sh`

```bash
#!/bin/bash
QUESTION_LEN=256
PLAN_LEN=256
EXEC_LEN=512
BLOCK_SIZE=16

python -u main.py \
    mode=train \
    model=small \
    model.length=1024 \
    algo=bd3lm \
    block_size=${BLOCK_SIZE} \
    training.hierarchical.enabled=true \
    training.hierarchical.question_len=${QUESTION_LEN} \
    training.hierarchical.plan_len=${PLAN_LEN} \
    training.hierarchical.exec_len=${EXEC_LEN}
```

### Bước 7: ✅ Testing & Documentation

**Files:**
- `test_hierarchical_mask.py`: Test script để verify mask
- `HIERARCHICAL_README.md`: Documentation đầy đủ bằng tiếng Việt

## 📋 CÁCH SỬ DỤNG NHANH

### 1. Chuẩn bị data

**Option A: Dữ liệu có sẵn cấu trúc**
```json
// data.json
[
  {
    "question": "What is the capital of France?",
    "plan": "I need to recall European geography and capitals.",
    "execution": "Paris is the capital and largest city of France, located in the north-central part of the country."
  }
]
```

**Option B: Tự động split từ văn bản**
```python
# Repo sẽ tự động chia:
# - 25% đầu → Question
# - 25% tiếp → Plan  
# - 50% còn lại → Execution
```

### 2. Training

```bash
# Chạy training script
sbatch scripts/train/train_hierarchical_bd3lm.sh

# Hoặc run trực tiếp
python main.py \
    mode=train \
    model=small \
    algo=bd3lm \
    block_size=16 \
    training.hierarchical.enabled=true \
    training.hierarchical.question_len=256 \
    training.hierarchical.plan_len=256 \
    training.hierarchical.exec_len=512
```

### 3. Test mask (trước khi train)

```bash
python test_hierarchical_mask.py
# Output: 
#  - ✅ Verification results
#  - 📊 Visualization: hierarchical_mask_test.png
```

### 4. Customize cho domain của bạn

**Sửa logic split trong `hierarchical_dataloader.py`:**

```python
def process_example(example):
    text = example['text']
    
    # TODO: Replace with your logic
    # Ví dụ:
    # - Parse từ markdown structure
    # - Dùng regex tách sections
    # - Dùng model khác để identify
    
    question = your_extract_question_logic(text)
    plan = your_extract_plan_logic(text)
    execution = your_extract_execution_logic(text)
    
    return {
        'question': tokenizer.encode(question),
        'plan': tokenizer.encode(plan),
        'execution': tokenizer.encode(execution),
    }
```

## 📁 CẤU TRÚC FILES MỚI

```
hdp-diffusion/
├── models/
│   ├── hierarchical_mask.py          # ✨ NEW: Hierarchical attention mask
│   └── dit.py                         # 🔧 MODIFIED: Added hierarchical support
│
├── hierarchical_dataloader.py         # ✨ NEW: Data collator & dataset utils
│
├── configs/
│   └── algo/
│       └── hierarchical.yaml          # ✨ NEW: Hierarchical config
│
├── scripts/
│   └── train/
│       └── train_hierarchical_bd3lm.sh # ✨ NEW: Training script
│
├── test_hierarchical_mask.py          # ✨ NEW: Test & verification
│
├── HIERARCHICAL_README.md             # ✨ NEW: Full documentation (Vietnamese)
│
└── IMPLEMENTATION_SUMMARY.md          # ✨ NEW: This file
```

## 🔍 ĐIỂM QUAN TRỌNG CẦN LƯU Ý

### 1. Attention Mask Structure

Mask phải đảm bảo:
```
✅ Plan có thể "đọc" Question (để hiểu ngữ cảnh)
✅ Execution có thể "đọc" cả Question và Plan
❌ Plan KHÔNG thể "đọc" Execution (tính nhân quả)
```

Kiểm tra bằng test:
```python
# Plan cannot see Execution
plan_to_exec = mask[question_end:plan_x0_end, plan_x0_end:]
assert not plan_to_exec.any(), "Plan should NOT see Execution!"
```

### 2. Data Format

**Sequence structure:**
```
Input:  [Q Q Q ... | P P P ... | E E E E ...]
        └─ 256 ───┘ └─ 256 ──┘ └── 512 ───┘

For training (with xt and x0):
[Q | Plan_xt | Plan_x0 | Exec_xt | Exec_x0]
 └─ 256 ─┘ └── 256 ──┘ └── 256 ──┘ └── 512 ──┘ └── 512 ──┘
```

### 3. Block Size Trade-off

| Block Size | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| 1 | Chậm nhất | Tốt nhất | Baseline (AR) |
| 4-8 | Trung bình | Rất tốt | Research |
| 16-32 | Nhanh | Tốt | Production |
| 1024 | Nhanh nhất | Thấp hơn | Fast inference |

**Khuyến nghị:** Bắt đầu với `block_size=16`

### 4. Integration với Codebase Gốc

**Không cần sửa nhiều!** Chỉ cần:

```python
# In diffusion.py or main training loop:
if config.training.hierarchical.enabled:
    # Use hierarchical mask
    model.backbone.gen_mask(
        seqlen=config.model.length,
        block_size=config.block_size,
        hierarchical_config={
            'question_len': config.training.hierarchical.question_len,
            'plan_len': config.training.hierarchical.plan_len,
            'exec_len': config.training.hierarchical.exec_len,
        }
    )
    
    # Use hierarchical collator
    from hierarchical_dataloader import HierarchicalDataCollator
    collator = HierarchicalDataCollator(tokenizer, ...)
else:
    # Original BD3-LM behavior
    model.backbone.gen_mask(seqlen, block_size)
```

## ⚠️ TROUBLESHOOTING

### Lỗi thường gặp:

1. **"Mask dimensions don't match"**
   - Kiểm tra: `total_len = question_len + plan_len*2 + exec_len*2`
   - Đảm bảo nhân 2 vì có cả xt và x0

2. **"CUDA out of memory"**
   - Giảm batch size: `loader.batch_size=32`
   - Giảm model size: `model=tiny`
   - Giảm sequence length

3. **Model không học được**
   - Test mask trước: `python test_hierarchical_mask.py`
   - Visualize attention patterns
   - Check data format: In ra vài samples
   - Giảm learning rate: `optim.lr=1e-4`

4. **"FlexAttention not available"**
   - Dùng SDPA thay vì: `model.attn_backend=sdpa`
   - Hoặc cài: `pip install flash-attn==2.5.6`

## 🚀 NEXT STEPS

### Để chạy thử ngay:

1. **Test mask (không cần GPU):**
   ```bash
   python test_hierarchical_mask.py
   ```

2. **Prepare data nhỏ để test:**
   ```python
   from hierarchical_dataloader import load_reasoning_dataset
   
   # Create small test dataset (100 examples)
   dataset = create_test_data(num_examples=100)
   ```

3. **Run training trên data nhỏ:**
   ```bash
   python main.py \
       mode=train \
       model=tiny \
       training.max_steps=1000 \
       training.hierarchical.enabled=true
   ```

4. **Verify loss giảm:**
   - Check tensorboard/wandb logs
   - Loss nên giảm sau ~100 steps

5. **Scale lên:**
   - Tăng data size
   - Tăng model size: `model=small`
   - Tăng training steps: `training.max_steps=100000`

### Để customize cho domain của bạn:

1. **Implement data parser:**
   - Sửa `hierarchical_dataloader.py`
   - Function `process_example()`
   - Parse theo format của bạn

2. **Tune hyperparameters:**
   - Độ dài: `question_len`, `plan_len`, `exec_len`
   - Block size: `4, 8, 16, 32`
   - Learning rate, warmup, etc.

3. **Add evaluation metrics:**
   - Đo chất lượng Plan riêng
   - Đo chất lượng Execution riêng
   - Đo coherence giữa Plan và Execution

## 📚 TÀI LIỆU THAM KHẢO

- **Paper gốc:** Block Diffusion (ICLR 2025)
- **Appendix B.6, B.7:** Hierarchical architecture design
- **Figure 4:** Attention mask visualization
- **HIERARCHICAL_README.md:** Full documentation

## ✅ CHECKLIST TRƯỚC KHI TRAIN

- [ ] Test mask: `python test_hierarchical_mask.py`
- [ ] Check visualization: `hierarchical_mask_test.png`
- [ ] Prepare data (100-1000 examples để test)
- [ ] Verify data format: Print ra vài samples
- [ ] Set hyperparameters trong config
- [ ] Tạo output directory: `mkdir -p outputs/hierarchical_test`
- [ ] Run small test: 1000 steps, tiny model
- [ ] Check loss giảm
- [ ] Scale lên full training

---

**Tóm lại:** Đã implement đầy đủ 3 bước theo yêu cầu của bạn:
1. ✅ Tìm và modify mask (Plan không thấy Execution)
2. ✅ Xử lý input data thành [Question, Plan, Execution]
3. ✅ Tắt arbitrary-length generation, giữ độ dài cố định

Code đã sẵn sàng để test và chạy thử! 🎉
