# Hierarchical Discrete Diffusion: Plan-then-Generate Architecture

## Tổng quan (Overview)

Repo này đã được mở rộng để hỗ trợ **mô hình suy luận phân tầng** (hierarchical reasoning) dựa trên cơ chế khối và mặt nạ chú ý (block diffusion with hierarchical attention masks).

### Kiến trúc chính:
```
[Question] → [Plan Block] → [Execution Block]
    ↓            ↓                  ↓
  Context    High-level      Detailed steps
             reasoning
```

### Nguyên tắc Attention Mask:
- ✅ **Plan Block** có thể nhìn thấy **Question**
- ✅ **Execution Block** có thể nhìn thấy **Question** và **Plan Block**  
- ❌ **Plan Block** KHÔNG thể nhìn thấy **Execution Block** (giữ tính nhân quả)

## Các file mới được thêm vào

### 1. `models/hierarchical_mask.py`
Định nghĩa attention mask phân tầng theo Figure 4 trong paper (Appendix B.6, B.7).

**Chức năng chính:**
- `hierarchical_block_diff_mask()`: Tạo mask cho cấu trúc Plan-then-Generate
- `create_hierarchical_mask()`: Helper function để khởi tạo mask

**Cấu trúc mask:**
```
Sequence: [Question | Plan_xt | Plan_x0 | Exec_xt | Exec_x0]

Question tokens: Full self-attention
Plan tokens:     Can see Question + Plan (block diffusion pattern)
Exec tokens:     Can see Question + Plan + Exec (block diffusion pattern)
```

### 2. `hierarchical_dataloader.py`
Xử lý dữ liệu đầu vào cho training và inference.

**Các class chính:**
- `HierarchicalDataCollator`: Collate data thành format [Question, Plan, Execution]
- `create_hierarchical_dataset()`: Tạo dataset từ dữ liệu thô
- `load_reasoning_dataset()`: Load dataset đã được format sẵn

**Input format:**
```json
{
  "question": "What is the capital of France?",
  "plan": "I need to recall European capitals...",
  "execution": "Paris is the capital and largest city..."
}
```

### 3. `configs/algo/hierarchical.yaml`
Config cho hierarchical training.

**Tham số chính:**
```yaml
hierarchical:
  question_len: 256  # Độ dài phần question
  plan_len: 256      # Độ dài Plan Block
  exec_len: 512      # Độ dài Execution Block
```

### 4. `scripts/train/train_hierarchical_bd3lm.sh`
Script để chạy training cho mô hình hierarchical.

## Hướng dẫn sử dụng (Quick Start)

### Bước 1: Chuẩn bị dữ liệu

Có 2 cách chuẩn bị dữ liệu:

#### Option A: Dữ liệu đã có cấu trúc [Question, Plan, Execution]

Tạo file JSON:
```json
[
  {
    "question": "Your question here...",
    "plan": "High-level plan...",
    "execution": "Detailed execution..."
  },
  ...
]
```

Sau đó load trong code:
```python
from hierarchical_dataloader import load_reasoning_dataset

dataset = load_reasoning_dataset(
    dataset_path='path/to/your/data.json',
    tokenizer=tokenizer,
    question_len=256,
    plan_len=256,
    exec_len=512
)
```

#### Option B: Tự động split từ văn bản dài

Nếu bạn có dữ liệu dạng văn bản thông thường (như OpenWebText), collator sẽ tự động chia:
- 25% đầu → Question
- 25% tiếp theo → Plan  
- 50% còn lại → Execution

**Lưu ý:** Cách này chỉ là placeholder. Bạn nên implement logic split phù hợp với domain của mình.

### Bước 2: Training

```bash
# Sửa tham số trong script nếu cần
vim scripts/train/train_hierarchical_bd3lm.sh

# Chạy training
sbatch scripts/train/train_hierarchical_bd3lm.sh
```

Hoặc chạy trực tiếp:
```bash
python main.py \
    mode=train \
    model=small \
    model.length=1024 \
    algo=bd3lm \
    block_size=16 \
    training.hierarchical.enabled=true \
    training.hierarchical.question_len=256 \
    training.hierarchical.plan_len=256 \
    training.hierarchical.exec_len=512
```

### Bước 3: Inference

```python
from models.hierarchical_mask import create_hierarchical_mask
from hierarchical_dataloader import HierarchicalDataCollator

# Initialize model with hierarchical mask
model.gen_mask(
    seqlen=1024,
    block_size=16,
    hierarchical_config={
        'question_len': 256,
        'plan_len': 256,
        'exec_len': 512
    }
)

# Prepare input
collator = HierarchicalDataCollator(tokenizer, 256, 256, 512)
batch = collator([{'question': question_tokens, ...}])

# Generate
samples = model.restore_model_and_sample(num_steps=50)
```

## Kiểm tra Attention Mask

Để visualize mask và đảm bảo nó đúng như mong muốn:

```python
from models.hierarchical_mask import create_hierarchical_mask
import matplotlib.pyplot as plt

mask = create_hierarchical_mask(
    seqlen=1024,
    block_size=16,
    question_len=256,
    plan_len=256,
    exec_len=512,
    attn_backend='sdpa'
)

# Visualize
plt.figure(figsize=(12, 12))
plt.imshow(mask.float(), cmap='binary')
plt.title('Hierarchical Attention Mask')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar(label='Can Attend')

# Add boundary lines
plt.axvline(x=256, color='r', linestyle='--', label='Question/Plan boundary')
plt.axvline(x=512, color='g', linestyle='--', label='Plan/Exec boundary')
plt.axhline(y=256, color='r', linestyle='--')
plt.axhline(y=512, color='g', linestyle='--')
plt.legend()
plt.savefig('hierarchical_mask_visualization.png')
```

## Tùy chỉnh cho domain của bạn

### 1. Sửa logic splitting trong DataCollator

File: `hierarchical_dataloader.py`

Tìm function `process_example()` và implement logic riêng:

```python
def process_example(example):
    text = example['text']
    
    # TODO: Implement your domain-specific logic
    # Ví dụ: 
    # - Dùng regex để tách sections
    # - Dùng model khác để identify plan vs execution
    # - Parse từ structured format
    
    question = extract_question(text)
    plan = extract_plan(text)
    execution = extract_execution(text)
    
    return {
        'question': tokenizer.encode(question),
        'plan': tokenizer.encode(plan),
        'execution': tokenizer.encode(execution),
    }
```

### 2. Điều chỉnh độ dài blocks

Tùy vào task, bạn có thể cần thay đổi:

```yaml
# configs/algo/hierarchical.yaml
hierarchical:
  question_len: 128   # Shorter questions
  plan_len: 384       # Longer plans
  exec_len: 512       # Keep execution the same
  total_len: 1024
```

### 3. Thay đổi block size

Block size ảnh hưởng đến trade-off giữa chất lượng và tốc độ:
- `block_size=1`: Autoregressive (chậm nhất, chất lượng cao nhất)
- `block_size=4,8,16`: BD3-LM (cân bằng)
- `block_size=1024`: Full diffusion (nhanh nhất, chất lượng thấp hơn)

```bash
# Trong training script
BLOCK_SIZE=8  # hoặc 4, 16, 32, ...
```

## Troubleshooting

### Lỗi: "Mask dimensions don't match"

Kiểm tra tổng độ dài:
```python
total = question_len + plan_len*2 + exec_len*2  # *2 vì có xt và x0
assert total == expected_mask_size
```

### Lỗi: "Unknown attention backend"

Đảm bảo bạn cài đặt đúng dependencies:
```bash
pip install torch>=2.0  # Cần cho SDPA
# Hoặc cho FlexAttention (tùy chọn):
pip install flash-attn==2.5.6
```

### Model không học được

1. Kiểm tra attention mask có đúng không (dùng visualization ở trên)
2. Giảm learning rate: `optim.lr=1e-4`
3. Tăng warmup steps: `training.warmup_steps=20000`
4. Check data quality: In ra vài samples để xem format có đúng không

## So sánh với baseline

| Model | Block Size | Speed | Quality | Hierarchical |
|-------|-----------|-------|---------|--------------|
| AR | 1 | ⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| MDLM | 1024 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ |
| BD3-LM | 16 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |
| **Hier-BD3-LM** | 16 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |

## Tính năng đã tắt (để đơn giản hóa)

Theo yêu cầu của bạn, các tính năng sau đã được tắt/đơn giản hóa:

1. ❌ **Arbitrary-length generation**: Fixed length (1024 tokens)
2. ❌ **Variable-length sampling**: Disabled by default
3. ✅ **KV caching**: Enabled để tăng tốc
4. ✅ **First-hitting sampler**: Enabled (faster than DDPM)

Nếu muốn bật lại, sửa trong config:
```yaml
sampling:
  var_length: true
  first_hitting: false
```

## Tham khảo Paper

Appendix liên quan:
- **Appendix B.6**: Block diffusion attention mask design
- **Appendix B.7**: Hierarchical reasoning architecture
- **Figure 4**: Visualization of attention patterns

## Citation

Nếu sử dụng code này, vui lòng cite paper gốc:
```bibtex
@inproceedings{arriola2025block,
  title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
  author={Arriola, Marianne and Gokaslan, Aaron and Chiu, Justin T and Yang, Zhihan and Qi, Zhixuan and Han, Jiaqi and Sahoo, Subham Sekhar and Kuleshov, Volodymyr},
  booktitle={ICLR},
  year={2025}
}
```

## Liên hệ

Nếu có câu hỏi hoặc gặp vấn đề, vui lòng:
1. Check documentation ở trên
2. Xem code examples trong `hierarchical_dataloader.py`
3. Test với data nhỏ trước (1000 examples)
4. Visualize attention mask để debug

Good luck với research! 🚀
