# GSM8K Hierarchical Training Guide

## 🎯 Mục tiêu

Train Hierarchical BD3-LM trên GSM8K để chứng minh:
- **H-Module (Plan)**: Học tư duy toán học trừu tượng
- **L-Module (Execution)**: Học cách tính toán cụ thể với số

## 📋 Pipeline Tổng quan

```
GSM8K Raw Data (7.5k examples)
    ↓
[1] Generate Plans với Llama-3-8B
    ↓
Hierarchical Format: [Question | Plan | Execution]
    ↓
[2] Train Hierarchical BD3-LM
    ↓
Evaluate reasoning quality
```

---

## 🚀 BƯỚC 1: Generate Plans cho GSM8K

### 1.1 Chuẩn bị môi trường

```bash
# Install vLLM (nếu chưa có)
pip install vllm

# Create directories
mkdir -p data/gsm8k logs scripts/data_prep
```

### 1.2 Chạy plan generation trên H200

**Script:** `scripts/data_prep/run_gsm8k_plan_generation.sh`

```bash
# Test với 100 examples trước
sbatch scripts/data_prep/run_gsm8k_plan_generation.sh

# Hoặc test nhanh (không qua SLURM):
python scripts/data_prep/generate_gsm8k_plans.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --split train \
    --batch_size 256 \
    --output_path data/gsm8k/gsm8k_hierarchical_train.json \
    --num_examples 100  # Test với 100 examples
```

**Thời gian ước tính:**
- 100 examples: ~2 phút
- 7,500 examples (full train): ~30-40 phút với H200
- 1,300 examples (test set): ~5-10 phút

### 1.3 Kiểm tra output

```bash
# Check file được tạo
ls -lh data/gsm8k/

# View statistics
jq 'length' data/gsm8k/gsm8k_hierarchical_train.json

# View first example
jq '.[0]' data/gsm8k/gsm8k_hierarchical_train.json
```

**Expected output format:**
```json
{
  "id": "gsm8k_train_0",
  "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "plan": "Calculate daily egg production. Determine total consumption from breakfast and baking. Subtract consumption from production to find remainder. Multiply remainder by unit price to get daily revenue.",
  "execution": "Janet gets 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, totaling 7 eggs. This leaves 16 - 7 = 9 eggs to sell. She sells them for $2 each, making 9 * 2 = $18 per day.",
  "answer_numerical": "18"
}
```

### 1.4 Analyze token lengths

```bash
# Analyze để xác định optimal max_len
python gsm8k_dataloader.py \
    --data_path data/gsm8k/gsm8k_hierarchical_train.json \
    --tokenizer gpt2 \
    --analyze
```

**Expected output:**
```
GSM8K Token Length Analysis
============================================================
Number of examples: 7473

Question:
  Mean: 65.3
  Median: 62.0
  95th percentile: 105.0

Plan:
  Mean: 42.1
  Median: 38.0
  95th percentile: 75.0

Execution:
  Mean: 98.7
  Median: 89.0
  95th percentile: 165.0

Recommended max_len settings (95th percentile):
  question_max_len: 128
  plan_max_len: 128
  exec_max_len: 256
  total: 512
```

---

## 🔧 BƯỚC 2: Training Hierarchical BD3-LM

### 2.1 Verify data và mask

```bash
# Test dataloader
python gsm8k_dataloader.py \
    --data_path data/gsm8k/gsm8k_hierarchical_train.json \
    --tokenizer gpt2

# Test hierarchical mask
python test_hierarchical_mask.py
```

### 2.2 Training configuration

**Default settings** (trong `scripts/train/train_gsm8k_hierarchical.sh`):

```bash
QUESTION_LEN=128   # Short math problems
PLAN_LEN=128       # Concise reasoning steps
EXEC_LEN=256       # Detailed calculations
BLOCK_SIZE=16      # Balance speed/quality

MAX_STEPS=50000    # ~6.7 epochs over 7.5k examples
BATCH_SIZE=32
LR=3e-4
```

### 2.3 Run training

```bash
# Full training
sbatch scripts/train/train_gsm8k_hierarchical.sh

# Or test với small run (1000 steps)
python main.py \
    mode=train \
    model=tiny \
    model.length=512 \
    algo=bd3lm \
    block_size=16 \
    training.max_steps=1000 \
    training.hierarchical.enabled=true \
    training.hierarchical.question_len=128 \
    training.hierarchical.plan_len=128 \
    training.hierarchical.exec_len=256 \
    data.name=gsm8k \
    data.train_path=data/gsm8k/gsm8k_hierarchical_train.json \
    data.test_path=data/gsm8k/gsm8k_hierarchical_test.json
```

### 2.4 Monitor training

```bash
# Watch logs
tail -f logs/gsm8k_hier_bd3lm_*.out

# Check tensorboard (if available)
tensorboard --logdir outputs/gsm8k_hierarchical_bd3lm_bs16
```

**Expected metrics:**
- Initial loss: ~8-9
- After 1K steps: ~6-7
- After 10K steps: ~5-6
- After 50K steps: ~4-5

---

## 📊 BƯỚC 3: Evaluation & Analysis

### 3.1 Generate samples

```bash
python main.py \
    mode=sample_eval \
    model=small \
    algo=bd3lm \
    eval.checkpoint_path=outputs/gsm8k_hierarchical_bd3lm_bs16/best.ckpt \
    sampling.num_sample_batches=100 \
    training.hierarchical.enabled=true
```

### 3.2 Analyze Plan quality

Tạo script để evaluate:

```python
# analyze_plans.py
import json

with open('generated_samples.json') as f:
    samples = json.load(f)

# Check if Plans are abstract (no specific numbers)
for sample in samples[:10]:
    plan = sample['plan']
    # Count numbers in plan
    import re
    numbers = re.findall(r'\d+', plan)
    print(f"Plan: {plan}")
    print(f"Contains {len(numbers)} numbers")
    print()
```

### 3.3 Compare với baseline

**Metrics để so sánh:**

1. **Perplexity**: NELBO trên test set
2. **Accuracy**: % correct answers (nếu có ground truth)
3. **Plan abstraction**: Số lượng numbers trong plan (càng ít càng tốt)
4. **Execution detail**: Numbers trong execution (càng nhiều càng tốt)

---

## 🎨 Prompt Engineering Tips

### Nếu Plans có quá nhiều số cụ thể:

**Cải thiện prompt:**

```python
PLAN_EXTRACTION_PROMPT = """...

**CRITICAL**: Replace ALL specific numbers with abstract terms:
- Instead of "5 apples" → "initial quantity"
- Instead of "add 3" → "add given amount"
- Instead of "multiply by 2" → "apply multiplication"
- Instead of "$10" → "unit price"

Your plan should read like a recipe that works for ANY numbers.

..."""
```

### Nếu Plans quá ngắn hoặc không rõ:

```python
# Tăng temperature
TEMPERATURE=0.8  # instead of 0.7

# Tăng max_tokens
MAX_TOKENS=384  # instead of 256
```

### Nếu Plans copy từ execution:

```python
# Thêm vào prompt:
"Do NOT copy the execution steps. Create NEW abstract reasoning steps."
```

---

## 🔬 Experiments để chứng minh trong Paper

### Experiment 1: Plan Abstraction

**Giả thuyết:** Plan module học được abstract reasoning

**Metrics:**
- Number density trong Plan vs Execution
- Plan similarity across different numbers (edit distance)

**Code:**
```python
import re

def count_numbers(text):
    return len(re.findall(r'\d+', text))

plan_numbers = [count_numbers(x['plan']) for x in samples]
exec_numbers = [count_numbers(x['execution']) for x in samples]

print(f"Avg numbers in Plan: {np.mean(plan_numbers):.1f}")
print(f"Avg numbers in Exec: {np.mean(exec_numbers):.1f}")
```

### Experiment 2: Transfer Learning

**Giả thuyết:** Pre-trained Plan module có thể transfer sang problems khác

**Setup:**
1. Train trên GSM8K
2. Freeze Plan module
3. Fine-tune Execution module trên SVAMP hoặc dataset khác
4. So sánh với baseline (không freeze)

### Experiment 3: Ablation Study

**So sánh:**
1. Hierarchical (Plan + Execution)
2. Flat (no Plan, chỉ có Question + Execution)
3. Baseline AR model

**Metrics:** Perplexity, accuracy, inference speed

---

## 📝 Expected Results (Ví dụ)

### Sample Generated Output:

**Question:**
```
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
```

**Generated Plan:**
```
Identify the quantity sold in the first period. Calculate the quantity for the second period as half of the first. Sum both quantities to get total.
```

**Generated Execution:**
```
Natalia sold 48 clips in April. In May, she sold half of that: 48 / 2 = 24 clips. Total clips sold: 48 + 24 = 72 clips.
```

**Answer:** 72 ✓

### Analysis:

✅ **Plan is abstract:** No specific numbers (48, 24, 72)
✅ **Plan is reusable:** Works for any "sell X then half" problem
✅ **Execution has details:** All calculations present
✅ **Answer is correct:** Matches ground truth

---

## 🐛 Troubleshooting

### Problem: vLLM installation fails

```bash
# Try with specific version
pip install vllm==0.4.0

# Or use Docker
docker pull vllm/vllm-openai:latest
```

### Problem: Plans are too similar to Execution

**Solution 1:** Improve prompt (add more emphasis on abstraction)

**Solution 2:** Use few-shot examples in prompt:
```python
PLAN_EXTRACTION_PROMPT = """...

Here are 3 good examples:
1. Question: "Add 5 and 3"
   Bad Plan: "Add 5 and 3 to get 8"
   Good Plan: "Apply addition operation to given numbers"

2. Question: "Multiply 4 by 7"
   Bad Plan: "Multiply 4 by 7 to get 28"
   Good Plan: "Use multiplication with provided operands"

..."""
```

### Problem: Training loss not decreasing

1. Check mask visualization
2. Verify data format (print samples)
3. Reduce learning rate: `LR=1e-4`
4. Increase warmup: `WARMUP_STEPS=10000`

### Problem: GPU OOM

```bash
# Reduce batch size
BATCH_SIZE=16

# Use smaller model
MODEL_SIZE=tiny

# Reduce sequence length
QUESTION_LEN=96
PLAN_LEN=96
EXEC_LEN=192
```

---

## 📊 Checklist

### Data Preparation
- [ ] Generate plans: `sbatch scripts/data_prep/run_gsm8k_plan_generation.sh`
- [ ] Verify JSON format: `jq '.[0]' data/gsm8k/gsm8k_hierarchical_train.json`
- [ ] Analyze lengths: `python gsm8k_dataloader.py --analyze`
- [ ] Check plan quality: Manually inspect 10-20 examples

### Training
- [ ] Test dataloader: `python gsm8k_dataloader.py`
- [ ] Test mask: `python test_hierarchical_mask.py`
- [ ] Small training test (1K steps): Check loss decreases
- [ ] Full training: Monitor for 50K steps
- [ ] Save checkpoints: Check `outputs/gsm8k_hierarchical_bd3lm_bs16/`

### Evaluation
- [ ] Generate samples from checkpoint
- [ ] Calculate metrics (perplexity, accuracy)
- [ ] Analyze plan abstraction
- [ ] Compare with baselines
- [ ] Prepare results for paper

---

## 🎯 Next Steps

### Immediate (Today)
1. Run plan generation: `sbatch scripts/data_prep/run_gsm8k_plan_generation.sh`
2. Check output quality
3. Start training test (1K steps)

### This Week
4. Full training (50K steps)
5. Generate evaluation samples
6. Analyze plan quality

### This Month
7. Run ablation studies
8. Compare with baselines
9. Write paper section

---

## 📚 References

- **GSM8K Dataset:** https://huggingface.co/datasets/gsm8k
- **vLLM Documentation:** https://docs.vllm.ai/
- **Llama-3-8B:** https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

---

**Summary:** Pipeline để train hierarchical BD3-LM trên GSM8K đã sẵn sàng. Bắt đầu bằng việc generate plans, sau đó training và evaluation! 🚀
