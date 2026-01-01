# 🎯 HIERARCHICAL BLOCK DIFFUSION - COMPLETE IMPLEMENTATION

## 📋 Tổng quan nhanh

Đây là implementation đầy đủ của **Hierarchical Discrete Diffusion Model** với kiến trúc **Plan-then-Generate** dựa trên BD3-LM (Block Discrete Denoising Diffusion Language Models).

**3 điểm cốt lõi:**
1. ✅ **Hierarchical Attention Mask** - Plan không thể nhìn thấy Execution
2. ✅ **Structured Data Format** - [Question | Plan | Execution]
3. ✅ **Simplified Generation** - Fixed length, no arbitrary-length

---

## 📚 Tài liệu đầy đủ (Đọc theo thứ tự)

### 🚀 Bắt đầu nhanh
- **[QUICKSTART.md](QUICKSTART.md)** ⭐ BẮT ĐẦU TỪ ĐÂY
  - 3 bước để chạy thử
  - Quick reference commands
  - Troubleshooting nhanh
  - **Thời gian đọc: 5 phút**

### 📖 Documentation chính
- **[HIERARCHICAL_README.md](HIERARCHICAL_README.md)** (Tiếng Việt)
  - Giải thích architecture chi tiết
  - Hướng dẫn sử dụng đầy đủ
  - Customization guide
  - Examples và best practices
  - **Thời gian đọc: 20 phút**

### 🔧 Implementation details
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
  - Chi tiết tất cả thay đổi
  - Code structure
  - Integration points
  - **Thời gian đọc: 15 phút**

### 📊 Visual reference
- **[architecture_diagram.py](architecture_diagram.py)**
  - ASCII diagrams
  - Attention flow visualization
  - **Run để xem:** `python architecture_diagram.py`

### 🎓 GSM8K Math Reasoning (NEW!)
- **[GSM8K_README.md](GSM8K_README.md)** ⭐ GSM8K Quick Start
  - Complete pipeline for math reasoning
  - Plan generation with vLLM + Llama-3
  - Training & evaluation on 7.5k problems
  - **Thời gian đọc: 10 phút**
- **[GSM8K_TRAINING_GUIDE.md](GSM8K_TRAINING_GUIDE.md)** 
  - Detailed step-by-step guide
  - Experiments for paper
  - Troubleshooting & tips
  - **Thời gian đọc: 20 phút**
- **[GSM8K_SETUP_SUMMARY.md](GSM8K_SETUP_SUMMARY.md)**
  - Quick setup summary
  - File structure & deliverables
  - **Thời gian đọc: 5 phút**

### ✅ Final summary
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)**
  - Verification checklist
  - Comparison với baseline
  - Next steps
  - **Thời gian đọc: 10 phút**

---

## 💻 Code Files

### Core Implementation
| File | Chức năng | Status |
|------|-----------|--------|
| **[models/hierarchical_mask.py](models/hierarchical_mask.py)** | Attention mask phân tầng | ✅ Complete |
| **[hierarchical_dataloader.py](hierarchical_dataloader.py)** | Data preprocessing | ✅ Complete |
| **[test_hierarchical_mask.py](test_hierarchical_mask.py)** | Unit tests & verification | ✅ Complete |

### Configuration
| File | Chức năng | Status |
|------|-----------|--------|
| **[configs/algo/hierarchical.yaml](configs/algo/hierarchical.yaml)** | Hierarchical config | ✅ Complete |
| **[scripts/train/train_hierarchical_bd3lm.sh](scripts/train/train_hierarchical_bd3lm.sh)** | Training script | ✅ Complete |

### Modified Files
| File | Thay đổi | Line |
|------|----------|------|
| **[models/dit.py](models/dit.py)** | Added `hierarchical_config` param | ~710 |

---

## ⚡ Quick Commands

### Test mask (không cần GPU)
```bash
python test_hierarchical_mask.py
```
**Output mong đợi:**
- ✅ All tests passed
- 📊 File `hierarchical_mask_test.png` được tạo

### Visualize architecture
```bash
python architecture_diagram.py
```
**Output:** ASCII diagrams showing attention flow

### Train với data test
```bash
# Edit để giảm steps (test nhanh)
vim scripts/train/train_hierarchical_bd3lm.sh

# Run
sbatch scripts/train/train_hierarchical_bd3lm.sh
```

### Train trực tiếp
```bash
python main.py \
    mode=train \
    model=tiny \
    algo=bd3lm \
    block_size=16 \
    training.max_steps=1000 \
    training.hierarchical.enabled=true \
    training.hierarchical.question_len=256 \
    training.hierarchical.plan_len=256 \
    training.hierarchical.exec_len=512
```

---

## 🎯 Cấu trúc Hierarchical Mask

### Sequence Structure
```
[Question | Plan_xt | Plan_x0 | Exec_xt | Exec_x0]
   256    |   256    |   256   |   512   |   512   = 1792 tokens
```

### Attention Rules
```
✅ Question → Question (full attention)
✅ Plan → Question (can see input)
✅ Plan → Plan (block diffusion)
❌ Plan → Execution (BLOCKED!)
✅ Execution → Question (can see input)
✅ Execution → Plan (can see high-level plan)
✅ Execution → Execution (block diffusion)
```

**Điểm quan trọng:** Plan KHÔNG thể nhìn thấy Execution → Giữ tính nhân quả

### Verification
```python
from models.hierarchical_mask import create_hierarchical_mask

mask = create_hierarchical_mask(
    seqlen=1024, block_size=16,
    question_len=256, plan_len=256, exec_len=512
)

# Verify Plan cannot see Execution
plan_to_exec = mask[256:768, 768:]  # Plan tokens → Exec tokens
assert not plan_to_exec.any(), "Plan should NOT see Execution!"
print("✅ Causal constraint verified!")
```

---

## 📦 Data Format

### Option A: Structured Data
```json
[
  {
    "question": "What is the capital of France?",
    "plan": "I need to recall European geography and capitals.",
    "execution": "Paris is the capital and largest city of France..."
  }
]
```

**Load:**
```python
from hierarchical_dataloader import load_reasoning_dataset

dataset = load_reasoning_dataset(
    dataset_path='data.json',
    tokenizer=tokenizer,
    question_len=256,
    plan_len=256,
    exec_len=512
)
```

### Option B: Auto-split
```python
from hierarchical_dataloader import HierarchicalDataCollator

collator = HierarchicalDataCollator(tokenizer, 256, 256, 512)
batch = collator([{'text': 'Your text here...'}])
# Auto-splits: 25% Q, 25% P, 50% E
```

### Option C: Custom Parser
```python
# In hierarchical_dataloader.py
def process_example(example):
    text = example['text']
    
    # TODO: Your custom logic
    question = your_extract_question(text)
    plan = your_extract_plan(text)
    execution = your_extract_execution(text)
    
    return {'question': ..., 'plan': ..., 'execution': ...}
```

---

## 🔧 Configuration Options

### Sequence Lengths (Adjustable)
```yaml
hierarchical:
  question_len: 256  # Question/context tokens
  plan_len: 256      # High-level plan tokens
  exec_len: 512      # Detailed execution tokens
  total_len: 1024    # Sum of above
```

### Block Size (Speed vs Quality)
```yaml
block_size: 16  # Options: 1, 4, 8, 16, 32, 64, 1024
```

**Trade-off:**
- `block_size=1`: Slowest, highest quality (Autoregressive)
- `block_size=16`: Balanced (Recommended)
- `block_size=1024`: Fastest, lower quality (Full diffusion)

### Training Settings
```yaml
training:
  max_steps: 100000
  warmup_steps: 10000
  batch_size: 64
  learning_rate: 5e-4
  ema: 0.9999
```

### Features Disabled (Simplified)
```yaml
sampling:
  var_length: false        # No variable-length
  arbitrary_length: false  # Fixed 1024 tokens
  first_hitting: true      # Faster sampling
  kv_cache: true          # Speed optimization
```

---

## ✅ Verification Checklist

### Pre-training
- [ ] Run: `python test_hierarchical_mask.py`
  - [ ] All tests passed
  - [ ] Visualization looks correct
- [ ] Check data format
  - [ ] Print 3-5 samples
  - [ ] Verify [Q, P, E] structure
- [ ] Review config
  - [ ] Lengths: 256/256/512
  - [ ] Block size: 16
  - [ ] LR, warmup, etc.

### During training
- [ ] Loss decreases (first 100 steps)
- [ ] No NaN/Inf values
- [ ] GPU utilization good (>80%)
- [ ] Checkpoints saving correctly

### Post-training
- [ ] Validate loss better than baseline
- [ ] Generate samples
  - [ ] Has hierarchical structure?
  - [ ] Plan makes sense?
  - [ ] Execution follows plan?
- [ ] Compare with AR/MDLM/SEDD

---

## 🐛 Common Issues

| Problem | Quick Fix |
|---------|-----------|
| "Mask dimensions don't match" | Check: `total = q_len + p_len*2 + e_len*2` |
| CUDA OOM | Reduce `batch_size` or use `model=tiny` |
| Loss not decreasing | Verify mask with test, reduce LR |
| "torch not found" | Install: `pip install -r requirements.txt` |
| Tests failing | Check Python/PyTorch version |

**Chi tiết:** See Troubleshooting section in [HIERARCHICAL_README.md](HIERARCHICAL_README.md)

---

## 🚀 Next Steps

### Ngay bây giờ (5 phút)
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `python test_hierarchical_mask.py`
3. Check visualization: `hierarchical_mask_test.png`

### Hôm nay (1-2 giờ)
4. Prepare 100-1000 examples of your data
5. Implement custom parser (if needed)
6. Run small training test (1000 steps)

### Tuần này (2-3 ngày)
7. Full training (100K steps)
8. Evaluate on test set
9. Compare with baselines

### Tháng này (1-2 tuần)
10. Tune hyperparameters
11. Try different block sizes
12. Write paper/report

---

## 📊 Expected Results

### Training Metrics
- **Initial loss:** ~8-10 (random)
- **After 1K steps:** ~6-7
- **After 10K steps:** ~5-6
- **After 100K steps:** ~4-5 (depends on data)

### Generation Quality
- **Plan coherence:** Should be high-level, logical
- **Execution detail:** Should follow plan structure
- **Causality:** Plan should not "leak" execution details

### Comparison
| Model | Perplexity | Speed | Hierarchical |
|-------|-----------|-------|--------------|
| AR | Best | Slow | ❌ |
| MDLM | Good | Fast | ❌ |
| BD3-LM | Good | Medium | ❌ |
| **Hier-BD3-LM** | Good | Medium | ✅ |

---

## 📚 References

### Papers
- **BD3-LM:** Block Diffusion (ICLR 2025)
- **Appendix B.6, B.7:** Hierarchical architecture
- **Figure 4:** Attention mask visualization

### Code
- **Original repo:** kuleshov-group/bd3lm
- **HuggingFace:** kuleshov-group/bd3-lms

### Citation
```bibtex
@inproceedings{arriola2025block,
  title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
  author={Arriola, Marianne and Gokaslan, Aaron and ...},
  booktitle={ICLR},
  year={2025}
}
```

---

## 💡 Tips & Tricks

### For faster debugging
```bash
# Use tiny model + small data
python main.py model=tiny training.max_steps=100 loader.batch_size=8
```

### For better quality
```bash
# Increase warmup, use EMA
training.warmup_steps=20000 training.ema=0.9999
```

### For faster inference
```bash
# Enable caching, use first-hitting
sampling.kv_cache=true sampling.first_hitting=true
```

### For customization
1. Edit mask: `models/hierarchical_mask.py`
2. Edit data: `hierarchical_dataloader.py`
3. Edit config: `configs/algo/hierarchical.yaml`

---

## 📞 Getting Help

### Documentation
1. **[QUICKSTART.md](QUICKSTART.md)** - Quick commands
2. **[HIERARCHICAL_README.md](HIERARCHICAL_README.md)** - Full guide
3. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Details

### Testing
```bash
python test_hierarchical_mask.py  # Verify implementation
python architecture_diagram.py    # View diagrams
```

### Common questions
- **How do I know mask is correct?** → Run tests, check visualization
- **My data format is different?** → Edit `process_example()` in dataloader
- **Training too slow?** → Reduce batch_size, use smaller model
- **How to evaluate?** → Check perplexity, generate samples, manual inspection

---

## ✨ Summary

**✅ Hoàn thành:**
- Hierarchical attention mask (Plan → Execution blocked)
- Data collator for [Question, Plan, Execution]
- Simplified generation (fixed length)
- Full documentation & tests

**📦 Deliverables:**
- 9 new files (code + docs)
- 1 modified file (dit.py)
- Working training script
- Comprehensive tests

**🎯 Ready for:**
- Small-scale testing
- Custom data integration
- Full training & evaluation

**🚀 Start here:** [QUICKSTART.md](QUICKSTART.md)

---

**Last updated:** 2026-01-01  
**Status:** ✅ Complete & Ready to Use  
**Contact:** See documentation for details
