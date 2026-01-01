# HDP-Diffusion: Hierarchical Dual-Process Diffusion for Mathematical Reasoning

**Research Implementation** - Novel architecture combining Block Diffusion with hierarchical attention for GSM8K math problems.

## 🔬 Key Innovation

HDP-Diffusion introduces a **3-block hierarchical structure** with **custom attention patterns** to model the dual-process theory of reasoning:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Question   │────▶│     Plan     │────▶│   Execution   │
│  (Context)  │     │ (Reasoning)  │     │ (Computation) │
│  128 tokens │     │  128 tokens  │     │  256 tokens   │
└─────────────┘     └──────────────┘     └───────────────┘
```

### Hierarchical Attention Rules

1. **Question Block**: Self-attention only (process context)
2. **Plan Block**: Attends to Question + Plan (abstract reasoning without seeing detailed calculations)
3. **Execution Block**: Attends to Question + Plan + Execution (full context for computation)

**Key Insight**: Plan cannot attend to Execution → preserves causal reasoning flow.

## 📁 File Structure

```
models/
  ├── hdp_attention_mask.py    # Custom attention mask generation
  ├── hdp_diffusion.py          # HDP model wrapper
  └── hierarchical_mask.py      # (legacy hierarchical implementation)

hdp_dataset.py                  # Dataset loader for HDP format
configs/data/hdp_diffusion.yaml # HDP configuration

scripts/train/
  ├── train_hdp_diffusion.sh    # Full training script
  └── test_hdp_diffusion.sh     # Quick test (50 steps)
```

## 🚀 Quick Start

### 1. Test the Implementation

Verify HDP attention and dataset work correctly:

```bash
# Test attention mask
python models/hdp_attention_mask.py

# Test dataset
python hdp_dataset.py

# Test model wrapper
python models/hdp_diffusion.py
```

### 2. Run Quick Training Test

```bash
bash scripts/train/test_hdp_diffusion.sh
```

This runs 50 steps to verify the full pipeline works.

### 3. Full Training

```bash
# On SLURM cluster
sbatch scripts/train/train_hdp_diffusion.sh

# Or locally
bash scripts/train/train_hdp_diffusion.sh
```

## 📊 Data Format

HDP-Diffusion expects JSON data with hierarchical structure:

```json
[
  {
    "id": "gsm8k_0",
    "question": "Natalia sold clips to 48 friends...",
    "plan": "Identify clips sold in April. Calculate May...",
    "execution": "48/2 = 24 clips in May. 48+24 = 72 total..."
  }
]
```

The existing GSM8K hierarchical data (`data/gsm8k/gsm8k_hierarchical_train.json`) already has this format!

## 🔧 Configuration

Key hyperparameters in `configs/data/hdp_diffusion.yaml`:

```yaml
hdp:
  enabled: true
  question_len: 128      # Question block size
  plan_len: 128          # Plan block size
  exec_len: 256          # Execution block size
  use_hdp_attention: true  # Enable hierarchical attention
  causal_within_block: false  # Bidirectional within blocks
```

## 🧪 Architecture Details

### Attention Mask Visualization

```
              Q   P   E
        ┌───┬───┬───┐
     Q  │ ✓ │ ✗ │ ✗ │  Question: self only
        ├───┼───┼───┤
     P  │ ✓ │ ✓ │ ✗ │  Plan: Q + P
        ├───┼───┼───┤
     E  │ ✓ │ ✓ │ ✓ │  Execution: full
        └───┴───┴───┘
```

Generate visualization:

```python
from models.hdp_attention_mask import visualize_hdp_mask
visualize_hdp_mask(seq_len=512, save_path="hdp_mask.png")
```

### Integration with BD3-LM

HDP-Diffusion wraps any BD3-LM backbone (DiT, GPT, etc.):

```python
from models.hdp_diffusion import create_hdp_model_from_backbone, HDPDiffusionConfig

# Create config
config = HDPDiffusionConfig(
    question_len=128,
    plan_len=128,
    exec_len=256,
    use_hdp_attention=True
)

# Wrap backbone
hdp_model = create_hdp_model_from_backbone(backbone, config)
```

## 📈 Training Pipeline

### Baseline Comparison

1. **BD3-LM Baseline** (simple Q&A): `bash scripts/train/train_gsm8k_bd3lm.sh`
2. **HDP-Diffusion** (hierarchical): `bash scripts/train/train_hdp_diffusion.sh`

### Ablation Studies

Test different configurations:

```bash
# No HDP attention (ablation)
python main.py ... +hdp.use_hdp_attention=false

# Different block sizes
python main.py ... +hdp.plan_len=64 +hdp.exec_len=320

# Causal within blocks
python main.py ... +hdp.causal_within_block=true
```

## 🎯 Expected Results

Monitor these metrics during training:

- **Training Loss**: Should decrease steadily
- **Validation PPL**: Lower is better
- **Plan Quality**: Can evaluate separately by masking execution
- **Execution Accuracy**: Final answer correctness

## 🔬 Research Notes

### Novel Contributions

1. **Hierarchical Attention for Diffusion**: First application of structured attention to discrete diffusion LMs
2. **Dual-Process Modeling**: Explicit separation of abstract reasoning (Plan) and concrete computation (Execution)
3. **Causal Reasoning**: Plan → Execution flow prevents "cheating" during training

### Potential Extensions

- Variable-length blocks based on problem complexity
- Multi-level hierarchies (more than 3 blocks)
- Cross-attention between Plan and Execution
- Curriculum learning: train Plan first, then Execution

## 🐛 Troubleshooting

### Common Issues

**Attention mask shape errors:**
- Ensure `model.length = question_len + plan_len + exec_len`
- Check backend compatibility: `flash_attn` vs `sdpa` vs `xformers`

**Data loading errors:**
- Verify JSON format has all required fields: `question`, `plan`, `execution`
- Check tokenizer padding: `tokenizer.pad_token = tokenizer.eos_token`

**OOM errors:**
- Reduce batch size: `loader.batch_size=16`
- Use gradient checkpointing
- Try `model.attn_backend=sdpa` instead of `flash_attn`

## 📚 Citation

If you use HDP-Diffusion in your research:

```bibtex
@article{hdp-diffusion2026,
  title={HDP-Diffusion: Hierarchical Dual-Process Diffusion for Mathematical Reasoning},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

Built on top of:

```bibtex
@inproceedings{arriola2025block,
  title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
  author={Arriola, Marianne and Gokaslan, Aaron and others},
  booktitle={ICLR},
  year={2025}
}
```

## 🤝 Contributing

This is research code. For questions or collaboration:
- Open an issue
- Check `HIERARCHICAL_README.md` for more implementation details

---

**Status**: 🚧 Active Research Implementation
**Last Updated**: January 2026
