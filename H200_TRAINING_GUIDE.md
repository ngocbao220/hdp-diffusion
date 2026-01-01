# H200 Training Guide - Optimized Configurations

## H200 GPU Specifications
- **VRAM**: 141GB HBM3
- **Memory Bandwidth**: 4.8 TB/s
- **FP16/BF16 Performance**: ~1979 TFLOPS
- **Best Use**: Large batch training, large models

## Optimized Training Configurations

### 1. GSM8K Baseline BD3-LM (H200)

**Script**: `scripts/train/train_gsm8k_bd3lm_h200.sh`

**Key Settings:**
```bash
BATCH_SIZE=128           # 4x larger (was 32)
GLOBAL_BATCH_SIZE=512    # 4x larger (was 128)
GRAD_ACCUM=4
LR=5e-4                  # Increased for larger batch
```

**Expected Performance:**
- **Speed**: 2-3x faster than 4xA100
- **Time**: ~12-18 hours for 50K steps
- **Memory Usage**: ~60-80GB VRAM

**Run:**
```bash
# SLURM
sbatch scripts/train/train_gsm8k_bd3lm_h200.sh

# Direct
bash scripts/train/train_gsm8k_bd3lm_h200.sh
```

---

### 2. HDP-Diffusion (H200)

**Script**: `scripts/train/train_hdp_diffusion_h200.sh`

**Key Settings:**
```bash
BATCH_SIZE=128           # Large batch for hierarchical model
GLOBAL_BATCH_SIZE=512
GRAD_ACCUM=4
SEQ_LEN=512             # Q(128) + P(128) + E(256)
```

**Expected Performance:**
- **Speed**: 2-3x faster than multi-GPU setup
- **Time**: ~12-18 hours for 50K steps
- **Memory**: ~70-90GB VRAM (hierarchical attention)

**Run:**
```bash
# SLURM
sbatch scripts/train/train_hdp_diffusion_h200.sh

# Direct
bash scripts/train/train_hdp_diffusion_h200.sh
```

---

## Batch Size Optimization Guide

### Conservative (Safe Start)
```bash
BATCH_SIZE=64
GLOBAL_BATCH_SIZE=256
GRAD_ACCUM=4
```
- Memory: ~40-50GB
- Good for debugging

### Recommended (Default)
```bash
BATCH_SIZE=128
GLOBAL_BATCH_SIZE=512
GRAD_ACCUM=4
```
- Memory: ~60-80GB
- Best balance of speed/stability

### Aggressive (Maximum Speed)
```bash
BATCH_SIZE=192
GLOBAL_BATCH_SIZE=768
GRAD_ACCUM=4
```
- Memory: ~90-110GB
- Requires careful tuning

### Extreme (If you have headroom)
```bash
BATCH_SIZE=256
GLOBAL_BATCH_SIZE=1024
GRAD_ACCUM=4
```
- Memory: ~110-130GB
- May require LR adjustment

---

## Learning Rate Scaling

When increasing batch size, scale learning rate:

```python
# Rule of thumb: LR scales with sqrt(batch_size)
base_lr = 3e-4
base_batch = 128
new_batch = 512

new_lr = base_lr * sqrt(new_batch / base_batch)
# new_lr = 3e-4 * sqrt(4) = 6e-4
```

**Recommended LR by batch size:**
- Batch 128: `3e-4` (base)
- Batch 256: `4e-4`
- Batch 512: `5e-4` (current setting)
- Batch 1024: `7e-4`

---

## Memory Usage Breakdown

**Small Model (768 hidden, 12 layers):**
- Model params: ~85M parameters
- Optimizer states (AdamW): ~340M params worth
- Activations (batch=128, seq=512): ~30-40GB
- Gradient accumulation: minimal

**Total for batch=128:** ~60-80GB

---

## Performance Optimization Tips

### 1. Use Flash Attention
```bash
model.attn_backend=flash_attn  # Already enabled
```

### 2. Mixed Precision Training
```bash
trainer.precision=bf16-mixed  # Already enabled
```

### 3. Gradient Checkpointing (if OOM)
```bash
+model.gradient_checkpointing=true
```

### 4. Compile Model (PyTorch 2.0+)
```bash
+trainer.compile=true
```

### 5. Increase num_workers
```bash
loader.num_workers=16  # Already set
```

---

## Monitoring & Debugging

### Check VRAM Usage
```bash
watch -n 1 nvidia-smi
```

### Wandb Metrics to Watch
- `train/loss` - should decrease steadily
- `train/throughput` - tokens/sec
- `system/gpu_memory_allocated` - VRAM usage
- `train/grad_norm` - gradient norm (should be stable)

### If OOM (Out of Memory)
1. Reduce `BATCH_SIZE` by half
2. Increase `GRAD_ACCUM` by 2x (keep global batch same)
3. Enable gradient checkpointing
4. Reduce `loader.num_workers`

---

## Benchmarking Results (Expected)

### Configuration Comparison

| Setup | Batch | VRAM | Speed | Time (50K steps) |
|-------|-------|------|-------|------------------|
| 4x A100 (80GB) | 32x4 | 60GB/GPU | 1x | ~24-36h |
| 1x H200 (141GB) | 128 | 70GB | 2-3x | ~12-18h |
| 1x H200 (aggr) | 192 | 100GB | 3-4x | ~8-12h |

### Cost Efficiency
- H200: ~$4-5/hour
- 4x A100: ~$10-12/hour
- **Savings**: 50-60% cost for same throughput!

---

## Quick Start Commands

### Baseline Training
```bash
# Start training immediately
bash scripts/train/train_gsm8k_bd3lm_h200.sh
```

### HDP-Diffusion Training
```bash
# Start hierarchical training
bash scripts/train/train_hdp_diffusion_h200.sh
```

### Custom Batch Size
```bash
# Edit the script first
nano scripts/train/train_gsm8k_bd3lm_h200.sh
# Change BATCH_SIZE=128 to your desired value
# Then run
bash scripts/train/train_gsm8k_bd3lm_h200.sh
```

---

## Troubleshooting

### Issue: OOM Error
```bash
# Solution: Reduce batch size
BATCH_SIZE=64  # Instead of 128
```

### Issue: Training too slow
```bash
# Check: Are you using Flash Attention?
model.attn_backend=flash_attn

# Check: Is bf16 enabled?
trainer.precision=bf16-mixed
```

### Issue: Loss not decreasing
```bash
# May need to adjust LR
LR=3e-4  # Try lower LR
# Or increase warmup
WARMUP_STEPS=10000  # Instead of 5000
```

---

## Recommended Workflow

1. **Start with default settings** (batch=128)
2. **Monitor first 1000 steps**
   - Check VRAM usage
   - Verify loss is decreasing
   - Check throughput
3. **Adjust if needed**
   - If VRAM < 80GB → increase batch size
   - If VRAM > 120GB → might OOM, reduce
4. **Let it run** for full 50K steps
5. **Evaluate** with inference scripts

---

Ready to train! 🚀 H200 will give you **2-3x speedup** over multi-GPU setups!
