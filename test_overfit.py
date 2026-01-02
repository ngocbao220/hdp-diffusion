"""
Single Batch Overfit Test for HDP-Diffusion

Mục đích: Train model trên ĐÚNG 1 mẫu duy nhất
Kỳ vọng: Model phải học thuộc lòng (loss → 0)
Nếu KHÔNG overfit được → CODE BỊ BUG!  
"""

import torch
import json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path. insert(0, os.path. dirname(os.path.abspath(__file__)))

# Import từ project
from hdp_dataset import HDPDataset, collate_hdp_batch
from diffusion import Diffusion
import models.dit  # Ensure model is imported

print("="*80)
print("🔬 SINGLE BATCH OVERFIT TEST - HDP DIFFUSION")
print("="*80)

# ============ CONFIG ============
DEVICE = 'cuda' if torch.cuda. is_available() else 'cpu'
BLOCK_SIZE = 4
SEQ_LENGTH = 512
QUESTION_LEN = 128
PLAN_LEN = 128
EXEC_LEN = 256

NUM_STEPS = 500  # Train 500 steps on 1 sample
PRINT_EVERY = 50
LR = 1e-3

print(f"\n📋 Config:")
print(f"  Device: {DEVICE}")
print(f"  Sequence Length: {SEQ_LENGTH}")
print(f"  Block Size: {BLOCK_SIZE}")
print(f"  Training Steps: {NUM_STEPS}")
print(f"  Learning Rate: {LR}")
print(f"  Expected:  Loss should go to ~0.0")

# ============ LOAD DATA ============
print(f"\n🗂️  Loading single sample...")

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create single-sample dataset
single_sample = {
    "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? ",
    "plan": "Identify the initial quantity of blue fiber.  Calculate the initial quantity of white fiber using the given ratio. Sum the quantities of both fibers to find the total.",
    "execution": "It takes 2/2=1 bolt of white fiber So the total amount of fabric is 2+1=3 bolts of fabric",
    "answer": "3"
}

# Save to temp file
os.makedirs('/tmp', exist_ok=True)
with open('/tmp/single_sample. json', 'w') as f:
    json.dump([single_sample], f)

# Load with HDPDataset
dataset = HDPDataset(
    data_path='/tmp/single_sample.json',
    tokenizer=tokenizer,
    block_sizes=(QUESTION_LEN, PLAN_LEN, EXEC_LEN),
    use_special_format=True,
    return_block_indices=True
)

print(f"✅ Loaded dataset with {len(dataset)} sample(s)")

# Create dataloader (batch_size=1)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_hdp_batch
)

# Get the single batch
single_batch = next(iter(dataloader))
for k, v in single_batch. items():
    if isinstance(v, torch.Tensor):
        single_batch[k] = v. to(DEVICE)

print(f"\n📦 Batch shapes:")
for k, v in single_batch.items():
    if isinstance(v, torch. Tensor):
        print(f"  {k}: {v.shape}")

# Decode để xem nội dung
input_ids = single_batch['input_ids'][0].cpu()
block_indices = single_batch['block_indices'][0].cpu()

# Decode each block separately
q_tokens = input_ids[block_indices == 0]
p_tokens = input_ids[block_indices == 1]
e_tokens = input_ids[block_indices == 2]

print(f"\n📝 Sample content:")
print(f"  Question:   {tokenizer.decode(q_tokens)[: 100]}...")
print(f"  Plan:      {tokenizer. decode(p_tokens)[:100]}...")
print(f"  Execution: {tokenizer.decode(e_tokens)[:100]}...")

# ============ CREATE MODEL CONFIG ============
print(f"\n🤖 Creating model config...")

# Create minimal config object
from types import SimpleNamespace

config = SimpleNamespace()

# Model config
config.model = SimpleNamespace()
config.model.length = SEQ_LENGTH
config.model.hidden_size = 768
config.model.n_embd = 768  # GPT2-small
config.model.n_layer = 12
config.model.n_head = 12
config.model.attn_backend = 'sdpa'
config.model.time_conditioning = True

# Algo config
config.algo = SimpleNamespace()
config.algo.name = 'bd3lm'
config.algo.backbone = 'dit'
config.algo.var_min = False
config.algo.parameterization = 'subs'
config.algo.T = 1000
config.algo.sampler = 'analytic'
config.algo.time_conditioning = True
config.algo.cross_attn = True
config.algo. ignore_bos = False
config.algo.mdlm_loss_scale = False
config.algo.clip_search_widths = [0.5, 0.6, 0.7, 0.8, 0.9]

# Data config
config.data = SimpleNamespace()
config.data.name = 'hdp_diffusion'
config.data.hdp = SimpleNamespace()
config.data.hdp.use_hdp_attention = True
config.data.hdp.question_len = QUESTION_LEN
config.data.hdp.plan_len = PLAN_LEN
config.data.hdp.exec_len = EXEC_LEN
config.data.hdp.use_special_format = True

# Training config
config.training = SimpleNamespace()
config.training.ema = 0  # Disable EMA for faster convergence
config.training.resample = True
config.training.from_pretrained = None
config.training.antithetic_sampling = False
config.training.sampling_eps_min = 0.001
config.training.sampling_eps_max = 1.0

# Noise config
config.noise = SimpleNamespace()
config.noise.type = 'loglinear'
config.noise.sigma_min = 0.0001
config.noise.sigma_max = 3.0

# Other required configs
config.mode = 'train'
config.seed = 42
config.block_size = BLOCK_SIZE
config.checkpointing = SimpleNamespace()
config.checkpointing.save_dir = '/tmp/overfit_test'

print(f"✅ Config created")
print(f"  model.length: {config.model.length}")
print(f"  block_size: {config.block_size}")
print(f"  use_hdp_attention: {config.data.hdp.use_hdp_attention}")

# ============ CREATE MODEL ============
print(f"\n🏗️  Creating model...")

try:
    model = Diffusion(config, tokenizer=tokenizer)
    model = model.to(DEVICE)
    model.train()
    
    print(f"✅ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  use_hdp_attention: {model.use_hdp_attention}")
    if hasattr(model, 'hdp_block_sizes'):
        print(f"  hdp_block_sizes: {model.hdp_block_sizes}")
    
except Exception as e:
    print(f"❌ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ SETUP OPTIMIZER ============
print(f"\n⚙️  Setting up optimizer...")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0,
    betas=(0.9, 0.999),
    eps=1e-8
)

print(f"✅ Optimizer created")

# ============ TRAINING LOOP ============
print(f"\n🚀 Starting single-batch overfitting...")
print(f"{'='*80}")
print(f"{'Step':<10} {'Loss':<15} {'Status'}")
print(f"{'-'*80}")

losses = []
best_loss = float('inf')

for step in range(NUM_STEPS):
    try:
        optimizer.zero_grad()
        
        # Forward pass
        loss_output = model._loss(
            single_batch['input_ids'],
            single_batch['attention_mask'],
            block_indices=single_batch. get('block_indices', None)
        )
        
        loss = loss_output.loss
        loss_val = loss.item()
        losses.append(loss_val)
        
        if loss_val < best_loss: 
            best_loss = loss_val
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf detected at step {step+1}!")
            break
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn. utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Print progress
        if (step + 1) % PRINT_EVERY == 0 or step == 0:
            status = ""
            if loss_val < 0.1:
                status = "✅ OVERFITTING!"
            elif loss_val < 1.0:
                status = "🟡 Decreasing..."
            elif step > 100 and len(losses) > 10:
                recent_avg = sum(losses[-10:]) / 10
                old_avg = sum(losses[: 10]) / 10
                if recent_avg > old_avg * 0.95:
                    status = "❌ NOT LEARNING!"
                else:
                    status = "🔵 Training..."
            else:
                status = "🔵 Training..."
            
            print(f"{step+1:<10} {loss_val:<15.6f} {status}")
    
    except Exception as e: 
        print(f"\n❌ Error at step {step+1}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"{'='*80}")

# ============ ANALYSIS ============
print(f"\n📊 ANALYSIS:")
print(f"{'='*80}")

if len(losses) == 0:
    print(f"❌ No losses recorded - training failed immediately!")
    sys.exit(1)

initial_loss = losses[0]
final_loss = losses[-1]
min_loss = min(losses)

print(f"Initial Loss:   {initial_loss:.6f}")
print(f"Final Loss:    {final_loss:.6f}")
print(f"Min Loss:      {min_loss:. 6f}")
print(f"Best Loss:     {best_loss:. 6f}")
print(f"Reduction:     {100*(initial_loss - final_loss)/initial_loss:.1f}%")

print(f"\n🎯 VERDICT:")
if final_loss < 0.1:
    print(f"✅✅✅ SUCCESS! Model CAN overfit on single sample.")
    print(f"   → Core logic is CORRECT")
    print(f"   → HDP attention + loss masking working!")
    print(f"   → Problem with full training is likely:")
    print(f"     • Data quality/diversity")
    print(f"     • Hyperparameters (LR, batch size)")
    print(f"     • Training time too short")
elif final_loss < 1.0:
    print(f"🟡 PARTIAL SUCCESS:  Loss decreased significantly.")
    print(f"   → Model IS learning but slowly")
    print(f"   → Possible issues:")
    print(f"     • Learning rate too low (try {LR*2})")
    print(f"     • Need more steps (try 1000)")
    print(f"     • Architecture bottleneck")
elif len(losses) > 100 and losses[-1] > losses[0] * 0.9:
    print(f"❌❌❌ FAILURE!  Model is NOT learning at all!")
    print(f"   → CRITICAL BUG in model logic")
    print(f"   → ACTION ITEMS:")
    print(f"     1. Check if HDP attention mask is being used")
    print(f"     2. Verify loss masking (Question should contribute 0)")
    print(f"     3. Check backbone forward() signature")
    print(f"     4. Verify gradients are flowing")
    print(f"     5. Try disabling HDP attention (set use_hdp_attention=False)")
else:
    print(f"🟠 UNCLEAR: Loss behavior is unstable")
    print(f"   → Need debugging:")
    if len(losses) < NUM_STEPS:
        print(f"     • Training stopped early - check error logs")
    else:
        print(f"     • Run for more steps (1000+)")
        print(f"     • Check for gradient issues")

# ============ SAMPLE TEST ============
print(f"\n🎲 Testing token predictions...")
print(f"{'='*80}")

model.eval()
with torch.no_grad():
    try:
        # Forward pass to get predictions
        sigma = torch.zeros(1, device=DEVICE)
        
        # Get logits
        logits = model. forward(
            single_batch['input_ids'],
            sigma,
            block_indices=single_batch.get('block_indices', None)
        )
        
        # Get predictions for each block
        all_preds = logits[0].argmax(dim=-1)
        
        # Calculate accuracy per block
        true_tokens = single_batch['input_ids'][0]
        
        for block_id, block_name in [(0, 'Question'), (1, 'Plan'), (2, 'Execution')]:
            mask = (single_batch['block_indices'][0] == block_id)
            block_preds = all_preds[mask]
            block_true = true_tokens[mask]
            
            matches = (block_preds == block_true).sum().item()
            total = len(block_preds)
            accuracy = 100 * matches / total if total > 0 else 0
            
            print(f"\n📊 {block_name} Block:")
            print(f"   Accuracy: {accuracy:.1f}% ({matches}/{total})")
            
            # Decode
            pred_text = tokenizer.decode(block_preds)
            true_text = tokenizer.decode(block_true)
            
            print(f"   True (first 80 chars): {true_text[:80]}...")
            print(f"   Pred (first 80 chars): {pred_text[:80]}...")
            
            if accuracy > 90:
                print(f"   ✅ Block MEMORIZED!")
            elif accuracy > 50:
                print(f"   🟡 Partially memorized")
            else:
                print(f"   ❌ Not memorized")
        
    except Exception as e: 
        print(f"❌ Error during sampling test: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print(f"🏁 Test complete!  See verdict above for next steps.")
print(f"{'='*80}\n")