"""
Visualization of Hierarchical Block Diffusion Architecture

This script generates a diagram showing:
1. Sequence structure
2. Attention flow
3. Block diffusion pattern
"""

def print_architecture_diagram():
    """Print ASCII diagram of the hierarchical architecture."""
    
    diagram = """
═══════════════════════════════════════════════════════════════════════════
                HIERARCHICAL BLOCK DIFFUSION ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════

1. INPUT STRUCTURE (Training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
    │  Question   │   Plan_xt    │   Plan_x0    │   Exec_xt    │   Exec_x0    │
    │   (256)     │    (256)     │    (256)     │    (512)     │    (512)     │
    └─────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
         ↓               ↓               ↓               ↓               ↓
    [Context]      [Noisy Plan]   [Clean Plan]   [Noisy Exec]  [Clean Exec]
    

2. ATTENTION FLOW (Who can see whom?)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Question ──────────────────────────────┐
        │                                  │
        ├──► Question (self-attention)    │
        │                                  │
        └──────────────────────────────────┼───────────────┐
                                           │               │
    Plan ─────────────────────────────┐    │               │
        │                             │    │               │
        ├──► Question ◄───────────────┘    │               │
        │                                  │               │
        ├──► Plan (block diffusion)       │               │
        │                                  │               │
        ╳──X Execution (BLOCKED!) ◄────────┼───────────────┤
                                           │               │
                                           │               │
    Execution ────────────────────────┐    │               │
        │                             │    │               │
        ├──► Question ◄───────────────┴────┘               │
        │                                                  │
        ├──► Plan ◄───────────────────────────────────────┘
        │
        └──► Execution (block diffusion)


3. BLOCK DIFFUSION PATTERN (within Plan/Exec)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Plan_xt blocks (noisy):
    ┌────┬────┬────┬────┐
    │ B1 │ B2 │ B3 │ B4 │  Each block: 16 tokens (if block_size=16)
    └────┴────┴────┴────┘
      ↓    ↓    ↓    ↓
    
    Within-block attention (block diagonal):
    ┌────┐
    │░░░░│ ← B1 attends to itself
    └────┘
         ┌────┐
         │░░░░│ ← B2 attends to itself
         └────┘
              ┌────┐
              │░░░░│ ← B3 attends to itself
              └────┘
                   ┌────┐
                   │░░░░│ ← B4 attends to itself
                   └────┘

    Plan_x0 blocks (clean):
    Can attend to previous Plan_x0 blocks (causal pattern)
    
    B1 → B1
    B2 → B1, B2
    B3 → B1, B2, B3
    B4 → B1, B2, B3, B4


4. ATTENTION MASK MATRIX (Simplified)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

               Q    Plan_xt  Plan_x0  Exec_xt  Exec_x0
            ┌────┬─────────┬────────┬────────┬────────┐
     Q      │ ✓✓ │         │        │        │        │ Full self-attention
            ├────┼─────────┼────────┼────────┼────────┤
  Plan_xt   │ ✓✓ │  ░░░░   │        │        │        │ Block diagonal
            ├────┼─────────┼────────┼────────┼────────┤
  Plan_x0   │ ✓✓ │   ✓✓✓   │  ░░░░  │        │        │ Causal + diagonal
            ├────┼─────────┼────────┼────────┼────────┤
  Exec_xt   │ ✓✓ │   ✓✓✓   │  ✓✓✓   │  ░░░░  │        │ Can see Q+P
            ├────┼─────────┼────────┼────────┼────────┤
  Exec_x0   │ ✓✓ │   ✓✓✓   │  ✓✓✓   │  ✓✓✓   │  ░░░░  │ Can see Q+P+E
            └────┴─────────┴────────┴────────┴────────┘
    
    Legend:
    ✓✓ = Full attention
    ░░░ = Block diffusion pattern (diagonal + causal)
    (empty) = Cannot attend (masked out)


5. TRAINING OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Given: Question
    
    Step 1: Model generates Plan
            ┌─────────┐
            │Question │──► Model ──► Plan_t ──► Denoise ──► Plan_0
            └─────────┘
    
    Step 2: Model generates Execution (conditioned on Question + Plan)
            ┌─────────┬──────┐
            │Question │ Plan │──► Model ──► Exec_t ──► Denoise ──► Exec_0
            └─────────┴──────┘
    
    Loss: NELBO = E[log p(Plan | Question)] + E[log p(Exec | Question, Plan)]


6. INFERENCE (Sampling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Option A: Full generation
    ┌─────────┐
    │Question │
    └────┬────┘
         │
         ├──► Sample Plan (T steps)
         │    ┌──────┐
         │    │ Plan │
         │    └──┬───┘
         │       │
         └───────┴──► Sample Execution (T steps)
                     ┌───────────┐
                     │ Execution │
                     └───────────┘
    
    Option B: Plan-only (for reasoning)
    ┌─────────┐
    │Question │──► Sample Plan ──► ┌──────┐
    └─────────┘                     │ Plan │
                                    └──────┘
    
    Option C: Execution-only (given plan)
    ┌─────────┬──────┐
    │Question │ Plan │──► Sample Execution ──► ┌───────────┐
    └─────────┴──────┘                          │ Execution │
                                                └───────────┘


7. KEY CONSTRAINTS (Causal Structure)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ✓ Question → Question    (Full attention - context understanding)
    ✓ Plan → Question        (Can read input)
    ✓ Plan → Plan            (Block diffusion within plan)
    ✗ Plan → Execution       (BLOCKED - maintains causality!)
    ✓ Execution → Question   (Can read input)
    ✓ Execution → Plan       (Can read high-level plan)
    ✓ Execution → Execution  (Block diffusion within execution)
    
    Why block Plan → Execution?
    - Ensures Plan is generated FIRST (high-level reasoning)
    - Prevents "cheating" by looking at detailed steps
    - Maintains hierarchical structure


═══════════════════════════════════════════════════════════════════════════
                            END OF DIAGRAM
═══════════════════════════════════════════════════════════════════════════
"""
    
    print(diagram)


def print_config_summary():
    """Print configuration summary."""
    
    summary = """
═══════════════════════════════════════════════════════════════════════════
                        CONFIGURATION SUMMARY
═══════════════════════════════════════════════════════════════════════════

Default Configuration:
───────────────────────────────────────────────────────────────────────────
    Question Length:    256 tokens
    Plan Length:        256 tokens
    Execution Length:   512 tokens
    Total Input:        1024 tokens
    
    Training Sequence:  1792 tokens (256 + 256*2 + 512*2)
    Block Size:         16 (adjustable: 4, 8, 16, 32, ...)
    
Model Settings:
───────────────────────────────────────────────────────────────────────────
    Backbone:           DiT (Diffusion Transformer)
    Algorithm:          BD3-LM (Block Discrete Denoising Diffusion)
    Attention:          Hierarchical + Block Diffusion
    
Training Settings:
───────────────────────────────────────────────────────────────────────────
    Learning Rate:      5e-4
    Warmup Steps:       10,000
    Max Steps:          100,000
    Batch Size:         64 (per GPU)
    EMA:                0.9999
    
Sampling Settings:
───────────────────────────────────────────────────────────────────────────
    Method:             First-hitting (faster than DDPM)
    KV Cache:           Enabled (for speedup)
    Var Length:         Disabled (fixed length)
    Nucleus P:          1.0 (no truncation)

Files:
───────────────────────────────────────────────────────────────────────────
    Mask:               models/hierarchical_mask.py
    Data:               hierarchical_dataloader.py
    Config:             configs/algo/hierarchical.yaml
    Training:           scripts/train/train_hierarchical_bd3lm.sh
    Testing:            test_hierarchical_mask.py
    Docs:               HIERARCHICAL_README.md (Vietnamese)
                        QUICKSTART.md (Quick reference)
                        IMPLEMENTATION_SUMMARY.md (Detailed)

═══════════════════════════════════════════════════════════════════════════
"""
    
    print(summary)


def print_usage_examples():
    """Print usage examples."""
    
    examples = """
═══════════════════════════════════════════════════════════════════════════
                           USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════

1. Create and Visualize Mask
───────────────────────────────────────────────────────────────────────────
    python
    >>> from models.hierarchical_mask import create_hierarchical_mask
    >>> mask = create_hierarchical_mask(
    ...     seqlen=1024, block_size=16,
    ...     question_len=256, plan_len=256, exec_len=512
    ... )
    >>> print(f"Mask shape: {mask.shape}")
    >>> # Verify Plan cannot see Execution
    >>> assert not mask[256:512, 512:].any()

2. Prepare Data
───────────────────────────────────────────────────────────────────────────
    python
    >>> from hierarchical_dataloader import HierarchicalDataCollator
    >>> from transformers import AutoTokenizer
    >>> 
    >>> tokenizer = AutoTokenizer.from_pretrained('gpt2')
    >>> collator = HierarchicalDataCollator(
    ...     tokenizer, question_len=256, plan_len=256, exec_len=512
    ... )
    >>> 
    >>> batch = collator([
    ...     {'text': 'Your input text here...'},
    ...     {'text': 'Another example...'}
    ... ])
    >>> print(batch['input_ids'].shape)  # [2, 1024]

3. Train Model
───────────────────────────────────────────────────────────────────────────
    bash
    # Edit configuration if needed
    $ vim scripts/train/train_hierarchical_bd3lm.sh
    
    # Run training
    $ sbatch scripts/train/train_hierarchical_bd3lm.sh
    
    # Or run directly
    $ python main.py \\
          mode=train \\
          model=small \\
          algo=bd3lm \\
          block_size=16 \\
          training.hierarchical.enabled=true

4. Test Before Training
───────────────────────────────────────────────────────────────────────────
    bash
    # Verify mask implementation
    $ python test_hierarchical_mask.py
    
    # Expected output:
    # ✅ Question tokens can see all other question tokens
    # ✅ Plan Block cannot see Execution Block
    # ✅ Plan Block can see Question
    # ✅ Execution Block can see Question
    # ✅ Execution Block can see Plan Block
    # All tests passed! ✅

5. Customize for Your Domain
───────────────────────────────────────────────────────────────────────────
    python
    # In hierarchical_dataloader.py
    def process_example(example):
        text = example['text']
        
        # Your custom logic here
        question = extract_question_with_regex(text)
        plan = identify_plan_section(text)
        execution = extract_remaining_as_execution(text)
        
        return {
            'question': tokenizer.encode(question),
            'plan': tokenizer.encode(plan),
            'execution': tokenizer.encode(execution),
        }

═══════════════════════════════════════════════════════════════════════════
"""
    
    print(examples)


if __name__ == '__main__':
    print_architecture_diagram()
    print()
    print_config_summary()
    print()
    print_usage_examples()
    
    print("\n" + "="*75)
    print("For more details, see:")
    print("  - QUICKSTART.md (Quick reference)")
    print("  - HIERARCHICAL_README.md (Full documentation in Vietnamese)")
    print("  - IMPLEMENTATION_SUMMARY.md (Detailed implementation notes)")
    print("="*75)
