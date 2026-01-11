import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

import dataloader
import diffusion
import utils

from hdp_dataset import HDPDataset

# Enable Tensor Cores optimization for H200/A100 GPUs
# Trade-off precision for performance on Ampere+ architectures
torch.set_float32_matmul_precision('high')  # Options: 'highest', 'high', 'medium'

# Use new TF32 API (Pytorch 2.0+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  # 🔧 FIX: Add special tokens to tokenizer to match training
  # Training adds: <|pad|>, <|plan|>, <|execution|>, <|answer|>
  # GPT-2 base vocab: 50257
  # After adding special tokens: 50257 + 4 = 50261
  # But checkpoint might have 50262 if extra token was added during training
  if not tokenizer.pad_token:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
  
  special_tokens_dict = {'additional_special_tokens': ['<|question|>', '<|plan|>', '<|execution|>', '<|answer|>']}
  num_added = tokenizer.add_special_tokens(special_tokens_dict)
  
  # Store marker IDs for hard position anchoring
  # Need to disable struct mode to add new keys
  import omegaconf
  omegaconf.OmegaConf.set_struct(config, False)
  config.model.question_token_id = tokenizer.additional_special_tokens_ids[0] # <|question|>
  config.model.plan_token_id = tokenizer.additional_special_tokens_ids[1]     # <|plan|>
  config.model.execution_token_id = tokenizer.additional_special_tokens_ids[2] # <|execution|>
  config.model.answer_token_id = tokenizer.additional_special_tokens_ids[3]    # <|answer|>
  omegaconf.OmegaConf.set_struct(config, True)
  
  print(f"\n🔧 Added {num_added} special tokens to tokenizer")
  print(f"   New tokenizer vocab_size: {len(tokenizer)}")
  print(f"   Special tokens: <|pad|>={tokenizer.pad_token_id}, <|plan|>={tokenizer.additional_special_tokens_ids[0]}, <|execution|>={tokenizer.additional_special_tokens_ids[1]}, <|answer|>={tokenizer.additional_special_tokens_ids[2]}, <|mask|>={tokenizer.mask_token_id}")
  
  # ✅ Tokenizer now has mask_token, so Diffusion.__init__ won't auto-increment vocab_size
  # Training and inference will use same vocab_size = 50262 (50257 base + 5 special tokens)

  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def generate_samples(config, logger, tokenizer):
    """
    Generate samples using the trained model.
    Now uses HDPDataset to ensure consistency between training and inference data formatting.
    """
    
    # 1. Load Model
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    # 2. Prepare Question Tokens (Conditioning)
    question_tokens = None
    
    # Check if HDP mode is enabled in config
    if hasattr(config.data, 'hdp') and config.data.hdp.get('use_hdp_attention', False):
        logger.info('HDP mode enabled - Loading test data via HDPDataset for consistency')

        # Determine test path
        test_path = None
        if hasattr(config.data, 'test_path') and config.data.test_path:
            test_path = config.data.test_path
        elif config.data.valid == 'gsm8k':
            test_path = 'data/gsm8k/test.json' # Default fallback
        
        if test_path and os.path.exists(test_path):
            try:
                # Get block sizes from config
                q_len = config.data.hdp.question_len
                p_len = config.data.hdp.plan_len
                e_len = config.data.hdp.execution_len
                
                # Initialize Dataset
                # IMPORTANT: This ensures <|question|> tag and padding are handled exactly like training
                eval_dataset = HDPDataset(
                    data_path=test_path,
                    tokenizer=tokenizer,
                    block_sizes=(q_len, p_len, e_len),
                    use_special_format=True, 
                    return_block_indices=True
                )

                # Determine how many samples to generate
                num_samples = config.sampling.num_sample_batches * config.loader.eval_batch_size
                num_samples = min(num_samples, len(eval_dataset))
                
                logger.info(f"Extracting questions from first {num_samples} samples in dataset...")
                
                batch_q_ids = []
                for i in range(num_samples):
                    sample = eval_dataset[i]
                    
                    # sample['input_ids'] contains [Question | Plan | Execution]
                    # We only need the Question part for the conditional input
                    full_seq = sample['input_ids']
                    
                    # Logic 1: Slice by length (Fastest & Safest for fixed block sizes)
                    q_ids = full_seq[:q_len]
                    
                    # Logic 2 (Alternative): Slice by block_indices if available (More robust)
                    # if 'block_indices' in sample:
                    #     q_ids = full_seq[sample['block_indices'] == 0]
                    
                    batch_q_ids.append(q_ids)
                
                if batch_q_ids:
                    question_tokens = torch.stack(batch_q_ids).to('cuda')
                    logger.info(f'✅ Loaded {len(question_tokens)} questions. Shape: {question_tokens.shape}')
                    # CHÈN VÀO MAIN.PY TRƯỚC KHI SAMPLING
                    print("\n=== DEBUG DIAGNOSTIC ===")
                    # 1. Check Tokenizer
                    print(f"Vocab Size: {len(tokenizer)}")
                    print(f"Plan Token ID: {tokenizer.convert_tokens_to_ids('<|plan|>')}")
                    print(f"Question Token ID: {tokenizer.convert_tokens_to_ids('<|question|>')}")

                    # 2. Check Input Data
                    if question_tokens is not None:
                        print("\nInput Tensor Shape:", question_tokens.shape)
                        print("Input Tensor First 10 IDs:", question_tokens[0, :10].tolist())
                        print("Decoded Input:", tokenizer.decode(question_tokens[0], skip_special_tokens=False))
                    else:
                        print("\n❌ CRITICAL: question_tokens is NONE! Model is generating from pure noise!")

                    print("========================\n")
                    logger.info('Generating samples...')
                
            except Exception as e:
                logger.error(f'Error utilizing HDPDataset for inference: {e}')
                logger.warning('Falling back to unconditional generation (or ensure test.json has required keys)')
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f'Test path not found: {test_path}. Performing unconditional generation.')

    # 3. Generate Samples
    # Pass seqlen explicitly for semi-AR sampler or correct diffusion length
    seq_len = config.model.length
    print("DEBUG INFERENCE INPUT:")
    print(tokenizer.decode(question_tokens[0], skip_special_tokens=False))
    logger.info(f"Starting sampling loop (Steps={config.sampling.get('num_steps', config.algo.T)})...")
    text_samples = model.restore_model_and_sample(
        num_steps=config.sampling.get('num_steps', config.algo.T),
        seqlen=seq_len, 
        question_tokens=question_tokens
    )

    # 4. Metrics & Logging
    print('Text samples:', text_samples)
    
    # Calculate metrics if available
    try:
        gen_ppl = model.metrics.gen_ppl.compute()
        print('Generative perplexity:', gen_ppl)
        gen_entropy = model.metrics.gen_entropy.compute()
        print('Entropy:', gen_entropy)
    except Exception:
        print("Metrics computation skipped (not enough data or metric error).")

    # 5. Save to CSV
    csv_path = config.sampling.logdir
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    save_dict = {
        'gen_ppl': model.metrics.gen_ppls,
        'gen_nfes': model.metrics.gen_nfes,
        'gen_entropy': model.metrics.gen_entropies,
        'gen_lengths': model.metrics.gen_lengths,
        'samples': [[i] for i in text_samples],
        'seed': [config.seed for _ in range(len(text_samples))]
    }

    if config.sampling.var_length:
        # If variable length, we might want to store raw strings differently
        # But keeping structure consistent
        pass

    utils.update_and_save_csv(save_dict, csv_path)
    logger.info(f"Samples saved to {csv_path}")
    
    return text_samples

def _display_hdp_samples(samples, validation_data, tokenizer):
    """Display HDP samples with question context, clearly separating <|plan|>, <|execution|>, <|answer|>."""
    print("\n" + "="*100)
    print("HDP-DIFFUSION SAMPLES WITH QUESTION CONTEXT")
    print("="*100)

    for idx, sample_text in enumerate(samples[:5]):  # Show first 5 samples
        truth = validation_data[idx] if idx < len(validation_data) else None

        plan_marker = "<|plan|>"
        exec_marker = "<|execution|>"
        answer_marker = "<|answer|>"
        question = truth['question'] if truth else "N/A"

        print(f"\n{'─'*100}")
        print(f"📊 SAMPLE #{idx + 1}")
        print(f"{'─'*100}")
        print(f"\n🔍 RAW GENERATED TEXT (first 500 chars):")
        print(f"   {repr(sample_text[:500])}")

        # Extract PLAN, EXECUTION, ANSWER
        plan = ""
        execution = ""
        answer = ""
        has_plan_marker = plan_marker in sample_text
        has_exec_marker = exec_marker in sample_text
        has_answer_marker = answer_marker in sample_text

        if has_plan_marker and has_exec_marker:
            _, rest = sample_text.split(plan_marker, 1)
            plan_part, exec_rest = rest.split(exec_marker, 1)
            plan = plan_part.strip()
            # Try to extract [ANSWER] from execution part
            if answer_marker in exec_rest:
                exec_part, ans_part = exec_rest.split(answer_marker, 1)
                execution = exec_part.strip()
                answer = ans_part.strip()
            else:
                execution = exec_rest.strip()
                answer = "[Not found in output]"
        else:
            # Try to extract by block position if markers not found
            plan = "[Markers not found - Model may not have learned format]"
            execution = "[Markers not found - Model may not have learned format]"
            answer = "[Markers not found - Model may not have learned format]"
            try:
                tokens = tokenizer.encode(sample_text)
                if len(tokens) >= 384:
                    # Assume: tokens[0:128]=question, [128:256]=plan, [256:384]=execution, [384:]=answer
                    plan_tokens = tokens[128:256]
                    exec_tokens = tokens[256:384]
                    ans_tokens = tokens[384:]
                    plan = tokenizer.decode(plan_tokens).strip()
                    execution = tokenizer.decode(exec_tokens).strip()
                    answer = tokenizer.decode(ans_tokens).strip() if ans_tokens else "[Not found in output]"
                elif len(tokens) >= 256:
                    plan_tokens = tokens[128:256]
                    exec_tokens = tokens[256:]
                    plan = tokenizer.decode(plan_tokens).strip()
                    execution = tokenizer.decode(exec_tokens).strip()
                    answer = "[Not found in output]"
            except Exception:
                pass

        # Clean padding
        plan = plan.replace(tokenizer.eos_token, '').strip()
        execution = execution.replace(tokenizer.eos_token, '').strip()
        answer = answer.replace(tokenizer.eos_token, '').strip()

        print(f"\n📋 QUESTION (from validation set):\n   {question}")
        print(f"\n🧠 <|plan|>:\n   {plan}")
        print(f"\n🔢 <|execution|>:\n   {execution}")
        print(f"\n✅ <|answer|>:\n   {answer}")

        if truth:
            print(f"\n🏷️  GROUND TRUTH:")
            # Check if HDP format (with plan/execution) or baseline format (just answer)
            if 'plan' in truth and 'execution' in truth:
                print(f"   [PLAN]: {truth['plan']}")
                print(f"   [EXECUTION]: {truth['execution']}")
                print(f"   [ANSWER]: {truth.get('answer', 'N/A')}")
            else:
                # Baseline format: just question + answer
                print(f"   [ANSWER]: {truth.get('answer', 'N/A')}")

        print(f"\n📈 FORMAT ANALYSIS:")
        print(f"   [PLAN] marker found: {'✅' if has_plan_marker else '❌'}")
        print(f"   [EXECUTION] marker found: {'✅' if has_exec_marker else '❌'}")
        print(f"   [ANSWER] marker found: {'✅' if has_answer_marker else '❌'}")
        if not (has_plan_marker and has_exec_marker):
            print(f"   ⚠️  Model has not learned the hierarchical format yet!")
            print(f"   💡 Suggestion: Train longer or check if training data has [PLAN]/[EXECUTION]/[ANSWER] markers")

    print("\n" + "="*100 + "\n")

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)

def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  print(f"\n{'='*60}")
  print(f"DEBUG:  Dataloader Info")
  print(f"{'='*60}")
  print(f"Train dataloader:")
  print(f"  - len(train_ds): {len(train_ds)}")
  print(f"  - len(train_ds.dataset): {len(train_ds.dataset)}")
  print(f"  - batch_size: {train_ds. batch_size}")
  print(f"Valid dataloader:")
  print(f"  - len(valid_ds): {len(valid_ds)}")
  print(f"  - len(valid_ds.dataset): {len(valid_ds.dataset)}")
  print(f"  - batch_size: {valid_ds.batch_size}")
  print(f"{'='*60}\n")
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.from_pretrained is not None and ckpt_path is None:
    logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
    # load pretraining checkpoint
    if 'kuleshov-group/' in config.training.from_pretrained:
      # load from hf
      model = diffusion.Diffusion(config, tokenizer=tokenizer)
      state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
          config.training.from_pretrained,
          trust_remote_code=True
      ).state_dict()
      model.load_state_dict(state_dict)
    else:
      model = diffusion.Diffusion.load_from_checkpoint(
        config.training.from_pretrained,
        tokenizer=tokenizer,
        config=config,
        strict=False)
    # add buffers for grid search
    model.register_buffer('sampling_eps_min', torch.tensor(
      config.training.sampling_eps_min))
    model.register_buffer('sampling_eps_max', torch.tensor(
      config.training.sampling_eps_max))
  else:
    logger.info(f'Initializing new model')
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)

  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    config.wandb = None
    # 1. Sinh ra các mẫu văn bản
    samples = generate_samples(config, logger, tokenizer)

    # 2. Tải dữ liệu gốc (Ground Truth) để so sánh (nếu có)
    # Phần này mô phỏng logic đọc file giống trong generate_samples
    validation_data = []
    if hasattr(config, 'data') and hasattr(config.data, 'test_path') and config.data.test_path:
        import json
        try:
            with open(config.data.test_path, 'r') as f:
                validation_data = json.load(f)
            logger.info(f"Loaded {len(validation_data)} validation samples for display comparison.")
        except Exception as e:
            logger.warning(f"Could not load validation data for display: {e}")
    
    # 3. Gọi hàm hiển thị format HDP
    _display_hdp_samples(samples, validation_data, tokenizer)

  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()