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
  # Training adds: [PAD], [PLAN], [EXECUTION], [ANSWER]
  # This increases vocab from 50257 → 50261
  if not tokenizer.pad_token:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  
  special_tokens_dict = {'additional_special_tokens': ['[PLAN]', '[EXECUTION]', '[ANSWER]']}
  num_added = tokenizer.add_special_tokens(special_tokens_dict)
  
  print(f"\n🔧 Added {num_added} special tokens to tokenizer")
  print(f"   New tokenizer vocab_size: {len(tokenizer)}")
  print(f"   Special tokens: [PAD]={tokenizer.pad_token_id}, [PLAN]={tokenizer.additional_special_tokens_ids[0]}, [EXECUTION]={tokenizer.additional_special_tokens_ids[1]}, [ANSWER]={tokenizer.additional_special_tokens_ids[2]}")

  # Extract vocab_size from checkpoint BEFORE creating model
  import torch
  import omegaconf
  ckpt = torch.load(config.eval.checkpoint_path, map_location='cpu', weights_only=False)
  if 'state_dict' in ckpt:
    # Get vocab_size from embedding layer shape
    embed_key = 'backbone.vocab_embed.embedding'
    if embed_key in ckpt['state_dict']:
      checkpoint_vocab_size = ckpt['state_dict'][embed_key].shape[0]
      print(f"\n🔧 Checkpoint vocab_size: {checkpoint_vocab_size}")
      print(f"   Tokenizer vocab_size: {len(tokenizer)}")
      
      # Override config vocab_size to match checkpoint
      if checkpoint_vocab_size != len(tokenizer):
        print(f"   ⚠️  MISMATCH DETECTED! Adjusting model vocab_size to {checkpoint_vocab_size}")
        # Temporarily disable struct mode to add new key
        omegaconf.OmegaConf.set_struct(config, False)
        config.model.vocab_size = checkpoint_vocab_size
        omegaconf.OmegaConf.set_struct(config, True)

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
    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    if config.eval. disable_ema:
        logger.info('Disabling EMA.')
        model. ema = None

    # Check if HDP mode with test questions
    question_tokens = None
    if hasattr(config.data, 'hdp') and config.data.hdp.get('use_hdp_attention', False):
        logger.info('HDP mode enabled - loading test questions')

        # Determine test data path
        test_path = None
        if hasattr(config.data, 'test_path') and config.data.test_path:
            test_path = config.data.test_path
        elif config.data.valid == 'gsm8k':
            # Use overfit data for testing
            test_path = 'data/gsm8k/gsm8k_overfit.json'
        
        if test_path:
            import json
            try:
                with open(test_path, 'r') as f:
                    test_data = json.load(f)

                # Get first few questions for sampling
                num_samples = config.sampling.num_sample_batches * config.loader.eval_batch_size
                questions = [sample['question'] for sample in test_data[:num_samples]]

                # Tokenize questions
                q_len = config.data.hdp.question_len
                question_tokens = tokenizer(
                    questions,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=q_len
                )['input_ids'].to('cuda')

                logger.info(f'✅ Loaded {len(questions)} test questions for conditional generation')
                logger.info(f'   Question length: {q_len} tokens')
            except Exception as e:
                logger.warning(f'Could not load test questions: {e}')
                logger.warning('Falling back to unconditional generation')
        else:
            logger.warning('No test data path configured - unconditional generation')

    # Generate samples
    text_samples = model.restore_model_and_sample(
        num_steps=config.sampling.get('num_steps', config.algo.T),
        question_tokens=question_tokens
    )

    print('Text samples:', text_samples)
    print('Generative perplexity:', model.metrics. gen_ppl. compute())
    print('Entropy:', model.metrics. gen_entropy.compute())

    csv_path = config. sampling.logdir
    save_dict = {
        'gen_ppl': model.metrics.gen_ppls,
        'gen_nfes': model. metrics.gen_nfes,
        'gen_entropy': model.metrics. gen_entropies,
        'gen_lengths': model.metrics. gen_lengths,
        'samples': [[i] for i in text_samples],
        'seed': [config.seed for _ in range(len(text_samples))]
    }

    if config.sampling.var_length:
        print(text_samples)
        save_dict['samples'] = ['' for _ in range(len(text_samples))]

    utils.update_and_save_csv(save_dict, csv_path)
    return text_samples

def _display_hdp_samples(samples, validation_data, tokenizer):
    """Display HDP samples with question context, clearly separating [PLAN], [EXECUTION], [ANSWER]."""
    print("\n" + "="*100)
    print("HDP-DIFFUSION SAMPLES WITH QUESTION CONTEXT")
    print("="*100)

    for idx, sample_text in enumerate(samples[:5]):  # Show first 5 samples
        truth = validation_data[idx] if idx < len(validation_data) else None

        plan_marker = "[PLAN]"
        exec_marker = "[EXECUTION]"
        answer_marker = "[ANSWER]"
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
        print(f"\n🧠 [PLAN]:\n   {plan}")
        print(f"\n🔢 [EXECUTION]:\n   {execution}")
        print(f"\n✅ [ANSWER]:\n   {answer}")

        if truth:
            print(f"\n🏷️  GROUND TRUTH:")
            print(f"   [PLAN]: {truth['plan']}")
            print(f"   [EXECUTION]: {truth['execution']}")
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