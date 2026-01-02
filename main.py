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
    if hasattr(config, 'hdp') and config.hdp. get('enabled', False):
        logger.info('HDP mode enabled - loading test questions')

        # Load test questions from dataset
        if hasattr(config. data, 'test_path') and config.data. test_path:
            import json
            try:
                with open(config.data. test_path, 'r') as f:
                    test_data = json. load(f)

                # Get first few questions for sampling
                num_samples = config.sampling.num_sample_batches * config.loader.eval_batch_size
                questions = [sample['question'] for sample in test_data[: num_samples]]

                # Tokenize questions
                q_len = config. hdp. question_len
                question_tokens = tokenizer(
                    questions,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=q_len
                )['input_ids'].to('cuda')

                logger.info(f'Loaded {len(questions)} test questions for conditional generation')
            except Exception as e:
                logger.warning(f'Could not load test questions:  {e}')
                logger.warning('Falling back to unconditional generation')

    # Generate samples
    text_samples = model. restore_model_and_sample(
        num_steps=config. algo.T,
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
  """Display HDP samples with question context."""
  print("\n" + "="*100)
  print("HDP-DIFFUSION SAMPLES WITH QUESTION CONTEXT")
  print("="*100)

  for idx, sample_text in enumerate(samples[:5]):  # Show first 5 samples
    truth = validation_data[idx] if idx < len(validation_data) else None

    # Parse sample
    plan_marker = "[PLAN]"
    exec_marker = "[EXECUTION]"

    question = truth['question'] if truth else "N/A"

    # Show raw sample for debugging
    print(f"\n{'─'*100}")
    print(f"📊 SAMPLE #{idx + 1}")
    print(f"{'─'*100}")
    print(f"\n🔍 RAW GENERATED TEXT (first 500 chars):")
    print(f"   {repr(sample_text[:500])}")

    if plan_marker in sample_text and exec_marker in sample_text:
      _, rest = sample_text.split(plan_marker, 1)
      plan_part, exec_part = rest.split(exec_marker, 1)
      plan = plan_part.strip()
      execution = exec_part.strip()
    else:
      # Try to parse by position if markers not found
      plan = "[Markers not found - Model may not have learned format]"
      execution = "[Markers not found - Model may not have learned format]"

      # Try to extract by counting tokens (128, 128, 256 blocks)
      try:
        tokens = tokenizer.encode(sample_text)
        if len(tokens) >= 256:
          # Assume: tokens[0:128] = question, [128:256] = plan, [256:] = execution
          plan_tokens = tokens[128:256]
          exec_tokens = tokens[256:]
          plan = tokenizer.decode(plan_tokens).strip()
          execution = tokenizer.decode(exec_tokens).strip()
      except:
        pass

    # Clean padding
    plan = plan.replace(tokenizer.eos_token, '').strip()
    execution = execution.replace(tokenizer.eos_token, '').strip()

    print(f"\n📋 QUESTION (from validation set):")
    print(f"   {question}")
    print(f"\n🧠 GENERATED PLAN:")
    print(f"   {plan}")
    print(f"\n🔢 GENERATED EXECUTION:")
    print(f"   {execution}")

    if truth:
      print(f"\n✅ GROUND TRUTH:")
      print(f"   Plan: {truth['plan']}")
      print(f"   Execution: {truth['execution']} [ANSWER] {truth.get('answer', 'N/A')}")

    # Analysis
    has_plan_marker = plan_marker in sample_text
    has_exec_marker = exec_marker in sample_text
    print(f"\n📈 FORMAT ANALYSIS:")
    print(f"   [PLAN] marker found: {'✅' if has_plan_marker else '❌'}")
    print(f"   [EXECUTION] marker found: {'✅' if has_exec_marker else '❌'}")

    if not (has_plan_marker and has_exec_marker):
      print(f"   ⚠️  Model has not learned the hierarchical format yet!")
      print(f"   💡 Suggestion: Train longer or check if training data has [PLAN]/[EXECUTION] markers")

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

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)

  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    config.wandb = None
    samples = generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()