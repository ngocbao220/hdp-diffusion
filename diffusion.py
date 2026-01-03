import itertools
from dataclasses import dataclass
import logging

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict

import dataloader
import metrics
import models
import noise_schedule
import utils
import numpy as np
import itertools
from models.hdp_attention_mask import get_hdp_attention_bias
# Note: hdp_temporal_mask not used with sequential denoising approach

LOGGER = logging.getLogger(__name__)

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  
  # Debug first call
  if not hasattr(_sample_categorical, '_debug_printed'):
    _sample_categorical._debug_printed = True
    print(f"\n🔍 [_sample_categorical] FIRST CALL DEBUG:")
    print(f"   categorical_probs.shape: {categorical_probs.shape}")
    print(f"   samples.shape: {samples.shape}")
    print(f"   samples unique values: {torch.unique(samples).tolist()[:20]}")  # First 20
    
    # Show first 10 samples (or less if sequence is shorter)
    seq_len = samples.shape[1]
    show_len = min(10, seq_len)
    print(f"   First {show_len} samples: {samples[0, :show_len].tolist()}")
    
    # Check if we sampled high indices
    max_sample = samples.max().item()
    print(f"   Max sampled value: {max_sample}")
    if max_sample > 50000:
      print(f"   ✅ Sampled tokens in high range (50000+)")
    else:
      print(f"   ⚠️  Max sample {max_sample} < 50000!")
      
    # Check probs at position 128 (first Plan token) - only if sequence is long enough
    if seq_len > 128:
      print(f"   Samples at positions 128-138 (Plan block): {samples[0, 128:min(138, seq_len)].tolist()}")
      probs_at_128 = categorical_probs[0, 128]
      top5 = probs_at_128.topk(5)
      print(f"   Probs at position 128 (first Plan token):")
      print(f"     Top 5 indices: {top5.indices.tolist()}")
      print(f"     Top 5 probs: {top5.values.tolist()}")
    else:
      print(f"   ℹ️  Sequence length {seq_len} < 128 (semi_ar block sampling)")
  
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor

class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    # self.vocab_size = self.tokenizer.vocab_size
    self.vocab_size = len(self.tokenizer) 
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.parameterization == 'ar':
      self.block_size = 1
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      #  egenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.get('var_min', False)  # Default False
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    
    # HDP attention configuration
    self.use_hdp_attention = False
    self.hdp_block_sizes = None

    if hasattr(self.config, 'data') and hasattr(self.config.data, 'hdp'):
      hdp_config = self.config.data.hdp
      self.use_hdp_attention = self.config.data.hdp.get('use_hdp_attention', False)

      if self.use_hdp_attention:
        q_len = hdp_config.get('question_len', 128)
        p_len = hdp_config.get('plan_len', 128)
        e_len = hdp_config. get('exec_len', 256)

        self.hdp_block_sizes = (
          q_len,  # Question block
          p_len,  # Plan block
          e_len   # Execution block
        )

        print(f"✅ HDP Attention enabled with block sizes: {self.hdp_block_sizes}")
        print(f"   Question: {q_len}, Plan: {p_len}, Execution: {e_len}")
        print(f"   Total seq length: {sum(self.hdp_block_sizes)}")
    
    self._validate_configuration()

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    # For sample_eval mode with first_hitting, check batch size only if not HDP
    is_hdp = (hasattr(self.config.data, 'name') and self.config.data.name == 'hdp_diffusion')
    if self.config.mode == 'sample_eval' and \
        hasattr(self.config.sampling, 'first_hitting') and \
        self.config.sampling.first_hitting and \
        not is_hdp:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {
      'dit', 'ar', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if hasattr(self.config.sampling, 'kv_cache') and self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
      
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma
  
  def _create_hdp_bd3lm_mask(self, block_indices, seq_len, device):
    """
    Create combined HDP + BD3-LM attention mask.
    
    For BD3-LM, the sequence is [xt | x0] where: 
    - xt tokens attend to:  their own block (diagonal) + previous x0 blocks
    - x0 tokens attend to:  their own block + previous x0 blocks (block causal)
    
    We need to combine this with HDP constraints: 
    - Question:  only attend to Question
    - Plan: attend to Question + Plan
    - Execution:  attend to all
    
    Note: With sequential denoising, this is a STATIC mask.
    The hierarchy is enforced by freezing blocks, not by changing mask.
    """
    n = seq_len // 2  # Original sequence length (before [xt, x0] concat)
    
    # Static HDP mask (hierarchy enforced by sequential denoising)
    hdp_mask = get_hdp_attention_bias(
        block_indices=block_indices,
        seq_len=seq_len,
        block_sizes=self.hdp_block_sizes,
        causal_within_block=False,
        device=device,
        dtype=torch.float32
    )
    
    # Get BD3-LM block diffusion mask
    # This should already be created in backbone, but we need to combine
    if hasattr(self.backbone, 'block_diff_mask'):
        bd3_mask = self.backbone.block_diff_mask
        if bd3_mask is not None:
            # Ensure same device
            bd3_mask = bd3_mask.to(device)

            # Check if shapes match
            if bd3_mask.shape[-2:] == hdp_mask.shape[-2:]:
                # Combine masks: both must allow attention
                # hdp_mask: 0 = allow, -inf = mask
                # bd3_mask: True = allow, False = mask
              
                if bd3_mask.dtype == torch.bool:
                    bd3_bias = torch.zeros_like(hdp_mask)
                    bd3_bias = bd3_bias.masked_fill(~bd3_mask, float('-inf'))
                else:
                    bd3_bias = bd3_mask
                    print(f"   ❌ SHAPES MISMATCH - BD3 mask IGNORED!")
                    print(f"   This is a BUG! HDP attention will not work correctly!")
                
                # Combined: take minimum (most restrictive)
                hdp_mask = torch.minimum(hdp_mask, bd3_bias)
            # else: shapes don't match, skip BD3 mask combination
    
    # Add head dimension:  (batch, seq, seq) -> (batch, 1, seq, seq)
    hdp_mask = hdp_mask.unsqueeze(1)
    
    return hdp_mask

  def forward(self, x, sigma, sample_mode=False, store_kv=False, block_indices=None):
    """Returns log score.
    
    Args:
        x: Input tensor
        sigma: Noise level
        sample_mode: Whether in sampling mode
        store_kv: Whether to store key-value cache
        block_indices: HDP block assignments
    """
    # Only print debug info once at the start of sampling (not every step!)
    if sample_mode and not hasattr(self, '_forward_debug_printed'): 
        self._forward_debug_printed = True
        print(f"\n🔍 [Diffusion.forward] First call in sample mode:")
        print(f"   x.shape: {x.shape}")
        print(f"   use_hdp_attention: {self.use_hdp_attention}")
        print(f"   block_indices: {'present' if block_indices is not None else 'None'}")
        if block_indices is not None:
            print(f"   block_indices.shape: {block_indices.shape}")
            print(f"   unique blocks: {torch.unique(block_indices).tolist()}")

    sigma = self._process_sigma(sigma)
    
    # Create HDP attention mask if enabled and block_indices provided
    hdp_mask = None
    if self.use_hdp_attention and block_indices is not None:
      # For BD3-LM cross attention, x is concatenated (xt, x0), so block_indices needs to be duplicated
      if self.cross_attn and x.shape[1] == block_indices.shape[1] * 2:
        block_indices_full = torch.cat((block_indices, block_indices), dim=1)
      else:
        block_indices_full = block_indices
      
      # Create HDP attention bias for SDPA
      hdp_mask = self._create_hdp_bd3lm_mask(
        block_indices=block_indices_full,
        seq_len=x.shape[1],
        device=x.device
      )
    elif sample_mode and not hasattr(self, '_forward_debug_printed2'):
        self._forward_debug_printed2 = True
        print(f"   ⚠️  HDP mask not created (missing block_indices or HDP disabled)")
    
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name == 'bd3lm':
        logits = self.backbone(x, sigma,
                              store_kv=store_kv,
                              sample_mode=sample_mode,
                              hdp_mask=hdp_mask)
      elif self.config.algo.name == 'ar':
        if self.config.algo.backbone == 'hf_dit':
          logits = self.backbone(x, None)     
        else:
          logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits.log_softmax(-1)
      else:
        logits = self.backbone(x, sigma)

    if self.cross_attn:
      x = x[:, :self.config.model.length]
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                      xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                        xt=x,
                                        sigma=sigma)
    return logits
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    block_indices = batch.get('block_indices', None)
    
    # 🔍 DEBUG: Print once to verify block_indices is passed
    if not hasattr(self, '_training_debug_printed'):
      self._training_debug_printed = True
      print("\n" + "="*80)
      print("🔍 [training_step] FIRST TRAINING BATCH DEBUG")
      print("="*80)
      print(f"Batch keys: {batch.keys()}")
      print(f"block_indices present: {block_indices is not None}")
      if block_indices is not None:
        print(f"block_indices.shape: {block_indices.shape}")
        print(f"unique blocks: {torch.unique(block_indices).tolist()}")
      print(f"use_hdp_attention: {self.use_hdp_attention}")
      print("="*80 + "\n")
    
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'],
                        block_indices=block_indices)
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    
    # 🔍 DEBUG: Check if model learned anything
    if self.global_step % 100 == 0:
      with torch.no_grad():
        print("\n" + "="*80)
        print(f"🔍 [training_step] MODEL LEARNING CHECK at step {self.global_step}")
        print("="*80)
        
        # ✅ FIX: Use ACTUAL training forward pass, not sample mode!
        # Training uses cross_attn: model([xt, x0], sigma) where xt is noisy
        x0 = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Sample t randomly (same as training!)
        # Training uses t ~ Uniform[sampling_eps_min, sampling_eps_max] = [0.001, 1.0]
        # Average t ≈ 0.5, so we get realistic noise level matching training distribution
        t_random = self._sample_t(x0.shape, self.device, 
                                   sampling_eps_min=1e-3, 
                                   sampling_eps_max=1.0)
        _, p_random = self.noise(t_random)
        # Add noise (same distribution as training)
        xt_random = self.q_xt(x0, p_random, sampling_eps_min=1e-3, sampling_eps_max=1.0)
        
        # ✅ CRITICAL: Keep Question block CLEAN (same as training!)
        # Question is input, should NOT be masked
        if block_indices is not None:
          # Handle both 1D [seq_len] and 2D [batch, seq_len] block_indices
          if block_indices.ndim == 1:
            question_mask = (block_indices == 0).unsqueeze(0).expand_as(x0)
          else:
            question_mask = (block_indices == 0)
          # Restore Question tokens to clean (no masking)
          xt_random = torch.where(question_mask, x0, xt_random)
        
        # 🔍 DEBUG: Log noise level and masking statistics
        avg_t = t_random.mean().item()
        num_masked = (xt_random == self.mask_index).sum().item()
        total_tokens = xt_random.numel()
        print(f"🔍 Accuracy check noise level:")
        print(f"   Average t: {avg_t:.3f}")
        print(f"   Masked tokens: {num_masked}/{total_tokens} ({num_masked/total_tokens*100:.1f}%)")
        if block_indices is not None:
          for block_id in range(3):
            if block_indices.ndim == 1:
              block_mask_indices = (block_indices == block_id)
              block_tokens = xt_random[:, block_mask_indices]
            else:
              block_mask_indices = (block_indices == block_id)
              block_tokens = xt_random[block_mask_indices]
            num_masked_block = (block_tokens == self.mask_index).sum().item()
            total_block = block_tokens.numel()
            print(f"   Block {block_id}: {num_masked_block}/{total_block} masked ({num_masked_block/total_block*100:.1f}%)")
        
        # Compute sigma from p
        sigma_random = self._sigma_from_p(p_random[:, 0].unsqueeze(-1))
        
        # Use training mode: [xt, x0] concatenation if cross_attn
        if self.cross_attn:
          x_input = torch.cat((xt_random, x0), dim=-1)
        else:
          x_input = xt_random
        
        # Forward pass (same as training)
        model_output = self.forward(x_input, sigma=sigma_random, block_indices=block_indices)
        pred_tokens = model_output.argmax(dim=-1)
        
        # Compare with ground truth (EXCLUDE PAD tokens!)
        gt_tokens = x0
        # ✅ FIX: Use attention_mask from dataset (correct PAD detection!)
        # attention_mask: 1 for content, 0 for PAD
        if attention_mask is not None:
          non_pad_mask = attention_mask.bool()
        else:
          # Fallback: assume all tokens are content
          non_pad_mask = torch.ones_like(gt_tokens, dtype=torch.bool)
        if non_pad_mask.sum() > 0:
          accuracy = (pred_tokens[non_pad_mask] == gt_tokens[non_pad_mask]).float().mean().item()
          num_content_tokens = non_pad_mask.sum().item()
        else:
          accuracy = 0.0
          num_content_tokens = 0
        
        print(f"Token-level accuracy: {accuracy*100:.2f}% (on {num_content_tokens} content tokens, excluding PAD)")
        print(f"\nFirst 50 tokens comparison:")
        print(f"Ground Truth: {gt_tokens[0, :50].tolist()}")
        print(f"Prediction  : {pred_tokens[0, :50].tolist()}")
        
        # Decode to text
        if hasattr(self, 'tokenizer'):
          gt_text = self.tokenizer.decode(gt_tokens[0, :50])
          pred_text = self.tokenizer.decode(pred_tokens[0, :50])
          print(f"\nGround Truth Text: {gt_text}")
          print(f"Prediction Text  : {pred_text}") 
        
        # Check by block if HDP (also exclude PAD)
        if block_indices is not None:
          print(f"\nAccuracy by block (excluding PAD):")
          for block_id in range(3):
            block_mask = (block_indices == block_id) & non_pad_mask  # AND with non-PAD mask
            if block_mask.sum() > 0:
              block_acc = (pred_tokens[block_mask] == gt_tokens[block_mask]).float().mean().item()
              num_tokens = block_mask.sum().item()
              print(f"  Block {block_id}: {block_acc*100:.2f}% ({num_tokens} tokens)")
            else:
              print(f"  Block {block_id}: N/A (no content tokens)")
        
        print("="*80 + "\n")
    
    return losses.loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    
    # Generate sample during validation to check quality
    if not self.trainer.sanity_checking and self.T > 0:  # Check T is valid
      try:
        print(f"\n{'='*80}")
        print(f"🔍 VALIDATION SAMPLE (Epoch {self.current_epoch})")
        print(f"{'='*80}")
        
        # Ensure model is in eval mode with EMA weights (already applied in on_validation_epoch_start)
        self.eval()
        
        # Get one validation batch
        val_dataloader = self.trainer.val_dataloaders
        if val_dataloader is not None:
          batch = next(iter(val_dataloader))
          
          # Move to device
          if isinstance(batch, dict):
            input_ids = batch['input_ids'][:1].to(self.device)  # Take first sample
            block_indices = batch.get('block_indices', None)
            if block_indices is not None:
              block_indices = block_indices[:1].to(self.device)
          else:
            input_ids = batch[:1].to(self.device)
            block_indices = None
          
          # First check: Forward pass prediction (like validation step)
          with torch.no_grad():
            t = torch.randint(0, self.T, (input_ids.shape[0],), device=self.device).long()
            log_x_t = self.log_sample_categorical(self._forward_diffusion(input_ids, t))
            log_x0_pred = self.forward(log_x_t, t, block_indices=block_indices)
            pred_tokens = log_x0_pred.exp().argmax(dim=-1)
          
          print(f"\n🔍 FORWARD PASS CHECK (t={t.item()}):")
          forward_pred_text = self.tokenizer.decode(pred_tokens[0, :50], skip_special_tokens=False)
          forward_gt_text = self.tokenizer.decode(input_ids[0, :50], skip_special_tokens=False)
          print(f"   Ground Truth: {forward_gt_text}")
          print(f"   Prediction  : {forward_pred_text}")
          
          # Second check: Full sampling
          print(f"\n🔍 FULL SAMPLING CHECK:")
          with torch.no_grad():
            samples = self._sample(
              seqlen=input_ids.shape[1],
              num_steps=100,  # Quick sampling for validation
              batch_size_per_gpu=1,
              question_tokens=None
            )
          
          # _sample returns list of decoded strings, not tokens
          if isinstance(samples, list) and len(samples) > 0:
            if isinstance(samples[0], str):
              # Already decoded
              generated = samples[0]
            else:
              # Token tensor, need to decode
              generated = self.tokenizer.decode(samples[0], skip_special_tokens=False)
          else:
            generated = "[Empty sample]"
            
          ground_truth = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
          
          print(f"\n📝 GROUND TRUTH:")
          print(f"   {ground_truth[:200]}...")
          print(f"\n🤖 GENERATED:")
          print(f"   {generated[:200]}...")
          print(f"\n{'='*80}\n")
      except Exception as e:
        print(f"⚠️  Failed to generate validation sample: {e}")
    
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    block_indices = batch.get('block_indices', None)
    is_hdp = (hasattr(self.config.data, 'name') and 
              self.config.data.name == 'hdp_diffusion')
    if self.var_min and not is_hdp:
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
          # compute and record nelbo
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max,
                            block_indices=block_indices)
          losses = Loss(
            nlls=losses_clip.nlls.clone(),
            token_mask=losses_clip.token_mask,
            loss=losses_clip.loss.clone())
        elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # elbo from clipped schedule (biased estimate)
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max,
                            block_indices=block_indices)
        if len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # only report variance over 100 batches
          nlls = losses_clip.nlls
          self.metrics.valid_vars[noise_clip_start].append(
            nlls.reshape(
              nlls.shape[0], -1, self.block_size).mean(-1))
    elif self.block_size == 1 and not is_hdp:
      # nll
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1,
                          sampling_eps_max=1,
                          block_indices=block_indices)
    else:
      # nelbo
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1,
                          block_indices=block_indices)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]
  
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    return xt
  
  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    if block_size is None:
      block_size = self.block_size
  
    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_index, x)

    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

    # no need to resample for bounds 1e-3, 1
    if self.config.training.resample and \
      not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               sampling_eps_min,
                               sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      if self.config.sampling.kv_cache:
        p_x0 = self.forward(x[:, -self.block_size:],
                        sigma_t,
                        sample_mode=True).to(torch.float64)
      else:   
        p_x0 = self.forward(x,
                          sigma_t,
                          sample_mode=True).to(torch.float64)
        p_x0 = p_x0[:, -self.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0)
      # randomly and uniformly select an index in the block (among masked tokens)
      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    # compute kv cache if all tokens in a block are sampled
    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    else:
      return p_x0, x_new

  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # precompute token buffer
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros(
        (bsz, num_pred_tokens + 1),
        dtype=torch.long,
        device=self.device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, self.vocab_size))
                .to(self.device))
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        # check if we need to resample (or stop sampling for variable-length sampling)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop and self.config.sampling.var_length:
            # For variable-length sampling, use truncated output
            x = x_out
            break
          # For fixed-length sampling, continue even if stop conditions met
        x[:, i + 1] = y
      return x
  
  @torch.no_grad()
  def _sample(
      self, seqlen=None, num_steps=None, eps=1e-5, 
      batch_size_per_gpu=None, question_tokens=None
  ):
      """Generate samples from the model. 
      
      Args: 
          seqlen: Sequence length
          num_steps:  Number of denoising steps
          eps:  Minimum noise level
          batch_size_per_gpu:  Batch size per GPU
          question_tokens:  (batch, q_len) - Optional question tokens for HDP conditional generation
      
      Returns:
          List of decoded text samples
      """
      # DEBUG: Check which sampler will be used

      
      if seqlen is None:
          seqlen = self.config.model.length
      if batch_size_per_gpu is None: 
          batch_size_per_gpu = self.config.loader.eval_batch_size
      samples = []
      
      if self.parameterization == 'ar':
          for _ in range(self.config.sampling.num_sample_batches):
              sample_i, num_tries = None, 0
              while sample_i is None:
                  num_tries += 1
                  sample_i = self._ar_sampler(batch_size_per_gpu)
                  if num_tries > 10: 
                      raise ValueError('Sampling failed.')
              samples.append(sample_i)
              self.metrics.gen_nfes. append(self.config.model.length)
          samples = torch.cat(samples, dim=0) 
          return self. tokenizer.batch_decode(samples)
      
      if self.sampler == 'semi_ar':
          for _ in range(self.config.sampling. num_sample_batches):
              sample_i, num_tries = None, 0
              while sample_i is None: 
                  num_tries += 1
                  sample_i, nfes = self._semi_ar_sampler(
                      n_samples=batch_size_per_gpu,
                      num_strides=(seqlen // self.block_size), 
                      num_steps=num_steps,
                      seqlen=seqlen,
                      question_tokens=question_tokens)  # ✅ Pass question tokens
                  if num_tries > 10:
                      raise ValueError('Sampling failed.')
              samples.append(sample_i)
              self.metrics.nfes.update(nfes)
              self.metrics.gen_nfes. append(nfes)
      else:
          nfes = num_steps
          for _ in range(self.config.sampling. num_sample_batches):
              sample_i, num_tries = None, 0
              while sample_i is None: 
                  sample_i = self._analytic_sampler(
                      n_samples=batch_size_per_gpu,
                      num_steps=num_steps,
                      seqlen=seqlen,
                      eps=eps,
                      question_tokens=question_tokens  # NEW:  Pass question tokens
                  )
                  num_tries += 1
                  if num_tries > 10 and sample_i is None:
                      raise ValueError('Sampling failed.')
              samples. append(sample_i)
              self.metrics.nfes.update(nfes)
              self.metrics.gen_nfes.append(nfes)
      
      samples = torch.cat(samples, dim=0) 
      return self.tokenizer.batch_decode(samples)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None, question_tokens=None):
    """Generate samples from the model. 
    
    Args:
        num_steps: Number of denoising steps
        eps: Minimum noise level  
        seqlen: Sequence length
        question_tokens:  (batch, q_len) - Optional question tokens for HDP
    
    Returns: 
        List of decoded text samples
    """
    if self. ema:   
        self.ema.store(self._get_parameters())
        self.ema.copy_to(self._get_parameters())
    self. backbone.eval()
    self.noise. eval()
    samples = self._sample(
        seqlen=seqlen,
        batch_size_per_gpu=self.config.loader.eval_batch_size,
        num_steps=num_steps,
        eps=eps,
        question_tokens=question_tokens  # NEW: Pass question tokens
    )
    self.metrics.record_generative_perplexity(
        samples,
        self.config.model.length,
        self.config.loader.eval_batch_size,
        self.device)
    return samples

  def get_score(self, x, sigma, block_indices=None):
    """Get score with optional HDP attention mask."""
    model_output = self.forward(
        x, sigma, 
        sample_mode=True,
        block_indices=block_indices
    ).to(torch.float64)
    
    if self.config.sampling. nucleus_p == 1.0:
        return model_output. exp()
    model_output = model_output - model_output. logsumexp(-1, keepdim=True)
    model_output = self._nucleus_sample(model_output.exp())
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt, block_indices=None):
    """Analytic update step with optional HDP mask."""
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t, block_indices=block_indices)
    
    # 🔍 DEBUG: Check score for mask_index positions (first step only)
    if not hasattr(self, '_analytic_update_debug_printed'):
      self._analytic_update_debug_printed = True
      mask_positions = (x == self.mask_index)
      if mask_positions.any():
        print(f"\n🔍 [_analytic_update] FIRST STEP DEBUG:")
        print(f"   sigma_t: {sigma_t[0, 0].item():.4f}, sigma_s: {sigma_s[0, 0].item():.4f}")
        print(f"   dsigma: {dsigma[0, 0].item():.4f}, exp(dsigma): {dsigma[0, 0].exp().item():.6f}")
        print(f"   x has {mask_positions.sum().item()} mask_index tokens")
        print(f"   mask_index: {self.mask_index}")
        
        # Check score at first mask position
        first_mask_pos = mask_positions[0].nonzero()[0].item()
        score_at_mask = score[0, first_mask_pos]  # (vocab_size,)
        print(f"   Score at first mask position ({first_mask_pos}):")
        print(f"     Top 5 tokens: {score_at_mask.topk(5).indices.tolist()}")
        print(f"     Top 5 scores: {score_at_mask.topk(5).values.tolist()}")
        print(f"     Score[0]: {score_at_mask[0].item():.4f}")
        print(f"     Score[mask_index={self.mask_index}]: {score_at_mask[self.mask_index].item():.4f}")
    
    stag_score = self._staggered_score(score, dsigma)
    
    # DEBUG: Check stag_score after adding mask probability
    if not hasattr(self, '_stag_score_debug_printed'):
      self._stag_score_debug_printed = True
      mask_positions = (x == self.mask_index)
      if mask_positions.any():
        first_mask_pos = mask_positions[0].nonzero()[0].item()
        stag_at_mask = stag_score[0, first_mask_pos]
        print(f"\n🔍 [_staggered_score] AFTER applying mask probability:")
        print(f"   Top 5 tokens: {stag_at_mask.topk(5).indices.tolist()}")
        print(f"   Top 5 scores: {stag_at_mask.topk(5).values.tolist()}")
        print(f"   Score[mask_index={self.mask_index}]: {stag_at_mask[self.mask_index].item():.4f}")
        print(f"   Mask probability should be high (close to 1.0) early in sampling")
    
    probs = stag_score * self._transp_transition(x, dsigma)
    
    # Sample new tokens
    x_new = _sample_categorical(probs)
    
    # FIX: Keep non-mask tokens unchanged (like DDPM sampler)
    # Only update positions that were mask_index
    copy_flag = (x != self.mask_index).to(x.dtype)
    x_out = copy_flag * x + (1 - copy_flag) * x_new
    
    return x_out


  def _denoiser_update(self, x, t, block_indices=None):
    """Final denoising step with optional HDP mask."""
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma, block_indices=block_indices)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples


  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    
    # Debug first call
    if not hasattr(self, '_transp_debug_printed'):
      self._transp_debug_printed = True
      print(f"\n🔍 [_transp_transition] DEBUG:")
      print(f"   i.shape: {i.shape}")
      print(f"   i unique values: {torch.unique(i).tolist()}")
      print(f"   self.vocab_size: {self.vocab_size}")
      print(f"   self.mask_index: {self.mask_index}")
      print(f"   First 10 i values: {i[0, :10].tolist()}")
    
    # Create transition matrix
    # For regular tokens: edge[i] has probability at position i
    # For mask_index: allow transition to ANY token (uniform distribution)
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    
    # Fix: When i == mask_index, set edge to ones (allow all transitions)
    # Original code only added scalar, not broadcast across vocab
    mask_positions = (i == self.mask_index)
    if mask_positions.any():
      # Set edge to 1.0 for all vocab positions when input is mask_index
      edge[mask_positions] = torch.ones(
        self.vocab_size, device=edge.device, dtype=edge.dtype)
    
    return edge

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    # ← ADD CHECK:  If num_blocks is 0
    if num_blocks == 0:
        raise ValueError(
            f"num_blocks = 0!  batch_dims={batch_dims}, "
            f"block_size={block_size}, n={n}.  "
            f"This usually happens when seq_len < block_size."
        )

    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
        t_out = torch.ones_like(t)
        # ← ENSURE correct shape
        assert t_out.shape == (batch_dims[0], batch_dims[-1]), \
            f"Shape mismatch: {t_out.shape} vs expected {(batch_dims[0], batch_dims[-1])}"
        return t_out
    
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None, block_indices=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma) # used for sedd

    # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
    # (numerical imprecision computing probs under loglinear schedule)
    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    
    # ✅ HDP: Keep Question block CLEAN (no masking) since it's the input!
    # Only mask Plan (block 1) and Execution (block 2)
    if block_indices is not None:
      # Handle both 1D [seq_len] and 2D [batch, seq_len] block_indices
      if block_indices.ndim == 1:
        # Expand to [batch, seq_len]
        question_mask = (block_indices == 0).unsqueeze(0).expand_as(x0)
      else:
        question_mask = (block_indices == 0)
      xt = torch.where(question_mask, x0, xt)
    
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    model_output = self.forward(x_input, sigma=sigma, block_indices=block_indices)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(
        model_output, sigma, xt, x0)

    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta
    return loss

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None, block_indices=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)
    if self.parameterization == 'ar':
      output = self.forward(input_tokens, None)
      loss = - output. gather(
        -1, output_tokens[: , : , None])[: , :, 0]
    else:
      loss = self._forward_pass_diffusion(
        input_tokens,
        sampling_eps_min=sampling_eps_min,
        sampling_eps_max=sampling_eps_max,
        block_indices=block_indices)
    
    if self.ignore_bos and not self.training:
      attention_mask[: , 0] = 0
    
    # ============ HDP LOSS MASKING - NEW CODE ============
    # ⚠️ FIX: Include Question in loss with lower weight to prevent collapse
    # Without this, Question block outputs random tokens (often '!' or PAD)
    if self.use_hdp_attention and block_indices is not None:
      # ⚠️ CRITICAL FIX: Zero out PAD tokens completely for HDP
      # PAD tokens (self.mask_index) should NOT contribute to loss
      # This prevents model from "cheating" by learning PAD patterns
      # Use x0 (original tokens before subsampling) to identify PAD tokens
      pad_mask = (x0 != self.mask_index).float()
      
      # 🔍 DEBUG: Print loss masking info on first step
      if self.training and not hasattr(self, '_loss_mask_debug_printed'):
        self._loss_mask_debug_printed = True
        print("\n" + "="*80)
        print("🔍 [_loss] HDP LOSS MASKING DEBUG (with PAD removal)")
        print("="*80)
        print(f"block_indices.shape: {block_indices.shape}")
        print(f"attention_mask.shape: {attention_mask.shape}")
        print(f"PAD tokens (mask_index={self.mask_index})")
        print(f"  Total tokens: {x0.numel()}")
        print(f"  PAD tokens: {(x0 == self.mask_index).sum().item()}")
        print(f"  Content tokens: {(x0 != self.mask_index).sum().item()}")
        print(f"\nActive tokens by block (AFTER removing PAD):")
        for block_id in range(3):
          block_mask = (block_indices == block_id).float()
          active_before = (attention_mask * block_mask).sum().item()
          active_after = (attention_mask * pad_mask * block_mask).sum().item()
          total_tokens = block_mask.sum().item()
          print(f"  Block {block_id}: {active_after:.0f}/{total_tokens:.0f} tokens (was {active_before:.0f} before PAD removal)")
        print("="*80 + "\n")
      
      # block_indices: 0=Question, 1=Plan, 2=Execution
      # Use weighted loss: Question=0.0x (no loss!), Plan=1x, Exec=1x
      # ✅ Question is INPUT - model should NOT learn to generate it!
      loss_weights = torch.ones_like(block_indices, dtype=torch.float32)
      loss_weights[block_indices == 0] = 0.0  # NO loss for Question (it's input!)
      loss_weights[block_indices == 1] = 1.0  # Full weight for Plan
      loss_weights[block_indices == 2] = 1.0  # Full weight for Execution
      
      # Combine: attention_mask * pad_mask * loss_weights
      # This ensures PAD tokens have ZERO contribution to loss
      attention_mask = attention_mask * pad_mask * loss_weights
      
      # Optional: Log separate losses for debugging
      if self.training and hasattr(self, 'global_step') and self.global_step % 100 == 0:
        with torch.no_grad():
          q_mask = (block_indices == 0).float() * attention_mask
          plan_mask = (block_indices == 1).float() * attention_mask
          exec_mask = (block_indices == 2).float() * attention_mask
          if q_mask.sum() > 0:
            q_loss = (loss * q_mask).sum() / q_mask.sum()
            self.log('trainer/question_loss', q_loss.item(), on_step=True, on_epoch=False)
          if plan_mask.sum() > 0:
            plan_loss = (loss * plan_mask).sum() / plan_mask.sum()
            self.log('trainer/plan_loss', plan_loss.item(), on_step=True, on_epoch=False)
          if exec_mask.sum() > 0:
            exec_loss = (loss * exec_mask).sum() / exec_mask.sum()
            self.log('trainer/exec_loss', exec_loss.item(), on_step=True, on_epoch=False)
    # ============ END HDP LOSS MASKING ============
       
    nlls = (loss * attention_mask)
    token_nll = nlls. sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def _analytic_sampler(
      self, n_samples, num_steps, seqlen, eps=1e-5,
      question_tokens=None
  ): 
      """
      Analytic sampler with HDP support.
      
      Args:
          n_samples: Number of samples to generate
          num_steps: Number of denoising steps
          seqlen:  Sequence length
          eps: Minimum noise level
          question_tokens: (batch, q_len) - Optional fixed question tokens for HDP
      
      Returns:
          Generated samples tensor
      """
      x = self._sample_prior(n_samples, seqlen).to(self.device)
      print(f"🔍 [_analytic_sampler] Starting sampling")
      print(f"{'='*70}")
      print(f"use_hdp_attention: {self.use_hdp_attention}")
      print(f"seqlen: {seqlen}, n_samples: {n_samples}")
      print(f"has hdp_block_sizes attr: {hasattr(self, 'hdp_block_sizes')}")
      if hasattr(self, 'hdp_block_sizes'):
          print(f"hdp_block_sizes value: {self.hdp_block_sizes}")
          print(f"hdp_block_sizes is None: {self.hdp_block_sizes is None}")

      # Ensure hdp_block_sizes exists and matches seqlen
      if self.use_hdp_attention:
        if not hasattr(self, 'hdp_block_sizes') or self.hdp_block_sizes is None:
          print(f"⚠️  WARNING: use_hdp_attention=True but hdp_block_sizes not set!")
          print(f"   Attempting to reconstruct from config...")
          
          if hasattr(self.config, 'data') and hasattr(self.config.data, 'hdp'):
              self.hdp_block_sizes = (
                  self.config.data.hdp.question_len,
                  self.config.data.hdp.plan_len,
                  self.config.data.hdp.exec_len
              )
          else:
              self.use_hdp_attention = False
        
        # Validate block sizes match seqlen
        if self.use_hdp_attention and self.hdp_block_sizes is not None:
          expected_len = sum(self.hdp_block_sizes)
          if expected_len != seqlen:
            if seqlen < 50:  # Too short for HDP
              self.use_hdp_attention = False
            else:
              q_len, p_len, e_len = self.hdp_block_sizes
              ratio = seqlen / expected_len
              q_len = max(1, int(q_len * ratio))
              p_len = max(1, int(p_len * ratio))
              e_len = max(1, seqlen - q_len - p_len)
              self.hdp_block_sizes = (q_len, p_len, e_len)

      # HDP-aware initialization
      block_indices = None
      q_len = 0
      
      if self.use_hdp_attention and self.hdp_block_sizes is not None:
          q_len, p_len, e_len = self.hdp_block_sizes
          
          # Create block_indices for HDP attention mask
          block_indices = torch.cat([
              torch.zeros(q_len, dtype=torch.long, device=self.device),
              torch.ones(p_len, dtype=torch.long, device=self.device),
              torch.full((e_len,), 2, dtype=torch.long, device=self.device)
          ]).unsqueeze(0).repeat(n_samples, 1)
          
          # If question_tokens provided, use them as fixed context
          if question_tokens is not None: 
              # Ensure correct batch size
              if question_tokens.shape[0] == 1 and n_samples > 1:
                  question_tokens = question_tokens.repeat(n_samples, 1)
              
              # Pad/truncate question to q_len
              if question_tokens.shape[1] < q_len:
                  pad_len = q_len - question_tokens.shape[1]
                  pad_token = self.tokenizer.pad_token_id
                  if pad_token is None:
                      pad_token = self.tokenizer.eos_token_id
                  question_tokens = F.pad(
                      question_tokens, (0, pad_len), 
                      value=pad_token
                  )
              else:
                  question_tokens = question_tokens[:, :q_len]
              
              # Set question tokens (they will be kept fixed)
              x[:, :q_len] = question_tokens
          else:
              # No question provided - set BOS at start
              x[:, 0] = self.tokenizer.bos_token_id
      else:
          # Standard (non-HDP) sampling
          x[:, 0] = self.tokenizer.bos_token_id
      
      timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
      dt = (1 - eps) / num_steps
      
      # ============================================================
      # SEQUENTIAL DENOISING: DISABLED for now (train/test mismatch)
      # TODO: Implement sequential training before enabling this
      # ============================================================
      USE_SEQUENTIAL_DENOISING = False  # Set to True when sequential training ready
      
      if USE_SEQUENTIAL_DENOISING and self.use_hdp_attention and block_indices is not None:
          # [Sequential denoising code remains but disabled]
          phase1_steps = num_steps // 2
          phase2_steps = num_steps - phase1_steps
          
          q_len, p_len, e_len = self.hdp_block_sizes
          plan_start = q_len
          plan_end = q_len + p_len
          exec_start = plan_end
          
          has_question = (question_tokens is not None and q_len > 0)
          
          if has_question:
              print(f"\n📊 HDP Sequential Denoising (with Question input):")
              print(f"   Phase 1 (steps 0-{phase1_steps}): Denoise PLAN only (Question frozen)")
              print(f"   Phase 2 (steps {phase1_steps}-{num_steps}): Denoise EXEC only (Q+Plan frozen)")
          else:
              print(f"\n📊 HDP Sequential Denoising (no Question input):")
              print(f"   Phase 1 (steps 0-{phase1_steps}): Denoise QUESTION + PLAN")
              print(f"   Phase 2 (steps {phase1_steps}-{num_steps}): Denoise EXEC (Q+Plan frozen)")
          
          x_exec_frozen = x[:, exec_start:].clone()
          
          if has_question:
              x_question_frozen = question_tokens.clone()
          
          phase1_desc = 'Phase 1: Plan' if has_question else 'Phase 1: Q+Plan'
          print(f"\n🔵 {phase1_desc}...")
          for i in tqdm(range(phase1_steps), desc=phase1_desc):
              t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
              x_updated = self._analytic_update(x=x, t=t, dt=dt, block_indices=block_indices)
              
              if has_question:
                  x_updated[:, :q_len] = x_question_frozen
              
              x_updated[:, exec_start:] = x_exec_frozen
              x = x_updated
          
          x_question_plan_clean = x[:, :exec_start].clone()
          x_plan_clean = x[:, plan_start:plan_end].clone()
          print(f"✅ Phase 1 complete: Q+Plan denoised, Exec still at noise")
          
          print(f"\n🟢 Phase 2: Denoising Execution (conditioned on clean Q+Plan)...")
          for i in tqdm(range(phase2_steps), desc='Phase 2: Exec'):
              step_idx = phase1_steps + i
              t = timesteps[step_idx] * torch.ones(x.shape[0], 1, device=self.device)
              x_updated = self._analytic_update(x=x, t=t, dt=dt, block_indices=block_indices)
              x_updated[:, :exec_start] = x_question_plan_clean
              x = x_updated
          
          print(f"✅ Phase 2 complete: Exec denoised (conditioned on clean Q+Plan)")
          
      else:
          # ============================================================
          # PARALLEL DENOISING (Standard - currently active)
          # All blocks denoised simultaneously with attention mask
          # ============================================================
          for i in tqdm(range(num_steps), desc='HDP Sampling' if block_indices is not None else 'Sampling'):
              t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

              x = self._analytic_update(x=x, t=t, dt=dt, block_indices=block_indices)
              
              # 🔍 DEBUG: Check Plan/Exec token distribution
              if i % 200 == 0 and block_indices is not None:  # Every 200 steps
                  q_len, p_len, e_len = self.hdp_block_sizes
                  plan_tokens = x[0, q_len:q_len+p_len]
                  exec_tokens = x[0, q_len+p_len:]
                  plan_mask_count = (plan_tokens == self.mask_index).sum().item()
                  exec_mask_count = (exec_tokens == self.mask_index).sum().item()
                  
                  # Count most frequent tokens
                  unique_plan, plan_counts = torch.unique(plan_tokens, return_counts=True)
                  unique_exec, exec_counts = torch.unique(exec_tokens, return_counts=True)
                  top_plan_token = unique_plan[plan_counts.argmax()].item()
                  top_exec_token = unique_exec[exec_counts.argmax()].item()
                  
                  print(f"\n   Step {i}:")
                  print(f"     Plan: {plan_mask_count}/{p_len} mask_index, most common token: {top_plan_token} ({plan_counts.max().item()} times)")
                  print(f"     Exec: {exec_mask_count}/{e_len} mask_index, most common token: {top_exec_token} ({exec_counts.max().item()} times)")
                  if i == 0:
                      print(f"     First 10 Plan tokens: {plan_tokens[:10].tolist()}")
                      print(f"     First 10 Exec tokens: {exec_tokens[:10].tolist()}")
              
              # Keep question tokens fixed if provided
              if self.use_hdp_attention and question_tokens is not None:
                  q_len = self.hdp_block_sizes[0]
                  x[:, :q_len] = question_tokens
      
      # Final denoising step
      t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
      x = self._denoiser_update(x=x, t=t, block_indices=block_indices)
      
      # Final: preserve Question if provided
      if self.use_hdp_attention and question_tokens is not None:
          q_len = self.hdp_block_sizes[0]
          x[:, :q_len] = question_tokens
      
      stop, x = self._check_stop_conds(x)
      if stop:
          return None
      return x

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, num_steps, num_strides, seqlen, context_size=1024, question_tokens=None):
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0
          
    mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
    if mdlm_semi_ar:
      # sliding window of length 512 for mdlm semi-ar decoding
      num_strides = self.config.model.length // 512
      num_strides -= 1

    ones = torch.ones((n_samples,1), dtype=self.dtype,
                      device=self.device)
    
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

    for stride_num in tqdm(range(num_strides)):
      # sample next block
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum[:, 0] = self.tokenizer.bos_token_id
      else:
        if mdlm_semi_ar:
          x = self._sample_prior(n_samples, 512).to(self.device)
        else:
          x = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum = torch.cat((x_accum, x), dim=1)

      # compute logits in a sliding window (context passed to model can't exceed context_size)
      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      fwd_idx = torch.arange(start_idx, end_idx)
      if mdlm_semi_ar and stride_num > 0: # MDLM
        fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        # faster (equivalent) sampler from zheng et al (2025)
        if self.config.sampling.first_hitting:
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          t *= u**(1 / num_masked)
              
        elif not self.config.sampling.first_hitting:
          t = timesteps[i]

        p_x0_cache, x_next = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,)
        if p_x0_cache is None:
          sampling_steps += 1
       
        x_accum[:, fwd_idx] = x_next

      # check if we need to resample (or stop sampling for variable-length sampling)
      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(x_accum)
        if stop and self.config.sampling.var_length:
          # For variable-length sampling, break when stop conditions met
          break
        # For fixed-length sampling, continue even if stop conditions met
    return x_accum, sampling_steps
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    # ⚠️ DISABLED for early checkpoints - entropy threshold too strict
    # entropy = self._compute_entropy(x[:, -256:])
    # if entropy < 4:
    #   stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x

  # ============ HDP SAMPLING HELPERS ============

  def sample_hdp_conditional(
    self, 
    question_text: str,
    num_steps: int = None,
    eps: float = 1e-5
  ) -> str:
    """
    Convenience method: Generate Plan + Execution given a question text.
    
    Args:
        question_text: The question/problem to solve
        num_steps: Number of denoising steps (default: config.algo.T)
        eps: Minimum noise level
    
    Returns: 
        Generated text including Question, Plan, and Execution
    
    Example:
        >>> output = model.sample_hdp_conditional(
        ...     "Janet's ducks lay 16 eggs per day.  She eats 3 for breakfast..."
        ... )
        >>> print(output)
    """
    if not self.use_hdp_attention: 
        raise ValueError("HDP attention not enabled. Set config.data.hdp.use_hdp_attention=True")
    
    if num_steps is None:
        num_steps = self.config.algo.T
    
    # Tokenize question
    question_tokens = self.tokenizer(
        question_text,
        return_tensors='pt',
        add_special_tokens=False,
        truncation=True,
        max_length=self.hdp_block_sizes[0]  # q_len
    )['input_ids'].to(self.device)
    
    # Generate
    if self.ema:
        self.ema.store(self._get_parameters())
        self.ema.copy_to(self._get_parameters())
    
    self.backbone.eval()
    self.noise.eval()
    
    samples = self._sample(
        seqlen=self.config.model.length,
        batch_size_per_gpu=1,
        num_steps=num_steps,
        eps=eps,
        question_tokens=question_tokens
    )
    
    return samples[0] if samples else None

  def sample_hdp_batch(
      self,
      questions: list,
      num_steps: int = None,
      eps: float = 1e-5
  ) -> list:
    """
    Generate Plan + Execution for a batch of questions. 
    
    Args: 
        questions: List of question strings
        num_steps: Number of denoising steps
        eps: Minimum noise level
    
    Returns:
        List of generated texts
    """
    if not self.use_hdp_attention:
        raise ValueError("HDP attention not enabled.")
    
    if num_steps is None:
        num_steps = self.config.algo.T
    
    q_len = self.hdp_block_sizes[0]
    
    # Tokenize all questions
    tokenized = self.tokenizer(
        questions,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=q_len
    )['input_ids'].to(self.device)
    
    # Generate
    if self.ema:
        self.ema.store(self._get_parameters())
        self.ema.copy_to(self._get_parameters())
    
    self.backbone.eval()
    self.noise.eval()
    
    samples = self._sample(
        seqlen=self.config.model.length,
        batch_size_per_gpu=len(questions),
        num_steps=num_steps,
        eps=eps,
        question_tokens=tokenized
    )
    
    return samples
