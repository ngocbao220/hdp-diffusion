import math
import typing

import einops
from einops import rearrange
from functools import partial
try:
  import flash_attn
  import flash_attn.layers.rotary
  FLASH_ATTN_AVAILABLE = True
except:
  FLASH_ATTN_AVAILABLE = False

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except:
  FLEX_ATTN_AVAILABLE = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
  """
  Constructs the specialized block diffusion attention mask.
  """
  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

  # 1. Block Diagonal Mask (M_BD)
  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
  # 2. Offset Block-Causal Mask (M_OBC)
  offset_block_causal = ((block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0))
  # 3. Block-Causal Mask (M_BC)
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

  return block_diagonal | offset_block_causal | block_causal

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(x, bias, scale, residual, prob, training)
  return _bias_dropout_add


def modulate(x, shift, scale):
  return x * (1 + scale) + shift

# @torch.jit.script
def bias_dropout_add_scale_fused_train(x, bias, scale, residual, prob: float):
  return bias_dropout_add_scale(x, bias, scale, residual, prob, True)

# @torch.jit.script
def bias_dropout_add_scale_fused_inference(x, bias, scale, residual, prob: float):
  return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

# @torch.jit.script
def modulate_fused(x, shift, scale):
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v

# =========================================================================
# 🚑 FIX: ROBUST ROPE IMPLEMENTATION (Broadcasting & Dim Mismatch Fix)
# =========================================================================
def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    # qkv shape: [Batch, Seq, 3, Heads, HeadDim] (5D)
    # cos shape: [Batch, Seq, 3, Heads, HeadDim] or [Batch, Seq, HeadDim] or [Seq, HeadDim]
    
    # 1. Fix Dimension Mismatch (e.g., cos=16, qkv=32)
    if cos.shape[-1] != qkv.shape[-1]:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

    # 2. Fix Broadcasting (Ensure cos/sin matches 5D qkv)
    if qkv.ndim == 5: 
        # Nếu cos là [Seq, Dim] -> [1, Seq, 1, 1, Dim]
        if cos.ndim == 2:
            cos = cos[None, :, None, None, :]
            sin = sin[None, :, None, None, :]
        # Nếu cos là [Batch, Seq, Dim] -> [Batch, Seq, 1, 1, Dim]
        elif cos.ndim == 3:
            cos = cos[:, :, None, None, :]
            sin = sin[:, :, None, None, :]
        # Nếu cos là [Batch, Seq, 3, 1, Dim] -> OK
        
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (qkv * cos) + (rotate_half(qkv) * sin)

def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  try:
    if not FLASH_ATTN_AVAILABLE: raise NameError
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
  except (NameError, AttributeError):
    return apply_rotary_pos_emb_torchscript(qkv, cos, sin)
# =========================================================================

class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]

# ... (Giữ nguyên các class TimestepEmbedder, LabelEmbedder, DDiTBlockCausal) ...
# Để tiết kiệm không gian, tôi chỉ liệt kê những phần thay đổi quan trọng ở dưới.
# Bạn hãy giữ nguyên code EmbeddingLayers của bạn.

class TimestepEmbedder(nn.Module):
  # ... (Giữ nguyên code cũ) ...
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half).to(t.dtype).to(t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb

class LabelEmbedder(nn.Module):
  # ... (Giữ nguyên code cũ) ...
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes
  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings

# ------------------------------------------------------------------------
# DDiTBlock: Sửa logic chọn backend attention
# ------------------------------------------------------------------------
class DDiTBlock(nn.Module):
  def __init__(self, n, dim, n_heads, adaLN,
               latent_dim=None, cond_dim=None,
               latent_conditioning=-1, mlp_ratio=4,
               dropout=0.1, block_size=1,
               max_batch_size=64, max_seqlen=1024, attn_backend='flash_attn'):
    super().__init__()
    self.max_seqlen = max_seqlen
    self.n = n
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.latent_conditioning = latent_conditioning
    self.block_size = block_size

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.kv_cache = None
    self.cache_idx = 0

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    
    # FIX: Tự động chuyển về SDPA nếu không có flash_attn
    if attn_backend == 'flash_attn' and not FLASH_ATTN_AVAILABLE:
        attn_backend = 'sdpa'
    self.attn_backend = attn_backend

  def _get_bias_dropout_scale(self):
    if self.training: return bias_dropout_add_scale_fused_train
    else: return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x)
      self.kv_cache[:, self.cache_idx:self.cache_idx+self.block_size] = new_qkv
      qkv = self.kv_cache[:, :self.cache_idx+self.block_size].clone()
    else:
      qkv = self.attn_qkv(x)
    if store_kv:
      self.cache_idx += self.block_size
      if self.cache_idx >= self.max_seqlen:
        self.cache_idx = self.max_seqlen - self.block_size
        self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

    qkv = einops.rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    
    # RoPE call
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn' and FLASH_ATTN_AVAILABLE:
        qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv
  
  def attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    
    # Handle mask: Nếu mask là Tensor -> Boolean hoặc Float bias
    # Nếu dùng SDPA, nó hỗ trợ cả boolean và additive mask
    if mask is not None:
        if mask.dtype == torch.bool:
             pass # SDPA nhận tốt
        elif mask.dtype in [torch.float16, torch.float32, torch.bfloat16]:
             pass # SDPA nhận tốt additive bias
    
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=False, # Mask HDP đã tự lo causal
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def cross_attn_flex(self, qkv, mask=None):
    qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
    x = fused_flex_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
    x = rearrange(x, 'b h s d -> b s (h d)')
    return x

  def forward(self, x, rotary_cos_sin, c, causal=False, mask=None, sample_mode=False, store_kv=False):
    batch_size, seq_len = x.shape[0], x.shape[1]

    # ... (Modulation logic giữ nguyên) ...
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size).chunk(6, dim=-1)

    x_skip = x
    if c is not None: x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else: x = self.norm1(x)

    # get qkvs
    if mask is not None and not sample_mode:
        # Training with cross-attention:  split [xt | x0]
        qkv_x = self.get_qkv(x[:, :self.n], rotary_cos_sin)
        qkv_x0 = self. get_qkv(x[: , self.n:], rotary_cos_sin)
        qkv = torch.cat((qkv_x, qkv_x0), dim=1)
    elif sample_mode and mask is not None: 
        # ✅ NEW: Sampling with HDP mask - treat as single sequence
        qkv = self. get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    else:
        # Standard (no mask or inference without cross-attn)
        qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
      
    # attention
    # FIX: Nếu có mask (HDP) thì KHÔNG DÙNG flash_attn mặc định (vì nó ko hỗ trợ custom mask)
    # Ta ép nó dùng 'sdpa' nếu mask != None
    backend_to_use = self.attn_backend
    if mask is not None and backend_to_use == 'flash_attn':
        backend_to_use = 'sdpa'

    if backend_to_use == 'flash_attn' and mask is None:
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0., causal=causal)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)     
    elif backend_to_use == 'flex' and FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif backend_to_use == 'sdpa':
      x = self.cross_attn(qkv, mask=mask)
    else:
      # Fallback cuối cùng cho trường hợp config flash_attn mà mask != None
      x = self.cross_attn(qkv, mask=mask)

    if self.kv_cache is not None:
      x = x[:, -self.block_size:]
    x = self.attn_mlp(x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip)
    return x

# ... (Class EmbeddingLayer và DDiTFinalLayer giữ nguyên) ...
class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
  def forward(self, x):
    return self.embedding[x]

class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, adaLN, tie_word_embeddings=False):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.tie_word_embeddings = tie_word_embeddings
  def forward(self, x, c):
    x = self.norm_final(x)
    if c is not None:
      if c.shape[0] == x.shape[0]: shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      else: shift, scale = rearrange(self.adaLN_modulation(c), '(b h) d -> b h d', b=x.shape[0]).chunk(2, dim=-1)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x

# ------------------------------------------------------------------------
# DIT: Main Backbone
# ------------------------------------------------------------------------
class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict: config = omegaconf.OmegaConf.create(config)
    self.causal = getattr(config.model, 'causal_attention', config.algo.parameterization == 'ar')
    self.n = config.model.length
    self.adaLN = not self.causal or getattr(config.model, 'adaln', False)
    self.config = config
    self.vocab_size = vocab_size
    self.block_size = config.block_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.n_heads = config.model.n_heads
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if self.adaLN == True: self.sigma_map = TimestepEmbedder(cond_dim)
    if not self.causal: self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)
    
    # FIX: Tự động dùng sdpa nếu không có flash_attn
    raw_backend = getattr(config.model, 'attn_backend', 'flash_attn')
    if raw_backend == 'flash_attn' and not FLASH_ATTN_AVAILABLE:
        raw_backend = 'sdpa'
    self.attn_backend = raw_backend
    
    self.max_seqlen = 1024

    blocks = []
    for _ in range(config.model.n_blocks):
      # Lưu ý: Code gốc của bạn có DDiTBlockCausal nhưng trong HDP thường dùng DDiTBlock (non-causal)
      # Tôi giữ nguyên logic lựa chọn block của bạn
      if self.causal:
        block = DDiTBlockCausal(
          n=config.model.length, dim=dim, n_heads=config.model.n_heads,
          dropout=config.model.dropout, max_batch_size=config.loader.eval_batch_size,
          adaLN=self.adaLN, cond_dim=cond_dim, attn_backend=self.attn_backend)
      else:
        block = DDiTBlock(
          n=config.model.length, dim=dim, n_heads=config.model.n_heads,
          cond_dim=cond_dim, adaLN=self.adaLN, dropout=config.model.dropout,
          block_size=self.block_size, attn_backend=self.attn_backend,
          max_seqlen=self.max_seqlen)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim, out_channels=vocab_size, cond_dim=cond_dim,
      adaLN=self.adaLN, tie_word_embeddings=config.model.tie_word_embeddings)
    if config.algo.cross_attn:
      self.gen_mask(config.model.length, self.block_size, self.attn_backend)

  def _get_bias_dropout_scale(self):
    if self.training: return bias_dropout_add_scale_fused_train
    else: return bias_dropout_add_scale_fused_inference
    
  def gen_mask(self, seqlen, block_size, attn_backend='sdpa', hierarchical_config=None):
    if hierarchical_config is not None:
      from models.hierarchical_mask import create_hierarchical_mask
      self.block_diff_mask = create_hierarchical_mask(
        seqlen=seqlen, block_size=block_size,
        question_len=hierarchical_config.get('question_len', 256),
        plan_len=hierarchical_config.get('plan_len', 256),
        exec_len=hierarchical_config.get('exec_len', 512),
        attn_backend=attn_backend)
      self.hierarchical_config = hierarchical_config
    elif attn_backend == 'flash_attn':
      pass
    elif attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      self.block_diff_mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=seqlen),
        B=None, H=None, Q_LEN=seqlen*2, KV_LEN=seqlen*2)
    elif attn_backend == 'sdpa':
      self.block_diff_mask = block_diff_mask(
        b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
        kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)
    else:
      # Fallback SDPA
      self.block_diff_mask = block_diff_mask(
        b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
        kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)
    
  def reset_kv_cache(self):
    for block in self.blocks:
      block.kv_cache = torch.zeros(
        self.config.loader.eval_batch_size, self.max_seqlen,
        self.config.model.hidden_size * 3, device='cuda', dtype=torch.bfloat16)
      block.cache_idx = 0

  def forward(self, indices, sigma, sample_mode=False, store_kv=False, hdp_mask=None):
    x = self.vocab_embed(indices)
    if sigma is None: t_cond = None
    else: t_cond = F.silu(self.sigma_map(sigma))

    # ✅ FIX: Set self.n BEFORE mask logic
    if hasattr(self, 'block_diff_mask') or hdp_mask is not None: 
        # For cross-attention, x is [xt | x0], so n = len(xt)
        self.n = indices.shape[1] // 2 if not sample_mode else indices.shape[1]
    else:
        self.n = indices.shape[1]

    if sample_mode and hdp_mask is not None: 
        print(f"\n🔍 DiT Backbone receiving HDP mask:")
        print(f"   hdp_mask.shape: {hdp_mask.shape}")
        print(f"   hdp_mask device: {hdp_mask.device}")
        print(f"   Masked positions: {(hdp_mask == float('-inf')).sum()}")
        
    if hdp_mask is not None:
      cross_attn = True
      mask = hdp_mask
      # FIX: In sampling mode, x is single sequence, not [xt | x0] concatenated
      if sample_mode:
        self.n = x.shape[1]
        rotary_cos_sin = self.rotary_emb(x)
      else:
        # Training mode: x is [xt | x0] concatenated
        self.n = x.shape[1] // 2
        rotary_cos_sin = self.rotary_emb(x[:, :self.n])
    else:
      cross_attn = hasattr(self, 'block_diff_mask')
      if cross_attn:
        mask = self.block_diff_mask
        if sample_mode:
          if self.config.sampling.kv_cache:
            mask = None
            accum_length = self.blocks[0].cache_idx + self.block_size
            x_full = torch.zeros((x.shape[0], accum_length, x.shape[2]), device=x.device)
            rotary_cos_sin = self.rotary_emb(x_full)
          else:
            mask = mask[self.n:self.n+x.shape[1], self.n:self.n+x.shape[1]]
            rotary_cos_sin = self.rotary_emb(x)
        else:
          rotary_cos_sin = self.rotary_emb(x[:, :self.n])
      else:
        rotary_cos_sin = self.rotary_emb(x)
        mask = None

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x, rotary_cos_sin, c=t_cond, causal=self.causal,
          sample_mode=sample_mode, mask=mask, store_kv=store_kv)
      x = self.output_layer(x, t_cond)
      
    if cross_attn and not sample_mode:
      x = x[:, :self.n]
    return x