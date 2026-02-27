import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional

from ..utils.tokenizer.rope import RotaryPositionalEmbedding

class GroupedQueryAttention(nn.Module):
  def __init__(self, config: Any) -> None:
    super().__init__()

    self.num_heads: int = config.num_heads
    self.kv_heads: int = config.kv_heads          
    self.head_dim: int = config.head_dim
    self.emb_dim: int = config.emb_dim
    self.dropout: float = config.dropout
    self.groups: int = self.num_heads // self.kv_heads

    self.q_proj = nn.Linear(self.emb_dim, self.num_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(self.emb_dim, self.kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(self.emb_dim, self.kv_heads * self.head_dim, bias=False)
    self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    self.rope: Optional[RotaryPositionalEmbedding] = (
      RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
      if config.use_rotary_embeddings
      else None
    )

  def _expand_kv(self, kv: Tensor) -> Tensor:
    # kv: (B, kv_heads, T, head_dim)
    B, _, T, D = kv.shape
    kv = kv.unsqueeze(2).expand(B, self.kv_heads, self.groups, T, D)
    return kv.reshape(B, self.num_heads, T, D)

  def forward(self, x: Tensor) -> Tensor:
    B, T, _ = x.shape

    q: Tensor = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k: Tensor = self.k_proj(x).view(B, T, self.kv_heads,  self.head_dim).transpose(1, 2)
    v: Tensor = self.v_proj(x).view(B, T, self.kv_heads,  self.head_dim).transpose(1, 2)

    if self.rope is not None:
      q, k = self.rope(q, k)

    k = self._expand_kv(k)
    v = self._expand_kv(v)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
      y: Tensor = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,
      )

    y = y.transpose(1, 2).contiguous().view(B, T, self.emb_dim)
    return self.out_proj(y)