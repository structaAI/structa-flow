import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any
import math

class FlashAttention(nn.Module):
  def __init__(self, config: Any)-> None:
    super().__init__()

    self.num_heads: int = config.num_heads
    self.emb_dim: int = config.emb_dim
    self.dropout: float = config.dropout

    self.causal_attention: nn.Linear = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=False)
    self.causal_projection: nn.Linear = nn.Linear(config.emb_dim, config.emb_dim, bias=False)

  def forward(self, x: torch.Tensor)-> Tensor:
    B, T, C = x.shape

    qkv: Tensor = self.causal_attention(x)
    q, k, v = qkv.split(self.emb_dim)

    q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

    with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
      y: Tensor = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True
      )
    
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.causal_projection(y)