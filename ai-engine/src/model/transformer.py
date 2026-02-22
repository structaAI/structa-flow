import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any

from .rms_norm import RMSNorm
from .flash_attention import FlashAttention

class TransformerBlock(nn.Module):
  def __init__(self, config: Any)-> None:
    super().__init__()

    self.emb_dim: int = config.emb_dim
    self.eps: float = config.eps
    self.num_heads: int = config.num_heads
    self.dropout: float = config.dropout if self.training else 0.0

    self.norm1: RMSNorm = RMSNorm(config.emb_dim, config.eps)
    self.attention: FlashAttention = FlashAttention(config)
    self.norm2: RMSNorm = RMSNorm(self.emb_dim, self.eps)

    self.ffn: nn.Module = nn.Sequential(
      nn.Linear(self.emb_dim, self.emb_dim * 4, bias=False),
      nn.SiLU(),
      nn.Linear(self.emb_dim * 4, self.emb_dim, bias=False)
    )

  def forward(self, x: Tensor)-> Tensor:
    x = x + self.attention(self.norm1(x))
    x = x + self.ffn(self.norm2(x))
    
    return x
