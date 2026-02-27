import torch
import torch.nn as nn
from torch import Tensor
from typing import Any

from .rms_norm import RMSNorm
from .flash_attention import GroupedQueryAttention
from .ffn import SwiGLUFeedForward


class TransformerBlock(nn.Module):
  def __init__(self, config: Any) -> None:
    super().__init__()

    self.norm1 = RMSNorm(config.emb_dim, config.norm_eps)
    self.attention = GroupedQueryAttention(config)

    self.norm2 = RMSNorm(config.emb_dim, config.norm_eps)
    self.ffn = SwiGLUFeedForward(config)

    # Residual dropout applied after each sub-layer output
    self.resid_dropout = nn.Dropout(p=config.dropout)

  def forward(self, x: Tensor) -> Tensor:
    x = x + self.resid_dropout(self.attention(self.norm1(x)))
    x = x + self.resid_dropout(self.ffn(self.norm2(x)))
    return x