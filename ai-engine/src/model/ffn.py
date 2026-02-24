import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.ModelConfig import ModelConfig


class SwiGLUFeedForward(nn.Module):
  def __init__(self, config: ModelConfig) -> None:
    super().__init__()
    self.gate_proj = nn.Linear(config.emb_dim, config.ffn_hidden_size, bias=False)
    self.up_proj   = nn.Linear(config.emb_dim, config.ffn_hidden_size, bias=False)
    self.down_proj = nn.Linear(config.ffn_hidden_size, config.emb_dim, bias=False)

  def forward(self, x: Tensor) -> Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))