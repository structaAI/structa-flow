import torch
import torch.nn as nn
from torch import Tensor
from typing import Any

from .transformer import TransformerBlock
from .rms_norm import RMSNorm
from .adaptive_bridge import AdaptiveThinkingGate

class ReasoningModelCore(nn.Module):
  def __init__(self, config: Any)-> None:
    super().__init__()

    self.config: Any = config
    self.num_heads: int = self.config.num_heads
    self.emb_dim: int = self.config.emb_dim
    self.vocab_size: int = self.config.vocab_size
    self.num_layers: int = self.config.num_layers

    self.token_embedding: nn.Embedding = nn.Embedding(self.vocab_size, self.emb_dim)
    self.layers: nn.ModuleList = nn.ModuleList([TransformerBlock(config) for _ in range(self.num_layers)])
    self.norm_f: RMSNorm = RMSNorm(self.emb_dim)
    self.output: nn.Linear = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    self.think_gate: AdaptiveThinkingGate = AdaptiveThinkingGate(self.emb_dim)
  
  def forward(self, idx: Tensor)-> Tensor:
    x: Tensor = self.token_embedding(idx)

    for layer in self.layers:
      x = layer(x)
    
    x = self.norm_f(x)
    logits: Tensor = self.output(x)

    return logits