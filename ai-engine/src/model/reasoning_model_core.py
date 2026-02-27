import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Tuple

from .transformer import TransformerBlock
from .rms_norm import RMSNorm
from .adaptive_bridge import AdaptiveThinkingGate


class ReasoningModelCore(nn.Module):
  def __init__(self, config: Any) -> None:
    super().__init__()

    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
    self.embed_dropout = nn.Dropout(p=config.dropout)
    self.layers = nn.ModuleList(
      [TransformerBlock(config) for _ in range(config.num_layers)]
    )
    self.norm_f = RMSNorm(config.emb_dim, config.norm_eps)

    self.output = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    self.output.weight = self.token_embedding.weight

    self.think_gate = AdaptiveThinkingGate(config.emb_dim)

    self.last_think_prob: Optional[Tensor] = None

    self._init_weights()

  def _init_weights(self) -> None:
    std = self.config.initializer_range
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)

  def forward(self, idx: Tensor) -> Tensor:
    x: Tensor = self.embed_dropout(self.token_embedding(idx))
    _should_think, think_prob = self.think_gate(x)
    self.last_think_prob = think_prob          # detach before logging

    for layer in self.layers:
      x = layer(x)

    x = self.norm_f(x)
    logits: Tensor = self.output(x)
    return logits

  def num_parameters(self, trainable_only: bool = True) -> int:
    params = (
      self.parameters() if not trainable_only
      else filter(lambda p: p.requires_grad, self.parameters())
    )
    return sum(p.numel() for p in params)