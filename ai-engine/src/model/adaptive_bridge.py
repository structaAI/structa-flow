import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class AdaptiveThinkingGate(nn.Module):
  def __init__(self, emb_dim: int) -> None:
    super().__init__()

    self.gate = nn.Sequential(
      nn.Linear(emb_dim, emb_dim // 4),
      nn.GELU(),
      nn.Linear(emb_dim // 4, 1),
      nn.Sigmoid(),
    )

  def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
    # Mean-pool over the sequence dimension → (B, emb_dim)
    pooled: Tensor = hidden_states.mean(dim=1)

    # (B, 1)
    thinking_prob: Tensor = self.gate(pooled)

    # (B,) — per-sample boolean mask, no index-0 shortcut
    should_think: Tensor = thinking_prob.squeeze(-1) > 0.5

    return should_think, thinking_prob