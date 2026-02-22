import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Any

class AdaptiveThinkingGate(nn.Module):
  def __init__(self, emb_dim: int)-> None:
    super().__init__()

    self.gate: nn.Sequential = nn.Sequential(
      nn.Linear(emb_dim, emb_dim // 4),
      nn.GELU(),
      nn.Linear(emb_dim // 4, 1),
      nn.Sigmoid()
    )
  
  def forward(self, hidden_states: Tensor)-> Tuple[bool, Tensor]:
    pooled_query: Tensor = hidden_states.mean(dim=1)
    thinking_prob: Tensor = self.gate(pooled_query)

    should_think: bool = bool(thinking_prob[0].item() > 0.5)

    return  should_think, thinking_prob