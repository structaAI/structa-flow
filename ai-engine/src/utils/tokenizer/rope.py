import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class RotaryPositionalEmbedding(nn.Module):
  inv_freq: Tensor    
  cos_cached: Tensor     
  sin_cached: Tensor     

  def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
    super().__init__()
    self.head_dim = head_dim

    inv_freq: Tensor = 1.0 / (
      base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)

    self._build_cache(max_seq_len)

  def _build_cache(self, seq_len: int) -> None:
    t: Tensor = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
    freqs: Tensor = torch.outer(t, self.inv_freq)
    self.register_buffer("cos_cached", torch.cat([freqs.cos(), freqs.cos()], dim=-1), persistent=False)
    self.register_buffer("sin_cached", torch.cat([freqs.sin(), freqs.sin()], dim=-1), persistent=False)

  def _rotate_half(self, x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

  def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
    seq_len = q.size(2)

    if seq_len > self.cos_cached.size(0):
      self._build_cache(seq_len)

    cos = self.cos_cached[:seq_len].to(q.dtype)
    sin = self.sin_cached[:seq_len].to(q.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + self._rotate_half(q) * sin
    k_rot = k * cos + self._rotate_half(k) * sin

    return q_rot, k_rot