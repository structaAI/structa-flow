import torch
import torch.nn as nn
from torch import Tensor

class RotartPositionEmbedding(nn.Module):
  def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0)-> None:
    super().__init__()

    self.head_dim = head_dim
    self.inv_freq: Tensor = 1.0 / (
      base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    self.register_buffer("inv_freq", self.inv_freq, persistent=False)

    self._build_cache(max_seq_len)

  def _build_cache(self, seq_len: int) -> None:
    t: Tensor = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
    freqs: Tensor = torch.outer(t, self.inv_freq)          # (seq_len, head_dim // 2)
    emb: Tensor = torch.cat([freqs, freqs], dim=-1)        # (seq_len, head_dim)
    self.register_buffer("cos_cached", emb.cos(), persistent=False)
    self.register_buffer("sin_cached", emb.sin(), persistent=False)
  
  def _rotate_half():
    pass