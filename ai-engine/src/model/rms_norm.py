import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
  def __init__(self, emb_dim: int, eps: float = 1e-6)->None:
    super().__init__()

    self.eps: float = eps
    self.weight: nn.Parameter = nn.Parameter(torch.ones(emb_dim))

  def _norm(self, x: Tensor)-> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: Tensor)-> Tensor:
    output_dtype = x.dtype
        
    norm_x = self._norm(x.to(torch.float32))

    return (norm_x.to(output_dtype) * self.weight)