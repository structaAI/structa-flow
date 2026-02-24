from dataclasses import dataclass

@dataclass
class ModelConfig:
  vocab_size: int = 32000
  emb_dim: int = 1024
  num_layers: int = 28
  num_heads: int = 32
  num_query_groups: int = 8
  max_seq_len: int = 2048
  ffn_hidden_size: int = 4096
  norm_eps: float = 1e-6
  dropout: float = 0.1
  initializer_range: float = 0.02

  use_rope: bool = True

  def __post_init__(self) -> None:
    assert self.num_heads % self.num_query_groups == 0, (
      f"num_heads ({self.num_heads}) must be divisible by "
      f"num_query_groups ({self.num_query_groups})"
    )
    assert self.emb_dim % self.num_heads == 0, (
      f"emb_dim ({self.emb_dim}) must be divisible by num_heads ({self.num_heads})"
    )
  
  @property
  def head_dim(self)-> int:
    return self.emb_dim // self.num_heads
  
  @property
  def kv_heads(self)-> int:
    return self.num_query_groups