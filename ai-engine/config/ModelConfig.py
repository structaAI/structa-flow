from __future__ import annotations


class ModelConfig:
  def __init__(
      self,
      vocab_size: int = 32000,
      emb_dim: int = 1024,
      num_layers: int = 28,
      num_heads: int = 32,
      num_query_groups: int = 8,
      max_seq_len: int = 2048,
      ffn_hidden_size: int = 4096,
      norm_eps: float = 1e-5,
      dropout: float = 0.1,
      initializer_range: float = 0.02,
      use_rotary_embeddings: bool = True,
  ) -> None:
    assert num_heads % num_query_groups == 0, (
        f"num_heads ({num_heads}) must be divisible by "
        f"num_query_groups ({num_query_groups})"
    )
    assert emb_dim % num_heads == 0, (
        f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
    )

    self.vocab_size: int = vocab_size
    self.emb_dim: int = emb_dim
    self.num_layers: int = num_layers
    self.num_heads: int = num_heads
    self.num_query_groups: int = num_query_groups
    self.max_seq_len: int = max_seq_len
    self.ffn_hidden_size: int = ffn_hidden_size
    self.norm_eps: float = norm_eps
    self.dropout: float = dropout
    self.initializer_range: float = initializer_range
    self.use_rotary_embeddings: bool = use_rotary_embeddings

  @property
  def head_dim(self) -> int:
    return self.emb_dim // self.num_heads

  @property
  def kv_heads(self) -> int:
    return self.num_query_groups

  def __repr__(self) -> str:
    return (
      f"ModelConfig(vocab_size={self.vocab_size}, emb_dim={self.emb_dim}, "
      f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
      f"num_query_groups={self.num_query_groups}, max_seq_len={self.max_seq_len}, "
      f"ffn_hidden_size={self.ffn_hidden_size}, dropout={self.dropout})"
    )