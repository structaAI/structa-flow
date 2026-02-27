from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml # type: ignore[import]


@dataclass
class ModelConfig:

  vocab_size: int = 32000
  emb_dim: int = 1024
  num_layers: int = 28
  num_heads: int = 32
  num_query_groups: int = 8         
  max_seq_len: int = 2048
  ffn_hidden_size: int = 4096
  norm_eps: float = 1e-5
  dropout: float = 0.1
  initializer_range: float = 0.02
  use_rotary_embeddings: bool = True

  def __post_init__(self) -> None:
    if self.num_heads % self.num_query_groups != 0:
      raise ValueError(
        f"num_heads ({self.num_heads}) must be divisible by "
        f"num_query_groups ({self.num_query_groups})"
      )
    if self.emb_dim % self.num_heads != 0:
      raise ValueError(
        f"emb_dim ({self.emb_dim}) must be divisible by "
        f"num_heads ({self.num_heads})"
      )

  @property
  def head_dim(self) -> int:
    return self.emb_dim // self.num_heads

  @property
  def kv_heads(self) -> int:
    return self.num_query_groups

  @property
  def kv_dim(self) -> int:
    return self.kv_heads * self.head_dim

  @classmethod
  def from_dict(cls, cfg: dict) -> "ModelConfig":
    alias_map = {
      "hidden_size": "emb_dim",
      "num_attention_heads": "num_heads",
      "max_position_embeddings": "max_seq_len",
      "rotary_embeddings": "use_rotary_embeddings",
      "norm_type": None,          
      "activation_function": None,  
      "type": None,
    }
    normalised: dict = {}
    for k, v in cfg.items():
      mapped = alias_map.get(k, k)
      if mapped is not None:
        normalised[mapped] = v

    return cls(**normalised)

  @classmethod
  def from_yaml(cls, path: str | Path) -> "ModelConfig":
    with open(path, "r") as fh:
      raw = yaml.safe_load(fh)
    if "model" not in raw:
      raise KeyError(f"Expected a top-level 'model' key in {path}")
    return cls.from_dict(raw["model"])