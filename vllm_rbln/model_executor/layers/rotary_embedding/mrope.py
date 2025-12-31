from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import (rotate_gptj,
                                                                rotate_neox)

mrope_original__init__ = MRotaryEmbedding.__init__


def mrope__custom_init__(
    self: MRotaryEmbedding,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    mrope_section: Optional[list[int]] = None,
):
    mrope_original__init__(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        mrope_section,
    )

    # For best compatibility with rbln, we use the rotate_half-style RoPE.
    cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

    # Pre-split cos/sin for mrope sections to avoid split in forward
    if mrope_section:
        cos_parts = cos.split(mrope_section, dim=-1)
        sin_parts = sin.split(mrope_section, dim=-1)
        for i, (c_part, s_part) in enumerate(zip(cos_parts, sin_parts)):
            self.register_buffer(f"cos_cache_{i}", c_part, persistent=False)
            self.register_buffer(f"sin_cache_{i}", s_part, persistent=False)

    if self.is_neox_style:
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
    else:
        cos = torch.stack([cos, cos], dim=-1).reshape(cos.shape[0], -1)
        sin = torch.stack([sin, sin], dim=-1).reshape(sin.shape[0], -1)
    self.register_buffer("cos_cache", cos, persistent=False)
    self.register_buffer("sin_cache", sin, persistent=False)


def mrope_forward_oot(
    self: MRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions:
            [num_tokens,] (text only) or
            [3, num_tokens] (T/H/W positions with multimodal inputs)
        query: [num_tokens, num_heads * head_size]
        key: [num_tokens, num_kv_heads * head_size]
    """
    positions_ndim = positions.ndim
    if offsets is not None:
        positions = positions + offsets

    rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
    query_shape = query.shape
    
    # Infer seq_len and batch_size robustly
    if positions.ndim >= 2:
        seq_len = positions.shape[-1]
    else:
        seq_len = positions.shape[0]

    if query.ndim == 2:
        batch_size = 1
    elif query.ndim == 3:
        batch_size = query_shape[0]
    elif query.ndim == 4:
        batch_size = query_shape[0]
        if query_shape[2] == seq_len:
            query = query.transpose(1, 2).flatten(2)
        elif query_shape[3] == seq_len:
            query = query.transpose(2, 3).transpose(1, 2).flatten(2)
        else:
            query = query.transpose(1, 2).flatten(2)
    else:
        raise ValueError(
            "MRoPE only supports 2D, 3D or 4D queries (got shape "
            f"{query_shape}).")

    if positions_ndim == 2:
        assert self.mrope_section

        # Avoid split in forward by using pre-split caches
        cos_list = []
        sin_list = []
        for i in range(3):
            c_cache = getattr(self, f"cos_cache_{i}")
            s_cache = getattr(self, f"sin_cache_{i}")
            c_part = c_cache.index_select(0, positions[i])
            s_part = s_cache.index_select(0, positions[i])
            cos_list.append(c_part)
            sin_list.append(s_part)
            
        cos = torch.cat(cos_list, dim=-1)
        sin = torch.cat(sin_list, dim=-1)
        
        # Re-apply neox style
        if self.is_neox_style:
            cos = cos.repeat(1, 2)
            sin = sin.repeat(1, 2)

        # Reshape for broadcasting
        # (seq_len, 128) -> (1, 1, seq_len, 128)
        cos = cos.view(batch_size, 1, seq_len, -1)
        sin = sin.view(batch_size, 1, seq_len, -1)

    else:
        positions_flat = positions.flatten()
        cos = self.cos_cache.index_select(0, positions_flat).view(
            batch_size, 1, seq_len, -1)
        sin = self.sin_cache.index_select(0, positions_flat).view(
            batch_size, 1, seq_len, -1)

    query = query.view(batch_size, seq_len, -1, self.head_size)
    query_rot = query[..., :self.rotary_dim]
    query_rot = query_rot.transpose(1, 2)
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    query_rot = query_rot.transpose(1, 2)

    if self.head_size == self.rotary_dim:
        query = query_rot.reshape(batch_size, seq_len, -1)
    else:
        query_pass = query[..., self.rotary_dim:]
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(batch_size, seq_len, -1)

    key_shape = key.shape
    # Handle key 4D input similarly (assuming same layout as query)
    if key.ndim == 4:
         if key_shape[2] == seq_len:
             key = key.transpose(1, 2).flatten(2)
         elif key_shape[3] == seq_len:
             key = key.transpose(2, 3).transpose(1, 2).flatten(2)
         else:
             key = key.transpose(1, 2).flatten(2)
             
    key = key.view(batch_size, seq_len, -1, self.head_size)
    key_rot = key[..., :self.rotary_dim]
    key_rot = key_rot.transpose(1, 2)
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin
    key_rot = key_rot.transpose(1, 2)

    if self.head_size == self.rotary_dim:
        key = key_rot.reshape(batch_size, seq_len, -1)
    else:
        key_pass = key[..., self.rotary_dim:]
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(batch_size, seq_len, -1)

    return query, key


MRotaryEmbedding.__init__ = mrope__custom_init__
MRotaryEmbedding.forward_oot = mrope_forward_oot
