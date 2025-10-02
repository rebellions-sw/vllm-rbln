# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding, _rotate_gptj,
    _rotate_neox)

rope_original__init__ = RotaryEmbedding.__init__


def rope__custom_init__(
    self: RotaryEmbedding,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
):
    rope_original__init__(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
    )

    # For best compatibility with rbln, we use the rotate_half-style RoPE.
    cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
    if self.is_neox_style:
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
    else:
        cos = torch.stack([cos, cos], dim=-1).reshape(cos.shape[0], -1)
        sin = torch.stack([sin, sin], dim=-1).reshape(sin.shape[0], -1)
    self.register_buffer("cos_cache", cos, persistent=False)
    self.register_buffer("sin_cache", sin, persistent=False)


def rope_forward_oot(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A PyTorch-native implementation of forward()."""
    # NOTE(RBLN): For best compatibility with rbln,
    # tensors are reshaped/transposed as follows:
    # - cos, sin: (1, 1, num_tokens, rotary_dim * 2)
    # - query, key: (1, num_heads, num_tokens, head_size)

    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]

    rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj

    cos = self.cos_cache.index_select(0, positions)
    sin = self.sin_cache.index_select(0, positions)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., :self.rotary_dim]
    query_rot = query_rot.unsqueeze(0).transpose(1, 2)
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    query_rot = query_rot.transpose(1, 2).squeeze(0)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        query = query_rot.reshape(query_shape)
    else:
        query_pass = query[..., self.rotary_dim:]
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)
    key_rot = key[..., :self.rotary_dim]
    key_rot = key_rot.unsqueeze(0).transpose(1, 2)
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin
    key_rot = key_rot.transpose(1, 2).squeeze(0)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        key = key_rot.reshape(key_shape)
    else:
        key_pass = key[..., self.rotary_dim:]
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


def deepseek_scaling_rope_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-native implementation equivalent to forward()."""
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()

    query_rot = query[..., :self.rotary_dim]
    key_rot = key[..., :self.rotary_dim]
    if self.rotary_dim < self.head_size:
        query_pass = query[..., self.rotary_dim:]
        key_pass = key[..., self.rotary_dim:]

    self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    if self.is_neox_style:
        # NOTE(woosuk): Here we assume that the positions tensor has the
        # shape [batch_size, seq_len].
        cos = cos.repeat(1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 2).unsqueeze(-2)
    else:
        cos = torch.stack([cos, cos],
                          dim=-1).reshape(cos_sin.shape).unsqueeze(-2)
        sin = torch.stack([sin, sin],
                          dim=-1).reshape(cos_sin.shape).unsqueeze(-2)

    rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if self.rotary_dim < self.head_size:
        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)
    else:
        query = query_rot
        key = key_rot
    return query, key


RotaryEmbedding.__init__ = rope__custom_init__
RotaryEmbedding.forward_oot = rope_forward_oot
DeepseekScalingRotaryEmbedding.forward = deepseek_scaling_rope_forward
