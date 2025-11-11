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

from typing import Optional

import torch
from vllm.model_executor.layers.rotary_embedding.common import (rotate_gptj,
                                                                rotate_neox)
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding


def mrope_forward_oot(
    self: MRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        positions:
            [batch_size, seq_len] (text only) or
            [3, batch_size, seq_len] (T/H/W positions with multimodal inputs)
        query: [batch_size, seq_len, num_heads * head_size]
        key: [batch_size, seq_len, num_kv_heads * head_size]
    """

    assert positions.ndim == 2 or positions.ndim == 3
    assert key is not None

    batch_size, seq_len = positions.shape[-2], positions.shape[-1]
    rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj

    cos = self.cos_cache[positions]
    sin = self.sin_cache[positions]

    if positions.ndim == 3:
        assert self.mrope_section
        # We use the rotate_half-style RoPE,
        # so mrope_section needs to be repeated as well.
        mrope_section = self.mrope_section * 2
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
            dim=-1)
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
            dim=-1)

    cos = cos.view(batch_size, 1, seq_len, -1)
    sin = sin.view(batch_size, 1, seq_len, -1)

    query_shape = query.shape
    query = query.view(batch_size, seq_len, -1, self.head_size)
    query_rot = query[..., :self.rotary_dim]
    query_rot = query_rot.transpose(1, 2)
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    query_rot = query_rot.transpose(1, 2)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        query = query_rot.reshape(query_shape)
    else:
        query_pass = query[..., self.rotary_dim:]
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(batch_size, seq_len, -1, self.head_size)
    key_rot = key[..., :self.rotary_dim]
    key_rot = key_rot.transpose(1, 2)
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin
    key_rot = key_rot.transpose(1, 2)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        key = key_rot.reshape(key_shape)
    else:
        key_pass = key[..., self.rotary_dim:]
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


MRotaryEmbedding.forward_oot = mrope_forward_oot
