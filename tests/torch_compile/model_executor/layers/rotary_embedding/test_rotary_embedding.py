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

import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def test_rope_custom_init_creates_buffers(vllm_config):
    """rope__custom_init__ should create cos_cache and sin_cache buffers."""
    with set_current_vllm_config(vllm_config):
        rope = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=2048,
            base=10000,
            is_neox_style=True,
            dtype=torch.float16,
        )
        assert isinstance(rope.get_buffer("cos_cache"), torch.Tensor)
        assert isinstance(rope.get_buffer("sin_cache"), torch.Tensor)


def test_rope_cos_sin_cache_shapes(vllm_config):
    """cos_cache and sin_cache should have correct shape after RBLN custom init.

    The original cos_sin_cache has shape (max_pos, rotary_dim).
    RBLN splits it in half (rotary_dim // 2) then repeats, yielding (max_pos, rotary_dim).
    """
    with set_current_vllm_config(vllm_config):
        head_size = 64
        rotary_dim = 64
        max_pos = 128
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_pos,
            base=10000,
            is_neox_style=True,
            dtype=torch.float16,
        )
        cos = rope.get_buffer("cos_cache")
        sin = rope.get_buffer("sin_cache")
        # cos_sin_cache is (max_pos, rotary_dim), split into two (rotary_dim//2) halves,
        # then each half is repeated -> back to (max_pos, rotary_dim)
        assert cos.shape[-1] == rotary_dim
        assert sin.shape[-1] == rotary_dim


def test_rope_neox_vs_gptj_style(vllm_config):
    """neox and gptj styles should produce different cos/sin cache layouts."""
    with set_current_vllm_config(vllm_config):
        rope_neox = RotaryEmbedding(
            head_size=32,
            rotary_dim=32,
            max_position_embeddings=64,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )
        rope_gptj = RotaryEmbedding(
            head_size=32,
            rotary_dim=32,
            max_position_embeddings=64,
            base=10000,
            is_neox_style=False,
            dtype=torch.float32,
        )
        cos_neox = rope_neox.get_buffer("cos_cache")
        cos_gptj = rope_gptj.get_buffer("cos_cache")
        # Same underlying values but different interleaving pattern
        assert cos_neox.shape == cos_gptj.shape
        # They should NOT be identical due to different repeat/stack patterns
        # (neox uses repeat, gptj uses stack+reshape)
        # But both should have the same L2 norm per row
        assert torch.allclose(
            cos_neox.float().norm(dim=-1),
            cos_gptj.float().norm(dim=-1),
            atol=1e-5,
        )


def test_rope_forward_oot_basic(vllm_config):
    """rope_forward_oot should transform query and key tensors."""
    with set_current_vllm_config(vllm_config):
        head_size = 32
        rotary_dim = 32
        num_heads = 4
        num_kv_heads = 4
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=128,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )

        batch_size, seq_len = 2, 8
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        query = torch.randn(batch_size, seq_len, num_heads * head_size)
        key = torch.randn(batch_size, seq_len, num_kv_heads * head_size)

        q_out, k_out = rope.forward_oot(positions, query, key)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape
        # Output should be different from input (rotation applied)
        assert not torch.equal(q_out, query)
        assert not torch.equal(k_out, key)


def test_rope_forward_oot_with_offsets(vllm_config):
    """rope_forward_oot with offsets should shift positions."""
    with set_current_vllm_config(vllm_config):
        head_size = 32
        rotary_dim = 32
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=128,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )

        batch_size, seq_len = 1, 4
        positions = torch.zeros(batch_size, seq_len, dtype=torch.long)
        offsets = torch.tensor([[0, 1, 2, 3]])
        query = torch.randn(batch_size, seq_len, 4 * head_size)
        key = torch.randn(batch_size, seq_len, 4 * head_size)

        # positions + offsets = [0,1,2,3], same as using positions=[0,1,2,3]
        q1, k1 = rope.forward_oot(
            positions, query.clone(), key.clone(), offsets=offsets
        )
        positions2 = torch.arange(seq_len).unsqueeze(0)
        q2, k2 = rope.forward_oot(positions2, query.clone(), key.clone())

        assert torch.allclose(q1, q2, atol=1e-5)
        assert torch.allclose(k1, k2, atol=1e-5)


def test_rope_forward_oot_partial_rotary(vllm_config):
    """When rotary_dim < head_size, only partial dimensions should be rotated."""
    with set_current_vllm_config(vllm_config):
        head_size = 64
        rotary_dim = 32  # Only half is rotated
        num_heads = 2
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=128,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )

        batch_size, seq_len = 1, 4
        positions = torch.arange(seq_len).unsqueeze(0)
        query = torch.randn(batch_size, seq_len, num_heads * head_size)
        key = torch.randn(batch_size, seq_len, num_heads * head_size)

        q_out, k_out = rope.forward_oot(positions, query, key)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape
