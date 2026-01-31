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
"""Unit tests for RBLN shared-storage KV connector helpers."""

import torch

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_shared_storage_connector import (  # noqa: E501
    RBLNReqMeta, RBLNSharedStorageConnector)


class TestRBLNReqMeta:
    """Tests for request metadata and slot mapping."""

    def test_slot_mapping_basic(self):
        """Verify slot mapping aligns tokens to block boundaries."""
        block_size = 4
        token_ids = list(range(10))  # 10 tokens
        block_ids = [0, 1]  # 2 blocks = 8 slots
        meta = RBLNReqMeta.make_meta(token_ids,
                                     block_ids,
                                     block_size,
                                     is_store=True)
        # 10 tokens aligned to block_size=4 -> 8 tokens
        assert meta.token_ids.shape[0] == 8
        assert meta.slot_mapping.shape[0] == 8
        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        assert torch.equal(meta.slot_mapping, expected)

    def test_slot_mapping_noncontiguous_blocks(self):
        """Block IDs need not be contiguous."""
        block_size = 2
        token_ids = list(range(4))
        block_ids = [5, 10]
        meta = RBLNReqMeta.make_meta(token_ids,
                                     block_ids,
                                     block_size,
                                     is_store=False)
        expected = torch.tensor([10, 11, 20, 21], dtype=torch.long)
        assert torch.equal(meta.slot_mapping, expected)


class TestKVExtractInject:
    """Round-trip tests for RBLN KV cache extract and inject."""

    @staticmethod
    def _make_kv_layer(
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_size: int,
    ) -> torch.Tensor:
        """Create RBLN KV cache tensor with known values.

        Shape: (2, num_blocks, num_kv_heads, 1, block_size, head_size)
        """
        return torch.randn(2, num_blocks, num_kv_heads, 1, block_size,
                           head_size)

    def test_extract_inject_roundtrip(self):
        """Extract then inject should reproduce the original data."""
        num_blocks, num_kv_heads, block_size, head_size = 4, 2, 4, 8
        kv_layer = self._make_kv_layer(num_blocks, num_kv_heads, block_size,
                                       head_size)
        original = kv_layer.clone()

        # Extract tokens 0..7 from blocks 0,1
        slot_mapping = torch.arange(0, 8, dtype=torch.long)
        extracted = RBLNSharedStorageConnector._extract_kv(
            kv_layer, slot_mapping)
        assert extracted.shape == (2, 8, num_kv_heads * head_size)

        # Zero out and inject back
        kv_layer.zero_()
        RBLNSharedStorageConnector._inject_kv(kv_layer, extracted,
                                              slot_mapping)

        # Blocks 0,1 should match original; blocks 2,3 remain zero
        assert torch.allclose(
            kv_layer[:, :2, :, :, :, :],
            original[:, :2, :, :, :, :],
        )
        assert torch.all(kv_layer[:, 2:, :, :, :, :] == 0)

    def test_extract_noncontiguous_slots(self):
        """Extract from non-contiguous block slots."""
        num_blocks, num_kv_heads, block_size, head_size = 8, 1, 2, 4
        kv_layer = self._make_kv_layer(num_blocks, num_kv_heads, block_size,
                                       head_size)

        # Take slots from block 3 (slots 6,7) and block 5 (slots 10,11)
        slot_mapping = torch.tensor([6, 7, 10, 11], dtype=torch.long)
        extracted = RBLNSharedStorageConnector._extract_kv(
            kv_layer, slot_mapping)
        assert extracted.shape == (2, 4, num_kv_heads * head_size)

    def test_inject_preserves_other_blocks(self):
        """Inject should only modify the targeted slots."""
        num_blocks, num_kv_heads, block_size, head_size = 4, 1, 2, 4
        kv_layer = self._make_kv_layer(num_blocks, num_kv_heads, block_size,
                                       head_size)
        original = kv_layer.clone()

        # Inject into block 2 only (slots 4, 5)
        slot_mapping = torch.tensor([4, 5], dtype=torch.long)
        new_kv = torch.ones(2, 2, num_kv_heads * head_size)
        RBLNSharedStorageConnector._inject_kv(kv_layer, new_kv, slot_mapping)

        # Blocks 0,1,3 should be unchanged
        assert torch.equal(
            kv_layer[:, 0, :, :, :, :],
            original[:, 0, :, :, :, :],
        )
        assert torch.equal(
            kv_layer[:, 1, :, :, :, :],
            original[:, 1, :, :, :, :],
        )
        assert torch.equal(
            kv_layer[:, 3, :, :, :, :],
            original[:, 3, :, :, :, :],
        )
