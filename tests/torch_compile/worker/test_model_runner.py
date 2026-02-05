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

"""Tests for v0 model_runner (vllm_rbln.worker.model_runner).

NOTE: The v0 model_runner imports vllm.attention.AttentionMetadata which was
removed in vLLM v0.13. These tests use importlib to verify the module state
and skip if the import fails.
"""

import importlib

import pytest

# The v0 model runner cannot be imported under vLLM v0.13+
# because it depends on vllm.attention.AttentionMetadata which was removed.
_v0_model_runner_available = True
try:
    _mod = importlib.import_module("vllm_rbln.worker.model_runner")
except ImportError:
    _v0_model_runner_available = False

pytestmark = pytest.mark.skipif(
    not _v0_model_runner_available,
    reason="v0 model_runner requires vllm.attention.AttentionMetadata (removed in v0.13)",
)

import torch


class TestModelInputForRebel:
    """Test ModelInputForRebel dataclass methods."""

    def test_default_fields(self):
        """Default fields should be None."""
        from vllm_rbln.worker.model_runner import ModelInputForRebel

        m = ModelInputForRebel()
        assert m.input_tokens is None
        assert m.input_positions is None
        assert m.attn_metadata is None
        assert m.virtual_engine is None
        assert m.seq_lens is None
        assert m.query_lens is None

    def test_with_values(self):
        """Should store provided values."""
        from vllm_rbln.worker.model_runner import ModelInputForRebel

        tokens = torch.tensor([[1, 2, 3]])
        positions = torch.tensor([[0, 1, 2]])
        m = ModelInputForRebel(
            input_tokens=tokens,
            input_positions=positions,
            seq_lens=[3],
            query_lens=[3],
        )
        assert torch.equal(m.input_tokens, tokens)
        assert torch.equal(m.input_positions, positions)
        assert m.seq_lens == [3]
        assert m.query_lens == [3]

    def test_as_broadcastable_tensor_dict(self):
        """Should produce dict with input_tokens and input_positions."""
        from vllm_rbln.worker.model_runner import ModelInputForRebel

        tokens = torch.tensor([[1, 2]])
        positions = torch.tensor([[0, 1]])
        m = ModelInputForRebel(
            input_tokens=tokens,
            input_positions=positions,
        )
        d = m.as_broadcastable_tensor_dict()
        assert "input_tokens" in d
        assert "input_positions" in d
        assert torch.equal(d["input_tokens"], tokens)

    def test_from_broadcasted_tensor_dict_no_backend(self):
        """Should reconstruct from tensor dict without attn backend."""
        from vllm_rbln.worker.model_runner import ModelInputForRebel

        tokens = torch.tensor([[5, 6]])
        positions = torch.tensor([[0, 1]])
        d = {"input_tokens": tokens, "input_positions": positions}
        m = ModelInputForRebel.from_broadcasted_tensor_dict(d)
        assert torch.equal(m.input_tokens, tokens)
        assert torch.equal(m.input_positions, positions)


class TestModelInputForRebelWithSamplingMetadata:
    """Test ModelInputForRebelWithSamplingMetadata dataclass."""

    def test_default_fields(self):
        """Sampling metadata defaults should be None."""
        from vllm_rbln.worker.model_runner import (
            ModelInputForRebelWithSamplingMetadata,
        )

        m = ModelInputForRebelWithSamplingMetadata()
        assert m.sampling_metadata is None
        assert m.is_prompt is None

    def test_as_broadcastable_includes_sampling(self):
        """Dict should include sampling metadata keys."""
        from vllm_rbln.worker.model_runner import (
            ModelInputForRebelWithSamplingMetadata,
        )

        tokens = torch.tensor([[1]])
        positions = torch.tensor([[0]])
        m = ModelInputForRebelWithSamplingMetadata(
            input_tokens=tokens,
            input_positions=positions,
        )
        d = m.as_broadcastable_tensor_dict()
        assert "input_tokens" in d
        assert "input_positions" in d


class TestModelInputForRebelBuilderModelInputData:
    """Test the inner ModelInputData class."""

    def test_init_default(self):
        """ModelInputData should initialize empty lists."""
        from vllm_rbln.worker.model_runner import ModelInputForRebelBuilder

        data = ModelInputForRebelBuilder.ModelInputData(use_mrope=False)
        assert data.input_tokens == []
        assert data.input_positions == []
        assert data.seq_lens == []
        assert data.query_lens == []
        assert data.num_prefills == 0
        assert data.num_prefill_tokens == 0
        assert data.num_decode_tokens == 0
        assert data.slot_mapping == []
        assert data.max_decode_seq_len == 0

    def test_init_mrope(self):
        """With use_mrope=True, should init mrope positions."""
        from vllm_rbln.worker.model_runner import ModelInputForRebelBuilder

        data = ModelInputForRebelBuilder.ModelInputData(use_mrope=True)
        assert data.use_mrope is True
        assert len(data.input_mrope_positions) == 3
        for pos_list in data.input_mrope_positions:
            assert pos_list == []
