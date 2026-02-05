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

import numpy as np
import torch

from vllm_rbln.v1.worker.rbln_model_runner import (
    AsyncRBLNModelRunnerOutput,
    DummyRunState,
    ExecuteModelState,
    create_lora_mask,
    create_sampler_indices_padded,
)


class TestExecuteModelState:
    """ExecuteModelState is a NamedTuple holding ephemeral cached state."""

    def test_fields(self):
        state = ExecuteModelState(
            scheduler_output=None,
            logits=torch.zeros(2, 4),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.zeros(2, 8),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
        )
        assert state.scheduler_output is None
        assert state.logits.shape == (2, 4)
        assert state.hidden_states.shape == (2, 8)
        assert state.spec_decode_metadata is None
        assert state.sample_hidden_states is None
        assert state.aux_hidden_states is None
        assert state.kv_connector_output is None

    def test_unpack(self):
        logits = torch.randn(3, 5)
        hidden = torch.randn(3, 10)
        state = ExecuteModelState(
            scheduler_output="sched",
            logits=logits,
            spec_decode_metadata="spec",
            spec_decode_common_attn_metadata="common",
            hidden_states=hidden,
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
        )
        (sched, l, sd, sdc, h, sh, ah, kv) = state
        assert sched == "sched"
        assert torch.equal(l, logits)
        assert sd == "spec"


class TestDummyRunState:
    """DummyRunState is a NamedTuple for dummy run inputs."""

    def test_fields(self):
        state = DummyRunState(
            attn_metadata={0: {"key": "val"}},
            num_input_tokens=128,
            input_ids={0: torch.zeros(128)},
            positions={0: torch.arange(128)},
        )
        assert state.num_input_tokens == 128
        assert 0 in state.attn_metadata
        assert state.input_ids[0].shape == (128,)
        assert state.positions[0].shape == (128,)


class TestAsyncRBLNModelRunnerOutput:
    """Test AsyncRBLNModelRunnerOutput init stores fields correctly."""

    def test_init(self):
        from unittest.mock import MagicMock

        model_output = MagicMock()
        sampled_ids = torch.tensor([[1], [2], [3]])
        invalid = [1]
        stream = MagicMock()

        out = AsyncRBLNModelRunnerOutput(
            model_runner_output=model_output,
            sampled_token_ids=sampled_ids,
            invalid_req_indices=invalid,
            async_output_copy_stream=stream,
        )
        assert out._model_runner_output is model_output
        assert torch.equal(out._sampled_token_ids, sampled_ids)
        assert out._invalid_req_indices == [1]


class TestCreateLoraMask:
    """Test create_lora_mask standalone function."""

    def test_no_lora(self):
        """When all lora_ids are 0, mask should be all zeros."""
        input_ids = torch.zeros(2, 4, dtype=torch.long)
        lora_ids = [0, 0]
        lora_index_to_id = [0, 1, 2]
        mask = create_lora_mask(
            input_ids=input_ids,
            lora_ids=lora_ids,
            lora_index_to_id=lora_index_to_id,
            max_loras=3,
            max_lora_rank=8,
            lora_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert mask.shape == (8, 24)  # 2*4, 3*8
        assert mask.sum().item() == 0.0

    def test_single_lora(self):
        """Single active LoRA should populate correct mask region."""
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        lora_ids = [2]
        lora_index_to_id = [0, 1, 2]
        mask = create_lora_mask(
            input_ids=input_ids,
            lora_ids=lora_ids,
            lora_index_to_id=lora_index_to_id,
            max_loras=3,
            max_lora_rank=4,
            lora_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        # lora_id=2 is at index 2, so cols 8..11 should be 1 for rows 0..2
        assert mask.shape == (3, 12)  # 1*3, 3*4
        assert mask[:, 8:12].sum().item() == 3.0 * 4.0  # 3 tokens * rank 4
        assert mask[:, :8].sum().item() == 0.0

    def test_multi_lora(self):
        """Multiple LoRAs in a batch."""
        input_ids = torch.zeros(2, 2, dtype=torch.long)
        lora_ids = [1, 2]
        lora_index_to_id = [0, 1, 2]
        mask = create_lora_mask(
            input_ids=input_ids,
            lora_ids=lora_ids,
            lora_index_to_id=lora_index_to_id,
            max_loras=3,
            max_lora_rank=2,
            lora_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        # Shape: (4, 6) — 2*2 tokens, 3*2 rank
        assert mask.shape == (4, 6)
        # Batch 0 (rows 0,1): lora_id=1 → index=1 → cols 2..3
        assert mask[0, 2:4].sum().item() == 2.0
        assert mask[1, 2:4].sum().item() == 2.0
        # Batch 1 (rows 2,3): lora_id=2 → index=2 → cols 4..5
        assert mask[2, 4:6].sum().item() == 2.0
        assert mask[3, 4:6].sum().item() == 2.0


class TestCreateSamplerIndicesPadded:
    """Test create_sampler_indices_padded standalone function."""

    def test_prefill_single_lora(self):
        """Prefill with a single LoRA."""
        indices = create_sampler_indices_padded(
            lora_ids=[1],
            lora_index_to_id=[0, 1],
            max_num_seqs=4,
            is_prefill=True,
            max_loras=2,
            device=torch.device("cpu"),
        )
        # Single request: lora_id=1, index_in_lora_index_to_id=1
        # padded = 0 + (1 * 1) = 1
        assert indices.shape == (1,)
        assert indices[0].item() == 1

    def test_decode_multiple(self):
        """Decode with multiple requests."""
        indices = create_sampler_indices_padded(
            lora_ids=[1, 0, 2],
            lora_index_to_id=[0, 1, 2],
            max_num_seqs=4,
            is_prefill=False,
            max_loras=3,
            device=torch.device("cpu"),
        )
        # Length = max_num_seqs = 4
        assert indices.shape == (4,)

    def test_decode_no_lora(self):
        """Decode with no active LoRAs — all should map to max_loras slot."""
        indices = create_sampler_indices_padded(
            lora_ids=[0, 0],
            lora_index_to_id=[0, 1],
            max_num_seqs=2,
            is_prefill=False,
            max_loras=2,
            device=torch.device("cpu"),
        )
        # For lora_id=0, prompt_mapping=-1, clamped to max_loras=2
        # indices[i] = i + (2 * 2) = i + 4
        assert indices.shape == (2,)
        assert indices[0].item() == 0 + (2 * 2)
        assert indices[1].item() == 1 + (2 * 2)
