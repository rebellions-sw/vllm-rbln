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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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


class TestSelectCommonBlockSize:
    """Test select_common_block_size pure logic: int sizes, MultipleOf, errors."""

    def _make_backend(self, supported_sizes):
        """Create a mock AttentionBackend with given supported sizes."""
        from unittest.mock import MagicMock

        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = supported_sizes
        return backend

    def _make_attn_group(self, backend):
        from unittest.mock import MagicMock

        group = MagicMock()
        group.backend = backend
        return group

    def test_kv_manager_size_supported_by_all(self):
        """If kv_manager_block_size is directly supported, return it."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        b1 = self._make_backend([16, 32])
        b2 = self._make_backend([16, 64])
        groups = [self._make_attn_group(b1), self._make_attn_group(b2)]

        result = RBLNModelRunner.select_common_block_size(16, groups)
        assert result == 16

    def test_kv_manager_size_with_multiple_of(self):
        """kv_manager_block_size supported via MultipleOf."""
        from vllm.attention.backends.abstract import MultipleOf
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        b1 = self._make_backend([MultipleOf(8)])  # 32 % 8 == 0
        b2 = self._make_backend([MultipleOf(16)])  # 32 % 16 == 0
        groups = [self._make_attn_group(b1), self._make_attn_group(b2)]

        result = RBLNModelRunner.select_common_block_size(32, groups)
        assert result == 32

    def test_fallback_to_largest_int_factor(self):
        """kv_manager_block_size not directly supported → find largest int factor."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        # kv_manager_block_size = 64
        # b1 supports [16, 32], b2 supports [32, 64]
        # 64 not in b1, so case 1 fails
        # case 2: int sizes = {16, 32, 64}
        #   64: 64 % 64 == 0, but b1 only supports [16, 32] → 64 not in b1 → fail
        #   32: 64 % 32 == 0, b1 supports 32, b2 supports 32 → pass
        b1 = self._make_backend([16, 32])
        b2 = self._make_backend([32, 64])
        groups = [self._make_attn_group(b1), self._make_attn_group(b2)]

        result = RBLNModelRunner.select_common_block_size(64, groups)
        assert result == 32

    def test_no_common_block_size_raises(self):
        """No common block size → ValueError."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        b1 = self._make_backend([16])
        b2 = self._make_backend([32])
        groups = [self._make_attn_group(b1), self._make_attn_group(b2)]

        with pytest.raises(ValueError, match="No common block size"):
            RBLNModelRunner.select_common_block_size(64, groups)

    def test_mixed_int_and_multiple_of(self):
        """One backend uses int, another uses MultipleOf — find common size."""
        from vllm.attention.backends.abstract import MultipleOf
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        # kv_manager_block_size = 48
        # b1: [16] (int only), b2: [MultipleOf(16)] (48 % 16 == 0)
        # Case 1: 48 not in b1's int sizes → check MultipleOf: b1 has no MultipleOf
        #   So case 1 fails for b1.
        # Case 2: int sizes = {16}, 48 % 16 == 0
        #   16 supported by b1 (int 16), supported by b2 (MultipleOf(16), 16%16==0)
        b1 = self._make_backend([16])
        b2 = self._make_backend([MultipleOf(16)])
        groups = [self._make_attn_group(b1), self._make_attn_group(b2)]

        result = RBLNModelRunner.select_common_block_size(48, groups)
        assert result == 16

    def test_single_backend_single_size(self):
        """Single backend with single size that matches kv_manager."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        b1 = self._make_backend([16])
        groups = [self._make_attn_group(b1)]

        result = RBLNModelRunner.select_common_block_size(16, groups)
        assert result == 16


class TestGetDpPadding:
    """Test get_dp_padding DP/MoE branching, bucketing."""

    def _make_runner(
        self,
        dp_size=1,
        dp_rank=0,
        specialized_moe_decode=False,
        max_num_batched_tokens=128,
    ):
        from unittest.mock import MagicMock

        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = MagicMock(spec=RBLNModelRunner)
        runner.vllm_config = MagicMock()
        runner.vllm_config.parallel_config.data_parallel_size = dp_size
        runner.vllm_config.parallel_config.data_parallel_rank = dp_rank
        runner.specialized_moe_decode = specialized_moe_decode
        runner.max_num_batched_tokens = max_num_batched_tokens
        runner.bucketing_manager = MagicMock()
        return runner

    def test_non_dp_passthrough(self):
        """dp_size=1 → return batch_bucket_size unchanged, no padded tokens."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = self._make_runner(dp_size=1)
        result = RBLNModelRunner.get_dp_padding(
            runner, num_tokens=10, batch_bucket_size=32
        )
        assert result == (32, None, None)

    def test_non_dp_rejects_padded_tokens(self):
        """dp_size=1 with num_padded_tokens → assertion error."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = self._make_runner(dp_size=1)
        with pytest.raises(AssertionError, match="should not be applied"):
            RBLNModelRunner.get_dp_padding(
                runner,
                num_tokens=10,
                batch_bucket_size=32,
                num_padded_tokens=128,
            )

    @patch("vllm_rbln.v1.worker.rbln_model_runner.RBLNDPMetadata")
    def test_dp_with_moe_padded_tokens(self, mock_dp_meta):
        """DP with specialized MOE decode and num_padded_tokens set."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_dp_meta.num_tokens_across_dp.return_value = torch.tensor(
            [5, 10], dtype=torch.int32
        )

        runner = self._make_runner(
            dp_size=2,
            dp_rank=0,
            specialized_moe_decode=True,
            max_num_batched_tokens=128,
        )
        result = RBLNModelRunner.get_dp_padding(
            runner,
            num_tokens=5,
            batch_bucket_size=32,
            num_padded_tokens=128,
            is_prefill=False,
        )
        batch_bucket, padded, tokens_across = result
        assert batch_bucket == 32
        assert padded == 128
        assert torch.equal(
            tokens_across, torch.tensor([5, 10], dtype=torch.int32)
        )

    @patch("vllm_rbln.v1.worker.rbln_model_runner.RBLNDPMetadata")
    def test_dp_prefill_uses_max_batched_tokens(self, mock_dp_meta):
        """DP prefill (any_prefill=True) → num_padded_tokens = max_num_batched_tokens."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_dp_meta.num_tokens_across_dp_with_max_decode_tokens.return_value = (
            torch.tensor([5, 10], dtype=torch.int32),
            None,  # max_decode_tokens=None → any_prefill=True
        )

        runner = self._make_runner(
            dp_size=2,
            dp_rank=0,
            specialized_moe_decode=False,
            max_num_batched_tokens=256,
        )
        result = RBLNModelRunner.get_dp_padding(
            runner,
            num_tokens=5,
            batch_bucket_size=32,
        )
        batch_bucket, padded, tokens_across = result
        assert batch_bucket == 32
        assert padded == 256  # max_num_batched_tokens

    @patch("vllm_rbln.v1.worker.rbln_model_runner.RBLNDPMetadata")
    def test_dp_decode_no_moe_uses_max_batched_tokens(self, mock_dp_meta):
        """DP decode without specialized MOE → num_padded_tokens = max_num_batched_tokens."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_dp_meta.num_tokens_across_dp_with_max_decode_tokens.return_value = (
            torch.tensor([5, 10], dtype=torch.int32),
            10,  # max_decode_tokens is not None → not any_prefill
        )

        runner = self._make_runner(
            dp_size=2,
            dp_rank=0,
            specialized_moe_decode=False,
            max_num_batched_tokens=256,
        )
        result = RBLNModelRunner.get_dp_padding(
            runner,
            num_tokens=5,
            batch_bucket_size=32,
        )
        _, padded, _ = result
        assert padded == 256  # not specialized_moe → max_num_batched_tokens

    @patch("vllm_rbln.v1.worker.rbln_model_runner.RBLNDPMetadata")
    def test_dp_decode_moe_uses_bucketing(self, mock_dp_meta):
        """DP decode with specialized MOE → uses bucketing manager."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_dp_meta.num_tokens_across_dp_with_max_decode_tokens.return_value = (
            torch.tensor([5, 10], dtype=torch.int32),
            10,  # max_decode_tokens=10
        )

        runner = self._make_runner(
            dp_size=2,
            dp_rank=0,
            specialized_moe_decode=True,
            max_num_batched_tokens=256,
        )
        runner.bucketing_manager.find_decode_batch_bucket.return_value = 16

        result = RBLNModelRunner.get_dp_padding(
            runner,
            num_tokens=5,
            batch_bucket_size=32,
        )
        batch_bucket, padded, _ = result
        assert batch_bucket == 16  # from bucketing
        assert padded == 16
        runner.bucketing_manager.find_decode_batch_bucket.assert_called_once_with(
            10
        )


class TestMayReorderBatch:
    """Test _may_reorder_batch stable sort + cycle swap logic."""

    def _make_runner(self, sort_batch=True, num_kv_cache_groups=1):
        from unittest.mock import MagicMock

        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = MagicMock(spec=RBLNModelRunner)
        runner.kv_cache_config = MagicMock()
        runner.kv_cache_config.kv_cache_groups = [
            MagicMock()
        ] * num_kv_cache_groups
        runner.input_batch = MagicMock()
        return runner

    @patch("vllm_rbln.v1.worker.rbln_model_runner.envs")
    def test_sort_disabled_noop(self, mock_envs):
        """VLLM_RBLN_SORT_BATCH=False → no reordering."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_envs.VLLM_RBLN_SORT_BATCH = False

        runner = self._make_runner()
        scheduler_output = MagicMock()

        RBLNModelRunner._may_reorder_batch(runner, scheduler_output)
        runner.input_batch.swap_states.assert_not_called()

    @patch("vllm_rbln.v1.worker.rbln_model_runner.envs")
    def test_no_kv_cache_groups_noop(self, mock_envs):
        """Zero kv_cache_groups → no reordering."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_envs.VLLM_RBLN_SORT_BATCH = True

        runner = self._make_runner(num_kv_cache_groups=0)
        scheduler_output = MagicMock()

        RBLNModelRunner._may_reorder_batch(runner, scheduler_output)
        runner.input_batch.swap_states.assert_not_called()

    @patch("vllm_rbln.v1.worker.rbln_model_runner.envs")
    def test_already_sorted_no_swaps(self, mock_envs):
        """Already sorted batch (descending num_tokens) → no swaps needed."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_envs.VLLM_RBLN_SORT_BATCH = True

        runner = self._make_runner()
        # 3 requests, already sorted descending by num_tokens
        runner.input_batch.req_ids = ["a", "b", "c"]
        runner.input_batch.num_tokens = np.array([100, 50, 10])

        scheduler_output = MagicMock()
        RBLNModelRunner._may_reorder_batch(runner, scheduler_output)
        runner.input_batch.swap_states.assert_not_called()

    @patch("vllm_rbln.v1.worker.rbln_model_runner.envs")
    def test_unsorted_batch_swaps(self, mock_envs):
        """Unsorted batch → swap_states called to sort descending by num_tokens."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_envs.VLLM_RBLN_SORT_BATCH = True

        runner = self._make_runner()
        # 3 requests: [10, 100, 50] → sorted desc: [100, 50, 10] = indices [1, 2, 0]
        runner.input_batch.req_ids = ["a", "b", "c"]
        runner.input_batch.num_tokens = np.array([10, 100, 50])

        scheduler_output = MagicMock()
        RBLNModelRunner._may_reorder_batch(runner, scheduler_output)
        # Verify that swap_states was called (at least once)
        assert runner.input_batch.swap_states.call_count >= 1

    @patch("vllm_rbln.v1.worker.rbln_model_runner.envs")
    def test_two_element_swap(self, mock_envs):
        """Two elements reversed → exactly one swap to fix."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        mock_envs.VLLM_RBLN_SORT_BATCH = True

        runner = self._make_runner()
        # [10, 50] → sorted desc: [50, 10] = indices [1, 0]
        runner.input_batch.req_ids = ["a", "b"]
        runner.input_batch.num_tokens = np.array([10, 50])

        scheduler_output = MagicMock()
        RBLNModelRunner._may_reorder_batch(runner, scheduler_output)
        # The cycle swap: src=1→dst=0, then src=0→dst=1 (but 1 is now done)
        assert runner.input_batch.swap_states.call_count >= 1


class TestGetPromptLogprobsDict:
    """Test _get_prompt_logprobs_dict chunk boundaries, accumulation, cleanup."""

    def _make_runner(self):
        from unittest.mock import MagicMock

        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = MagicMock(spec=RBLNModelRunner)
        runner.device = torch.device("cpu")
        runner.requests = {}
        runner.input_batch = MagicMock()
        runner.input_batch.in_progress_prompt_logprobs_cpu = {}
        runner.input_batch.req_id_to_index = {}
        runner.query_start_loc = MagicMock()
        runner.model = MagicMock()
        runner.sampler = MagicMock()
        runner._sync_device = MagicMock()
        return runner

    def test_empty_dict_when_no_prompt_logprobs(self):
        """No num_prompt_logprobs → empty dict."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = self._make_runner()
        runner.num_prompt_logprobs = {}

        result = RBLNModelRunner._get_prompt_logprobs_dict(
            runner,
            hidden_states=torch.zeros(10, 8),
            num_scheduled_tokens={"req1": 5},
        )
        assert result == {}

    def test_preempted_request_skipped(self):
        """Request not in num_scheduled_tokens → skipped."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = self._make_runner()
        runner.num_prompt_logprobs = {"req1": 3}

        result = RBLNModelRunner._get_prompt_logprobs_dict(
            runner,
            hidden_states=torch.zeros(10, 8),
            num_scheduled_tokens={},  # req1 not here
        )
        assert result == {}

    def test_completed_request_cleanup(self):
        """Completed prefill → logprobs returned and cleaned up from dicts."""
        from vllm.v1.outputs import LogprobsTensors
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = self._make_runner()
        num_prompt_logprobs = 2  # top-2 logprobs
        runner.num_prompt_logprobs = {"req1": num_prompt_logprobs}

        # Request with 5 prompt tokens, starting from 0
        request = MagicMock()
        request.prompt_token_ids = [10, 20, 30, 40, 50]  # 5 tokens
        request.num_computed_tokens = 0
        runner.requests = {"req1": request}
        runner.input_batch.req_id_to_index = {"req1": 0}

        # query_start_loc for req_idx=0
        runner.query_start_loc.np = np.array([0, 5])

        # num_prompt_tokens=5, start_tok=1, num_remaining=4
        # num_scheduled=5 > 4 → last chunk, num_logits=4
        num_logits = 4
        k = num_prompt_logprobs + 1  # 3 (top-2 + 1 target)

        # Model returns logits, sampler returns logprobs data
        logits = torch.randn(num_logits, 100)
        runner.model.compute_logits.return_value = logits
        runner.sampler.compute_logprobs.return_value = torch.randn(
            num_logits, 100
        )

        # gather_logprobs returns: token_ids(N,K), logprobs(N,K), ranks(N,)
        token_ids = torch.zeros(num_logits, k, dtype=torch.int32)
        logprobs_vals = torch.zeros(num_logits, k, dtype=torch.float32)
        ranks = torch.zeros(num_logits, dtype=torch.int32)
        runner.sampler.gather_logprobs.return_value = (
            token_ids,
            logprobs_vals,
            ranks,
        )

        result = RBLNModelRunner._get_prompt_logprobs_dict(
            runner,
            hidden_states=torch.zeros(10, 8),
            num_scheduled_tokens={"req1": 5},
        )

        assert "req1" in result
        # Cleanup: req1 removed from num_prompt_logprobs and in_progress
        assert "req1" not in runner.num_prompt_logprobs
        assert "req1" not in runner.input_batch.in_progress_prompt_logprobs_cpu
        # _sync_device called because result is non-empty
        runner._sync_device.assert_called_once()

    def test_chunked_request_stays_in_progress(self):
        """Chunked prefill (not all tokens scheduled) → stays in progress."""
        from vllm.v1.outputs import LogprobsTensors
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        runner = self._make_runner()
        num_prompt_logprobs = 2
        runner.num_prompt_logprobs = {"req1": num_prompt_logprobs}

        request = MagicMock()
        request.prompt_token_ids = [10, 20, 30, 40, 50, 60, 70, 80]  # 8 tokens
        request.num_computed_tokens = 0
        runner.requests = {"req1": request}
        runner.input_batch.req_id_to_index = {"req1": 0}
        runner.query_start_loc.np = np.array([0, 3])

        # num_prompt_tokens=8, start_tok=1, num_remaining=7
        # num_scheduled=3 <= 7 → chunk, num_logits=3
        num_logits = 3
        k = num_prompt_logprobs + 1  # 3

        logits = torch.randn(num_logits, 100)
        runner.model.compute_logits.return_value = logits
        runner.sampler.compute_logprobs.return_value = torch.randn(
            num_logits, 100
        )

        token_ids = torch.zeros(num_logits, k, dtype=torch.int32)
        logprobs_vals = torch.zeros(num_logits, k, dtype=torch.float32)
        ranks = torch.zeros(num_logits, dtype=torch.int32)
        runner.sampler.gather_logprobs.return_value = (
            token_ids,
            logprobs_vals,
            ranks,
        )

        result = RBLNModelRunner._get_prompt_logprobs_dict(
            runner,
            hidden_states=torch.zeros(10, 8),
            num_scheduled_tokens={"req1": 3},
        )

        # Not completed → not in result, but still in num_prompt_logprobs
        assert "req1" not in result
        assert "req1" in runner.num_prompt_logprobs
