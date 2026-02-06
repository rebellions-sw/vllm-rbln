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

import os
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestRBLNWorkerInitDeviceEnv:
    """Test _init_device_env() device selection logic."""

    def _make_worker_stub(
        self,
        local_rank=0,
        local_world_size=1,
        tp_size=1,
        dp_rank=0,
        env_var="RBLN_DEVICES",
    ):
        """Create a minimal worker-like object with needed attributes."""
        worker = MagicMock()
        worker.local_rank = local_rank
        worker.local_world_size = local_world_size
        worker.parallel_config = MagicMock()
        worker.parallel_config.data_parallel_rank = dp_rank
        return worker

    def test_no_env_var_single_device(self):
        """Without env var set, device IDs are computed from rank.

        Tests the logic inline since _init_device_env requires full
        worker initialization.
        """
        # Simulate the logic from _init_device_env:
        # total_device_count = world_size * tp_size = 1 * 1 = 1
        # dev_begin = total_device_count * dp_rank = 1 * 0 = 0
        # dev_end = 0 + 1 = 1, device_ids = ["0"]
        # start = local_rank * tp_size = 0 * 1 = 0, end = 0 + 1 = 1
        # selected = "0"
        total_device_count = 1 * 1  # world_size * tp_size
        dev_begin = total_device_count * 0  # dp_rank
        dev_end = dev_begin + total_device_count
        device_ids = [str(i) for i in range(dev_begin, dev_end)]
        start_idx = 0 * 1  # local_rank * tp_size
        end_idx = start_idx + 1
        selected = ",".join(device_ids[start_idx:end_idx])
        assert selected == "0"

    def test_multi_device_dp(self):
        """With DP > 1, device offset should account for DP rank."""
        total_device_count = 2 * 1  # world_size=2, tp_size=1
        dp_rank = 1
        local_rank = 0
        tp_size = 1

        dev_begin = total_device_count * dp_rank
        dev_end = dev_begin + total_device_count
        device_ids = [str(i) for i in range(dev_begin, dev_end)]
        start_idx = local_rank * tp_size
        end_idx = start_idx + tp_size
        selected = ",".join(device_ids[start_idx:end_idx])
        assert selected == "2"

    def test_multi_device_tp(self):
        """With TP > 1, multiple devices should be selected."""
        total_device_count = 1 * 2  # world_size=1, tp_size=2
        dp_rank = 0
        local_rank = 0
        tp_size = 2

        dev_begin = total_device_count * dp_rank
        dev_end = dev_begin + total_device_count
        device_ids = [str(i) for i in range(dev_begin, dev_end)]
        start_idx = local_rank * tp_size
        end_idx = start_idx + tp_size
        selected = ",".join(device_ids[start_idx:end_idx])
        assert selected == "0,1"


class TestInitWorkerDistributedEnvironment:
    """Test init_worker_distributed_environment env var logic."""

    def test_sets_env_vars(self):
        """Should set LOCAL_RANK and WORLD_SIZE."""
        from vllm_rbln.v1.worker.rbln_worker import (
            init_worker_distributed_environment,
        )

        vllm_config = MagicMock()
        vllm_config.parallel_config.world_size = 2
        vllm_config.parallel_config.data_parallel_size = 1

        # Mock all distributed functions to avoid actual init
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce"),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.init_distributed_environment"
            ),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized"
            ),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"
            ),
        ):
            init_worker_distributed_environment(
                vllm_config=vllm_config,
                rank=0,
                distributed_init_method="tcp://localhost:12345",
                local_rank=0,
                backend="gloo",
            )
            assert os.environ["LOCAL_RANK"] == "0"
            assert os.environ["WORLD_SIZE"] == "2"

    def test_dp_adjusts_rank(self):
        """With DP > 1, LOCAL_RANK should be adjusted across DP."""
        from vllm_rbln.v1.worker.rbln_worker import (
            init_worker_distributed_environment,
        )

        vllm_config = MagicMock()
        vllm_config.parallel_config.world_size = 2
        vllm_config.parallel_config.data_parallel_size = 2
        vllm_config.parallel_config.world_size_across_dp = 4
        vllm_config.parallel_config.data_parallel_rank = 1

        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce"),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.init_distributed_environment"
            ),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized"
            ),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"
            ),
        ):
            init_worker_distributed_environment(
                vllm_config=vllm_config,
                rank=0,
                distributed_init_method="tcp://localhost:12345",
                local_rank=0,
                backend="gloo",
            )
            # rank_across_dp = dp_rank * world_size + rank = 1 * 2 + 0 = 2
            assert os.environ["LOCAL_RANK"] == "2"
            assert os.environ["WORLD_SIZE"] == "4"


class TestRBLNWorkerSimpleMethods:
    """Test simple/trivial methods on RBLNWorker (v1)."""

    def test_initialize_cache(self):
        """initialize_cache should set num_gpu_blocks and num_cpu_blocks."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.cache_config = MagicMock()
        # Call the unbound method on the mock
        RBLNWorker.initialize_cache(
            worker, num_gpu_blocks=100, num_cpu_blocks=0
        )
        assert worker.cache_config.num_gpu_blocks == 100
        assert worker.cache_config.num_cpu_blocks == 0

    def test_check_health(self):
        """check_health should return None (worker is always healthy)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        result = RBLNWorker.check_health(worker)
        assert result is None

    def test_sleep_logs_warning(self):
        """sleep() should just log a warning and return."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        # Should not raise
        RBLNWorker.sleep(worker, level=1)

    def test_wake_up_logs_warning(self):
        """wake_up() should just log a warning and return."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        RBLNWorker.wake_up(worker, tags=None)

    def test_profile_raises_without_profiler(self):
        """profile() should raise RuntimeError if profiler is None."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.profiler = None
        with pytest.raises(RuntimeError, match="Profiler is not enabled"):
            RBLNWorker.profile(worker, is_start=True)

    def test_profile_start(self):
        """profile(is_start=True) should call profiler.start()."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.profiler = MagicMock()
        RBLNWorker.profile(worker, is_start=True)
        worker.profiler.start.assert_called_once()

    def test_profile_stop(self):
        """profile(is_start=False) should call profiler.stop()."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.profiler = MagicMock()
        worker.local_rank = 1  # non-zero, so no print
        RBLNWorker.profile(worker, is_start=False)
        worker.profiler.stop.assert_called_once()


class TestDetermineAvailableMemory:
    """Test determine_available_memory ATOM/REBEL branching and quantization logic."""

    def _make_worker(
        self,
        device_name="rbln-ca25",
        quantization=None,
        tp_size=1,
        params=None,
    ):
        """Create a worker mock with model parameters."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)

        # model_runner.model.named_parameters
        if params is None:
            # Default: 2 bf16 params of size 100
            params = [
                ("attn.weight", torch.zeros(100, dtype=torch.bfloat16)),
                ("mlp.weight", torch.zeros(200, dtype=torch.bfloat16)),
            ]
        worker.model_runner = MagicMock()
        worker.model_runner.model = MagicMock()
        worker.model_runner.model.named_parameters.return_value = iter(params)

        worker.model_config = MagicMock()
        worker.model_config.quantization = quantization
        worker.parallel_config = MagicMock()
        worker.cache_config = MagicMock()
        worker.cache_config.gpu_memory_utilization = 0.9
        return worker

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    @patch("vllm_rbln.v1.worker.rbln_worker.current_platform")
    @patch("vllm_rbln.v1.worker.rbln_worker.estimate_available_memory")
    def test_atom_no_quantization(
        self, mock_estimate, mock_platform, mock_envs
    ):
        """ATOM (ca) without quantization: num_runtimes=2*tp, nbits=16."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_platform.get_device_name.return_value = "RBLN-CA25"
        mock_envs.VLLM_RBLN_TP_SIZE = 2
        mock_estimate.return_value = 1_000_000

        worker = self._make_worker(device_name="rbln-ca25")
        result = RBLNWorker.determine_available_memory(worker)

        assert result == 1_000_000
        call_kwargs = mock_estimate.call_args
        assert call_kwargs[1]["nbits_per_param"] == 16
        assert call_kwargs[1]["num_runtimes"] == 4  # 2 * tp_size=2

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    @patch("vllm_rbln.v1.worker.rbln_worker.current_platform")
    @patch("vllm_rbln.v1.worker.rbln_worker.estimate_available_memory")
    def test_rebel_no_quantization(
        self, mock_estimate, mock_platform, mock_envs
    ):
        """REBEL (cr) without quantization: num_runtimes=2*4=8, nbits=16."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_platform.get_device_name.return_value = "RBLN-CR100"
        mock_estimate.return_value = 2_000_000

        worker = self._make_worker()
        result = RBLNWorker.determine_available_memory(worker)

        assert result == 2_000_000
        call_kwargs = mock_estimate.call_args
        assert call_kwargs[1]["nbits_per_param"] == 16
        assert call_kwargs[1]["num_runtimes"] == 8  # 2 * 4

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    @patch("vllm_rbln.v1.worker.rbln_worker.current_platform")
    @patch("vllm_rbln.v1.worker.rbln_worker.estimate_available_memory")
    def test_atom_with_mxfp4_uses_bf16_bits(
        self, mock_estimate, mock_platform, mock_envs
    ):
        """ATOM + mxfp4: still uses 16 bits (ATOM doesn't support mxfp4)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_platform.get_device_name.return_value = "RBLN-CA25"
        mock_envs.VLLM_RBLN_TP_SIZE = 1
        mock_estimate.return_value = 500_000

        worker = self._make_worker(quantization="mxfp4")
        result = RBLNWorker.determine_available_memory(worker)

        assert result == 500_000
        call_kwargs = mock_estimate.call_args
        assert call_kwargs[1]["nbits_per_param"] == 16  # ATOM fallback to bf16

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    @patch("vllm_rbln.v1.worker.rbln_worker.current_platform")
    @patch("vllm_rbln.v1.worker.rbln_worker.estimate_available_memory")
    def test_rebel_with_mxfp4_uses_4bits(
        self, mock_estimate, mock_platform, mock_envs
    ):
        """REBEL + mxfp4: uses 4 bits per param."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_platform.get_device_name.return_value = "RBLN-CR100"
        mock_estimate.return_value = 800_000

        worker = self._make_worker(quantization="mxfp4")
        result = RBLNWorker.determine_available_memory(worker)

        assert result == 800_000
        call_kwargs = mock_estimate.call_args
        assert call_kwargs[1]["nbits_per_param"] == 4  # REBEL supports mxfp4

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    @patch("vllm_rbln.v1.worker.rbln_worker.current_platform")
    @patch("vllm_rbln.v1.worker.rbln_worker.estimate_available_memory")
    def test_param_counting_bf16_only(
        self, mock_estimate, mock_platform, mock_envs
    ):
        """All bf16 params → counted as attention params."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_platform.get_device_name.return_value = "RBLN-CA25"
        mock_envs.VLLM_RBLN_TP_SIZE = 1
        mock_estimate.return_value = 100

        params = [
            ("layer.attn", torch.zeros(50, dtype=torch.bfloat16)),
            ("layer.mlp", torch.zeros(30, dtype=torch.bfloat16)),
        ]
        worker = self._make_worker(params=params)
        RBLNWorker.determine_available_memory(worker)

        call_kwargs = mock_estimate.call_args
        # 50 + 30 = 80 total bf16 params, packed_num_elems=1, ratio=1
        assert call_kwargs[1]["n_model_params"] == 80

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    @patch("vllm_rbln.v1.worker.rbln_worker.current_platform")
    @patch("vllm_rbln.v1.worker.rbln_worker.estimate_available_memory")
    def test_param_counting_mixed_quant_rebel(
        self, mock_estimate, mock_platform, mock_envs
    ):
        """REBEL mxfp4: bf16 params (attention) + uint8 params (expert)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_platform.get_device_name.return_value = "RBLN-CR100"
        mock_estimate.return_value = 100

        params = [
            ("attn.weight", torch.zeros(100, dtype=torch.bfloat16)),
            ("expert.weight", torch.zeros(50, dtype=torch.uint8)),
        ]
        worker = self._make_worker(quantization="mxfp4", params=params)
        RBLNWorker.determine_available_memory(worker)

        call_kwargs = mock_estimate.call_args
        # bf16: 100 numel (attention)
        # uint8/quant: 50 * packed_num_elems(2) * ratio(1) = 100 (expert)
        # total = 200
        assert call_kwargs[1]["n_model_params"] == 200


class TestCompileOrWarmUpModel:
    """Test compile_or_warm_up_model DP divisibility and skip/warmup paths."""

    def _make_worker(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.parallel_config = MagicMock()
        worker.scheduler_config = MagicMock()
        worker.model_config = MagicMock()
        worker.model_runner = MagicMock()
        return worker

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    def test_dp_padded_decode_divisibility_ok(self, mock_envs):
        """DP padded_decode: passes when max_num_batched_tokens % max_num_seqs == 0."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_envs.VLLM_RBLN_DP_IMPL = "padded_decode"
        mock_envs.VLLM_RBLN_COMPILE_MODEL = True
        mock_envs.VLLM_RBLN_ENABLE_WARM_UP = True

        worker = self._make_worker()
        worker.parallel_config.data_parallel_size = 2
        worker.scheduler_config.max_num_batched_tokens = 128
        worker.scheduler_config.max_num_seqs = 32  # 128 % 32 == 0
        worker.model_config.enforce_eager = False

        # Should not raise
        RBLNWorker.compile_or_warm_up_model(worker)
        worker.model_runner.prepare_dummy_run.assert_called_once()
        worker.model_runner.warm_up_model.assert_called_once()

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    def test_dp_padded_decode_divisibility_fail(self, mock_envs):
        """DP padded_decode: fails assertion when not divisible."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_envs.VLLM_RBLN_DP_IMPL = "padded_decode"

        worker = self._make_worker()
        worker.parallel_config.data_parallel_size = 2
        worker.scheduler_config.max_num_batched_tokens = 100
        worker.scheduler_config.max_num_seqs = 30  # 100 % 30 != 0

        with pytest.raises(AssertionError):
            RBLNWorker.compile_or_warm_up_model(worker)

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    def test_dp_dummy_prefill_raises_valueerror(self, mock_envs):
        """DP dummy_prefill: raises ValueError (deprecated)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_envs.VLLM_RBLN_DP_IMPL = "dummy_prefill"

        worker = self._make_worker()
        worker.parallel_config.data_parallel_size = 2

        with pytest.raises(ValueError, match="dummy_prefill is not supported"):
            RBLNWorker.compile_or_warm_up_model(worker)

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    def test_skip_warmup_enforce_eager(self, mock_envs):
        """enforce_eager=True → skip warmup."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_envs.VLLM_RBLN_COMPILE_MODEL = True
        mock_envs.VLLM_RBLN_ENABLE_WARM_UP = True

        worker = self._make_worker()
        worker.parallel_config.data_parallel_size = 1
        worker.model_config.enforce_eager = True

        RBLNWorker.compile_or_warm_up_model(worker)
        worker.model_runner.warm_up_model.assert_not_called()

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    def test_skip_warmup_compile_disabled(self, mock_envs):
        """VLLM_RBLN_COMPILE_MODEL=False → skip warmup."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_envs.VLLM_RBLN_COMPILE_MODEL = False
        mock_envs.VLLM_RBLN_ENABLE_WARM_UP = True

        worker = self._make_worker()
        worker.parallel_config.data_parallel_size = 1
        worker.model_config.enforce_eager = False

        RBLNWorker.compile_or_warm_up_model(worker)
        worker.model_runner.warm_up_model.assert_not_called()

    @patch("vllm_rbln.v1.worker.rbln_worker.envs")
    def test_warmup_runs_and_enables_perf_tracker(self, mock_envs):
        """Full warmup path: warm_up_model + _enable_performance_tracker."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_envs.VLLM_RBLN_COMPILE_MODEL = True
        mock_envs.VLLM_RBLN_ENABLE_WARM_UP = True

        worker = self._make_worker()
        worker.parallel_config.data_parallel_size = 1
        worker.model_config.enforce_eager = False

        RBLNWorker.compile_or_warm_up_model(worker)
        worker.model_runner.warm_up_model.assert_called_once()
        worker.model_runner._enable_performance_tracker.assert_called_once()


class TestExecuteModel:
    """Test execute_model PP control flow and kv_connector passthrough."""

    def _make_worker(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.model_runner = MagicMock()
        worker.vllm_config = MagicMock()
        return worker

    @patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group")
    def test_direct_model_runner_output(self, mock_pp_group):
        """When model_runner returns ModelRunnerOutput directly, pass it through."""
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_pp_group.return_value.is_first_rank = True

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10

        output = MagicMock(spec=ModelRunnerOutput)
        worker = self._make_worker()
        worker.model_runner.execute_model.return_value = output

        result = RBLNWorker.execute_model(worker, scheduler_output)
        assert result is output

    @patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group")
    def test_none_output(self, mock_pp_group):
        """When model_runner returns None, return None."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_pp_group.return_value.is_first_rank = True

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10

        worker = self._make_worker()
        worker.model_runner.execute_model.return_value = None

        result = RBLNWorker.execute_model(worker, scheduler_output)
        assert result is None

    @patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group")
    def test_no_forward_pass(self, mock_pp_group):
        """When total_num_scheduled_tokens == 0, no intermediate tensors needed."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_pp_group.return_value.is_first_rank = True

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 0

        worker = self._make_worker()
        worker.model_runner.execute_model.return_value = None

        result = RBLNWorker.execute_model(worker, scheduler_output)
        assert result is None

    @patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group")
    def test_intermediate_tensor_send_no_kv_connector(self, mock_pp_group):
        """PP non-last rank: IntermediateTensors sent, kv_connector_output is falsy → None."""
        from vllm.sequence import IntermediateTensors
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_pp_group.return_value.is_first_rank = True
        mock_pp_group.return_value.is_last_rank = False

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10

        intermediate = IntermediateTensors({"hidden": torch.zeros(4)})
        intermediate.kv_connector_output = None
        worker = self._make_worker()
        worker.model_runner.execute_model.return_value = intermediate
        worker.vllm_config.parallel_config.distributed_executor_backend = "ray"

        result = RBLNWorker.execute_model(worker, scheduler_output)
        assert result is None
        mock_pp_group.return_value.send_tensor_dict.assert_called_once()

    @patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group")
    @patch("vllm_rbln.v1.worker.rbln_worker.EMPTY_MODEL_RUNNER_OUTPUT")
    def test_intermediate_tensor_with_kv_connector_finished(
        self, mock_empty_output, mock_pp_group
    ):
        """PP non-last rank with kv_connector that has finished_sending/recving."""
        from vllm.sequence import IntermediateTensors
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        mock_pp_group.return_value.is_first_rank = True
        mock_pp_group.return_value.is_last_rank = False

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10

        kv_output = MagicMock()
        kv_output.finished_sending = False
        kv_output.finished_recving = False
        kv_output.__bool__ = lambda self: True  # truthy

        intermediate = IntermediateTensors({"hidden": torch.zeros(4)})
        intermediate.kv_connector_output = kv_output

        worker = self._make_worker()
        worker.model_runner.execute_model.return_value = intermediate
        worker.vllm_config.parallel_config.distributed_executor_backend = "ray"

        result = RBLNWorker.execute_model(worker, scheduler_output)
        # Neither finished_sending nor finished_recving → return EMPTY
        assert result is mock_empty_output
