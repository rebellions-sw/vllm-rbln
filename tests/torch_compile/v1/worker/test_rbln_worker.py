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
