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
"""Tests for RBLNWorker."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_rbln.v1.worker.rbln_worker import (
    RBLNWorker, init_worker_distributed_environment)

BLOCK_SIZE = 1024
DEVICE = current_platform.device_type


def get_vllm_config():
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def worker():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config), patch.object(
            RBLNWorker, '_init_device_env'):
        w = RBLNWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:18080",
        )
        # Mock model_runner
        w.model_runner = MagicMock()
        return w


@pytest.fixture
def worker_with_profiler():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config), patch.object(
            RBLNWorker, '_init_device_env'), patch(
                'vllm_rbln.rbln_envs.VLLM_TORCH_PROFILER_DIR',
                '/tmp/profiler'):
        w = RBLNWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:18080",
        )
        return w


def test_init_creates_worker_with_correct_attributes():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config), patch.object(
            RBLNWorker, '_init_device_env'):
        worker = RBLNWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
            is_driver_worker=True,
        )

        assert worker.local_rank == 0
        assert worker.rank == 0
        assert worker.is_driver_worker is True
        assert worker.device == torch.device(current_platform.device_type)
        assert worker.parallel_config.disable_custom_all_reduce is True
        assert worker.profiler is None


def test_init_with_ray_backend_skips_device_env_setup():
    vllm_config = get_vllm_config()
    vllm_config.parallel_config.distributed_executor_backend = "ray"

    with set_current_vllm_config(vllm_config), patch.object(
            RBLNWorker, '_init_device_env') as mock_init_device:
        worker = RBLNWorker(  # noqa: F841
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
        )
        # _init_device_env should not be called for Ray backend
        mock_init_device.assert_not_called()


def test_initialize_cache_sets_block_counts(worker):
    num_gpu_blocks = 100
    num_cpu_blocks = 50

    worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    assert worker.cache_config.num_gpu_blocks == num_gpu_blocks
    assert worker.cache_config.num_cpu_blocks == num_cpu_blocks


def test_load_model_calls_model_runner(worker):
    worker.load_model()
    worker.model_runner.load_model.assert_called_once()


def test_get_model_returns_model_runner_model(worker):
    mock_model = MagicMock()
    worker.model_runner.get_model.return_value = mock_model

    result = worker.get_model()

    assert result == mock_model
    worker.model_runner.get_model.assert_called_once()


def test_get_supported_tasks_delegates_to_model_runner(worker):
    expected_tasks = ("generate", )
    worker.model_runner.get_supported_tasks.return_value = expected_tasks

    result = worker.get_supported_tasks()

    assert result == expected_tasks


def test_get_kv_cache_spec_delegates_to_model_runner(worker):
    expected_spec = {"layer.0": MagicMock()}
    worker.model_runner.get_kv_cache_spec.return_value = expected_spec

    result = worker.get_kv_cache_spec()

    assert result == expected_spec


def test_add_lora_delegates_to_model_runner(worker):
    lora_request = MagicMock()
    worker.model_runner.add_lora.return_value = True

    result = worker.add_lora(lora_request)

    assert result is True
    worker.model_runner.add_lora.assert_called_once_with(lora_request)


def test_remove_lora_delegates_to_model_runner(worker):
    worker.model_runner.remove_lora.return_value = True

    result = worker.remove_lora(lora_id=1)

    assert result is True
    worker.model_runner.remove_lora.assert_called_once_with(1)


def test_list_loras_delegates_to_model_runner(worker):
    expected_loras = {1, 2, 3}
    worker.model_runner.list_loras.return_value = expected_loras

    result = worker.list_loras()

    assert result == expected_loras


def test_pin_lora_delegates_to_model_runner(worker):
    worker.model_runner.pin_lora.return_value = True

    result = worker.pin_lora(lora_id=1)

    assert result is True
    worker.model_runner.pin_lora.assert_called_once_with(1)


def test_profile_raises_when_profiler_not_enabled(worker):
    with pytest.raises(RuntimeError, match="Profiler is not enabled"):
        worker.profile(is_start=True)


def test_profile_start_calls_profiler_start(worker_with_profiler):
    worker_with_profiler.profiler = MagicMock()
    worker_with_profiler.profile(is_start=True)
    worker_with_profiler.profiler.start.assert_called_once()


def test_profile_stop_calls_profiler_stop(worker_with_profiler):
    worker_with_profiler.profiler = MagicMock()
    worker_with_profiler.profile(is_start=False)
    worker_with_profiler.profiler.stop.assert_called_once()


def test_check_health_returns_none(worker):
    result = worker.check_health()
    assert result is None


def test_execute_dummy_batch_returns_none(worker):
    result = worker.execute_dummy_batch()
    assert result is None


def test_compile_or_warm_up_skips_when_enforce_eager(worker):
    worker.model_config.enforce_eager = True

    worker.compile_or_warm_up_model()
    worker.model_runner.warm_up_model.assert_not_called()


def test_compile_or_warm_up_calls_warm_up_when_enabled(worker):
    worker.model_config.enforce_eager = False

    with patch('vllm_rbln.rbln_envs.VLLM_RBLN_COMPILE_MODEL',
               True), patch('vllm_rbln.rbln_envs.VLLM_RBLN_ENABLE_WARM_UP',
                            True):
        worker.compile_or_warm_up_model()

    worker.model_runner.warm_up_model.assert_called_once()


def test_execute_model_with_model_runner_output(worker):
    from vllm.v1.outputs import ModelRunnerOutput

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    mock_output = MagicMock(spec=ModelRunnerOutput)
    worker.model_runner.execute_model.return_value = mock_output

    with patch(
            'vllm.distributed.parallel_state.get_pp_group') as mock_pp_group:
        mock_pp_group.return_value.is_first_rank = True
        result = worker.execute_model(scheduler_output)

    assert result == mock_output


def test_sets_environment_variables():
    vllm_config = get_vllm_config()

    with patch(
            'vllm_rbln.v1.worker.rbln_worker.init_distributed_environment'
    ), patch('vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce'), patch(
            'vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized'
    ), patch('vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized'):
        init_worker_distributed_environment(
            vllm_config=vllm_config,
            rank=0,
            distributed_init_method="tcp://localhost:18080",
            local_rank=0,
            backend="gloo",
        )

        assert os.environ.get('LOCAL_RANK') == '0'
        assert os.environ.get('WORLD_SIZE') == '1'


def test_data_parallel_sets_correct_env_vars():
    vllm_config = get_vllm_config()
    vllm_config.parallel_config.data_parallel_size = 2

    with patch(
            'vllm_rbln.v1.worker.rbln_worker.init_distributed_environment'
    ), patch('vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce'), patch(
            'vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized'
    ), patch('vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized'):
        init_worker_distributed_environment(
            vllm_config=vllm_config,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
            local_rank=0,
            backend="gloo",
        )

        assert 'LOCAL_RANK' in os.environ
        assert 'WORLD_SIZE' in os.environ
