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

"""Tests for v0 worker (vllm_rbln.worker.worker).

NOTE: The v0 worker imports vllm.attention.get_attn_backend which was
removed in vLLM v0.13. Tests skip if the import fails.
"""

import importlib

import pytest

_v0_worker_available = True
try:
    _mod = importlib.import_module("vllm_rbln.worker.worker")
except ImportError:
    _v0_worker_available = False

pytestmark = pytest.mark.skipif(
    not _v0_worker_available,
    reason="v0 worker requires vllm.attention.get_attn_backend (removed in v0.13)",
)

from unittest.mock import MagicMock, patch

import torch


class TestRBLNCacheEngineSwap:
    """RBLNCacheEngine should raise NotImplementedError for swap ops."""

    def test_swap_in_raises(self):
        from vllm_rbln.worker.worker import RBLNCacheEngine

        engine = MagicMock(spec=RBLNCacheEngine)
        with pytest.raises(NotImplementedError, match="Swap is not supported"):
            RBLNCacheEngine.swap_in(engine, {0: 1})

    def test_swap_out_raises(self):
        from vllm_rbln.worker.worker import RBLNCacheEngine

        engine = MagicMock(spec=RBLNCacheEngine)
        with pytest.raises(NotImplementedError, match="Swap is not supported"):
            RBLNCacheEngine.swap_out(engine, {0: 1})


class TestRBLNCacheEngineGetCacheBlockSize:
    """Test the static get_cache_block_size calculation."""

    def test_basic_calculation(self):
        from vllm_rbln.worker.worker import RBLNCacheEngine, RBLNWorker

        model_config = MagicMock()
        model_config.get_head_size.return_value = 64
        model_config.get_num_kv_heads.return_value = 8
        model_config.get_total_num_kv_heads.return_value = 8
        model_config.get_num_layers.return_value = 2

        parallel_config = MagicMock()
        parallel_config.enable_expert_parallel = False

        with patch.object(RBLNWorker, "disable_tp", False):
            block_size_bytes = RBLNCacheEngine.get_cache_block_size(
                block_size=16,
                cache_dtype="auto",
                model_config=model_config,
                parallel_config=parallel_config,
            )
        assert block_size_bytes == 131072


class TestRBLNWorkerV0Properties:
    """Test v0 RBLNWorker properties and simple methods."""

    def test_do_metadata_broadcast_tp1(self):
        from vllm_rbln.worker.worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.parallel_config = MagicMock()
        worker.parallel_config.tensor_parallel_size = 1
        result = RBLNWorker.do_metadata_broadcast.fget(worker)
        assert result is False

    def test_do_metadata_broadcast_tp2(self):
        from vllm_rbln.worker.worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.parallel_config = MagicMock()
        worker.parallel_config.tensor_parallel_size = 2
        result = RBLNWorker.do_metadata_broadcast.fget(worker)
        assert result is True

    def test_kv_cache_returns_cpu_cache(self):
        from vllm_rbln.worker.worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        fake_cache = [[torch.zeros(1)]]
        worker.cpu_cache = fake_cache
        result = RBLNWorker.kv_cache.fget(worker)
        assert result is fake_cache

    def test_start_profile_raises_without_profiler(self):
        from vllm_rbln.worker.worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.profiler = None
        with pytest.raises(RuntimeError, match="Profiler is not enabled"):
            RBLNWorker.start_profile(worker)

    def test_stop_profile_raises_without_profiler(self):
        from vllm_rbln.worker.worker import RBLNWorker

        worker = MagicMock(spec=RBLNWorker)
        worker.profiler = None
        with pytest.raises(RuntimeError, match="Profiler is not enabled"):
            RBLNWorker.stop_profile(worker)
