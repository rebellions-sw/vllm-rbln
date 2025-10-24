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

from typing import Any, Optional

import torch
import torch.distributed
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.core.kv_cache_utils import get_uniform_page_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

import vllm_rbln.rbln_envs as envs
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

logger = init_logger(__name__)


class RBLNOptimumWorker(WorkerBase):

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
            # Additional parameters for compatibility with vllm
            **kwargs):
        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.profiler = None

    def init_device(self) -> None:
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        self.device = self.vllm_config.device_config.device
        self.model_runner = RBLNOptimumModelRunner(self.vllm_config,
                                                   self.device)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """It follows the way to calculate num_blocks in vLLM.
        """
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        num_layers = len(kv_cache_spec)
        page_size = get_uniform_page_size(kv_cache_spec)

        adapter = self.model_runner.model.kv_block_adapter
        num_gpu_blocks = adapter.get_available_num_blocks()

        return num_gpu_blocks * page_size * num_layers

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        # TODO setting intermediate_tensors for PP

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)
        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        raise RuntimeError("Profiler is not enabled.")

    def load_model(self):
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        if not envs.VLLM_RBLN_ENABLE_WARM_UP:
            logger.info(
                "Warm up is disabled. " \
                "Set VLLM_RBLN_ENABLE_WARM_UP=1 to enable warm up."
            )
            return

        logger.info("Running dummy warm up.")
        self.model_runner.dummy_sampler_run()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        pass

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks."""
        pass

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise RuntimeError("It is not required in vLLM RBLN.")

    def remove_lora(self, lora_id: int) -> bool:
        raise RuntimeError("It is not required in vLLM RBLN.")

    def list_loras(self) -> set[int]:
        rbln_cfg = getattr(self.model_runner.model.model, "rbln_config", None)
        lora_cfg = getattr(rbln_cfg, "lora_config", None)
        if lora_cfg is None:
            raise ValueError("The model is not compiled with LoRA.")

        lora_adapters = getattr(lora_cfg, "adapters", [])

        adapter_ids = set(a.lora_int_id for a in lora_adapters)
        return adapter_ids

    def pin_lora(self, lora_id: int) -> bool:
        raise RuntimeError("It is not required in vLLM RBLN.")


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(
        world_size=1,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="gloo",
    )
    ensure_model_parallel_initialized(
        1,
        1,
    )
