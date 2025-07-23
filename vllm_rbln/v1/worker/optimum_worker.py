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
from typing import Optional

import torch
import torch.distributed
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.v1.core.kv_cache_utils import get_uniform_page_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

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

        self.model_runner = RBLNOptimumModelRunner(self.vllm_config,
                                                   self.device)
        self.profiler = None

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """It follows the way to calculate num_blocks in vLLM.
        """
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        num_layers = len(kv_cache_spec)
        page_size = get_uniform_page_size(kv_cache_spec)

        attn_impl = self.model_runner.model.model.get_attn_impl() if hasattr(
            self.model_runner.model.model, "get_attn_impl") else None

        if attn_impl is not None and attn_impl == "flash_attn":
            # We use the last block as dummy block
            num_gpu_blocks = (
                self.model_runner.model.model.get_kvcache_num_blocks() - 1)

            if npu_num_blocks := os.environ.get("VLLM_RBLN_NPU_NUM_BLOCKS"):
                num_gpu_blocks = int(npu_num_blocks) - 1

        else:
            num_gpu_blocks = self.scheduler_config.max_num_seqs

        # NOTE vLLM tried to leave a dummy block before execution.
        # It prevents vLLM from # of blocks == 0 in case of batch size is 1.
        if num_gpu_blocks == 1:
            num_gpu_blocks = 2
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

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise RuntimeError("LoRA is not enabled.")

    def load_model(self):
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        # TODO(eunji): warmup is required?
        # self.model_runner.warming_up_model()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError
