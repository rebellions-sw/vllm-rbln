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

from vllm.v1.worker.worker_base import WorkerBase
from vllm_rbln.worker.optimum_model_runner_v1 import RBLNOptimumModelRunner
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger

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

        self.model_runner = RBLNOptimumModelRunner(self.vllm_config, device)
        self.profiler = None

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    # FIXME(eunji) In V0, we returned both num_gpu_blocks and num_cpu_blocks.
    # Now we can only return num_gpu_blocks
    def determine_available_memory(self) -> int:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        """
        attn_impl = self.model_runner.model.model.get_attn_impl() if hasattr(
            self.model_runner.model.model, "get_attn_impl") else None

        if attn_impl is not None and attn_impl == "flash_attn":
            # We use the last block as dummy block
            num_gpu_blocks = (
                self.model_runner.model.model.get_kvcache_num_blocks() - 1)

            if npu_num_blocks := os.environ.get("VLLM_RBLN_NPU_NUM_BLOCKS"):
                num_gpu_blocks = int(npu_num_blocks) - 1
        else:
            # Set the number of GPU blocks to be the same as the maximum
            # number of sequences that can be processed in a single batch.
            # This is equivalent to schedule without PagedAttention.
            num_gpu_blocks = self.scheduler_config.max_num_seqs

        return num_gpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.
        """

        # Different values are not tested.
        assert num_cpu_blocks == 0

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    # TODO(eunji) implment following V1
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        self.model_runner.execute_model()

    def profile(self, is_start: bool = True):
        raise RuntimeError("Profiler is not enabled.")
    
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise RuntimeError("LoRA is not enabled.")

    def load_model(self):
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        raise RuntimeError("Compilation in vLLM is not supported yet.")

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def init_distributed_environment(self):
        """RBLN uses rebel-compiler for tensor parallelism.

        vLLM still needs the environment inited when TP/PP > 1
        """
        init_distributed_environment(
            world_size=1,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            1,
            1,
        )
