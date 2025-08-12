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
"""A RBLN worker class."""
from typing import List, Optional, Tuple

import torch
import torch.distributed
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoRANotSupportedWorkerBase, WorkerBase,
                                     WorkerInput)

from vllm_rbln.worker.optimum_model_runner import RBLNOptimumModelRunner

logger = init_logger(__name__)


class RBLNOptimumWorker(LoRANotSupportedWorkerBase,
                        LocalOrDistributedWorkerBase):
    """A worker class that executes the model on RBLN NPUs via optimum-rbln.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.model_runner: RBLNOptimumModelRunner = RBLNOptimumModelRunner(
            vllm_config=vllm_config)

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        """
        adapter = self.model_runner.model.kv_block_adapter
        num_gpu_blocks = adapter.get_available_num_blocks()
        # Swap not yet supported with RBLN backend.
        num_cpu_blocks = 0
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.
        """

        # Different values are not tested.
        assert num_cpu_blocks == 0

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    @property
    def do_metadata_broadcast(self) -> bool:
        return False

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        return WorkerInput(num_seq_groups=len(
            execute_model_req.seq_group_metadata_list), )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

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
