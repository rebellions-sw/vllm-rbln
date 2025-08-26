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
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, bind_kv_cache
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

try:
    from vllm.worker.worker_base import LoRANotSupportedWorkerBase
except ImportError:
    from vllm.worker.worker_base import (
        LoraNotSupportedWorkerBase as LoRANotSupportedWorkerBase, )

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.worker.model_runner import RBLNModelRunner
from vllm_rbln.worker.utils import get_maximum_num_blocks

logger = init_logger(__name__)


class RBLNCacheEngine:
    """Manages the KV cache for RBLN backend.

    This class is responsible for initializing and managing RBLN KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        assert "rbln" in current_platform.get_device_name().lower()
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        if RBLNWorker.disable_tp and parallel_config.enable_expert_parallel:
            self.num_heads = model_config.get_total_num_kv_heads()
        else:
            self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for CPU backend, because we want to reuse KV cache management
        # in the scheduler.
        self.num_cpu_blocks = cache_config.num_gpu_blocks

        # default cache type is bf16 (half precision)
        # FIXME - force cache data type into fp32 for graph compilation
        if cache_config.cache_dtype == "auto":
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE["float"]
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        logger.info("[RBLN] initialize cache engine")
        # Initialize the cache.
        # TODO : cpu_cache will be replaced with dev_cache
        self.cpu_cache = self._allocate_kv_cache()

    def _allocate_kv_cache(self, ) -> List[torch.Tensor]:
        """Allocates KV cache on RBLN."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            self.num_cpu_blocks, self.block_size, self.num_heads,
            self.head_size)
        kv_cache: List[torch.Tensor] = []
        logger.info("[RBLN] attention backend get_kv_cache_shape = %s",
                    kv_cache_shape)
        logger.info("[RBLN] allocate kv cache shape = %s", kv_cache_shape)
        kv_cache_size = 1
        for dim in kv_cache_shape:
            kv_cache_size *= dim
        logger.info("[RBLN] 1 layer : allocate kv cache size = %d",
                    kv_cache_size)
        kv_cache_size *= self.num_layers
        logger.info("[RBLN] all layers : allocate kv cache size = %d",
                    kv_cache_size)

        # allocate kv cache onto RBLN device
        # RBLN device tensor allocation
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape, dtype=self.dtype, device="cpu"))
        logger.info("[RBLN] allocate kv cache length = %d", len(kv_cache))

        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in RBLNCacheEngine.")

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in RBLNCacheEngine.")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        logger.info("[RBLN] copy kv cache")
        self.attn_backend.copy_blocks(self.cpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        if RBLNWorker.disable_tp and parallel_config.enable_expert_parallel:
            num_heads = model_config.get_total_num_kv_heads()
        else:
            num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        # default cache type is bf16 (half precision)
        # FIXME - force cache data type into fp32 for graph compilation
        if cache_dtype == "auto":
            dtype = STR_DTYPE_TO_TORCH_DTYPE["float"]
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        total_size = dtype_size * total
        logger.info("[RBLN] get kv cache block size = %d", total_size)
        return total_size


class RBLNWorker(LoRANotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    """A worker class that executes the model on RBLN NPUs."""

    disable_tp = os.environ.get("DISABLE_ATTN_TP",
                                "False").lower() in ("true", "1")

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        assert "rbln" in current_platform.get_device_name().lower()

        self.local_rank = local_rank
        self.rank = rank
        self.parallel_config.rank = rank

        if self.parallel_config.distributed_executor_backend == "ray":
            logger.info(
                "Running on Ray backend. Skipping device env var setup.")
        else:
            self.set_device()

        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        self.model_runner: RBLNModelRunner = RBLNModelRunner(
            vllm_config=vllm_config, is_driver_worker=is_driver_worker)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[RBLNCacheEngine]

        # TODO : cpu_cache will be replaced with dev cache
        self.cpu_cache: Optional[List[List[torch.Tensor]]] = None

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def set_device(self) -> None:
        world_size = self.parallel_config.world_size
        env_var = current_platform.device_control_env_var

        total_device_count = world_size * envs.RBLN_TP_SIZE

        if env_var not in os.environ:
            device_ids = [str(i) for i in range(total_device_count)]
        else:
            device_ids = os.environ[env_var].split(",")

        if len(device_ids) < total_device_count:
            raise RuntimeError(f"{env_var} has {len(device_ids)} devices"
                               " but required {total_device_count}")

        start_idx = self.local_rank * envs.RBLN_TP_SIZE
        end_idx = start_idx + envs.RBLN_TP_SIZE
        selected_devices = ",".join(device_ids[start_idx:end_idx])

        os.environ[env_var] = selected_devices
        logger.info(
            "Local rank: %d, Selected devices: %s",
            self.local_rank,
            selected_devices,
        )

    def init_device(self) -> None:
        # Note: unique identifier for creating allreduce shared memory
        # os.environ["VLLM_DIST_IDENT"] = (
        #     self.distributed_init_method.split(":")[-1]
        # )

        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        We configure num_gpu_blocks to be equal to max_num_seqs.
        """
        # Set the number of GPU blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.

        block_size = self.cache_config.block_size

        # This function comes from optimum-rbln.
        # We must keep it updated as optimum is upgraded.
        max_num_blocks = get_maximum_num_blocks(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            kvcache_block_size=block_size,
            # quantization : 4 (This is an ad-hoc value. Need to fix it)
            nbits_per_param=16 if not self.model_config.quantization else 4,
            n_model_params=sum(p.numel()
                               for p in self.model_runner.model.parameters()),
            # 1 : prefill
            num_runtimes=1 + self.scheduler_config.max_num_seqs)

        max_required_num_blocks = (self.model_config.max_model_len *
                                   self.scheduler_config.max_num_seqs //
                                   block_size)

        num_gpu_blocks = min(max_num_blocks - 1, max_required_num_blocks)

        if npu_num_blocks := os.environ.get("VLLM_RBLN_NPU_NUM_BLOCKS"):
            num_gpu_blocks = int(npu_num_blocks) - 1

        # Swap not yet supported with RBLN backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache."""

        # Different values are not tested.
        assert num_cpu_blocks == 0

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._init_cache_engine()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            RBLNCacheEngine(
                self.cache_config,
                self.model_config,
                self.parallel_config,
                self.device_config,
            ) for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.cpu_cache = [
            self.cache_engine[ve].cpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
        bind_kv_cache(self.compilation_config.static_forward_context,
                      self.cpu_cache)
        self.model_runner.block_size = self.cache_engine[0].block_size

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        """Get KV cache"""
        # TODO : cpu_cache will be replaced with dev_cache
        return self.cpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        return WorkerInput(num_seq_groups=len(
            execute_model_req.seq_group_metadata_list), )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass
        """ KV cache copy
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[worker_input.virtual_engine].copy(
                worker_input.blocks_to_copy)
            """

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes."""
        return RBLNCacheEngine.get_cache_block_size(
            self.model_runner.block_size,
            self.cache_config.cache_dtype,
            self.model_config,
            self.parallel_config,
        )

    def init_distributed_environment(self):
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )

        # warm up test for torch.distributed
        # all_reduce test
        torch.distributed.all_reduce(torch.zeros(1, dtype=torch.float16).cpu())
        # broadcast test
        # NOTE - broadcast DOES NOT support torch.int16 data type
        tensor_temp = torch.empty((1, 128), dtype=torch.int)
        torch.distributed.broadcast(tensor_temp, src=0)
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )
