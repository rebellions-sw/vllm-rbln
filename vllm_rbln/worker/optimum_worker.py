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
        self.check_rbln_config()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        """
        attn_impl = self.model_runner.model.model.get_attn_impl() if hasattr(
            self.model_runner.model.model, "get_attn_impl") else None

        if attn_impl is not None and attn_impl == "flash_attn":
            # We use the last block as dummy block
            num_gpu_blocks = (
                self.model_runner.model.model.get_kvcache_num_blocks() - 1) \
                if self.model_runner.model.model.rbln_config.batch_size > 1 \
                else (self.model_runner.model.model.get_kvcache_num_blocks())

            if npu_num_blocks := os.environ.get("VLLM_RBLN_NPU_NUM_BLOCKS"):
                num_gpu_blocks = int(npu_num_blocks) - 1
        else:
            # Set the number of GPU blocks to be the same as the maximum
            # number of sequences that can be processed in a single batch.
            # This is equivalent to schedule without PagedAttention.
            num_gpu_blocks = self.scheduler_config.max_num_seqs
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

    def check_rbln_config(self):
        rbln_config = self.model_runner.model.rbln_model_config
        submodules = rbln_config.submodules
        batch_size = rbln_config.batch_size
        kvcache_partition_len = getattr(rbln_config, "kvcache_partition_len",
                                        None)
        max_seq_len = getattr(rbln_config, "max_seq_len", None)
        dec_max_seq_len = getattr(rbln_config, "dec_max_seq_len", None)

        max_model_len = self.model_config.max_model_len
        max_num_seqs = self.scheduler_config.max_num_seqs
        block_size = self.vllm_config.cache_config.block_size
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens

        # NOTE It is based on the decoder submodule is only one.
        for submodule in submodules:
            submodule_config = getattr(rbln_config, submodule)
            if kvcache_partition_len is None:
                kvcache_partition_len = getattr(submodule_config,
                                                "kvcache_partition_len", None)
            if max_seq_len is None:
                max_seq_len = getattr(submodule_config, "max_seq_len", None)

            if dec_max_seq_len is None:
                dec_max_seq_len = getattr(submodule_config, "dec_max_seq_len",
                                          None)

            submodule_batch_size = getattr(submodule_config, "batch_size",
                                           None)
            # FIXME
            if submodule_batch_size is not None \
                and batch_size != submodule_batch_size:
                batch_size = submodule_batch_size

        if kvcache_partition_len is None:
            if max_seq_len is not None:
                assert block_size == max_seq_len, (
                    f"`block_size({block_size})` must match "
                    f"`max_seq_len({max_seq_len})` "
                    "of the compiled RBLN model.")
            elif dec_max_seq_len is not None:
                assert block_size == dec_max_seq_len, (
                    f"`block_size({block_size})` must match "
                    f"`dec_max_seq_len({dec_max_seq_len})` "
                    "of the compiled RBLN model.")
        else:
            assert block_size == kvcache_partition_len, (
                f"`block_size({block_size})` must match "
                f"the `kvcache_partition_len({kvcache_partition_len})` "
                "of the compiled RBLN model.")

        if dec_max_seq_len is not None:  # encoder-decoder
            assert max_num_batched_tokens == max_seq_len, (
                f"`max_num_batched_tokens({max_num_batched_tokens})` "
                f"must match the `dec_max_seq_len({dec_max_seq_len})` "
                "of the compiled RBLM model.")
            assert max_model_len == max_seq_len, (
                f"`max_model_len({max_model_len})` must match "
                f"the `dec_max_seq_len({dec_max_seq_len})` "
                "of the compiled RBLM model.")
        elif max_seq_len:  # including decoder
            assert max_num_batched_tokens == max_seq_len, (
                f"`max_num_batched_tokens({max_num_batched_tokens})` "
                f"must match the `max_seq_len({max_seq_len})` "
                "of the compiled RBLM model.")
            assert max_model_len == max_seq_len, (
                f"`max_model_len({max_model_len})` must match "
                f"the `max_seq_len({max_seq_len})` "
                "of the compiled RBLM model.")

        assert max_num_seqs == batch_size, (
            f"`max_num_seqs({max_num_seqs})` must match "
            f"the `batch_size({batch_size})` of the compiled RBLM model.")
