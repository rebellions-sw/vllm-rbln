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
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.distributed import (ensure_model_parallel_initialized, get_pp_group,
                              init_distributed_environment)
from vllm.forward_context import DPMetadata
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, bind_kv_cache
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

try:
    from vllm.worker.worker_base import LoRANotSupportedWorkerBase
except ImportError:
    from vllm.worker.worker_base import (
        LoraNotSupportedWorkerBase as LoRANotSupportedWorkerBase, )

from vllm.worker.model_runner_base import BroadcastableModelInput

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
        cpu_cache: Optional[list[torch.Tensor]] = None,
    ) -> None:
        assert "rbln" in current_platform.get_device_name().lower()
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

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
        num_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = num_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        # default cache type is bf16 (half precision)
        # FIXME - force cache data type into fp32 for graph compilation
        if cache_config.cache_dtype == "auto":
            # NOTE(jiwoo.park) Currently, eager mode can support only FP16 dtype
            # for the KV cache.
            if self.device_config.device_type == "rbln":
                self.dtype = torch.float16
            else:
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

        logger.info("initialize cache engine")
        # Initialize the cache.
        # TODO : cpu_cache will be replaced with dev_cache
        self.cpu_cache = self._allocate_kv_cache(
            num_blocks) if cpu_cache is None else cpu_cache

    def _allocate_kv_cache(
        self,
        num_blocks: int,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on RBLN."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        kv_cache: List[torch.Tensor] = []
        logger.info("attention backend get_kv_cache_shape = %s",
                    kv_cache_shape)
        logger.info("allocate kv cache shape = %s", kv_cache_shape)
        kv_cache_size = 1
        for dim in kv_cache_shape:
            kv_cache_size *= dim
        logger.info("1 layer : allocate kv cache size = %d", kv_cache_size)
        kv_cache_size *= self.num_layers
        logger.info("all layers : allocate kv cache size = %d", kv_cache_size)

        # allocate kv cache onto RBLN device
        # RBLN device tensor allocation
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype).to(self.device_config.device))
        logger.info("allocate kv cache length = %d", len(kv_cache))

        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in RBLNCacheEngine.")

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in RBLNCacheEngine.")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        logger.info("copy kv cache")
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
        logger.info("get kv cache block size = %d", total_size)
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

        self.local_rank = local_rank
        self.rank = rank
        self.parallel_config.rank = rank
        self.parallel_config.local_world_size = (
            self.parallel_config.world_size // envs.VLLM_RBLN_NUM_RAY_NODES)

        distributed_backend = self.parallel_config.distributed_executor_backend
        if distributed_backend == "mp":
            logger.info("distributed executor backend mp enabled")
            self.set_device()
        elif distributed_backend == "ray":
            logger.info("distributed executor backend ray enabled")
            self.set_device()
        else:
            logger.info(
                "Running on other backend. Skipping device env var setup.")

        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

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

        self.is_dummy_execute_phase = False
        self.dummy_execute_model_req: Optional[ExecuteModelRequest] = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def set_device(self) -> None:
        world_size = self.parallel_config.local_world_size
        env_var = current_platform.device_control_env_var

        total_device_count = world_size * envs.VLLM_RBLN_TP_SIZE

        distributed_backend = self.parallel_config.distributed_executor_backend
        if env_var not in os.environ or distributed_backend == "ray":
            device_ids = [str(i) for i in range(total_device_count)]
        else:
            device_ids = os.environ[env_var].split(",")

        # This check is only valid for single node mp backends, invalid for ray
        # ex) node#0 : RBLN_DEVICES=0,1
        #     node#1 : RBLN_DEVICES=2,3
        if distributed_backend == "mp" and len(
                device_ids) < total_device_count:
            raise RuntimeError(f"{env_var} has devices {device_ids}"
                               f" but required {total_device_count}")

        start_idx = self.local_rank * envs.VLLM_RBLN_TP_SIZE
        end_idx = start_idx + envs.VLLM_RBLN_TP_SIZE
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
        start_time = time.perf_counter()
        self.model_runner.load_model()
        elapsed_time = time.perf_counter() - start_time
        logger.info("load_model completed in %.6f seconds (%.3f ms)",
                    elapsed_time, elapsed_time * 1000)

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        We configure num_gpu_blocks to be equal to max_num_seqs.
        """
        ve_cnt = self.parallel_config.pipeline_parallel_size
        max_model_len = self.model_config.max_model_len
        block_size = self.cache_config.block_size
        num_blocks_per_seq = max_model_len // block_size

        # We always allocate this number of blocks, but the last one is
        # reserved for padding. As a result, the vLLM system should treat
        # it as if there is one fewer usable block than the number
        # actually allocated.
        if npu_num_blocks := os.environ.get("VLLM_RBLN_NPU_NUM_BLOCKS"):
            num_gpu_blocks = int(npu_num_blocks) - 1
        else:

            # This function comes from optimum-rbln.
            # We must keep it updated as optimum is upgraded.
            max_num_blocks = get_maximum_num_blocks(
                model_config=self.model_config,
                parallel_config=self.parallel_config,
                kvcache_block_size=block_size,
                # quantization : 4 (This is an ad-hoc value. Need to fix it)
                nbits_per_param=16
                if not self.model_config.quantization else 4,
                n_model_params=sum(
                    p.numel() for p in self.model_runner.model.parameters()),
                # 2 : 1 for prefill and decode each
                num_runtimes=2) - 1

            max_required_num_blocks = (num_blocks_per_seq *
                                       self.scheduler_config.max_num_seqs) + 1
            max_required_num_blocks = max_required_num_blocks * ve_cnt

            num_gpu_blocks = min(
                int(max_num_blocks * self.cache_config.gpu_memory_utilization),
                max_required_num_blocks)

        num_blocks_per_ve = num_gpu_blocks // ve_cnt
        assert num_blocks_per_seq <= num_blocks_per_ve, \
            "There must be at least enough blocks to handle one request." \
            "You may need to adjust max_model_len."

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

        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        # All cache engines share the same cpu_cache
        self.cache_engine = [
            RBLNCacheEngine(
                self.cache_config,
                self.model_config,
                self.parallel_config,
                self.device_config,
            )
        ]
        cpu_cache = self.cache_engine[0].cpu_cache
        self.cache_engine.extend([
            RBLNCacheEngine(
                self.cache_config,
                self.model_config,
                self.parallel_config,
                self.device_config,
                cpu_cache=cpu_cache,
            ) for _ in range(1, self.parallel_config.pipeline_parallel_size)
        ])
        self.cpu_cache = [
            self.cache_engine[ve].cpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]

        bind_kv_cache(self.compilation_config.static_forward_context,
                      self.cpu_cache)
        if not self.model_config.enforce_eager and envs.VLLM_RBLN_COMPILE_MODEL:
            for kv_cache in cpu_cache:
                self.model_runner.compile_context.mark_static_address(kv_cache)

    def _warm_up_model(self) -> None:
        model_inputs = self._prepare_dummy_input()
        if self.model_config.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL \
            or not envs.VLLM_RBLN_ENABLE_WARM_UP:
            logger.warning("skipping _warm_up_model")
            return

        assert self.kv_cache is not None

        start_time = time.perf_counter()
        self.model_runner._dummy_run(model_inputs, self.kv_cache[0])
        elapsed_time = time.perf_counter() - start_time
        logger.info("compilation completed in %.6f seconds (%.3f ms)",
                    elapsed_time, elapsed_time * 1000)

        # FIXME(RBLN): To reduce dynamo cache lookup overhead, make dyanmo
        # evaluate a minimal set of guards required for dispatching compiled
        # functions. This assumes that the model does not change.
        torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)

    def _prepare_dummy_input(
        self,
        max_num_batched_tokens: int = 1,
        max_num_seqs: int = 1
    ) -> Tuple[BroadcastableModelInput, BroadcastableModelInput]:
        prefill_req = None
        decode_req = None
        if self.is_driver_worker:
            num_blocks = self.cache_config.num_gpu_blocks
            sampling_params = SamplingParams()

            prefill_seqs: List[SequenceGroupMetadata] = []
            decode_seqs: List[SequenceGroupMetadata] = []

            for group_id in range(max_num_seqs):
                seq_len = max_num_batched_tokens

                dummy_data = self.model_runner.input_registry \
                    .dummy_data_for_profiling(self.model_config,
                                            seq_len,
                                            self.model_runner.mm_registry)
                prefill_seq_data = dummy_data.seq_data
                block_tables = {group_id: [num_blocks]}

                prefill_seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: prefill_seq_data},
                    sampling_params=sampling_params,
                    block_tables=block_tables,
                )
                prefill_seqs.append(prefill_seq)

                decode_seq_data = prefill_seq_data.from_seqs(
                    prefill_seq_data.get_prompt_token_ids(),
                    prefill_seq_data.get_prompt_token_ids()[-1:],
                )
                decode_seq_data.update_num_computed_tokens(seq_len)

                decode_seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=False,
                    seq_data={group_id: decode_seq_data},
                    sampling_params=sampling_params,
                    block_tables=block_tables,
                )
                decode_seqs.append(decode_seq)

            prefill_req = ExecuteModelRequest(prefill_seqs[:1])
            decode_req = ExecuteModelRequest(decode_seqs)

        if self.parallel_config.data_parallel_size > 1:
            self.dummy_execute_model_req = prefill_req
            max_num_batched_tokens = \
                self.scheduler_config.max_num_batched_tokens
            max_num_seqs = self.scheduler_config.max_num_seqs
            if envs.VLLM_RBLN_DP_IMPL == "padded_decode":
                # TODO: consider relaxing this constraint
                assert max_num_batched_tokens % max_num_seqs == 0, \
                    "max_num_batched_tokens must be divisible by max_num_seqs"

        prefill_inputs = self.prepare_input(prefill_req)
        decode_inputs = self.prepare_input(decode_req)
        prefill_model_input, _, _ = prefill_inputs
        decode_model_input, _, _ = decode_inputs
        return (prefill_model_input, decode_model_input)

    def _prepare_dp_model(self, model_input: BroadcastableModelInput) -> None:
        dp_size = self.parallel_config.data_parallel_size
        if self.is_dummy_execute_phase or dp_size <= 1:
            return

        dp_rank = self.parallel_config.data_parallel_rank
        assert model_input.attn_metadata is not None
        if model_input.attn_metadata.num_prefill_tokens > 0:
            num_tokens = \
                self.scheduler_config.max_num_batched_tokens
        else:
            num_tokens = self.scheduler_config.max_num_seqs

        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank)

        if model_input.attn_metadata.num_prefill_tokens > 0:
            return

        def is_same_phase(val, tensors):
            return all(t == val for t in tensors)

        same_phase = is_same_phase(num_tokens, num_tokens_across_dp)
        if same_phase:
            return

        self.is_dummy_execute_phase = True
        while not same_phase:
            self.execute_model(self.dummy_execute_model_req)
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                num_tokens, dp_size, dp_rank)
            same_phase = is_same_phase(num_tokens, num_tokens_across_dp)
        self.is_dummy_execute_phase = False

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

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps

        if envs.VLLM_RBLN_DP_IMPL == "dummy_prefill":
            self._prepare_dp_model(model_input)

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        if not get_pp_group().is_first_rank:
            # NOTE - DO NOT all_gather_group for RBLN pp
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict())
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                orig_model_execute_time = intermediate_tensors.tensors.get(
                    "model_execute_time", torch.tensor(0)).item()

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        model_execute_time = time.perf_counter() - start_time
        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                output.tensors["model_execute_time"] = torch.tensor(
                    model_execute_time + orig_model_execute_time)
            # NOTE - DO NOT all_gather_group for RBLN pp
            get_pp_group().send_tensor_dict(output.tensors)
            return [None]
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes."""
        return RBLNCacheEngine.get_cache_block_size(
            self.cache_config.block_size,
            self.cache_config.cache_dtype,
            self.model_config,
            self.parallel_config,
        )

    def init_distributed_environment(self):
        # Set envs for RCCL
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        os.environ['WORLD_SIZE'] = str(self.parallel_config.world_size)

        if self.parallel_config.data_parallel_size > 1:
            world_size = self.parallel_config.world_size
            rank = self.parallel_config.data_parallel_rank * world_size
            rank += self.local_rank
            world_size = self.parallel_config.world_size_across_dp
            os.environ['LOCAL_RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)

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

    def shutdown(self) -> None:
        logger.info("v0 rbln_worker shutdown called")
        if envs.VLLM_RBLN_METRICS:
            # FIXME - performance tracker atexit is not called
            self.model_runner.performance_tracker.print_final_stats()
