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
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.core.kv_cache_utils import get_uniform_page_size
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.worker_base import WorkerBase

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.worker.utils import get_maximum_num_blocks

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class RBLNWorker(WorkerBase):
    """A worker class that executes the model on RBLN NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        assert "rbln" in current_platform.get_device_name().lower()
        self.device = torch.device(current_platform.device_name)

        if self.parallel_config.distributed_executor_backend == "ray":
            logger.info(
                "Running on Ray backend. Skipping device env var setup.")
        else:
            self._init_device_env()

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True),
            )
        else:
            self.profiler = None

        self.parallel_config.disable_custom_all_reduce = True

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))

    def _init_device_env(self) -> None:
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
        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        self.model_runner: RBLNModelRunner = RBLNModelRunner(
            self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on RBLN, ignore it.")
        pass

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        logger.warning("sleep mode is not supported on RBLN, ignore it.")
        pass

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
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

        # for partition skip, we need dummy block slot.
        no_dummy_slots = 1
        max_required_num_blocks = (self.model_config.max_model_len *
                                   self.scheduler_config.max_num_seqs //
                                   block_size) + no_dummy_slots
        num_gpu_blocks = min(max_num_blocks, max_required_num_blocks)

        if npu_num_blocks := os.environ.get("VLLM_RBLN_NPU_NUM_BLOCKS"):
            num_gpu_blocks = int(npu_num_blocks)

        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        num_layers = len(kv_cache_spec)
        page_size = get_uniform_page_size(kv_cache_spec)
        return num_gpu_blocks * page_size * num_layers

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate RBLN KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        logger.warning("model warm-up is not supported on RBLN.")
        pass

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)
        parallel_config = self.vllm_config.parallel_config
        if (parallel_config.distributed_executor_backend != "external_launcher"
                and not get_pp_group().is_last_rank):
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None
        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "gloo",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config

    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        distributed_init_method,
        local_rank,
        backend,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
    )

    ensure_kv_transfer_initialized(vllm_config)
