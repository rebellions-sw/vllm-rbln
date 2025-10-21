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

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    VllmConfig = None

import rebel
from torch._dynamo import register_backend
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase)
from vllm.platforms import Platform, PlatformEnum, _Backend
from vllm.utils import FlexibleArgumentParser

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


# RBLN custom communicator (vllm/distributed/device_communicators/...)
class RblnCommunicator(DeviceCommunicatorBase):

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor,
                                    input_,
                                    group=self.device_group)
        if dim == -1:
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()

            output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
            if dim == 2:
                # output_tensor(dim=4).movedim(0, 2) == permute(1, 2, 0)
                # output_tensor = output_tensor.movedim(0, dim)
                output_tensor = output_tensor.permute(1, 2, 0, 3)
            else:
                assert False, "not yet implemented"
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                  (self.world_size *
                                                   input_size[dim], ) +
                                                  input_size[dim + 1:])
        elif dim == 0:
            pass
        else:
            assert False, "RBLN all_gather dim!=0 && dim!=-1, not yet implemented"
            # Reshape
            output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                  (self.world_size *
                                                   input_size[dim], ) +
                                                  input_size[dim + 1:])

        return output_tensor


def bypass_backend(graph_module: "torch.fx.GraphModule"):
    return graph_module.forward


register_backend(name="bypass", compiler_fn=bypass_backend)


class RblnPlatform(Platform):
    _enum = PlatformEnum.OOT

    # TODO(jiwoo.park): GroupCoordinator uses the device_name
    # when torch.device(device_name) is called.
    # But we don't support the 'rbln'' device yet.
    # To support this, we must use PyTorch-RBLN
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "RBLN"
    simple_compile_backend = "bypass"
    device_control_env_var: str = "RBLN_DEVICES"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return rebel.get_npu_name(device_id)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Check if the current platform supports async output.
        """
        return False

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on RBLN.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_rbln.platform.RblnCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        return True

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        if parser is None:
            return

        for action in parser._actions:
            if action.dest == "device":
                action.choices.append("rbln")

        for action in parser._actions:
            if action.dest == "block_size":
                action.choices = None  # Override choices

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if vllm_config.scheduler_config.is_multi_step:
            raise NotImplementedError(
                "Multi-step execution is not supported for RBLN")

        model_config = vllm_config.model_config
        task = model_config.task
        supported_tasks = set(model_config.supported_tasks)
        pooling_tasks = {"embed", "classify", "reward", "score"}

        if task == "auto":
            is_pooling = bool(pooling_tasks & supported_tasks)
            is_generate = "generate" in supported_tasks
        else:
            is_pooling = task in pooling_tasks
            is_generate = task == "generate"

        if is_pooling and envs.VLLM_USE_V1:
            raise ValueError("Pooling models are only supported on v0.")

        if is_generate and cls.supports_v1(
                model_config
        ) and not envs.VLLM_USE_V1 and not model_config.is_encoder_decoder:
            logger.warning("V0 support for decoder models is deprecated.")

        if envs.RBLN_ENFORCE_MODEL_FP32:
            logger.info("original model_config.dtype = %s", model_config.dtype)
            if model_config.dtype == torch.bfloat16:
                logger.warning("bfloat16 is not supported on RBLN.")

            # FIXME - force model dtype into fp32 for graph compilation
            model_config.dtype = torch.float
            assert model_config.dtype == torch.float
            logger.info("RBLN enforce model_config.dtype as torch.float")
        else:
            dtype = model_config.dtype
            logger.info("original model_config.dtype = %s", dtype)
            if dtype != torch.bfloat16 and dtype != torch.float16 and dtype != torch.float:
                logger.warning("%s is not supported on RBLN. only fp32,fp16,bf16 is supported", dtype)
                model_config.dtype = torch.float
            logger.info("RBLN use model_config.dtype = %s", model_config.dtype)

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if envs.RBLN_USE_VLLM_MODEL:
            if envs.VLLM_USE_V1:
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = (
                        "vllm_rbln.v1.worker.rbln_worker.RBLNWorker")
                scheduler_config.scheduler_cls = (
                    "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler")
            else:
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = (
                        "vllm_rbln.worker.worker.RBLNWorker")
                scheduler_config.scheduler_cls = (
                    "vllm_rbln.core.scheduler.RBLNScheduler")
        else:
            if envs.VLLM_USE_V1:
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = \
                        "vllm_rbln.v1.worker.optimum_worker.RBLNOptimumWorker"
                scheduler_config.scheduler_cls = \
                        "vllm_rbln.v1.core.optimum_scheduler.RBLNOptimumScheduler"
            else:
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = \
                        "vllm_rbln.worker.optimum_worker.RBLNOptimumWorker"
                scheduler_config.scheduler_cls = \
                    "vllm_rbln.core.optimum_scheduler.RBLNOptimumScheduler"

        if (parallel_config.distributed_executor_backend is not None
                and parallel_config.distributed_executor_backend != "mp"):
            logger.warning(
                ("%s is not supported on RBLN, fallback to mp "
                 "distributed executor backend."),
                parallel_config.distributed_executor_backend,
            )

        assert (vllm_config.lora_config
                is None), "LoRA is not supported for RBLN backend."
        assert (not vllm_config.speculative_config
                ), "Speculative decoding not yet supported for RBLN backend."

        cache_config = vllm_config.cache_config
        if cache_config:
            assert vllm_config.cache_config.block_size is not None, (
                "block_size must be configured for RBLN backend")
            cache_config.enable_prefix_caching = False

        if envs.VLLM_USE_V1 and envs.RBLN_USE_VLLM_MODEL:
            from vllm.config import CompilationLevel

            if (vllm_config.compilation_config.level
                    != CompilationLevel.NO_COMPILATION):
                logger.info("RBLN doesn't @support_torch_compile decorator")
                vllm_config.compilation_config.level = (
                    CompilationLevel.NO_COMPILATION)
                if (len(vllm_config.compilation_config.custom_ops) == 1
                        and vllm_config.compilation_config.custom_ops[0]
                        == "none"):
                    vllm_config.compilation_config.custom_ops = []

            if not model_config.disable_cascade_attn:
                logger.info("The cascade attention is disabled"
                            " because RBLN does not support it")
                model_config.disable_cascade_attn = True

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
    ) -> str:
        if envs.VLLM_USE_V1:
            attn_backend_cls = ("vllm_rbln.v1.attention.backends."
                                "flash_attention.RBLNAttentionBackend")
        else:
            attn_backend_cls = ("vllm_rbln.attention.backends."
                                "flash_attention.RBLNAttentionBackend")
        logger.info("Using RBLN Attention Backend: %s", attn_backend_cls)

        return attn_backend_cls

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True
