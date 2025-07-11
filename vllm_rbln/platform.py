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

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None
import os
from pathlib import Path

import rebel
from torch._dynamo import register_backend
from vllm.platforms import Platform, PlatformEnum, _Backend
from vllm.utils import FlexibleArgumentParser

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

_RBLN_TORCH_COMPILE_SUPPORTED = [
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen2MoeModel",
    "Qwen2MoeForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
]


def is_torch_compile_supported(vllm_config: VllmConfig) -> bool:
    model = vllm_config.model_config.model  # noqa

    # Compiled models (e.g., local paths) must use Optimum-rbln
    if isinstance(model, (str, Path)) and os.path.exists(model):
        return False

    if (vllm_config.additional_config is not None and
            vllm_config.additional_config.get("force_optimum_rbln", False)):
        return False

    # Check if the architecture supports torch.compile
    architectures = getattr(vllm_config.model_config.hf_config,
                            "architectures", [])
    return any(arch in _RBLN_TORCH_COMPILE_SUPPORTED for arch in architectures)


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

        # torch.compile() is currently disabled.
        # TODO: Replace with dynamic check via is_torch_compile_supported().
        is_torch_compile = False
        model_config = vllm_config.model_config
        logger.info("original model_config.dtype = %s", model_config.dtype)
        if model_config.dtype == torch.bfloat16:
            logger.warning("bfloat16 is not supported on RBLN.")

        # FIXME - force model dtype into fp32 for graph compilation
        model_config.dtype = torch.float
        assert model_config.dtype == torch.float
        logger.info("RBLN model_config.dtype = %s", model_config.dtype)

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm_rbln.worker.worker.RBLNWorker" if is_torch_compile else
                "vllm_rbln.worker.optimum_worker.RBLNOptimumWorker")

        scheduler_config = vllm_config.scheduler_config
        scheduler_config.scheduler_cls = (
            "vllm_rbln.core.scheduler.RBLNScheduler" if is_torch_compile else
            "vllm_rbln.core.optimum_scheduler.RBLNOptimumScheduler")

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
        attn_backend_cls = (
            "vllm_rbln.attention.backends.flash_attention.RBLNAttentionBackend"
        )
        logger.info("Using RBLN Attention Backend: %s", attn_backend_cls)

        return attn_backend_cls
