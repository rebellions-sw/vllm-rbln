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
    from vllm.config import ModelConfig, VllmConfig
else:
    VllmConfig = None

import rebel
from torch._dynamo import register_backend
from vllm.platforms import Platform, PlatformEnum, _Backend
from vllm.utils import FlexibleArgumentParser

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


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
        if envs.RBLN_USE_VLLM_MODEL:
            # patches
            if envs.VLLM_USE_V1:
                # FIXME(jiwoo.park):disable timeout for RBLN
                import vllm.v1.executor.multiproc_executor as mp
                mp.EXECUTE_MODEL_TIMEOUT_S = None

                import vllm_rbln.v1.attention.layer  # noqa
            else:
                import vllm_rbln.attention.layer  # noqa
            import vllm_rbln.model_executor.layers.fused_moe.layer  # noqa
            import vllm_rbln.model_executor.layers.logits_processor  # noqa
            import vllm_rbln.model_executor.layers.rotary_embedding  # noqa
            import vllm_rbln.model_executor.layers.vocab_parallel_embedding  # noqa
            import vllm_rbln.model_executor.model_loader.weight_loader  # noqa
            import vllm_rbln.models.deepseek_v2  # noqa
            import vllm_rbln.models.qwen2_moe  # noqa
            import vllm_rbln.models.qwen3_moe  # noqa
            import vllm_rbln.models.utils  # noqa

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

        logger.info("original model_config.dtype = %s", model_config.dtype)
        if model_config.dtype == torch.bfloat16:
            logger.warning("bfloat16 is not supported on RBLN.")

        # FIXME - force model dtype into fp32 for graph compilation
        model_config.dtype = torch.float
        assert model_config.dtype == torch.float
        logger.info("RBLN model_config.dtype = %s", model_config.dtype)

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

            # FIXME(jiwoo.park) This is a temporary workaround.
            if model_config.enforce_eager:
                RblnPlatform.device_type = "rbln"
                vllm_config.device_config.device_type = RblnPlatform.device_type
                vllm_config.device_config.device = (torch.device(
                    RblnPlatform.device_type))
                # NOTE - force dtype into fp16 for eager mode
                model_config.dtype = torch.float16
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
