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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import BatchedTensorInputs
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.worker.model_runner_base import ModelRunnerInputBase

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend


# FIXME(eunji): In original vLLM, this dataclasss is located in model_runner.
# And it makes available to decouple the vllm logic and hf model logic
@dataclass(frozen=True)
class ModelInputForRBLN(ModelRunnerInputBase):
    """
    Used by the RBLNModelRunner.
    """
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    block_tables: torch.Tensor
    running_requests_ids: List[str]
    finished_requests_ids: List[str]
    is_prompt: bool = False  # for V1
    cached_block_tables: List[int] = field(
        default_factory=list)  # for prefix caching
    cached_lengths: List[int] = field(
        default_factory=list)  # for prefix caching
    sampling_metadata: "SamplingMetadata" = None,  # for V0
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    pooling_metadata: Optional[PoolingMetadata] = None  # for V1
    lora_requests: Optional[List[LoRARequest]] = None  # for V0
    lora_mapping: Optional["LoRAMapping"] = None  # for V0
    dummy_block: Optional[int] = None  # for prefix caching

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        raise NotImplementedError("ModelInputForRBLN cannot be broadcast.")

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForRBLN":
        assert attn_backend is None
        return cls.from_broadcasted_tensor_dict(tensor_dict)


version_error = RuntimeError(
    "Incompatible vLLM version detected. "
    "This vLLM version is not compatible with optimum-rbln. "
    "Please verify that you are using a supported version.")
