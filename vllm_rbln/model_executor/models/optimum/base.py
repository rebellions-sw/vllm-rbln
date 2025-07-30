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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PretrainedConfig
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import BatchedTensorInputs
from vllm.worker.model_runner_base import ModelRunnerInputBase

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

# modified/customized models for RBLN
_RBLN_GENERATION_MODELS: Dict[str, Tuple[str, str]] = {
    "LlamaForCausalLM": (
        "llama",
        "RBLNLlamaForCausalLM",
    ),
    "GemmaForCausalLM": ("gemma", "RBLNGemmaForCausalLM"),
    "PhiForCausalLM": ("phi", "RBLNPhiForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "RBLNGPT2LMHeadModel"),
    "MidmLMHeadModel": ("midm", "RBLNMidmLMHeadModel"),
    "MistralForCausalLM": ("mistral", "RBLNMistralForCausalLM"),
    "ExaoneForCausalLM": ("exaone", "RBLNExaoneForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "RBLNQwen2ForCausalLM"),
    "OPTForCausalLM": ("opt", "RBLNOPTForCausalLM"),
    "Qwen3ForCausalLM": ("qwen3", "RBLNQwen3ForCausalLM"),
}

_RBLN_ENCODER_DECODER_MODELS: Dict[str, Tuple[str, str]] = {
    "BartForConditionalGeneration":
    ("bart", "RBLNBartForConditionalGeneration"),
    "T5ForConditionalGeneration": ("t5", "RBLNT5ForConditionalGeneration"),
    "T5WithLMHeadModel": ("t5", "RBLNT5ForConditionalGeneration"),
}

_RBLN_MULTIMODAL_MODELS = {
    "LlavaNextForConditionalGeneration":
    ("llava_next", "RBLNLlavaNextForConditionalGeneration"),
    "Qwen2_5_VLForConditionalGeneration":
    ("qwen2_5_vl", "RBLNQwen2_5_VLForConditionalGeneration"),
    "Idefics3ForConditionalGeneration":
    ("idefics3", "RBLNIdefics3ForConditionalGeneration"),
    "Blip2ForConditionalGeneration":
    ("blip2", "RBLNBlip2ForConditionalGeneration"),
    "Gemma3ForConditionalGeneration": ("gemma3",
                                       "RBLNGemma3ForConditionalGeneration"),
    "WhisperForConditionalGeneration": ("whisper",
                                        "RBLNWhisperForConditionalGeneration"),
    "LlavaForConditionalGeneration": ("llava",
                                      "RBLNLlavaForConditionalGeneration"),
}

_RBLN_EMBEDDING_MODELS = {
    "T5EncoderModel": ("t5_encoder", "RBLNT5EncoderModel"),
    "BertModel": ("bert_model", "RBLNBertModel"),
    "RobertaForSequenceClassification":
    ("roberta_classification", "RBLNRobertaForSequenceClassification"),
    "RobertaModel": ("roberta", "RBLNRobertaModel"),
    "XLMRobertaForSequenceClassification":
    ("xlm_roberta_classification", "RBLNXLMRobertaForSequenceClassification"),
    "XLMRobertaModel": ("xlm_roberta", "RBLNXLMRobertaModel"),
    "Qwen3Model": ("qwen3", "RBLNQwen3Model"),
}

_RBLN_SUPPORTED_MODELS = {
    **_RBLN_GENERATION_MODELS,
    **_RBLN_ENCODER_DECODER_MODELS,
    **_RBLN_MULTIMODAL_MODELS,
    **_RBLN_EMBEDDING_MODELS,
}


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
    sampling_metadata: "SamplingMetadata" = None,  # for V0
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    token_type_ids: Optional[torch.Tensor] = None
    pooling_metadata: Optional[PoolingMetadata] = None  # for V0

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


def is_multi_modal(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_MULTIMODAL_MODELS)


def is_pooling_arch(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_EMBEDDING_MODELS)


def is_enc_dec_arch(config: PretrainedConfig) -> bool:
    return is_arch_supported(config, _RBLN_ENCODER_DECODER_MODELS)


def is_arch_supported(config: PretrainedConfig,
                      model_set: Dict[str, Tuple[str, str]]) -> bool:
    architectures = getattr(config, "architectures", [])
    return any(arch in _RBLN_SUPPORTED_MODELS and arch in model_set
               for arch in architectures)


def get_rbln_model_info(config: PretrainedConfig) -> Tuple[str, str]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _RBLN_SUPPORTED_MODELS:
            model_name, model_cls_name = _RBLN_SUPPORTED_MODELS[arch]
            return model_name, model_cls_name

    raise ValueError(
        f"Model architectures {architectures} are not supported on RBLN "
        f"for now. Supported architectures: "
        f"{list(_RBLN_SUPPORTED_MODELS.keys())}")


version_error = RuntimeError(
    "Incompatible vLLM version detected. "
    "This vLLM version is not compatible with optimum-rbln. "
    "Please verify that you are using a supported version.")
