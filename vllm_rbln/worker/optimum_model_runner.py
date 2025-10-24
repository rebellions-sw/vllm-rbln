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

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap)
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase

from vllm_rbln.model_executor.model_loader.rbln_model_loader import (
    get_optimum_model)
from vllm_rbln.model_executor.models.optimum import (  # noqa
    ModelInputForRBLN, RBLNOptimumForEncoderModel)
from vllm_rbln.utils.optimum.registry import (get_rbln_model_info,
                                              is_enc_dec_arch)

logger = init_logger(__name__)


class RBLNOptimumModelRunner(ModelRunnerBase[ModelInputForRBLN]):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        # FIXME: For RBLN support Enc-only model which based on enc-dec config.
        # When using an encoder-only model (such as T5EncoderModel)
        # with a config designed for enc-dec architectures,
        # it’s important to set the is_encoder_decoder flag to False.
        # This prevents the scheduler from applying text generation settings.
        _, model_cls_name = get_rbln_model_info(vllm_config.model_config)
        if model_cls_name in ["RBLNT5EncoderModel"]:
            vllm_config.model_config.hf_config.__dict__[
                "is_encoder_decoder"] = False
        if model_cls_name in ["RBLNQwen3ForCausalLM"
                              ] and vllm_config.model_config.task == "embed":
            # NOTE The architecture of Qwen3-Embedding model in huggingface
            # is `Qwen3ForCausalLM`. But it have to be mapped to `Qwen3Model`
            # for optimum-rbln.
            vllm_config.model_config.hf_config.__dict__["architectures"] = [
                "Qwen3Model"
            ]

        ModelRunnerBase.__init__(self, vllm_config)

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY

        self.enable_lora = self.vllm_config.lora_config is not None

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set after load_model.
        self.sampler = get_sampler()

    def load_model(self) -> None:
        with set_current_vllm_config(self.vllm_config, check_compile=False):
            self.model = get_optimum_model(vllm_config=self.vllm_config)
        self.use_optimum_lora = getattr(self.model.rbln_model_config,
                                        "use_lora", None)
        if self.enable_lora and not self.use_optimum_lora:
            raise RuntimeError(
                "The compiled model is for LoRA."
                "Please compile the model with `rbln_lora_config`")
        if not self.enable_lora and self.use_optimum_lora:
            raise RuntimeError("The model is compiled for LoRA."
                               "Please set `enable_lora=True` in vLLM.")

        if self.use_optimum_lora:
            self.valid_lora_ids = list(
                range(len(self.model.rbln_model_config.lora_config.adapters)))

    def get_model(self):
        return self.model

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], BatchedTensorInputs,
               torch.Tensor, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []

        seq_lens: List[int] = []
        multi_modal_inputs_list: List[MultiModalKwargs] = []
        block_tables: List[List[int]] = []
        running_requests_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            # Check lora_int_id is valid
            if seq_group_metadata.lora_request and self.use_optimum_lora:
                lora_int_id = seq_group_metadata.lora_request.lora_int_id
                if lora_int_id >= len(self.valid_lora_ids):
                    raise RuntimeError(
                        f"Invalid `lora_int_id`: {lora_int_id}. "
                        f"Valid `lora_int_ids` are {self.valid_lora_ids} "
                        "(must be consistent with the compiled model).")

            seq_data = (seq_group_metadata.encoder_seq_data
                        if is_enc_dec_arch(self.model_config.hf_config) else
                        seq_group_metadata.seq_data[seq_id])

            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            input_tokens.append(prompt_tokens)
            positions = list(range(seq_len))
            input_positions.append(positions)

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            if block_table is not None:
                block_tables.append(block_table)

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                # Process multi-modal data
                mm_kwargs = self._compute_multi_modal_input(
                    positions, seq_group_metadata)

                multi_modal_inputs_list.append(mm_kwargs)

            request_id = seq_group_metadata.request_id
            running_requests_ids.append(request_id)

        max_seq_len = max(seq_lens)

        assert max_seq_len > 0
        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_len=max_seq_len,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               max_len=max_seq_len,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)
        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=self.model_config.max_model_len //
            self.cache_config.block_size,
            pad=-1,
            dtype=torch.int32,
            device=self.device,
        ).to(torch.int16) if len(block_tables) > 0 else None

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_inputs_list)
        return (input_tokens, input_positions, seq_lens, multi_modal_kwargs,
                block_tables, running_requests_ids)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        block_tables: List[List[int]] = []
        running_requests_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            assert len(seq_ids) == 1, (
                "For now, multiple sequences in a SequenceGroup ",
                "are not supported.")
            request_id = seq_group_metadata.request_id

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]

                if block_table is not None:
                    assert len(block_table) >= 1
                    block_tables.append(block_table)

                # NOTE It is based on the 1to1 correspondence
                # between sequence_id and request_id.
                running_requests_ids.append(request_id)

        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_len=1,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)

        input_positions = make_tensor_with_pad(input_positions,
                                               max_len=1,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)

        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=self.model_config.max_model_len //
            self.cache_config.block_size,
            pad=-1,
            dtype=torch.int32,
            device=self.device,
        ).to(torch.int16) if len(block_tables) > 0 else None
        return input_tokens, input_positions, block_tables, running_requests_ids

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForRBLN:
        return ModelInputForRBLN.from_broadcasted_tensor_dict(tensor_dict)

    def _compute_multi_modal_input(
            self, positions: list[int],
            seq_group_metadata: SequenceGroupMetadata
    ) -> Optional[Dict[str, Any]]:
        """If multi-modal data is given, add it to the input."""
        mm_kwargs, _ = MultiModalPlaceholderMap.from_seq_group(
            seq_group_metadata,
            range(positions[0], positions[0] + len(positions)))
        if not mm_kwargs:
            return None
        return mm_kwargs

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForRBLN:
        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, seq_lens, multi_modal_kwargs,
             block_tables, running_requests_ids
             ) = self._prepare_prompt(seq_group_metadata_list)

        else:
            (input_tokens, input_positions, block_tables, running_requests_ids
             ) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since rbln worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids)
        ) if seq_group_metadata_list[0].sampling_params is not None else None

        lora_requests = None
        lora_mapping = None
        if self.enable_lora:
            # LoRA data.
            lora_requests = [
                seq_group_metadata.lora_request
                for seq_group_metadata in seq_group_metadata_list
            ]
            lora_mapping = LoRAMapping(is_prefill=is_prompt,
                                       index_mapping=[],
                                       prompt_mapping=[])

        return ModelInputForRBLN(
            input_tokens=input_tokens,
            input_positions=input_positions,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            block_tables=block_tables,
            pooling_metadata=None,  # Pooling is deprecated in V0
            running_requests_ids=running_requests_ids,
            finished_requests_ids=finished_requests_ids,
            lora_requests=lora_requests,
            lora_mapping=lora_mapping)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForRBLN,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[SamplerOutput]:

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        hidden_states = self.model(model_input=model_input)

        # Compute the logits.
        # select last sequence in logits
        if hidden_states is not None:
            assert hidden_states.dim() == 3
            hidden_states = hidden_states[:, -1, :]
            assert hidden_states.dim() == 2

        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    def set_active_loras(self, lora_requests: List[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        is_prefill = lora_mapping.is_prefill
        max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        num_reqs = len(lora_requests)
        adapter_ids = [
            0 if lora_request is None else lora_request.adapter_id
            for lora_request in lora_requests
        ]
        # Padding
        if not is_prefill and num_reqs < max_num_reqs:
            adapter_ids += [0] * (max_num_reqs - num_reqs)
        self.model.model.set_lora_int_ids(adapter_ids)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise RuntimeError("It is not required in vLLM RBLN.")

    def remove_lora(self, lora_id: int) -> bool:
        raise RuntimeError("It is not required in vLLM RBLN.")

    def pin_lora(self, lora_id: int) -> bool:
        raise RuntimeError("It is not required in vLLM RBLN.")

    def list_loras(self) -> Set[int]:
        rbln_cfg = getattr(self.model, "rbln_model_config", None)
        lora_cfg = getattr(rbln_cfg, "lora_config", None)
        if lora_cfg is None:
            raise ValueError("The model is not compiled with LoRA.")

        lora_adapters = getattr(self.model.rbln_model_config.lora_config,
                                "adapters", [])
        adapter_ids = {a.lora_int_id for a in lora_adapters}
        return adapter_ids
