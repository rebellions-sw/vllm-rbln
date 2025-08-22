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

from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.pooling_params import PoolingParams
from vllm.sequence import (IntermediateTensors, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase

from vllm_rbln.model_executor.model_loader.rbln_model_loader import (
    get_optimum_model)
from vllm_rbln.model_executor.models.optimum import (
    ModelInputForRBLN, RBLNOptimumForEncoderModel, get_rbln_model_info,
    is_enc_dec_arch, is_pooling_arch)

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
            # NOTE It is for cases
            # where the pooling configuration files (modules.json, ...)
            # do not exist in the compiled Qwen3 embedding model directory.
            if vllm_config.model_config.pooler_config.pooling_type is None:
                vllm_config.model_config.pooler_config.pooling_type = "LAST"
                vllm_config.model_config.pooler_config.normalize = True

        ModelRunnerBase.__init__(self, vllm_config)

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
            .create_input_mapper(self.model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

    def load_model(self) -> None:
        self.model = get_optimum_model(vllm_config=self.vllm_config)

    def get_model(self):
        return self.model

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int],
               BatchedTensorInputs, torch.Tensor, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        token_type_ids: List[List[int]] = []

        seq_lens: List[int] = []
        multi_modal_inputs_list: List[MultiModalKwargs] = []
        block_tables: List[List[int]] = []
        running_requests_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = (seq_group_metadata.encoder_seq_data
                        if is_enc_dec_arch(self.model_config.hf_config) else
                        seq_group_metadata.seq_data[seq_id])

            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(seq_len)))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            if block_table is not None:
                block_tables.append(block_table)

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                # Process multi-modal data
                if self.mm_registry.has_processor(self.model_config):
                    mm_kwargs = mm_data
                else:
                    mm_kwargs = self.multi_modal_input_mapper(
                        mm_data,
                        seq_group_metadata.mm_processor_kwargs,
                    )
                multi_modal_inputs_list.append(mm_kwargs)

            if len(seq_group_metadata.token_type_ids) > 0:
                token_type_ids.append(seq_group_metadata.token_type_ids)

            request_id = seq_group_metadata.request_id
            running_requests_ids.append(request_id)

        max_seq_len = max(seq_lens) if not is_pooling_arch(
            self.model_config.hf_config) else self.model_config.max_model_len

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

        token_type_ids = make_tensor_with_pad(
            token_type_ids,
            max_len=max_seq_len,
            pad=0,
            dtype=torch.long,
            device=self.device) if len(token_type_ids) > 0 else None

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_inputs_list)
        return (input_tokens, input_positions, seq_lens, multi_modal_kwargs,
                block_tables, token_type_ids, running_requests_ids)

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
             block_tables, type_token_ids, running_requests_ids
             ) = self._prepare_prompt(seq_group_metadata_list)
            pooling_metadata = self._prepare_pooling(seq_group_metadata_list,
                                                     seq_lens)

        else:
            (input_tokens, input_positions, block_tables, running_requests_ids
             ) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None
            type_token_ids = None
            pooling_metadata = None

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

        return ModelInputForRBLN(input_tokens=input_tokens,
                                 input_positions=input_positions,
                                 sampling_metadata=sampling_metadata,
                                 multi_modal_kwargs=multi_modal_kwargs,
                                 block_tables=block_tables,
                                 token_type_ids=type_token_ids,
                                 pooling_metadata=pooling_metadata,
                                 running_requests_ids=running_requests_ids,
                                 finished_requests_ids=finished_requests_ids)

    def _prepare_pooling(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> PoolingMetadata:
        """Prepare PoolingMetadata for the sequence group metadata list."""
        seq_groups: List[Tuple[List[int], PoolingParams]] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            pooling_params = seq_group_metadata.pooling_params
            seq_groups.append((seq_ids, pooling_params))

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        pooling_metadata = PoolingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
        )

        return pooling_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForRBLN,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[SamplerOutput]:

        hidden_states = self.model(model_input=model_input)
        if isinstance(self.model, RBLNOptimumForEncoderModel):
            return [
                self.model.pool(hidden_states, model_input.pooling_metadata)
            ]

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
