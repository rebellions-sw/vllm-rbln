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

import dataclasses
import math
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union, cast)

import torch
from torch import nn
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalKwargs, MultiModalPlaceholderMap
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

TModelInputForRebel = TypeVar("TModelInputForRebel",
                              bound="ModelInputForRebel")
_PAD_SLOT_ID = -1


@dataclass(frozen=True)
class ModelInputForRebel(ModelRunnerInputBase):
    """
    Base class contains metadata needed for the base model forward pass on Rebel
    """

    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    virtual_engine: Optional[int] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "token_type_ids": self.token_type_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForRebel],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None
    ) -> TModelInputForRebel:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


@dataclass(frozen=True)
class ModelInputForRebelWithSamplingMetadata(ModelInputForRebel):
    """
    Used by the ModelRunner.
    """

    sampling_metadata: Optional["SamplingMetadata"] = None
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "token_type_ids": self.token_type_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForRebelWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForRebelBuilder(ModelRunnerInputBuilderBase[ModelInputForRebel]
                                ):

    class ModelInputData:

        def __init__(self, use_mrope: bool):
            self.use_mrope = use_mrope
            self.input_tokens: List[List[int]] = []
            self.input_positions: List[List[int]] = []
            self.token_type_ids: Optional[List[List[int]]] = []
            self.seq_lens: List[int] = []
            self.query_lens: List[int] = []
            self.prefill_block_tables: List[List[int]] = []
            self.decode_block_tables: List[List[int]] = []
            self.max_decode_seq_len: int = 0
            self.num_prefills: int = 0
            self.num_prefill_tokens: int = 0
            self.num_decode_tokens: int = 0
            # RBLN slot mapping modification
            # RBLN slot mapping is table between token ids
            # and [block_number, block_offset]
            self.slot_mapping: List[int] = []
            self.multi_modal_inputs_list: List[MultiModalKwargs] = []
            self.multi_modal_placeholder_maps: Dict[
                str, MultiModalPlaceholderMap] = defaultdict(
                    MultiModalPlaceholderMap)
            self.input_mrope_positions: List[List[int]] = [[]
                                                           for _ in range(3)]

    def __init__(self,
                 runner: "RBLNModelRunner",
                 finished_requests_ids: Optional[List[str]] = None) -> None:
        super().__init__()
        self.runner = runner
        self.max_num_seqs = runner.scheduler_config.max_num_seqs
        self.chunked_prefill = (runner.scheduler_config.chunked_prefill_enabled
                                or runner.cache_config.enable_prefix_caching)
        self.chunked_prefill_size = (
            runner.scheduler_config.max_num_batched_tokens)
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.cache_config.block_size
        self.device = self.runner.device
        self.max_model_len = self.runner.scheduler_config.max_model_len

        if self.runner.attn_backend is not None:
            # spec decode (e.g. Medusa) does not have atten backend
            attn_backend = self.runner.attn_backend
            self.attn_metadata_builder = attn_backend.get_builder_cls()(self)

    def prepare(self,
                finished_requests_ids: Optional[List[str]] = None) -> None:
        self.seq_group_metadata_list: List[SequenceGroupMetadata] = []
        self.input_data = ModelInputForRebelBuilder.ModelInputData(
            self.runner.model_config.uses_mrope)
        self.attn_metadata_builder.prepare()

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        self.seq_group_metadata_list.append(seq_group_metadata)

    def set_seq_group_list(
            self, seq_group_metadata_list: List[SequenceGroupMetadata]):
        self.seq_group_metadata_list = seq_group_metadata_list

    def _prepare_prompt(
        self,
        data: ModelInputData,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        list_input_block_ids: List[List[int]] = []

        block_size = self.runner.block_size
        assert (
            len(seq_group_metadata_list) == 1), f"seq_group_metadata_list: \
            len({len(seq_group_metadata_list)}) - {seq_group_metadata_list}"

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]

            token_chunk_size = seq_group_metadata.token_chunk_size
            computed_len = seq_data.get_num_computed_tokens()
            seq_len = min(seq_data.get_len(), computed_len + token_chunk_size)
            tokens = seq_data.get_token_ids()[computed_len:seq_len]

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            assert len(block_table) == math.ceil(seq_data.get_len() /
                                                 block_size)
            list_input_block_ids.append(block_table)
            data.input_tokens.append(tokens)
            data.input_positions.append(list(range(computed_len, seq_len)))
            data.num_prefills += 1
            data.num_prefill_tokens += len(tokens)
            data.query_lens.append(len(tokens))
            data.seq_lens.append(seq_len)
            for i, pos in enumerate(data.input_positions[0]):
                block_number = block_table[pos // block_size]
                block_offset = pos % block_size
                data.slot_mapping.append(block_number)
                data.slot_mapping.append(block_offset)

        max_seq_len = max(data.seq_lens)
        assert max_seq_len > 0

        num_partition = self.max_model_len // block_size
        dummy = self.runner.cache_config.num_gpu_blocks
        # make_tensor_with_pad takes List[List[]] as input
        # To make it work, input_block_ids is expanded
        input_block_ids = make_tensor_with_pad(list_input_block_ids,
                                               max_len=num_partition,
                                               pad=dummy,
                                               dtype=torch.long,
                                               device=self.device)
        # input_block_ids gets back in here.
        input_block_ids = input_block_ids.flatten().tolist()
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)

        prefill_size = (self.chunked_prefill_size if self.chunked_prefill else
                        1 << (math.ceil(math.log2(max_seq_len))))
        input_tokens = make_tensor_with_pad(data.input_tokens,
                                            max_len=prefill_size,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(data.input_positions,
                                               max_len=prefill_size,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)

        logger.info("[RBLN] model input builder, prepare_prompt")
        logger.info("\tpadded input_tokens = %s", input_tokens)
        logger.info("\tpadded input_positions = %s", input_positions)
        logger.info("\tinput_block_ids = %s", input_block_ids)
        logger.info("\tseq_lens = %s", data.seq_lens)
        logger.info("\tquery_lens = %s", data.query_lens)
        return (input_tokens, input_positions, input_block_ids)

    def _prepare_decode(
        self,
        data: ModelInputData,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0

        list_input_block_ids: List[List[int]] = []
        block_size = self.block_size
        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()

                seq_len = seq_data.get_len()
                token_position = seq_len - 1

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                assert len(block_table) >= 1

                list_input_block_ids.append(block_table)
                data.max_decode_seq_len = max(data.max_decode_seq_len, seq_len)
                data.input_tokens.append([generation_token])
                data.input_positions.append([token_position])
                data.num_decode_tokens += 1
                data.query_lens.append(1)
                data.seq_lens.append(seq_len)
                block_number = block_table[token_position // block_size]
                block_offset = token_position % block_size
                data.slot_mapping.append(block_number)
                data.slot_mapping.append(block_offset)

        # batch padding
        dummy = self.runner.cache_config.num_gpu_blocks
        batch_padding_size = self.max_num_seqs - len(data.input_tokens)
        data.input_tokens.extend([[0]] * batch_padding_size)
        data.input_positions.extend([[0]] * batch_padding_size)
        list_input_block_ids.extend([[dummy]] * batch_padding_size)

        num_partition = self.max_model_len // block_size
        input_block_ids = make_tensor_with_pad(list_input_block_ids,
                                               max_len=num_partition,
                                               pad=dummy,
                                               dtype=torch.long,
                                               device=self.device)

        input_tokens = make_tensor_with_pad(data.input_tokens,
                                            max_len=1,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(data.input_positions,
                                               max_len=1,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)

        logger.info("[RBLN] model input builder, prepare_decode")
        logger.info("\tpadded input_tokens = %s", data.input_tokens)
        logger.info("\tpadded input_positions = %s", data.input_positions)
        logger.info("\tinput_block_ids = %s", input_block_ids)
        logger.info("\tseq_lens = %s", data.seq_lens)
        logger.info("\tquery_lens = %s", data.query_lens)

        assert input_tokens.shape[0] == self.max_num_seqs
        assert input_positions.shape[0] == self.max_num_seqs
        assert input_block_ids.shape[0] == self.max_num_seqs

        return (input_tokens, input_positions, input_block_ids)

    def build(self) -> ModelInputForRebel:
        assert self.seq_group_metadata_list is not None
        seq_group_metadata_list = self.seq_group_metadata_list
        is_prompt = seq_group_metadata_list[0].is_prompt
        token_type_ids = seq_group_metadata_list[0].token_type_ids
        input_data = self.input_data
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions,
             input_block_ids) = self._prepare_prompt(input_data,
                                                     seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             input_block_ids) = self._prepare_decode(input_data,
                                                     seq_group_metadata_list)

        attn_metadata = self.attn_metadata_builder.build(
            input_data.seq_lens, input_data.query_lens, input_block_ids, -1)
        return self.model_input_cls(
            input_tokens=input_tokens,
            input_positions=input_positions,
            token_type_ids=token_type_ids,
            seq_lens=input_data.seq_lens,
            query_lens=input_data.query_lens,
            attn_metadata=attn_metadata,
        )


class RBLNModelRunner(ModelRunnerBase[ModelInputForRebelWithSamplingMetadata]):
    _model_input_cls: Type[ModelInputForRebelWithSamplingMetadata] = (
        ModelInputForRebelWithSamplingMetadata)
    _builder_cls: Type[ModelInputForRebelBuilder] = ModelInputForRebelBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
    ):
        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config

        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on RBLN. "
                           "The model will run without sliding window.")
        self.device = self.device_config.device
        self.pin_memory = False

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        needs_attn_backend = (num_attn_heads != 0
                              or self.model_config.is_attention_free)
        self.attn_backend = (get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        ) if needs_attn_backend else None)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        self.sampler = get_sampler()

        if hasattr(self, "_builder_cls"):
            # multi-step model runner does not have `_builder_cls`
            self.builder = self._builder_cls(
                cast(RBLNModelRunner, weakref.proxy(self)))

    def compile_model(self, model):
        if envs.RBLN_COMPILE_MODEL:
            if envs.RBLN_TP_SIZE > 1:
                compiled_model = torch.compile(
                    model,
                    backend="rbln",
                    options={
                        "compile_context": self.compile_context,
                        "cache_dir": "./rsd_cache_dir",
                        "tensor_parallel_size": envs.RBLN_TP_SIZE,
                    },
                    dynamic=False,
                )
            else:
                compiled_model = torch.compile(
                    model,
                    backend="rbln",
                    options={
                        "compile_context": self.compile_context,
                        "cache_dir": "./cache_dir",
                    },
                    dynamic=False,
                )

            return compiled_model
        else:
            return model

    # LLM attention block
    def attention_block(self, decoder_layer, hidden_states, residual,
                        input_positions):
        # attention input_layernorm
        if residual is None:
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = decoder_layer.input_layernorm(
                hidden_states, residual)
        # attention self_attn
        hidden_states = decoder_layer.self_attn(input_positions, hidden_states)
        # attention post_attention_layernorm
        hidden_states, residual = decoder_layer.post_attention_layernorm(
            hidden_states, residual)
        return hidden_states, residual

    # LLM decoder layer block
    def decoder_layer_block(self, decoder_layer, hidden_states, residual,
                            input_positions):
        hidden_states, residual = self.attention_block(decoder_layer,
                                                       hidden_states, residual,
                                                       input_positions)
        # mlp, fused_mode
        hidden_states = decoder_layer.mlp(hidden_states)
        return hidden_states, residual

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config).eval()

        logger.info("[RBLN] load_model = %s", self.model)
        logger.info("[RBLN] model_config.num_layers = %d",
                    self.model_config.get_num_layers(self.parallel_config))

        # NOTE - refer to pytorch 2.5 release notes
        # torch.compile regional compilation without recompilations
        # To prevent nn.modules parameters to be model input, set false
        # if this flag is set, nn.modules parameters are treated as model input
        torch._dynamo.config.inline_inbuilt_nn_modules = False
        # RBLN compile context to mark static address for kv cache tensor
        # if tensor is set to have static address,
        # similar to RBLN kv cache binding
        from rebel.compile_context import CompileContext

        self.compile_context = CompileContext(use_weight_sharing=True)
        compiled_graph = self.compile_model(self.model)
        self.model_executable = compiled_graph

    def get_model(self) -> nn.Module:
        return self.model

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForRebelWithSamplingMetadata:
        return (ModelInputForRebelWithSamplingMetadata.
                from_broadcasted_tensor_dict(
                    tensor_dict,
                    attn_backend=self.attn_backend,
                ))

    @torch.inference_mode()
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForRebelWithSamplingMetadata:
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        # Sampling metadata is only required for the final pp group
        generators = self.get_generators(finished_requests_ids)
        pin_memory = self.pin_memory
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            model_input.seq_lens,
            model_input.query_lens,
            self.device,
            pin_memory=pin_memory,
            generators=generators,
        )

        is_prompt = seq_group_metadata_list[
            0].is_prompt if seq_group_metadata_list else None
        logger.info("[RBLN] num_requests = %d", len(seq_group_metadata_list))
        logger.info("[RBLN] input_ids = %s", model_input.input_tokens)
        logger.info("[RBLN] positions = %s", model_input.input_positions)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   virtual_engine=virtual_engine,
                                   is_prompt=is_prompt)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForRebel,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[SamplerOutput]:
        assert kv_caches is not None
        if num_steps > 1:
            raise ValueError(
                "Rebel worker does not support multi-step execution.")

        execute_model_kwargs = {}
        if previous_hidden_states is not None:
            execute_model_kwargs.update(
                {"previous_hidden_states": previous_hidden_states})

        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            # RBLN compile context is much similar to vLLM forward context
            if model_input.attn_metadata is not None:
                model_input.attn_metadata.kv_caches = kv_caches
            for kv_cache in kv_caches:
                self.compile_context.mark_static_address(kv_cache)
            hidden_states = self.model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                **execute_model_kwargs,
            )
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            assert hidden_states.dim() == 2

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output = self.sampler(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            if model_input.is_prompt:
                output.prefill_hidden_states = hidden_states
            output.hidden_states = hidden_states
        return [output]

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForRebelWithSamplingMetadata:
        self.builder.prepare(finished_requests_ids)
        self.builder.set_seq_group_list(seq_group_metadata_list)

        return self.builder.build()  # type: ignore

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
