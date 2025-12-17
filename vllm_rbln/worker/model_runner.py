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
import os
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union, cast)

import torch
from torch import nn
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group, get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs,
                             MultiModalPlaceholderMap, MultiModalRegistry)
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
from vllm_rbln.worker.metrics import PerformanceTracker

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
    attn_metadata: Optional["AttentionMetadata"] = None
    virtual_engine: Optional[int] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
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
        self.num_partition = self.max_model_len // self.block_size

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
        virtual_engine: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        list_input_block_ids: List[List[int]] = []

        block_size = self.runner.block_size
        num_blocks = self.runner.cache_config.num_gpu_blocks
        num_blocks_per_ve = num_blocks // \
            self.runner.parallel_config.pipeline_parallel_size
        ve_offset = num_blocks_per_ve * virtual_engine
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
            block_table = list(map(lambda v: v + ve_offset, block_table))
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

        dummy = num_blocks
        # make_tensor_with_pad takes List[List[]] as input
        # To make it work, input_block_ids is expanded
        input_block_ids = make_tensor_with_pad(list_input_block_ids,
                                               max_len=self.num_partition,
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

        return (input_tokens, input_positions, input_block_ids)

    def _prepare_decode(
        self,
        data: ModelInputData,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0

        list_input_block_ids: List[List[int]] = []
        block_size = self.block_size
        num_blocks = self.runner.cache_config.num_gpu_blocks
        num_blocks_per_ve = num_blocks // \
            self.runner.parallel_config.pipeline_parallel_size
        ve_offset = num_blocks_per_ve * virtual_engine
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
                block_table = list(map(lambda v: v + ve_offset, block_table))
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
        dummy = num_blocks
        batch_padding_size = self.max_num_seqs - len(data.input_tokens)
        data.input_tokens.extend([[0]] * batch_padding_size)
        data.input_positions.extend([[0]] * batch_padding_size)
        list_input_block_ids.extend([[dummy]] * batch_padding_size)

        input_block_ids = make_tensor_with_pad(list_input_block_ids,
                                               max_len=self.num_partition,
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

        assert input_tokens.shape[0] == self.max_num_seqs
        assert input_positions.shape[0] == self.max_num_seqs
        assert input_block_ids.shape[0] == self.max_num_seqs

        return (input_tokens, input_positions, input_block_ids)

    def build(
        self,
        virtual_engine: int = 0,
    ) -> ModelInputForRebel:
        assert self.seq_group_metadata_list is not None
        seq_group_metadata_list = self.seq_group_metadata_list
        is_prompt = seq_group_metadata_list[0].is_prompt
        input_data = self.input_data
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions,
             input_block_ids) = self._prepare_prompt(input_data,
                                                     seq_group_metadata_list,
                                                     virtual_engine)
        else:
            (input_tokens, input_positions,
             input_block_ids) = self._prepare_decode(input_data,
                                                     seq_group_metadata_list,
                                                     virtual_engine)

        attn_metadata = self.attn_metadata_builder.build(
            input_data.seq_lens, input_data.query_lens, input_block_ids, -1)
        return self.model_input_cls(
            input_tokens=input_tokens,
            input_positions=input_positions,
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
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
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

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        self.sampler = get_sampler()

        # Lazy initialization
        self.compute_logits_model: nn.Module

        if hasattr(self, "_builder_cls"):
            # multi-step model runner does not have `_builder_cls`
            self.builder = self._builder_cls(
                cast(RBLNModelRunner, weakref.proxy(self)))
        if envs.VLLM_RBLN_METRICS:
            self.performance_tracker = PerformanceTracker()
            self.performance_tracker.register_cleanup()

    def compile_model(self, model):
        TP = get_tp_group()
        PP = get_pp_group()
        DP = get_dp_group()

        process_group_dict = {}
        process_group_dict[TP.device_group.group_name] = TP.ranks
        process_group_dict[TP.cpu_group.group_name] = TP.ranks
        process_group_dict[PP.device_group.group_name] = PP.ranks
        process_group_dict[PP.cpu_group.group_name] = PP.ranks
        process_group_dict[DP.device_group.group_name] = DP.ranks
        process_group_dict[DP.cpu_group.group_name] = DP.ranks

        options = {
            "compile_context": self.compile_context,
            "tensor_parallel_size": envs.VLLM_RBLN_TP_SIZE,
            "process_group_dict": process_group_dict
        }
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            options["cache_dir"] = os.path.join(
                envs.VLLM_CACHE_ROOT,
                ("rsd_cache_dir"
                 if envs.VLLM_RBLN_TP_SIZE > 1 else "cache_dir"))
        if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
            options["mode"] = "strict"

        compiled_model = torch.compile(
            model,
            backend="rbln",
            options=options,
            dynamic=False,
        )
        return compiled_model

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

        self.compute_logits_model = self.model
        if self.model_config.is_multimodal_model and hasattr(
                self.model.get_language_model(), "logits_processor"):
            self.compute_logits_model = self.model.get_language_model()

        logger.info("load_model = %s", self.model)
        logger.info("model_config.num_layers = %d",
                    self.model_config.get_num_layers(self.parallel_config))

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            selected_token_indices: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds)

            if get_pp_group().is_last_rank:
                # last rank create real model output
                if selected_token_indices is not None:
                    # aten::select -> adv_index -->
                    #     contrib_dynamic_take (tensor -> scalar)
                    # aten::index_select --> take -->
                    #     contrib_dynamic_take (tensor -> scalar)
                    model_output = model_output[:, selected_token_indices]
                logits = self.compute_logits_model.compute_logits(
                    model_output, None)
                return logits.view(-1, logits.size(-1))

            # non last rank create intermediate tensors, bypass it
            return model_output

        if self.model_config.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            self.model_executable = model_wrapper
        else:
            # NOTE - refer to pytorch 2.5 release notes
            # torch.compile regional compilation without recompilations
            # To prevent nn.modules parameters to be model input, set false
            # if this flag is set,
            # nn.modules parameters are treated as model input
            torch._dynamo.config.inline_inbuilt_nn_modules = False
            # RBLN compile context to mark static address for kv cache tensor
            # if tensor is set to have static address,
            # similar to RBLN kv cache binding
            from rebel.compile_context import CompileContext

            self.compile_context = CompileContext(use_weight_sharing=True)
            compiled_graph = self.compile_model(model_wrapper)
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
            seq_group_metadata_list, finished_requests_ids, virtual_engine)

        if get_pp_group().is_last_rank:
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
        else:
            sampling_metadata = None

        is_prompt = seq_group_metadata_list[
            0].is_prompt if seq_group_metadata_list else None
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   virtual_engine=virtual_engine,
                                   is_prompt=is_prompt)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForRebelWithSamplingMetadata,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        assert kv_caches is not None
        if num_steps > 1:
            raise ValueError(
                "Rebel worker does not support multi-step execution.")

        execute_model_kwargs = {}
        if previous_hidden_states is not None:
            execute_model_kwargs.update(
                {"previous_hidden_states": previous_hidden_states})

        assert model_input.attn_metadata is not None
        token_indices = None
        if get_pp_group().is_last_rank:
            assert model_input.sampling_metadata is not None
            num_prefills = model_input.attn_metadata.num_prefills
            selected_token_indices = \
                model_input.sampling_metadata.selected_token_indices
            len_token_indices = len(selected_token_indices)
            if num_prefills > 0:
                assert len_token_indices == 0 or len_token_indices == 1
                num_prefill_tokens = \
                    model_input.attn_metadata.num_prefill_tokens
                token_indices = torch.tensor(
                    [num_prefill_tokens - 1],
                    dtype=selected_token_indices.dtype)
                if len_token_indices == 1:
                    assert torch.equal(selected_token_indices, token_indices)

        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            # RBLN compile context is much similar to vLLM forward context
            if model_input.attn_metadata is not None:
                model_input.attn_metadata.kv_caches = kv_caches

            start_time = time.perf_counter()
            logits_or_intermediate_states = self.model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                selected_token_indices=token_indices,
                **execute_model_kwargs,
            )
            end_time = time.perf_counter()
            if envs.VLLM_RBLN_METRICS:
                # Record performance metrics
                execution_time = end_time - start_time
                if model_input.is_prompt:
                    total_tokens = sum(model_input.query_lens
                                       ) if model_input.query_lens else 0
                    self.performance_tracker.record_prefill(
                        execution_time, total_tokens)
                else:
                    num_seqs = len(
                        model_input.seq_lens) if model_input.seq_lens else 0
                    self.performance_tracker.record_decode(
                        execution_time, num_seqs)
            if get_pp_group(
            ).is_last_rank and not envs.VLLM_RBLN_LOGITS_ALL_GATHER:
                # Gather logits for TP
                logits_processor = self.compute_logits_model.logits_processor
                logits_or_intermediate_states = logits_or_intermediate_states \
                                                .unsqueeze(0)
                logits_or_intermediate_states = logits_processor._gather_logits(
                    logits_or_intermediate_states)
                logits_or_intermediate_states = logits_or_intermediate_states \
                                                .squeeze(0)

        if not get_pp_group().is_last_rank:
            intermediate_states = logits_or_intermediate_states
            assert isinstance(intermediate_states, IntermediateTensors)
            return intermediate_states

        # Compute the logits. -> moved to model executable
        if num_prefills > 0 and len_token_indices != 0:
            logits = logits_or_intermediate_states
        else:
            logits = logits_or_intermediate_states[selected_token_indices]

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output = self.sampler(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        assert self.return_hidden_states is False, \
            "Rebel worker does not support return_hidden_states."

        return [output]

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None,
        virtual_engine: int = 0,
    ) -> ModelInputForRebelWithSamplingMetadata:
        self.builder.prepare(finished_requests_ids)
        self.builder.set_seq_group_list(seq_group_metadata_list)

        return self.builder.build(virtual_engine)  # type: ignore

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    def _dummy_run(self,
                   model_inputs: Tuple[ModelInputForRebelWithSamplingMetadata,
                                       ModelInputForRebelWithSamplingMetadata],
                   kv_caches: List[torch.Tensor]) -> None:
        # Run the model with the dummy inputs.
        for model_input in model_inputs:
            assert model_input.input_tokens is not None
            batch_size, seq_len = model_input.input_tokens.shape
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = \
                    self.model.make_empty_intermediate_tensors(
                    batch_size=batch_size * seq_len,
                    dtype=self.model_config.dtype,
                    device=self.device)
                intermediate_tensors = IntermediateTensors({
                    key:
                    val.reshape((batch_size, seq_len, -1))
                    for key, val in intermediate_tensors.items()
                })

            self.execute_model(model_input, kv_caches, intermediate_tensors)

        return
