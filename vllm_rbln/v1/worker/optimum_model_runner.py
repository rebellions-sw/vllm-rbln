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
from typing import Optional, Union

import torch
import torch.distributed
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_rbln.model_executor.model_loader.rbln_model_loader import (
    get_optimum_model)
from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN


class RBLNOptimumModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        # self.compilation_config = vllm_config.compilation_config
        # self.lora_config = vllm_config.lora_config
        # self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        # self.speculative_config = vllm_config.speculative_config
        # self.prompt_adapter_config = vllm_config.prompt_adapter_config
        # self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_multimodal_model = model_config.is_multimodal_model
        self.is_pooling_model = model_config.pooler_config is not None
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=self.mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Sampler
        self.sampler = Sampler()
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        self.use_aux_hidden_state_outputs = False

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
        )

        # Cache the device properties.
        # self._init_device_properties()

        # NOTE The shape of input_ids, positions are modified for optimum
        self.input_ids = torch.zeros(self.max_num_reqs,
                                     self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_reqs,
                                     self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)

        # None in the first PP rank. The rest are set after load_model.
        # TODO(eunji) It will be implemented for PP
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)

    def load_model(self) -> None:
        self.model = get_optimum_model(model_config=self.model_config,
                                       scheduler_config=self.scheduler_config)

    def get_model(self) -> nn.Module:
        return self.model

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)
        sampling_metadata = self.input_batch.sampling_metadata
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT
        # Prepare the decoder inputs.
        model_input = self._prepare_inputs(scheduler_output)
        hidden_states = self.model(model_input)
        logits = self.model.compute_logits(hidden_states, None)
        sampler_output = self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        valid_sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = valid_sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = valid_sampled_token_ids.tolist()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
        )

    def mask_block_table(
        self,
        block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """This function serves as an interface to convert VLLM block tables
        to the format expected by Optimum-RBLN.

        In V1, the block with block_id 0 is used as a dummy block
        called null_block, so valid blocks start from 1.
        
        However, in Optimum-RBLN, the last block is used as the dummy block,
        and valid blocks start from 0.
        """
        block_ids = block_ids - 1
        dummy_block = self.cache_config.num_gpu_blocks
        block_ids[block_ids == -1] = dummy_block
        return block_ids

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelInputForRBLN:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            attention_cuda_graphs: whether attention can run in cudagraph
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        num_prefill_reqs = len(scheduler_output.scheduled_new_reqs)
        num_decode_reqs = len(scheduler_output.scheduled_cached_reqs)
        finished_requests_ids = scheduler_output.finished_req_ids
        is_prefill = False

        if num_prefill_reqs > 1 or (num_prefill_reqs >= 1
                                    and num_decode_reqs > 0):
            raise RuntimeError(
                "prefill stage request cannot processed with other requests.")

        if num_prefill_reqs > 0:
            is_prefill = True

        if is_prefill:
            input_ids, positions, block_tables, \
            multi_modal_kwargs, running_request_ids \
                = self._prepare_prefill(scheduler_output)
        else:
            input_ids, positions, block_tables, running_request_ids \
                = self._prepare_decode(scheduler_output)

        # TODO interemediate_tensor should be set
        model_input = ModelInputForRBLN(
            input_tokens=input_ids,
            input_positions=positions,
            sampling_metadata=None,
            multi_modal_kwargs=multi_modal_kwargs if is_prefill else None,
            block_tables=block_tables,
            running_requests_ids=running_request_ids,
            finished_requests_ids=list(finished_requests_ids),
            token_type_ids=None,
            pooling_metadata=None,  # FIXME
            is_prompt=is_prefill)
        return model_input

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        # TODO It is temporary basic attention setting
        head_size = self.model_config.get_head_size()
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)

        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_idx in range(num_layers):
            kv_cache_spec[str(layer_idx)] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=self.kv_cache_dtype,
                use_mla=False,
            )
        return kv_cache_spec

    def _prepare_prefill(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[MultiModalKwargs], list[str]]:
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        running_request_ids = []
        multi_modal_kwargs: Optional[MultiModalKwargs] = None

        assert len(scheduler_output.scheduled_new_reqs) == 1

        for req_id, scheduled in zip(self.input_batch.req_ids,
                                     scheduler_output.scheduled_new_reqs):
            req_index = self.input_batch.req_id_to_index[req_id]
            prompt_tokens = scheduled.prompt_token_ids
            input_tokens.append(prompt_tokens)
            seq_len = len(prompt_tokens)
            input_positions.append(list(range(seq_len)))
            block_table = self.input_batch.block_table.block_tables[
                0].block_table[req_index]
            block_table = self.mask_block_table(block_table)
            block_table = block_table.unsqueeze(0)
            running_request_ids.append(scheduled.req_id)

        if self.is_multimodal_model:
            raise NotImplementedError
        else:
            multi_modal_kwargs = None

        input_tokens = torch.tensor(input_tokens)
        input_positions = torch.tensor(input_positions)

        return input_tokens, input_positions, block_table, \
            multi_modal_kwargs, running_request_ids

    def _prepare_decode(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        block_tables_list = []
        running_request_ids = []
        # multi_modal_kwargs: Optional[MultiModalKwargs]

        for req_id, scheduled in zip(self.input_batch.req_ids,
                                     scheduler_output.scheduled_cached_reqs):
            req_index = self.input_batch.req_id_to_index[req_id]
            input_position = int(self.input_batch.num_tokens[req_index] - 1)
            input_tokens.append(
                [self.input_batch.token_ids_cpu[req_index][input_position]])
            input_positions.append([input_position])
            block_table = self.input_batch.block_table.block_tables[
                0].block_table[req_index]
            block_table = self.mask_block_table(block_table)
            block_tables_list.append(block_table)
            running_request_ids.append(scheduled.req_id)

        # if self.is_multimodal_model:
        #     raise NotImplementedError
        # else:
        #     multi_modal_kwargs = None

        input_tokens = torch.tensor(input_tokens)
        input_positions = torch.tensor(input_positions)
        block_tables = torch.stack(block_tables_list)

        return input_tokens, input_positions, block_tables, running_request_ids

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        pass

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input NPU tensors for the model.

        The SamplingMetadata is updated and copied to the NPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.extend(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_block_ids in zip(  # type: ignore[call-overload]
                        req_state.block_ids,
                        req_data.new_block_ids,
                        strict=True):
                    block_ids.extend(new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        # batch_changed = len(removed_req_indices) > 0
        # or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices.sort(reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        # batch_reordered = self._may_reorder_batch(scheduler_output)
        # if batch_changed or batch_reordered:
        #     self.input_batch.refresh_sampling_metadata()
