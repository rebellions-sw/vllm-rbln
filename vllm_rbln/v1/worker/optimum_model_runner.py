# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import logging
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import BatchedTensorInputs
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.model_loader.rbln_model_loader import (
    get_optimum_model)
from vllm_rbln.model_executor.models.optimum import ModelInputForRBLN
from vllm_rbln.v1.sample.sampler import WARM_UP_CONFIGS
from vllm_rbln.v1.sample.sampler import Sampler as RBLNSampler
from vllm_rbln.v1.worker.multimodal import RBLNOptimumMultiModalKwargs

logger = init_logger(__name__)


class RBLNOptimumModelRunner(LoRAModelRunnerMixin):

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

        # Sampler
        self.use_rbln_sampler = envs.RBLN_SAMPLER
        logger.info("Using RBLN sampler: %s", self.use_rbln_sampler)
        sampler = RBLNSampler() if self.use_rbln_sampler else Sampler()
        if self.use_rbln_sampler:
            # Use torch.compile for optimized RBLN sampler
            sampler = torch.compile(sampler, dynamic=False, fullgraph=False)

        self.sampler = sampler
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

    def load_model(self) -> None:
        self.model = get_optimum_model(vllm_config=self.vllm_config)

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
            # FIXME If local block table exists in the model,
            # clear the local block table.
            # Because in the case of LLM (not AsyncLLMEngine),
            # `finished_request_ids` is provided separately
            # from new requests.
            # It is a temporary solution.
            if getattr(self.model, "attention_manager", None):
                self.model.attention_manager.clear()
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT
        # Prepare the decoder inputs.
        model_input = self._prepare_inputs(scheduler_output)
        hidden_states = self.model(model_input)
        # FIXME [batch_size, 1, vocab_size] -> [batch_size, vocab_size]
        hidden_states = hidden_states.squeeze(1)
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
        num_blocks: int,
        *,
        pad_value: int = -1,
    ) -> torch.Tensor:
        """Mask (pad) unused block slots in-place.

        Sets entries beyond `num_blocks` to `pad_value`.
        Use `pad_value=0` for v1 (dummy block id 0), or pass your own padding.
        """
        if num_blocks < 0:
            raise ValueError("num_blocks must be >= 0")

        if block_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_ids must be int32 or int64")

        # In V1, block ID 0 is reserved as a dummy "null_block",
        # so valid blocks start from 1.
        # The compiler, however, expects valid blocks to start from 0.
        block_ids = block_ids - 1
        max_blocks = block_ids.size(-1)
        k = max(0, min(num_blocks, max_blocks))  # clamp to [0, max_blocks]
        if k < max_blocks:
            block_ids.narrow(-1, k, max_blocks - k).fill_(pad_value)

        return block_ids

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelInputForRBLN:
        """
        :return: ModelInputForRBLN[
            input_tokens: Token IDs,
            input_positions: Potision IDs,
            sampling_metadata, pooling_metadata: It is `None` in V1,
            multi_modal_kwargs: Batched multi-modal data,
            block_tables: [num_reqs, num_blocks_per_req] shaped tensor,
            running_requests_ids: RUNNING request IDs,
            finished_requests_ids: FINISHED request IDs in between
                the previous and the current steps,
            token_type_ids: Not used,
            is_prompt: It is used only in V1
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        num_prefill_reqs = len(scheduler_output.scheduled_new_reqs)
        num_decode_reqs = len(scheduler_output.scheduled_cached_reqs)
        finished_requests_ids = scheduler_output.finished_req_ids
        is_prefill = False

        if num_prefill_reqs > 1 or (num_prefill_reqs >= 1
                                    and num_decode_reqs > 0):
            raise RuntimeError(
                "Prefill stage request cannot processed with other requests.")

        if num_prefill_reqs > 0 or \
            (num_decode_reqs == 1 and \
            scheduler_output.scheduled_cached_reqs[0].resumed_from_preemption):
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

    def _get_multi_kwargs(self, scheduler_output: "SchedulerOutput") -> int:
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs = list[RBLNOptimumMultiModalKwargs]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        assert len(grouped_mm_inputs_list) == 1

        grouped_mm_inputs = grouped_mm_inputs_list[0]
        batched_mm_inputs = RBLNOptimumMultiModalKwargs.batch(
            grouped_mm_inputs)
        batched_mm_inputs = RBLNOptimumMultiModalKwargs.as_kwargs(
            batched_mm_inputs,
            device=self.device,
        )

        return batched_mm_inputs

    def _prepare_prefill(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[RBLNOptimumMultiModalKwargs], list[str]]:
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        running_request_ids = []
        batched_mm_inputs: Optional[BatchedTensorInputs] = None
        is_preempted = False

        if len(scheduler_output.scheduled_new_reqs) == 1:
            reqs = scheduler_output.scheduled_new_reqs
        elif len(scheduler_output.scheduled_cached_reqs) == 1:
            reqs = scheduler_output.scheduled_cached_reqs
            is_preempted = True
        else:
            raise RuntimeError(
                "Prefill stage request cannot processed with other requests.")

        req_id = self.input_batch.req_ids[0]
        num_blocks_per_req = self.input_batch.block_table.block_tables[
            0].num_blocks_per_row
        block_tables_cpu = self.input_batch.block_table.block_tables[
            0].get_cpu_tensor()

        for req_id, scheduled in zip(self.input_batch.req_ids, reqs):
            req_index = self.input_batch.req_id_to_index[req_id]
            if is_preempted:
                logger.warning("Request %s is resumed.", req_id)
                num_token = int(self.input_batch.num_tokens[req_index])
                prompt_tokens = self.input_batch.token_ids_cpu[
                    req_index][:num_token]
            else:
                prompt_tokens = np.array(scheduled.prompt_token_ids)
            seq_len = len(prompt_tokens)
            input_positions = list(range(seq_len))
            num_blocks = num_blocks_per_req[req_index]
            block_table = block_tables_cpu[req_index]
            block_table = self.mask_block_table(block_table, num_blocks)
            logger.debug(
                "Request %s is now scheduled. Prompt tokens: %s, "
                "Already generated tokens: %s, Allocated block(s): %s", req_id,
                len(self.requests[req_id].prompt_token_ids),
                len(self.requests[req_id].output_token_ids),
                block_table.tolist())
            running_request_ids.append(req_id)

        if self.is_multimodal_model:
            batched_mm_inputs = self._get_multi_kwargs(scheduler_output)

        input_tokens = torch.tensor(prompt_tokens).unsqueeze(0)
        input_positions = torch.tensor(input_positions).unsqueeze(0)
        block_table = block_table.unsqueeze(0)

        return input_tokens, input_positions, block_table, \
            batched_mm_inputs, running_request_ids

    def _prepare_decode(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        block_tables_list = []
        running_request_ids = []
        block_tables_cpu = self.input_batch.block_table.block_tables[
            0].get_cpu_tensor()
        num_blocks_per_req = self.input_batch.block_table.block_tables[
            0].num_blocks_per_row

        for req_id, scheduled in zip(self.input_batch.req_ids,
                                     scheduler_output.scheduled_cached_reqs):
            req_index = self.input_batch.req_id_to_index[req_id]
            input_position = int(self.input_batch.num_tokens[req_index] - 1)
            input_tokens.append(
                [self.input_batch.token_ids_cpu[req_index][input_position]])
            input_positions.append([input_position])
            num_blocks = num_blocks_per_req[req_index]
            block_table = block_tables_cpu[req_index]
            block_table = self.mask_block_table(block_table, num_blocks)
            block_tables_list.append(block_table)
            running_request_ids.append(req_id)

        input_tokens = torch.tensor(input_tokens)
        input_positions = torch.tensor(input_positions)
        block_tables = torch.stack(block_tables_list)

        return input_tokens, input_positions, block_tables, running_request_ids

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input NPU tensors for the model.

        The SamplingMetadata is updated and copied to the NPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        for req_id in scheduler_output.finished_req_ids:
            if logger.isEnabledFor(logging.DEBUG):
                block_ids = [
                    block_id - 1
                    for block_id in self.requests[req_id].block_ids[0]
                ]
                logger.debug(
                    "Request %s is finished. Prompt tokens: %s, "
                    "Generated tokens: %s, Freed block(s): %s", req_id,
                    len(self.requests[req_id].prompt_token_ids),
                    len(self.requests[req_id].output_token_ids), block_ids)
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
                        req_state.block_ids, req_data.new_block_ids):
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
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0
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
        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def dummy_sampler_run(self):
        if not self.use_rbln_sampler:
            return

        def set_sampling_tensors(input_batch, **params):
            input_batch.temperature_cpu_tensor.fill_(params["temperature"])
            input_batch.temperature.fill_(params["temperature"])

            optional_keys = [
                ("top_p", input_batch.top_p_cpu_tensor, input_batch.top_p),
                ("top_k", input_batch.top_k_cpu_tensor, input_batch.top_k),
                ("min_p", input_batch.min_p_cpu_tensor, input_batch.min_p),
                ("frequency_penalties",
                 input_batch.frequency_penalties_cpu_tensor,
                 input_batch.frequency_penalties),
                ("presence_penalties",
                 input_batch.presence_penalties_cpu_tensor,
                 input_batch.presence_penalties),
                ("repetition_penalties",
                 input_batch.repetition_penalties_cpu_tensor,
                 input_batch.repetition_penalties),
            ]

            for key, cpu_tensor, dev_tensor in optional_keys:
                val = params.get(key)
                if val is not None:
                    cpu_tensor.fill_(val)
                    dev_tensor.fill_(val)

        def populate_reqs(input_batch, base_config, batch_size):
            for i in range(batch_size):
                req_id = f"{base_config['name']}_req_{i}"
                input_batch._req_ids.append(req_id)
                input_batch.req_id_to_index[req_id] = i

                if base_config["all_greedy"]:
                    input_batch.greedy_reqs.add(req_id)
                elif base_config["all_random"]:
                    input_batch.random_reqs.add(req_id)

                for attr, req_set in [
                    ("top_p", input_batch.top_p_reqs),
                    ("top_k", input_batch.top_k_reqs),
                    ("frequency_penalties",
                     input_batch.frequency_penalties_reqs),
                    ("repetition_penalties",
                     input_batch.repetition_penalties_reqs),
                    ("presence_penalties",
                     input_batch.presence_penalties_reqs),
                ]:
                    if base_config.get(attr) is not None:
                        req_set.add(req_id)

        def clear_reqs(input_batch):
            input_batch._req_ids.clear()
            input_batch.req_id_to_index.clear()
            input_batch.greedy_reqs.clear()
            input_batch.random_reqs.clear()
            input_batch.top_p_reqs.clear()
            input_batch.top_k_reqs.clear()
            input_batch.frequency_penalties_reqs.clear()
            input_batch.repetition_penalties_reqs.clear()
            input_batch.presence_penalties_reqs.clear()

        def dummy_run_batches(base_config):
            for batch_size in range(1, self.input_batch.max_num_reqs + 1):
                input_batch = self.input_batch
                populate_reqs(input_batch, base_config, batch_size)

                metadata = input_batch._make_sampling_metadata()
                metadata.no_penalties = base_config["no_penalties"]
                metadata.all_greedy = base_config["all_greedy"]
                metadata.all_random = base_config["all_random"]

                if (not metadata.no_penalties
                        and metadata.prompt_token_ids is None):
                    metadata.prompt_token_ids = torch.zeros((batch_size, 1),
                                                            dtype=torch.long,
                                                            device="cpu")

                logger.info(
                    "Running dummy compile with batch_size=%d, vocab_size=%d",
                    batch_size, input_batch.vocab_size)
                logger.info("Sampling metadata: %s", metadata)

                with torch.inference_mode():
                    empty_logits = torch.empty(batch_size,
                                               input_batch.vocab_size,
                                               dtype=torch.float32)
                    _ = self.sampler(logits=empty_logits,
                                     sampling_metadata=metadata)

                clear_reqs(input_batch)

        for config in WARM_UP_CONFIGS:
            logger.info("Running dummy sampler config: %s", config["name"])

            set_sampling_tensors(
                self.input_batch,
                temperature=config["temperature"],
                top_p=config.get("top_p"),
                top_k=config.get("top_k"),
                min_p=config.get("min_p"),
                frequency_penalties=config.get("frequency_penalties"),
                repetition_penalties=config.get("repetition_penalties"),
                presence_penalties=config.get("presence_penalties"),
            )

            dummy_run_batches(config)
