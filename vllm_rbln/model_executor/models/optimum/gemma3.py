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
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm.config import ModelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma3_mm import (Gemma3ImageInputs,
                                                  Gemma3ImagePixelInputs)

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


@dataclass
class SlidingWindowEntry:
    local_table_id: int
    padded_cache_length: int
    attention_mask: torch.Tensor


class RBLNOptimumGemma3ForConditionalGeneration(RBLNOptimumModelBase,
                                                RBLNOptimumDecoderMixin):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        super().__init__(model_config=model_config,
                         scheduler_config=scheduler_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            padding_value=self.padding_value,
            vocab_size=model_config.get_vocab_size,
            use_multiple_decoder=getattr(self.model.rbln_config.language_model,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.
            decoder_batch_sizes,
        )

        self.sliding_window_table: Dict[str, SlidingWindowEntry] = {}

    def pad_local_table_items(
        self,
        sliding_window_table_ids: List[int],
        attention_masks: List[torch.Tensor],
        position_ids: torch.Tensor,
        padded_cache_lengths: List[int],
        request_nums: int,
        padded_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Validate input
        if request_nums > 0 and not attention_masks:
            raise ValueError(
                "attention_masks cannot be empty when request_nums > 0.")

        position_id_dtype = position_ids.dtype
        seq_len = attention_masks[0].shape[1] if attention_masks else 0

        # Determine padding value for local_block_table_id
        used_ids = set(sliding_window_table_ids)
        pad_value = next(
            (i for i in range(self.decoder_batch_size) if i not in used_ids),
            0)

        local_block_table_id = torch.full(
            (padded_batch_size, 1),
            pad_value,
            dtype=torch.int16,
        )
        local_block_table_id[:request_nums] = torch.tensor(
            sliding_window_table_ids, dtype=torch.int16).unsqueeze(1)

        padded_cache_lengths_tensor = torch.zeros(padded_batch_size,
                                                  1,
                                                  dtype=position_id_dtype)
        padded_cache_lengths_tensor[:request_nums] = torch.tensor(
            padded_cache_lengths, dtype=position_id_dtype).unsqueeze(1)

        attention_mask_dtype = attention_masks[
            0].dtype if attention_masks else torch.bool
        attention_mask = torch.zeros(padded_batch_size,
                                     seq_len,
                                     dtype=attention_mask_dtype)
        if attention_masks:
            attention_mask[:request_nums] = torch.cat(attention_masks)

        # cache_positions - the index including padding between text and image
        # padded_cache_lengths_tensor - the size of padding
        # position_ids - the index of the token to be decoded in the sequence.
        cache_positions = torch.zeros(padded_batch_size,
                                      1,
                                      dtype=position_id_dtype)
        cache_positions[:request_nums] = (
            position_ids[:request_nums] +
            padded_cache_lengths_tensor[:request_nums])

        return local_block_table_id, attention_mask, cache_positions

    def select_local_block_table_value(
        self,
        is_prompt: bool,
        input_ids: torch.Tensor,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
    ) -> Tuple[list[int], list[int], list[torch.Tensor]]:
        if is_prompt:
            # Generate attention mask without padding
            attention_mask = torch.ones_like(input_ids).squeeze(0)

            # Determine sliding_window_table_id
            # FIXME:
            # finished_requests_ids is typed as list[str],
            # but used as list[int].
            if finished_requests_ids:
                first_id = finished_requests_ids[0]
                local_table_id = self.sliding_window_table[
                    first_id].local_table_id

                for request_id in finished_requests_ids:
                    self.sliding_window_table.pop(request_id)
            else:
                used_ids = {
                    v.local_table_id
                    for v in self.sliding_window_table.values()
                }
                available_ids = set(range(self.decoder_batch_size)) - used_ids
                assert len(available_ids) > 0
                local_table_id = min(available_ids)

            if len(self.sliding_window_table) > self.decoder_batch_size:
                raise ValueError(
                    "Sliding window table size must not exceed the batch size."
                )

            return [local_table_id], [], [attention_mask]

        else:
            local_table_ids: List[int] = []
            padded_cache_lengths: List[int] = []
            attention_masks: List[torch.Tensor] = []

            for request_id in running_requests_ids:
                sliding_window = self.sliding_window_table[request_id]
                local_table_ids.append(sliding_window.local_table_id)
                padded_cache_lengths.append(sliding_window.padded_cache_length)
                attention_masks.append(sliding_window.attention_mask)

            return local_table_ids, padded_cache_lengths, attention_masks

    def get_pixel_values(self, model_input: ModelInputForRBLN):
        image_input = None

        if model_input.multi_modal_kwargs:
            image_input = self._parse_and_validate_image_input(
                **model_input.multi_modal_kwargs)
            if image_input is not None:
                assert image_input["type"] == "pixel_values"
                pixel_values = image_input["pixel_values"]

        else:
            pixel_values = None

        return pixel_values

    def forward(self, model_input: ModelInputForRBLN) -> torch.Tensor:
        input_ids = model_input.input_tokens
        position_ids = model_input.input_positions.to(torch.int32)
        block_tables = model_input.block_tables
        is_prompt = model_input.sampling_metadata.num_prompts > 0

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids, padded_cache_lengths, attention_masks = \
            self.select_local_block_table_value(
                is_prompt,
                input_ids,
                running_requests_ids,
                finished_requests_ids,
            )

        kwargs = self.preprocess_for_decoder(
            is_prompt,
            block_tables,
            input_ids,
            position_ids,
        )

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        if is_prompt:
            cache_position = kwargs.pop("cache_position")
        else:
            position_ids = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            inputs_embeds = None
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx],
                                                dtype=torch.int16)
            # token_type_ids model_input != token_type_ids of gemma3
            # https://github.com/huggingface/transformers/blob/d0c9c66d1c09df3cd70bf036e813d88337b20d4c/src/transformers/models/gemma3/processing_gemma3.py#L143
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[input_ids ==
                           self.model.config.image_token_index] = 1

            pixel_values = self.get_pixel_values(model_input)
            inputs_embeds = self.model._preprocess_prefill(
                input_ids, inputs_embeds, pixel_values)
            if self.model.language_model.prefill_decoder is None:
                raise version_error
            attention_mask = attention_masks[0]
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                attention_mask=attention_mask,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
                token_type_ids=token_type_ids,
            )
            logits = output.logits
            updated_attention_mask = output.attention_mask
            updated_padded_cache_length = output.padded_cache_lengths

            assert len(running_requests_ids) == 1
            self.sliding_window_table[
                running_requests_ids[0]] = SlidingWindowEntry(
                    sliding_window_table_ids[0], updated_padded_cache_length,
                    updated_attention_mask)
        else:
            if self.model.language_model.decoders is None:
                raise ValueError("Decoders is None")
            padded_batch_size = kwargs.pop("padded_batch_size",
                                           self.decoder_batch_size)
            self.model.language_model.decoder = \
                self.model.language_model.decoders[padded_batch_size]
            local_block_table_id, attention_mask, cache_position \
                    = self.pad_local_table_items(sliding_window_table_ids,
                                                 attention_masks,
                                                 position_ids,
                                                 padded_cache_lengths,
                                                 request_nums,
                                                 padded_batch_size)

            rows = torch.arange(attention_mask.size(0))
            cols = cache_position.squeeze(1)

            attention_mask[rows, cols] = 1

            logits = self.model.language_model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_table_id,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

            # Update attention mask of newly generated token
            for idx, request_id in enumerate(running_requests_ids):
                self.sliding_window_table[
                    request_id].attention_mask = attention_mask[idx:idx + 1]

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

    def _parse_and_validate_image_input(
            self, **kwargs: Any) -> Optional[Gemma3ImageInputs]:
        pixel_values: torch.Tensor = kwargs.get("pixel_values")
        num_crops: torch.Tensor = kwargs.get("num_crops")
        embed_is_patch = kwargs.get("embed_is_patch")
        num_embeds = kwargs.get("num_embeds")

        pixel_values = pixel_values.squeeze(0)

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        return Gemma3ImagePixelInputs(
            type="pixel_values",
            pixel_values=self._validate_pixel_values(pixel_values),
            num_patches=num_crops + 1,
            embed_is_patch=embed_is_patch,
            num_embeds=num_embeds,
        )

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.model.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data
