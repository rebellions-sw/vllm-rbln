# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import vllm.envs as env
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .base import ModelInputForRBLN, version_error
from .model_base import (RBLNOptimumDecoderMixin, RBLNOptimumDictTableMixin,
                         RBLNOptimumModelBase)

logger = init_logger(__name__)


@dataclass
class SlidingWindowEntry:
    local_table_id: int
    # padded_cache_length and attention mask are
    # for sliding window attention with image.
    padded_cache_length: Optional[int]
    attention_mask: Optional[torch.Tensor]


class RBLNOptimumSlidingWindowAttentionMixin(RBLNOptimumDictTableMixin):
    """
    It is for the model that supports Sliding Window Attention.
    """

    def setup_sliding_window_attention_mixin(self, vllm_config,
                                             sliding_window) -> None:
        # Check the sliding window configuration is set
        # if vllm_config.model_config.disable_sliding_window:
        # raise RuntimeError("Please set `disable_sliding_window`=False")
        if sliding_window and \
            vllm_config.model_config.disable_sliding_window:
            raise RuntimeError("The model requires sliding window attention."
                               "Please set `disable_sliding_window`=False.")

        # self.sliding_window_size = self.model_config.hf_config.sliding_window
        # if self.sliding_window_size:
        self.sliding_window_table: Dict[str, SlidingWindowEntry] = {}
        self.padding_images = vllm_config.model_config.is_multimodal_model

    def pad_local_table_items(
        self,
        sliding_window_table_ids: List[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        padded_cache_lengths: Optional[List[int]] = None,
        attention_masks: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.padding_images:
            assert padded_cache_lengths is not None
            assert attention_masks is not None
        else:
            assert padded_cache_lengths is None
            assert attention_masks is None

        position_id_dtype = cache_positions.dtype
        seq_len = attention_masks[0].shape[1] if attention_masks else 0

        # Determine padding value for local_block_table_id
        used_ids = set(sliding_window_table_ids)
        pad_value = next(
            (i for i in range(decoder_batch_size) if i not in used_ids), 0)

        local_block_table_id = torch.full(
            (decoder_batch_size, 1),
            pad_value,
            dtype=torch.int16,
        )
        local_block_table_id[:request_nums] = torch.tensor(
            sliding_window_table_ids, dtype=torch.int16).unsqueeze(1)
        padded_cache_positions = torch.zeros(decoder_batch_size,
                                             1,
                                             dtype=position_id_dtype)
        padded_cache_positions[:request_nums] = cache_positions[:request_nums]
        if self.padding_images:
            padded_cache_lengths_tensor = torch.zeros(decoder_batch_size,
                                                      1,
                                                      dtype=position_id_dtype)
            padded_cache_lengths_tensor[:request_nums] = torch.tensor(
                padded_cache_lengths, dtype=position_id_dtype).unsqueeze(1)

            attention_mask_dtype = (attention_masks[0].dtype
                                    if attention_masks else torch.bool)
            attention_mask = torch.zeros(decoder_batch_size,
                                         seq_len,
                                         dtype=attention_mask_dtype)
            if attention_masks:
                attention_mask[:request_nums] = torch.cat(attention_masks)

            # cache_positions:
            #  the index including padding between text and image
            # padded_cache_lengths_tensor:
            #   the size of padding
            # position_ids:
            #   the index of the token to be decoded in the sequence.
            position_ids = torch.zeros(decoder_batch_size,
                                       1,
                                       dtype=position_id_dtype)

            position_ids[:request_nums] = (
                padded_cache_positions[:request_nums] -
                padded_cache_lengths_tensor[:request_nums])
            cache_positions = padded_cache_positions
        return (
            local_block_table_id,
            cache_positions,
            position_ids if self.padding_images else None,
            attention_mask if self.padding_images else None,
        )

    def select_local_block_table_value(
        self,
        is_prompt: bool,
        input_ids: torch.Tensor,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
    ) -> Tuple[list[int], list[int], list[torch.Tensor]]:
        get_extra_values_fn = None
        attention_mask = None

        if is_prompt:
            if self.padding_images:
                attention_mask = ((input_ids != self.pad_token_id).to(
                    torch.int64).squeeze(0))
        else:
            if self.padding_images:
                get_extra_values_fn = lambda entry: (
                    entry.padded_cache_length,
                    entry.attention_mask,
                )

        result = self.get_table_mapping_values(
            self.sliding_window_table,
            decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
            get_entry_fn=lambda entry: entry.local_table_id,
            get_extra_values_fn=get_extra_values_fn,
        )

        if is_prompt:
            result = cast(list[int], result)
            table_ids = result
            return table_ids, [], [attention_mask]
        else:
            if self.padding_images:
                result = cast(Tuple[list[int], list[int], list[torch.Tensor]],
                              result)
                table_ids, padded_cache_lengths, attention_masks = result
                return table_ids, padded_cache_lengths, attention_masks
            else:
                result = cast(Tuple[list[int]], result)
                table_ids = result

            return table_ids, None, None

    def mask_attention_mask(attention_mask: torch.Tensor,
                            cache_position: torch.Tensor) -> torch.Tensor:
        """
        To enable attention for the newly generated tokens,
        set their corresponding `cache_position` values
        in the `attention_mask` to 1.
        """
        assert cache_position is not None and attention_mask is not None, \
            ("This function requires `cache_position` and `attention_mask`. "
            "Please check if the `padding_images` is True.")

        rows = torch.arange(attention_mask.shape[0])
        cols = cache_position.squeeze(1)

        attention_mask[rows, cols] = 1
        return attention_mask

    def clear_dict_table(self):
        self.sliding_window_table.clear()


class RBLNOptimumSlidingWindowAttentionForCausalLM(
        RBLNOptimumModelBase,
        RBLNOptimumDecoderMixin,
        RBLNOptimumSlidingWindowAttentionMixin,
):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            padding_value=self.padding_value,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(self.model.rbln_config,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
        )
        self.setup_sliding_window_attention_mixin(vllm_config=vllm_config)

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]
        if env.VLLM_USE_V1:
            is_prompt = model_input.is_prompt
        else:
            is_prompt = model_input.sampling_metadata.num_prompts > 0

        # In prefill phase, the length of list must be 1
        if self.padding_images:
            sliding_window_table_ids, padded_cache_lengths, attention_masks = (
                self.select_local_block_table_value(
                    is_prompt,
                    input_ids,
                    self.decoder_batch_size,
                    running_requests_ids,
                    finished_requests_ids,
                ))
        else:
            sliding_window_table_ids, _, _ = (
                self.select_local_block_table_value(
                    is_prompt,
                    input_ids,
                    self.decoder_batch_size,
                    running_requests_ids,
                    finished_requests_ids,
                ))

        kwargs = self.preprocess_for_decoder(
            is_prompt,
            block_tables,
            input_ids,
            cache_position,
        )

        padded_batch_size = kwargs.pop("padded_batch_size",
                                       self.decoder_batch_size)

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            if self.model.prefill_decoder is None:
                raise version_error
            prefill_batch_idx = sliding_window_table_ids[0]
            # attention_mask = attention_masks[0]
            local_block_table_id = torch.tensor([prefill_batch_idx],
                                                dtype=torch.int16)
            # from fpdb import ForkedPdb; ForkedPdb().set_trace()
            # return self.model.prefill_decoder(**kwargs).logits
            output = self.model.prefill_decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                attention_mask=attention_masks[0]
                if self.padding_images else None,
                local_block_tables=local_block_table_id,
                # block_tables=block_tables,
                # token_type_ids=token_type_ids,
            )
            logits = output.logits
            # FIXME
            # updated_padded_cache_length = output.padded_cache_lengths
            # print("updated_padded_cache_length", updated_padded_cache_length)
            assert len(running_requests_ids) == 1
            self.sliding_window_table[running_requests_ids[0]] = (
                SlidingWindowEntry(
                    sliding_window_table_ids[0],
                    output.padded_cache_lengths
                    if self.padding_images else None,
                    output.attention_mask if self.padding_images else None,
                ))
        else:
            # from fpdb import ForkedPdb; ForkedPdb().set_trace()
            self.model.decoder = self.model.decoders[padded_batch_size]
            (local_block_table_id, cache_position, position_ids,
             attention_mask) = self.pad_local_table_items(
                 sliding_window_table_ids,
                 cache_position,
                 request_nums,
                 padded_batch_size,
                 padded_cache_lengths if self.padding_images else None,
                 attention_masks if self.padding_images else None,
             )
            # ForkedPdb().set_trace()
            # print("cache_position", cache_position)
            logits = self.model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                # block_tables=block_tables,
                local_block_tables=local_block_table_id,
                attention_mask=attention_mask if self.padding_images else None,
                # position_ids=position_ids,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits
