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
from typing import Dict, List, Optional, Tuple, cast

import torch
from vllm.logger import init_logger

from .model_base import RBLNOptimumDictTableMixin

logger = init_logger(__name__)


@dataclass
class SlidingWindowAttentionEntry:
    local_table_id: int


@dataclass
class HybridAttentionImageEntry:
    local_table_id: int
    padded_cache_length: int
    attention_mask: torch.Tensor


class AttentionManager(RBLNOptimumDictTableMixin):

    def pad_list22dtensor(
        self,
        original_list: list[int],
        rows: int,
        cols: int,
        pad_value: int = 0,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.int16
        valid_nums = len(original_list)
        padded = torch.full((rows, cols), pad_value, dtype=dtype)
        original_tensor = torch.tensor(original_list, dtype=dtype).unsqueeze(1)
        padded[:valid_nums] = original_tensor
        return padded

    def pad_tensors2tensor(
        self,
        original_tensors: list[torch.Tensor],
        rows: int,
        cols: int,
        pad_value: int = 0,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = original_tensors[0].dtype
        valid_nums = len(original_tensors)
        padded = torch.full((rows, cols), pad_value, dtype=dtype)
        original_tensor = torch.cat(original_tensors)
        padded[:valid_nums] = original_tensor
        return padded

    def pad_tensor2tensor(
        self,
        original_tensor: torch.Tensor,
        rows: int,
        cols: int,
        pad_value: int = 0,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = original_tensor.dtype
        valid_nums = original_tensor.shape[0]
        padded = torch.full((rows, cols), pad_value, dtype=dtype)
        padded[:valid_nums] = original_tensor
        return padded


class SlidingWindowAttentionManager(AttentionManager):

    def __init__(self):
        self.sliding_window_table: Dict[str, SlidingWindowAttentionEntry] = {}

    def get(
        self,
        is_prompt: bool,
        input_ids: torch.Tensor,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
    ) -> list[int]:
        result = self.get_table_mapping_values(
            self.sliding_window_table,
            decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
            get_entry_fn=lambda entry: entry.local_table_id,
        )

        table_ids = cast(list[int], result)
        return table_ids

    def add(
        self,
        running_requests_id: str,
        local_table_id: int,
    ):
        self.sliding_window_table[running_requests_id] = \
            SlidingWindowAttentionEntry(
            local_table_id=local_table_id,
        )

    def preprocess_params(
        self,
        sliding_window_table_ids: List[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Determine padding value for local_block_table_id
        used_ids = set(sliding_window_table_ids)
        pad_value = next(
            (i for i in range(decoder_batch_size) if i not in used_ids), 0)

        local_block_table_id = self.pad_list22dtensor(sliding_window_table_ids,
                                                      decoder_batch_size, 1,
                                                      pad_value, torch.int16)
        padded_cache_positions = self.pad_tensor2tensor(
            cache_positions, decoder_batch_size, 1, 0)

        return (
            local_block_table_id,
            padded_cache_positions,
        )

    def clear_dict_table(self):
        self.sliding_window_table.clear()


class HybridAttentionImageManager(AttentionManager):

    def __init__(self, pad_token_id):
        self.hybrid_attention_table: Dict[str, HybridAttentionImageEntry] = {}
        self.pad_token_id = pad_token_id

    def get(
        self,
        is_prompt: bool,
        input_ids: torch.Tensor,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
    ) -> Tuple[list[int], list[int], list[torch.Tensor]]:
        get_extra_values_fn = None
        if is_prompt:
            attention_mask = ((input_ids != self.pad_token_id).to(
                torch.int64).squeeze(0))
        else:
            get_extra_values_fn = lambda entry: (
                entry.padded_cache_length,
                entry.attention_mask,
            )

        result = self.get_table_mapping_values(
            self.hybrid_attention_table,
            decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
            get_entry_fn=lambda entry: entry.local_table_id,
            get_extra_values_fn=get_extra_values_fn,
        )

        if is_prompt:
            table_ids = cast(list[int], result)
            return table_ids, [], [attention_mask]
        else:
            result = cast(Tuple[list[int], list[int], list[torch.Tensor]],
                          result)
            table_ids, padded_cache_lengths, attention_masks = result
            return table_ids, padded_cache_lengths, attention_masks

    def add(self,
            running_requests_id: str,
            local_table_id: int,
            padded_cache_length: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None):
        self.hybrid_attention_table[
            running_requests_id] = HybridAttentionImageEntry(
                local_table_id=local_table_id,
                padded_cache_length=padded_cache_length,
                attention_mask=attention_mask,
            )

    def preprocess_params(
        self,
        sliding_window_table_ids: List[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        padding_offsets: List[int],
        attention_masks: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert padding_offsets is not None
        assert attention_masks is not None

        position_id_dtype = cache_positions.dtype
        # Determine padding value for local_block_table_id
        used_ids = set(sliding_window_table_ids)
        pad_value = next(
            (i for i in range(decoder_batch_size) if i not in used_ids), 0)

        local_block_table_id = self.pad_list22dtensor(sliding_window_table_ids,
                                                      decoder_batch_size, 1,
                                                      pad_value, torch.int16)
        padded_padding_offsets = self.pad_list22dtensor(
            padding_offsets, decoder_batch_size, 1, 0)
        padded_cache_positions = self.pad_tensor2tensor(
            cache_positions, decoder_batch_size, 1, 0)
        padded_attention_mask = self.pad_tensors2tensor(
            attention_masks, decoder_batch_size, attention_masks[0].shape[1],
            0)

        # cache_positions:
        #  the index including padding between text and image
        # padding_offsets:
        #   the size of padding
        # position_ids:
        #   the index of the token to be decoded in the sequence.
        position_ids = torch.zeros(decoder_batch_size,
                                   1,
                                   dtype=position_id_dtype)

        position_ids = padded_cache_positions - padded_padding_offsets
        return (
            local_block_table_id,
            padded_cache_positions,
            position_ids,
            padded_attention_mask,
        )

    def update_hybrid_attention_table(self, running_requests_ids: list[str],
                                      attention_mask: torch.Tensor):
        """
        Update the sliding window table with a new attention mask.
        """
        for idx, request_id in enumerate(running_requests_ids):
            self.hybrid_attention_table[
                request_id].attention_mask = attention_mask[idx:idx + 1]

    def update_attention_mask(self, attention_mask: torch.Tensor,
                              cache_position: torch.Tensor) -> torch.Tensor:
        """
        To enable attention for the newly generated tokens,
        set their corresponding `cache_position` values
        in the `attention_mask` to 1.
        """

        rows = torch.arange(attention_mask.shape[0])
        cols = cache_position.squeeze(1)

        attention_mask[rows, cols] = 1
        return attention_mask

    def clear_dict_table(self):
        self.hybrid_attention_table.clear()
