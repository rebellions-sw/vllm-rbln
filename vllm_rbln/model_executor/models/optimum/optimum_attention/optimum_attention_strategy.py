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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Generic, List, Optional, TypeVar,
                    Union, cast)

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class InnerAttentionEntry:
    local_table_id: int


@dataclass
class HybridAttentionImageEntry(InnerAttentionEntry):
    pad_len: int
    attention_mask: torch.Tensor  # shape: (max_seq_len)


EntryT = TypeVar("EntryT", bound=InnerAttentionEntry)
Result1T = TypeVar("Result1T")
Result2T = TypeVar("Result2T")


class AttentionStrategy(ABC, Generic[EntryT, Result1T, Result2T]):

    def __init__(self, decoder_batch_size: int):
        self.table: Dict[str, EntryT] = {}
        self.decoder_batch_size = decoder_batch_size
        self.local_block_table_ids: torch.Tensor = torch.zeros(
            self.decoder_batch_size, 1, dtype=torch.int16)
        self.cache_positions: torch.Tensor = torch.zeros(
            self.decoder_batch_size, 1, dtype=torch.int32)
        self.mask = torch.zeros(self.decoder_batch_size, 1, dtype=torch.bool)

    def add(self, running_requests_id: str, local_table_id: int) -> None:
        self.table[running_requests_id] = self.entry_factory(local_table_id)

    @abstractmethod
    def entry_factory(self) -> EntryT:
        ...

    @abstractmethod
    def get(
        self,
        is_prompt: bool,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        **kwargs,
    ) -> Result1T:
        ...

    @abstractmethod
    def preprocess(
        self,
        local_block_table_ids: List[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        **kwargs,
    ) -> Result2T:
        ...

    def clear(self):
        self.table.clear()

    def find_available_table_id(
        self,
        decoder_batch_size: int,
        finished_requests_ids: list[str],
        get_entry_fn: Callable[[Any], Any],
    ) -> int:
        """
        Find an available table ID by reusing from
        finished requests or finding a new one.
        """
        if finished_requests_ids:
            # Reuse table_id from the first finished request
            first_id = finished_requests_ids[0]
            first_entry = self.table[first_id]
            table_id = get_entry_fn(
                first_entry) if get_entry_fn else first_entry

            # Clean up finished requests from table
            for request_id in finished_requests_ids:
                self.table.pop(request_id)
            return table_id
        else:
            # Find the minimum available table_id
            used_ids = {
                get_entry_fn(v) if get_entry_fn else v
                for v in self.table.values()
            }
            available_ids = set(range(decoder_batch_size)) - used_ids
            assert available_ids, "No available table IDs"
            return min(available_ids)

    def mask_local_block_table(self, request_nums: int,
                               valid_table_ids: set[int]) -> None:
        """Fill padding positions with a valid table_id."""
        self.mask.fill_(True)
        self.mask[:request_nums] = False
        fill_value = next(iter(valid_table_ids))
        self.local_block_table_ids[self.mask] = fill_value

    def handle_prefill(
        self,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        decoder_batch_size: int,
        get_entry_fn: Callable[[Any], Any],
    ) -> torch.Tensor:
        current_request_id = running_requests_ids[0]
        table_id = self.find_available_table_id(decoder_batch_size,
                                                finished_requests_ids,
                                                get_entry_fn)

        self.local_block_table_ids[0, 0] = table_id
        self.add(current_request_id, table_id)
        return self.local_block_table_ids[0, :]  # [1, 1] -> [1]

    def copy_extra_values_to_tensors(
        self,
        entry: Any,
        index: int,
        get_extra_values_fn: Callable[[Any], Union[Any, tuple[Any, ...]]],
        extra_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        for value, tensor in zip(get_extra_values_fn(entry), extra_tensors):
            if isinstance(value, torch.Tensor):
                tensor[index].copy_(value)
            else:
                tensor[index] = value

    def handle_decode(
        self,
        running_requests_ids: list[str],
        decoder_batch_size: int,
        get_entry_fn: Callable[[Any], Any],
        get_extra_values_fn: Optional[Callable[[Any], Union[Any, tuple[Any,
                                                                       ...]]]],
        extra_tensors: Optional[tuple[torch.Tensor, ...]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        request_nums = len(running_requests_ids)
        valid_table_ids = set(range(decoder_batch_size))
        has_extra_values = get_extra_values_fn is not None

        # Copy table_ids and extra values for each running request
        for i, request_id in enumerate(running_requests_ids):
            entry = self.table[request_id]
            table_id = get_entry_fn(entry)
            self.local_block_table_ids[i, 0] = table_id
            valid_table_ids.remove(table_id)
            if has_extra_values:
                self.copy_extra_values_to_tensors(entry, i,
                                                  get_extra_values_fn,
                                                  extra_tensors)

        # Fill padding positions if needed
        if request_nums < decoder_batch_size:
            self.mask_local_block_table(request_nums, valid_table_ids)

        if has_extra_values:
            return (self.local_block_table_ids[:decoder_batch_size], *extra_tensors)
        return self.local_block_table_ids[:decoder_batch_size]

    def get_table_mapping_values(
        self,
        decoder_batch_size: int,
        is_prompt: bool,
        finished_requests_ids: list[str],
        running_requests_ids: list[str],
        get_entry_fn: Callable[[Any], Any],
        get_extra_values_fn: Optional[Callable[[Any],
                                               Union[Any, tuple[Any,
                                                                ...]]]] = None,
        extra_tensors: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if is_prompt:
            return self.handle_prefill(
                running_requests_ids,
                finished_requests_ids,
                decoder_batch_size,
                get_entry_fn,
            )
        else:
            return self.handle_decode(
                running_requests_ids,
                decoder_batch_size,
                get_entry_fn,
                get_extra_values_fn,
                extra_tensors,
            )


InnerR1 = list[int]
InnerR2 = tuple[torch.Tensor, torch.Tensor]


class InnerAttentionStrategy(AttentionStrategy[InnerAttentionEntry, InnerR1,
                                               InnerR2]):

    def __init__(self, decoder_batch_size: int):
        super().__init__(decoder_batch_size=decoder_batch_size)

    def entry_factory(self, local_table_id: int) -> InnerAttentionEntry:
        return InnerAttentionEntry(local_table_id=local_table_id)

    def get(
        self,
        is_prompt: bool,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        **kwargs,
    ) -> list[int]:
        result = self.get_table_mapping_values(
            decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
            get_entry_fn=lambda entry: entry.local_table_id,
        )

        table_ids = cast(torch.Tensor, result)
        return table_ids

    def preprocess(
        self,
        local_block_table_ids: torch.Tensor,
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            local_block_table_ids,
            cache_positions,
        )


HybridR1 = tuple[list[int], list[int], list[torch.Tensor]]
HybridR2 = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class HybridAttentionImageStrategy(AttentionStrategy[HybridAttentionImageEntry,
                                                     HybridR1, HybridR2]):

    def __init__(self, decoder_batch_size: int, pad_token_id: int,
                 max_seq_len: int):
        super().__init__(decoder_batch_size=decoder_batch_size)
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.pad_lens: torch.Tensor = torch.zeros(self.decoder_batch_size,
                                                  dtype=torch.int16)
        self.attention_masks: torch.Tensor = torch.zeros(
            self.decoder_batch_size, self.max_seq_len, dtype=torch.float32)

    def entry_factory(self, local_table_id: int) -> HybridAttentionImageEntry:
        return HybridAttentionImageEntry(local_table_id=local_table_id,
                                         pad_len=0,
                                         attention_mask=None)

    def add_extra_values(self, running_requests_id: str, pad_len: int,
                         attention_mask: torch.Tensor) -> None:
        self.table[running_requests_id].pad_len = pad_len
        self.table[running_requests_id].attention_mask = attention_mask

    def get(
        self,
        is_prompt: bool,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        **kwargs,
    ) -> tuple[list[int], list[int], list[torch.Tensor]]:
        get_extra_values_fn = None
        extra_tensors = None
        input_ids: Optional[torch.Tensor] = kwargs.get("input_ids")
        assert input_ids is not None

        if is_prompt:
            attention_mask = ((input_ids != self.pad_token_id).to(
                torch.int64).squeeze(0))
        else:
            get_extra_values_fn = lambda entry: (
                entry.pad_len,
                entry.attention_mask,
            )
            extra_tensors = (self.pad_lens[:decoder_batch_size],
                             self.attention_masks[:decoder_batch_size])

        result = self.get_table_mapping_values(
            decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
            get_entry_fn=lambda entry: entry.local_table_id,
            get_extra_values_fn=get_extra_values_fn,
            extra_tensors=extra_tensors,
        )

        if is_prompt:
            table_ids = cast(torch.Tensor, result)
            return table_ids, [], [attention_mask]
        else:
            result = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          result)
            table_ids, pad_lens, attention_masks = result
            return table_ids, pad_lens, attention_masks

    def preprocess(
        self,
        local_block_table_ids: list[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_lens: Optional[torch.Tensor] = kwargs.get("pad_lens")
        attention_masks: Optional[torch.Tensor] = kwargs.get("attention_masks")

        assert pad_lens is not None
        assert attention_masks is not None
        # FIXME: if multi batch?
        # cache_positions (decoder_batch_size, 1):
        #  the index including padding between text and image
        # pad_lens (decoder_batch_size):
        #   the size of padding
        # position_ids (decoder_batch_size, 1):
        #   the index of the token to be decoded in the sequence.
        position_ids = cache_positions - pad_lens.unsqueeze(1)
        return (
            local_block_table_ids,
            cache_positions,
            position_ids,
            attention_masks,
        )

    def update_hybrid_attention_table(self, running_requests_ids: list[str],
                                      attention_mask: torch.Tensor) -> None:
        """
        Update the sliding window table with a new attention mask.
        """
        for idx, request_id in enumerate(running_requests_ids):
            self.table[request_id].attention_mask = attention_mask[idx]

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
