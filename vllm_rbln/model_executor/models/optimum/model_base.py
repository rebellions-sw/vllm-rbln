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
import bisect
import os
from functools import cache
from pathlib import Path
from typing import Any, Optional, Dict, Callable, Tuple, Union

import optimum.rbln
import torch
import torch.nn as nn
from vllm.config import ModelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

from .base import get_rbln_model_info

logger = init_logger(__name__)


class RBLNOptimumModelBase(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.init_model()
        self.padding_value = self.get_padding_value()
        self.batch_size = self.scheduler_config.max_num_seqs

    def init_model(self) -> None:
        config = self.model_config.hf_config
        model_name, model_cls_name = get_rbln_model_info(config)

        if isinstance(self.model_config.model,
                      (str, Path)) and os.path.exists(self.model_config.model):
            model_path = Path(self.model_config.model)
            if model_path.is_dir() and any(model_path.glob('*.rbln')):
                compiled_path = self.model_config.model
            else:
                compiled_path = self.model_config.compiled_model_dir
        else:
            compiled_path = self.model_config.compiled_model_dir

        if compiled_path is None or not os.path.exists(compiled_path):
            raise RuntimeError(
                f"Compiled model path does not exist: {compiled_path}")

        # huggingface model class name
        logger.info("model_name = %s, model_cls_name = %s, model_path = %s",
                    model_name, model_cls_name, compiled_path)

        # huggingface model class
        model_cls = getattr(optimum.rbln, model_cls_name)
        assert model_cls is not None
        model = model_cls.from_pretrained(compiled_path, export=False)
        self.model = model
        self.rbln_model_config = model.rbln_config
        self.attn_impl = model.get_attn_impl() if hasattr(
            model, "get_attn_impl") else None

    def get_padding_value(self):
        attn_impl = self.attn_impl
        padding = -1
        if attn_impl is not None and attn_impl == "flash_attn":
            # For flash attention, the last block is the dummy block
            padding = self.model.get_kvcache_num_blocks() - 1

            if npu_num_blocks := os.environ.get("VLLM_RBLN_NPU_NUM_BLOCKS"):
                padding = int(npu_num_blocks) - 1

        return padding


class RBLNOptimumDecoderMixin:

    def setup_decoder_mixin(
        self,
        attn_impl: str,
        padding_value: int,
        vocab_size: int,
        use_multiple_decoder: bool,
        default_batch_size: int,
        decoder_batch_sizes: list[int],
    ):
        self.attn_impl = attn_impl
        self.use_multiple_decoder = use_multiple_decoder
        # FIXME: self.batch_size != self.decoder_batch_size ?
        self.decoder_batch_size = default_batch_size
        if self.use_multiple_decoder:
            self.decoder_batch_sizes = tuple(reversed(decoder_batch_sizes))

        self.logits_processor = LogitsProcessor(vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()
        self.padding_value = padding_value

    def pad_decoder_items(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
        input_block_ids: Optional[torch.Tensor] = None,
        padded_batch_size: Optional[int] = None,
    ):
        assert input_ids.shape[1] == 1
        if input_block_ids is None and padded_batch_size is None:
            raise ValueError(
                "Either input_block_ids or padded_batch_size must be provided."
            )
        elif input_block_ids is not None and padded_batch_size is not None:
            raise ValueError(
                "Cannot provide both input_block_ids and padded_batch_size.")

        if padded_batch_size is None:
            padded_batch_size = self.decoder_batch_size

        original_batch_size = input_ids.shape[0]

        padded_input_ids = torch.zeros(padded_batch_size,
                                       1,
                                       dtype=input_ids.dtype)
        padded_position_ids = torch.zeros(padded_batch_size,
                                          1,
                                          dtype=positions.dtype)
        padded_block_tables = torch.zeros(padded_batch_size,
                                          block_tables.shape[1],
                                          dtype=block_tables.dtype).fill_(
                                              self.padding_value)

        if self.attn_impl != "flash_attn":
            available_blocks = torch.arange(0,
                                            padded_batch_size,
                                            dtype=block_tables.dtype)
            mask = torch.ones(padded_batch_size, dtype=torch.bool)
            unused_blocks = available_blocks[
                ~torch.isin(available_blocks, block_tables.flatten())]

            if input_block_ids is None:
                padded_input_ids[:original_batch_size] = input_ids
                padded_position_ids[:original_batch_size] = positions
                padded_block_tables[:original_batch_size] = block_tables
                mask[:original_batch_size] = False
            else:
                padded_input_ids[input_block_ids] = input_ids
                padded_position_ids[input_block_ids] = positions
                padded_block_tables[input_block_ids] = block_tables
                mask[input_block_ids] = False

            if unused_blocks.numel() > 0:
                padded_block_tables[mask] = unused_blocks[0]

        else:
            padded_input_ids[:original_batch_size] = input_ids
            padded_position_ids[:original_batch_size] = positions
            padded_block_tables[:original_batch_size] = block_tables

        return padded_input_ids, padded_position_ids, padded_block_tables

    def preprocess_for_decoder(
        self,
        is_prompt: bool,
        block_tables: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        input_block_ids: Optional[list[int]] = None,
    ):
        padded_batch_size = None
        # 1. Set the type
        # TODO: Does it require changing the dtype dynamically?
        input_ids = input_ids.to(
            torch.int64) if input_ids is not None else None
        cache_position = cache_position.to(
            torch.int32) if cache_position is not None else None
        block_tables = block_tables.to(torch.int16)

        # 2. Adjust the shape of tensors by squeezing and padding
        if is_prompt:
            block_tables = block_tables.squeeze(0)
            padded_batch_size = 1
        else:
            if input_block_ids is None:
                padded_batch_size = self.decoder_batch_size
                if input_ids is not None:
                    request_nums = input_ids.shape[0]
                # Select lower-bounded batch size in case of multiple decoders
                if self.use_multiple_decoder:
                    padded_batch_size = self.select_lower_bounded_batch_size(
                        request_nums, self.decoder_batch_sizes)

            input_ids, cache_position, block_tables = self.pad_decoder_items(
                input_ids,
                cache_position,
                block_tables,
                input_block_ids=input_block_ids,
                padded_batch_size=padded_batch_size)

        kwargs = {
            "block_tables": block_tables,
            "padded_batch_size": padded_batch_size,
            "input_ids": input_ids,
            "cache_position": cache_position,
        }
        return kwargs

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.logits_processor(None, hidden_states, sampling_metadata)

    @classmethod
    @cache
    def select_lower_bounded_batch_size(self, original_batch_size: int,
                                        decoder_batch_sizes: tuple):
        index = bisect.bisect_left(decoder_batch_sizes, original_batch_size)
        return decoder_batch_sizes[index]

class RBLNOptimumLocalBlockTableMixin:
    def get_table_mapping_values(
        self,
        table_mapping: Dict[str, Any],
        decoder_batch_size: int,
        is_prompt: bool,
        finished_requests_ids: list[str],
        running_requests_ids: list[str],
        get_entry_fn: Optional[Callable[[Any], Any]] = None,
        get_extra_values_fn: Optional[Callable[[Any], Union[Any, Tuple[Any, ...]]]] = None,
    ) -> Union[list[int], Tuple[list[int], ...]]:
        if is_prompt:
            if finished_requests_ids:
                first_id = finished_requests_ids[0]
                first_entry = table_mapping[first_id]
                table_id = get_entry_fn(first_entry) if get_entry_fn else first_entry

                for request_id in finished_requests_ids:
                    table_mapping.pop(request_id)
            else:
                used_ids = {
                    get_entry_fn(v) if get_entry_fn else v
                    for v in table_mapping.values()
                }
                available_ids = set(range(decoder_batch_size)) - used_ids
                assert available_ids, "No available table IDs"
                table_id = min(available_ids)
            return [table_id]

        else:
            table_ids = []
            extra_values = []

            for request_id in running_requests_ids:
                entry = table_mapping[request_id]
                table_id = get_entry_fn(entry) if get_entry_fn else entry
                table_ids.append(table_id)

                if get_extra_values_fn:
                    result = get_extra_values_fn(entry)
                    if not isinstance(result, tuple):
                        result = (result,)
                    extra_values.append(result)

            if get_extra_values_fn:
                extra_values_lists = list(zip(*extra_values)) if extra_values else []
                return (table_ids, *extra_values_lists)
            return table_ids

    def clear_table_mapping(self, table_mapping: Dict[str, Any]):
        table_mapping.clear()