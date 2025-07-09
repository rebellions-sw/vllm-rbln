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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from vllm.config import ModelConfig, SchedulerConfig
from vllm.logger import init_logger

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)

class RBLNOptimumWhisperForConditionalGeneration(RBLNOptimumModelBase, RBLNOptimumDecoderMixin):
    INVALID_TOKEN = 100
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
    ):
        super().__init__(model_config=model_config,
                         scheduler_config=scheduler_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            padding_value=self.padding_value,
            vocab_size=model_config.get_vocab_size,
            use_multiple_decoder=False,
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=[self.batch_size],
        )
        self.dec_max_seq_len = self.model_config.max_model_len
        self.dec_lengths = [0] * self.batch_size
        self.table_mapping: Dict[str, int] = {}

    def get_table_id(
        self,
        is_prompt: True,
        finished_requests_ids: bool,
        running_requests_ids: list[str],
    ) -> list[int]:
        if is_prompt:
            if finished_requests_ids:
                first_id = finished_requests_ids[0]
                table_id = self.table_mapping[first_id]

                for request_id in finished_requests_ids:
                    self.table_mapping.pop(request_id)
            else:
                used_ids = set(self.table_mapping.values())
                available_ids = set(range(self.decoder_batch_size)) - used_ids
                assert len(available_ids) > 0
                table_id = min(available_ids)
            return [table_id]
        else:
            table_ids = []
            for request_id in running_requests_ids:
                table_id = self.table_mapping[request_id]
                table_ids.append(table_id)
            return table_ids

    def pad_tensors(self, input_ids, valid_block_ids):
        # request_nums = input_ids.shape[0]
        padded_input_ids = torch.zeros(
            self.batch_size,
            input_ids.shape[1],
            dtype=input_ids.dtype
        )
        # WIP
        # padded_cache_position = torch.zeros(
        #     self.batch_size,
        #     1,
        #     dtype=cache_position.dtype
        # )
        padded_input_ids[valid_block_ids] = input_ids
        # padded_cache_position[:request_nums] = cache_position

        return padded_input_ids


    def forward(self, model_input: ModelInputForRBLN) -> torch.Tensor:
        input_ids = model_input.input_tokens
        # cache_position = model_input.input_positions
        is_prompt = model_input.sampling_metadata.num_prompts > 0
        block_tables = model_input.block_tables

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        # request_nums = input_ids.shape[0]

        table_ids = self.get_table_id(is_prompt, finished_requests_ids, running_requests_ids)
        valid_block_ids = torch.tensor(table_ids)
        model_kwargs = {}

        if is_prompt:
            try:
                input_features = model_input.multi_modal_kwargs["input_features"]
            except (AttributeError, KeyError):
                raise ValueError("Whisper requires audio data as an input.")
    
        input_ids = self.pad_tensors(input_ids, [valid_block_ids])
        cache_position = torch.zeros(
            self.batch_size,
            1,
            dtype=torch.int32
        )
        # NOTE Is it ok generate torch.zero tensor for each forward? 
        decoder_attention_mask = torch.zeros(
            self.batch_size,
            self.dec_max_seq_len,
            dtype=torch.int32
        )
        if is_prompt:
            # encoder is inplace function
            # there's no need to return the value.
            # block_tables = torch.tensor([[table_ids[0]]], dtype=torch.int16)
            self.model.encoder(input_features=input_features[0], block_tables=block_tables.squeeze(0))
            self.model.is_language_detected = True
            lm_logits = torch.zeros(
                1, 1, self.model.config.vocab_size + self.INVALID_TOKEN)
            # Set the probability of INVALID_TOKEN (the last token in
            # the logits tensor) to 1.0.
            lm_logits[0][0][-1] = 1
            self.table_mapping[running_requests_ids[0]] = table_ids[0]
            self.dec_lengths[table_ids[0]] = 0
        else:
            # extract cache_position from dec_lengths
            # breakpoint()
            input_ids[input_ids == (
                self.model.config.vocab_size + self.INVALID_TOKEN -
                1)] = self.model.config.decoder_start_token_id

            for batch_idx in valid_block_ids:
                cache_position[batch_idx] = self.dec_lengths[batch_idx]
                decoder_attention_mask[batch_idx, : cache_position[batch_idx] + 1] = 1
                self.dec_lengths[batch_idx] += 1
            # breakpoint()
            padded_block_tables = torch.zeros(self.batch_size, 1, dtype=torch.int16).fill_(self.padding_value)
            # block_tables = []
            padded_block_tables[valid_block_ids] = block_tables
            # for i, table_id in enumerate(table_ids):
            #     block_tables[i]

            block_tables = torch.tensor(block_tables, dtype=torch.int16)
            decoder_output = self.model.decoder(
                decoder_input_ids=input_ids.contiguous(),
                decoder_attention_mask=decoder_attention_mask,
                cache_position=cache_position,
                block_tables=padded_block_tables,
            )
            lm_logits = decoder_output.logits
            # breakpoint()
            lm_logits = lm_logits[valid_block_ids]
        return lm_logits